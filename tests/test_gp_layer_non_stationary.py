"""Tests for the non-stationary Gibbs kernel (issue #49).

These tests exercise the kernel itself, the external L-BFGS-B fit, and
the `--kernel non_stationary` plumbing through `fit_compliance_gp` and
the `fit-gp` CLI.

The acceptance-criteria test is
`test_non_stationary_lml_higher_than_stationary_on_two_region_data`:
on a synthetic dataset with anisotropic compliance geometry — sharp
score variation in one region of input space, smooth in another —
the Gibbs kernel must achieve strictly higher log-marginal-likelihood
than the stationary RBF on the same data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sklearn.gaussian_process.kernels import RBF
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy, load_policy
from maimonedes.monitor.gp_layer import (
    GibbsKernel,
    _fit_gibbs_hyperparameters,
    _fit_gp_arrays,
    fit_compliance_gp,
    non_stationary_kernel,
    quartile_diagnostics,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import (
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from tests.fakes import FakeEmbedClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"
PROBES_PATH = PROJECT_ROOT / "config" / "probes" / "anchors_v1.yaml"

runner = CliRunner()


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "gp_ns.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


@pytest.fixture
def policy() -> Policy:
    return load_policy(POLICY_PATH, RUBRIC_PATH)


def _two_region_data(
    n: int = 100, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 2D dataset with anisotropic score response.

    Sharp tanh transition in the right half of input space (x[0] > 0),
    gentle transition in the left half. A stationary RBF must compromise
    its single length-scale; a non-stationary kernel can adapt.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    sharpness = np.where(X[:, 0] > 0, 6.0, 0.8)
    y = 0.5 + 0.45 * np.tanh(sharpness * X[:, 0])
    y = y + rng.normal(0.0, 0.02, size=n)
    return X, y


# ---- GibbsKernel core --------------------------------------------------------


def test_gibbs_kernel_is_not_stationary() -> None:
    assert GibbsKernel().is_stationary() is False


def test_gibbs_kernel_diag_is_sigma_squared() -> None:
    kernel = GibbsKernel(sigma=2.0, a=0.5, b=0.3, c=-0.4)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [-0.5, 0.5]])
    diag = kernel.diag(X)
    assert diag.shape == (3,)
    assert np.allclose(diag, 4.0)  # sigma^2 = 4


def test_gibbs_kernel_call_is_symmetric_and_diag_matches_diag() -> None:
    kernel = GibbsKernel(sigma=1.5, a=0.0, b=0.4, c=-0.2)
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(8, 3))
    K = kernel(X)
    assert K.shape == (8, 8)
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.allclose(np.diag(K), kernel.diag(X), atol=1e-10)


def test_gibbs_kernel_call_two_input_arrays() -> None:
    kernel = GibbsKernel(sigma=1.0, a=0.0, b=0.2, c=-0.1)
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    Y = np.array([[0.5, 0.5], [-0.5, -0.5]])
    K = kernel(X, Y)
    assert K.shape == (3, 2)
    # All entries must be finite and within [0, σ²] (kernel is bounded above by σ²).
    assert np.isfinite(K).all()
    assert (K > 0.0).all()
    assert (K <= kernel.sigma ** 2 + 1e-9).all()


def test_gibbs_kernel_eval_gradient_raises() -> None:
    kernel = GibbsKernel()
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(NotImplementedError, match="externally optimized"):
        kernel(X, eval_gradient=True)


def test_gibbs_kernel_handles_1d_input() -> None:
    """ℓ(x) gracefully drops the c·x[1] term when feature_dim < 2."""
    kernel = GibbsKernel(sigma=1.0, a=0.0, b=0.5, c=0.7)
    X = np.array([[0.5], [1.0], [-0.5]])
    K = kernel(X)
    assert K.shape == (3, 3)
    assert np.isfinite(K).all()


def test_gibbs_kernel_kernel_matrix_is_psd_under_alpha_jitter() -> None:
    """K + α·I admits a Cholesky factorization for diverse hyperparameters."""
    rng = np.random.default_rng(7)
    X = rng.normal(0.0, 1.0, size=(15, 4))
    for sigma, a, b, c in [
        (1.0, 0.0, 0.0, 0.0),  # essentially stationary
        (0.5, 0.5, 0.5, -0.5),
        (3.0, -1.0, -0.3, 0.2),
    ]:
        kernel = GibbsKernel(sigma=sigma, a=a, b=b, c=c)
        K = kernel(X) + 1e-2 * np.eye(15)
        np.linalg.cholesky(K)  # raises if not PD


def test_gibbs_kernel_repr_shows_hyperparameters() -> None:
    rep = repr(GibbsKernel(sigma=1.23, a=0.45, b=-0.67, c=0.89))
    assert "GibbsKernel" in rep
    for token in ["1.23", "0.45", "-0.67", "0.89"]:
        assert token in rep


def test_non_stationary_kernel_factory_returns_gibbs() -> None:
    k = non_stationary_kernel(sigma=2.0, a=0.1)
    assert isinstance(k, GibbsKernel)
    assert k.sigma == 2.0
    assert k.a == 0.1


# ---- External hyperparameter optimization ------------------------------------


def test_fit_gibbs_hyperparameters_returns_finite_optimum() -> None:
    X, y = _two_region_data(n=60, seed=0)
    y_norm = (y - np.mean(y)) / (np.std(y) or 1.0)
    sigma, a, b, c, lml = _fit_gibbs_hyperparameters(
        X, y_norm, alpha=1e-2, n_restarts=2, seed=0
    )
    for v in (sigma, a, b, c, lml):
        assert np.isfinite(v)
    # σ must respect bounds; a, b, c stay in their declared ranges.
    assert 0.01 <= sigma <= 100.0
    assert -3.0 <= a <= 3.0
    assert -1.0 <= b <= 1.0
    assert -1.0 <= c <= 1.0


def test_fit_gibbs_hyperparameters_is_deterministic_for_same_seed() -> None:
    X, y = _two_region_data(n=60, seed=0)
    y_norm = (y - np.mean(y)) / (np.std(y) or 1.0)
    a_run = _fit_gibbs_hyperparameters(
        X, y_norm, alpha=1e-2, n_restarts=2, seed=0
    )
    b_run = _fit_gibbs_hyperparameters(
        X, y_norm, alpha=1e-2, n_restarts=2, seed=0
    )
    for x, y_ in zip(a_run, b_run):
        assert x == pytest.approx(y_, rel=1e-9, abs=1e-12)


# ---- Acceptance criterion: LML comparison on two-region data -----------------


def test_non_stationary_lml_higher_than_stationary_on_two_region_data() -> None:
    """Issue #49 acceptance criterion: on anisotropic data the Gibbs
    kernel achieves strictly higher log-marginal likelihood than the
    default stationary RBF using the same alpha and same y normalisation.
    """
    X, y = _two_region_data(n=100, seed=42)
    gp_stat = _fit_gp_arrays(X, y, kernel="stationary", n_restarts_optimizer=2)
    gp_ns = _fit_gp_arrays(X, y, kernel="non_stationary", n_restarts_optimizer=2)

    lml_stat = float(gp_stat.log_marginal_likelihood_value_)
    lml_ns = float(gp_ns.log_marginal_likelihood_value_)
    assert lml_ns > lml_stat, (
        f"non-stationary LML {lml_ns:.3f} should exceed stationary {lml_stat:.3f}"
    )


def test_non_stationary_posterior_std_drops_at_training_points() -> None:
    """Sanity check: predictive σ at a training point is smaller than at
    a far-away test point (uncertainty should grow away from data)."""
    X, y = _two_region_data(n=80, seed=11)
    gp = _fit_gp_arrays(X, y, kernel="non_stationary", n_restarts_optimizer=1)

    # Training point.
    _, std_train = gp.predict(X[:1], return_std=True)
    # Far-away test point — well outside the [-1.5, 1.5] training cube.
    far = np.array([[10.0, -10.0]])
    _, std_far = gp.predict(far, return_std=True)
    assert std_train[0] < std_far[0]


def test_fit_gp_arrays_uses_external_optimizer_for_non_stationary() -> None:
    """The fitted GaussianProcessRegressor for `kernel="non_stationary"`
    must carry a GibbsKernel; sklearn's L-BFGS path is bypassed
    (optimizer=None) and our external fit is the source of σ, a, b, c."""
    X, y = _two_region_data(n=50, seed=3)
    gp = _fit_gp_arrays(X, y, kernel="non_stationary", n_restarts_optimizer=1)
    assert isinstance(gp.kernel_, GibbsKernel)
    assert gp.optimizer is None


def test_fit_gp_arrays_rejects_unknown_kernel_string() -> None:
    X, y = _two_region_data(n=20, seed=0)
    with pytest.raises(ValueError, match="unknown kernel selector"):
        _fit_gp_arrays(X, y, kernel="hyperbolic")


def test_fit_gp_arrays_keeps_stationary_default() -> None:
    X, y = _two_region_data(n=30, seed=0)
    gp = _fit_gp_arrays(X, y, kernel=None)
    assert not isinstance(gp.kernel_, GibbsKernel)
    assert isinstance(gp.kernel_.k2, RBF)


# ---- Integration through fit_compliance_gp -----------------------------------


def _seed_two_region_pairs(
    policy: Policy,
    *,
    n: int = 60,
    seed: int = 0,
) -> int:
    """Seed `compliance_scores` rows that follow the two-region structure
    when their texts are embedded by `FakeEmbedClient` (which deterministically
    embeds text → vector). The texts encode the synthetic (x, y) so the GP
    fit pipeline sees the anisotropic geometry end to end."""
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    rng = np.random.default_rng(seed)
    payloads: list[tuple[str, int, float]] = []
    with get_session() as session:
        for i in range(n):
            x0 = float(rng.uniform(-1.5, 1.5))
            x1 = float(rng.uniform(-1.5, 1.5))
            sharp = 6.0 if x0 > 0 else 0.8
            y_val = 0.5 + 0.45 * float(np.tanh(sharp * x0))
            y_val = max(0.0, min(1.0, y_val + float(rng.normal(0, 0.02))))
            text = f"region-x0={x0:.4f}-x1={x1:.4f}-i={i}"
            llm = LLMCall(
                backend_name="ollama-supervised",
                model="llama:test",
                request_messages_json=json.dumps([]),
                response_content=text,
                raw_response_json="{}",
                prompt_tokens=10,
                completion_tokens=10,
                latency_ms=1.0,
                request_hash=f"ns-hash-{i}",
            )
            session.add(llm)
            session.flush()
            payloads.append(("A1", llm.id, y_val))
    for anchor_id, llm_id, agg in payloads:
        record_score(
            ComplianceScore(
                anchor_id=anchor_id,
                policy_id=policy.id,
                per_sub_condition={sid: agg for sid in sub_ids},
                aggregate=agg,
                judge_model="judge:test",
                supervised_model="llama:test",
                llm_call_id=llm_id,
            )
        )
    return len(payloads)


def test_fit_compliance_gp_accepts_non_stationary_kernel_string(
    db: str, policy: Policy
) -> None:
    _seed_two_region_pairs(policy, n=40)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
        kernel="non_stationary",
        n_restarts_optimizer=1,
    )
    assert "GibbsKernel" in gp.kernel_repr


def test_fit_compliance_gp_accepts_stationary_kernel_string(
    db: str, policy: Policy
) -> None:
    _seed_two_region_pairs(policy, n=40)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
        kernel="stationary",
        n_restarts_optimizer=1,
    )
    assert "GibbsKernel" not in gp.kernel_repr


def test_fit_compliance_gp_rejects_unknown_kernel_name(
    db: str, policy: Policy
) -> None:
    _seed_two_region_pairs(policy, n=40)
    fake_embed = FakeEmbedClient(default_dim=8)
    with pytest.raises(ValueError, match="unknown kernel selector"):
        fit_compliance_gp(
            embed_client=fake_embed,
            policy=policy,
            embedding_model="fake-embed",
            min_samples=10,
            kernel="not_a_kernel",
        )


# ---- Quartile diagnostic -----------------------------------------------------


def test_quartile_diagnostics_reports_four_quartiles_for_non_stationary_fit(
    db: str, policy: Policy
) -> None:
    _seed_two_region_pairs(policy, n=80)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
        kernel="non_stationary",
        n_restarts_optimizer=1,
    )
    rows = quartile_diagnostics(gp)
    assert len(rows) == 4
    assert [r.quartile for r in rows] == [1, 2, 3, 4]
    # n must sum to the total training set.
    assert sum(r.n for r in rows) == gp.n_samples
    # ell_mean must be strictly positive everywhere.
    assert all(r.ell_mean > 0.0 for r in rows)


def test_quartile_diagnostics_falls_back_for_stationary_kernel(
    db: str, policy: Policy
) -> None:
    """For a non-Gibbs kernel, ell_mean is the kernel's scalar length_scale —
    not informative as a non-stationarity diagnostic, but the report should
    still produce four valid quartile rows."""
    _seed_two_region_pairs(policy, n=80)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
        kernel="stationary",
        n_restarts_optimizer=1,
    )
    rows = quartile_diagnostics(gp)
    assert len(rows) == 4
    # Stationary kernel → ell is a constant scalar across all quartiles.
    ell_values = {round(r.ell_mean, 6) for r in rows}
    assert len(ell_values) == 1


# ---- CLI ---------------------------------------------------------------------


def test_fit_gp_cli_accepts_kernel_non_stationary(
    db: str,
    policy: Policy,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _seed_two_region_pairs(policy, n=40)
    fake_embed = FakeEmbedClient(default_dim=8)
    monkeypatch.setattr(cli, "_embed_factory", lambda s: fake_embed)
    output = tmp_path / "gp_ns_cli.pkl"
    result = runner.invoke(
        app,
        [
            "fit-gp",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--output",
            str(output),
            "--min-samples",
            "10",
            "--kernel",
            "non_stationary",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "GibbsKernel" in result.output
    assert output.exists()


def test_fit_gp_cli_diagnose_emits_quartile_table(
    db: str,
    policy: Policy,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _seed_two_region_pairs(policy, n=80)
    fake_embed = FakeEmbedClient(default_dim=8)
    monkeypatch.setattr(cli, "_embed_factory", lambda s: fake_embed)
    output = tmp_path / "gp_ns_diag.pkl"
    result = runner.invoke(
        app,
        [
            "fit-gp",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--output",
            str(output),
            "--min-samples",
            "10",
            "--kernel",
            "non_stationary",
            "--diagnose",
        ],
    )
    assert result.exit_code == 0, result.output
    for header_token in ["pc1_mean", "ell_mean", "sigma_pred", "score_var"]:
        assert header_token in result.output
    # Four quartile rows expected.
    for tag in ["Q1", "Q2", "Q3", "Q4"]:
        assert tag in result.output


def test_fit_gp_cli_rejects_unknown_kernel_name(
    db: str,
    policy: Policy,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_embed = FakeEmbedClient(default_dim=8)
    monkeypatch.setattr(cli, "_embed_factory", lambda s: fake_embed)
    result = runner.invoke(
        app,
        [
            "fit-gp",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--output",
            str(tmp_path / "x.pkl"),
            "--min-samples",
            "10",
            "--kernel",
            "matern",
        ],
    )
    assert result.exit_code == 5
