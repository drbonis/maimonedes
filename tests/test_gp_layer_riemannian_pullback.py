"""Tests for the Riemannian-pullback kernel (issue #62, §4.5.3, §6.8).

The kernel warps the GP's input space by routing each embedding
through a Stage-2 head into score space and applying a learned metric
tensor at the score-space midpoint:

    k(e_i, e_j) = σ² · exp(−½ · (c_i − c_j)ᵀ · g(m_c) · (c_i − c_j) / ℓ²)
                + ε · exp(−½ · ‖e_i − e_j‖² / ℓ_E²)

These tests cover:
  - Basic mechanics (symmetry, diag, eval_gradient guard).
  - PD-after-jitter on synthetic data (acceptance criterion).
  - Ranking: a kernel built on a high-curvature metric produces larger
    posterior std at otherwise-equivalent test points than the same
    kernel under a low-curvature metric (acceptance criterion).
  - Both `mode="score"` and `mode="jacobian"` round-trip.
  - `_fit_gp_arrays(kernel="riemannian_pullback")` requires stage2 + metric.
  - ComplianceGP carries `kernel_kind="riemannian_pullback"` and the
    `record_gp_fit(kernel_kind=..., metric_fit_id=...)` columns persist.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from maimonedes.models.stage2 import Stage2Model
from maimonedes.monitor.gp_layer import (
    ComplianceGP,
    GPTarget,
    RiemannianPullbackKernel,
    _fit_gp_arrays,
    propose_targets,
    riemannian_pullback_kernel,
)
from maimonedes.monitor.metric import (
    RiemannianMetric,
    fit_metric_from_pairs,
)
from maimonedes.settings import Settings
from maimonedes.storage.gp_fits import (
    list_gp_fits,
    record_gp_fit,
)
from maimonedes.storage.metric_fits import record_metric_fit
from maimonedes.storage.repo import (
    get_engine,
    init_engine,
    reset_engine_for_tests,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "gp_riem.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


# ---------------------------------------------------------------------------
# Test fixtures: synthetic Stage-2 head + RiemannianMetric
# ---------------------------------------------------------------------------


def _make_identity_stage2(k: int, d_emb: int, *, seed: int = 0) -> Stage2Model:
    """A Stage-2 surrogate whose axis i predicts embedding coordinate i.

    Trained as `Pipeline(StandardScaler, Ridge)` per axis on synthetic
    `(X, y_i = X[:, i])` data, so f is approximately a coordinate
    projection. Sufficient for kernel-mechanics tests; the exact
    fidelity to the projection doesn't matter as long as `f` is
    deterministic and Lipschitz.
    """
    if k > d_emb:
        raise ValueError(f"need k<=d_emb; got k={k}, d_emb={d_emb}")
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(0.0, 1.0, size=(64, d_emb))
    heads: dict[str, Pipeline] = {}
    for i in range(k):
        y_train = X_train[:, i]
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("head", Ridge(alpha=1e-3, random_state=seed)),
            ]
        )
        pipeline.fit(X_train, y_train)
        heads[f"axis_{i}"] = pipeline
    return Stage2Model(
        policy_id="riem_test",
        trained_at=datetime.now(timezone.utc),
        n_samples=64,
        n_train=64,
        n_eval=0,
        embedding_model="synthetic",
        feature_dim=d_emb,
        heads=heads,
        agreement_metrics=[],
        agreement_status="green",
        head_kind="ridge",
    )


def _make_constant_g_metric(
    k: int, *, scale: float = 1.0, seed: int = 0
) -> RiemannianMetric:
    """Train a `MetricMLP` on `g_target = scale·I` everywhere.

    The trained MLP returns `g(c) ≈ scale·I` at any score-space coord —
    a constant-curvature metric. Used to vary curvature across test
    runs without introducing position-dependence (which would force
    extra fixtures).
    """
    rng = np.random.default_rng(seed)
    n = 40
    c_array = rng.uniform(0.0, 1.0, size=(n, k))
    g_target = scale * np.eye(k)
    g_array = np.tile(g_target, (n, 1, 1))
    return fit_metric_from_pairs(
        c_array,
        g_array,
        k=k,
        hidden=8,
        depth=1,
        epochs=300,
        l2=0.0,
        lr=0.05,
        seed=seed,
        policy_id="riem_test",
        n_anchors=n,
        n_jacobians=n * 4,
        sub_condition_ids=tuple(f"axis_{i}" for i in range(k)),
    )


# ---------------------------------------------------------------------------
# Kernel mechanics
# ---------------------------------------------------------------------------


def test_kernel_factory_inherits_metric_axis_order() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    kernel = riemannian_pullback_kernel(stage2_model=stage2, metric=metric)
    assert isinstance(kernel, RiemannianPullbackKernel)
    assert kernel.sub_condition_axes == ("axis_0", "axis_1")


def test_kernel_rejects_axis_missing_from_stage2() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    with pytest.raises(ValueError, match="not in stage2_model.heads"):
        RiemannianPullbackKernel(
            stage2_model=stage2,
            metric=metric,
            sub_condition_axes=("axis_0", "missing_axis"),
        )


def test_kernel_rejects_metric_k_mismatch() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=3, d_emb=4)
    with pytest.raises(ValueError, match="metric.k=2 does not match"):
        RiemannianPullbackKernel(
            stage2_model=stage2,
            metric=metric,
            sub_condition_axes=("axis_0", "axis_1", "axis_2"),
        )


def test_kernel_rejects_unknown_mode() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    with pytest.raises(ValueError, match="unknown mode"):
        RiemannianPullbackKernel(
            stage2_model=stage2,
            metric=metric,
            sub_condition_axes=("axis_0", "axis_1"),
            mode="bogus",
        )


def test_kernel_is_not_stationary() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    kernel = riemannian_pullback_kernel(stage2_model=stage2, metric=metric)
    assert kernel.is_stationary() is False


def test_kernel_diag_is_sigma_squared_plus_epsilon() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    kernel = riemannian_pullback_kernel(
        stage2_model=stage2, metric=metric, sigma=2.0, epsilon=1e-2
    )
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(5, 4))
    diag = kernel.diag(X)
    assert diag.shape == (5,)
    # k(e, e) = σ²·exp(0) + ε·exp(0) = σ² + ε
    assert np.allclose(diag, 4.0 + 1e-2)


def test_kernel_call_is_symmetric_score_mode() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    kernel = riemannian_pullback_kernel(stage2_model=stage2, metric=metric)
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(6, 4))
    K = kernel(X)
    assert K.shape == (6, 6)
    assert np.allclose(K, K.T, atol=1e-10)
    assert np.allclose(np.diag(K), kernel.diag(X), atol=1e-10)
    assert np.isfinite(K).all()


def test_kernel_call_is_symmetric_jacobian_mode() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=3)
    kernel = riemannian_pullback_kernel(
        stage2_model=stage2, metric=metric, mode="jacobian"
    )
    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, size=(5, 3))
    K = kernel(X)
    assert K.shape == (5, 5)
    assert np.allclose(K, K.T, atol=1e-9)
    assert np.allclose(np.diag(K), kernel.diag(X), atol=1e-9)
    assert np.isfinite(K).all()


def test_kernel_call_two_input_arrays() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    kernel = riemannian_pullback_kernel(stage2_model=stage2, metric=metric)
    rng = np.random.default_rng(3)
    X = rng.uniform(0.0, 1.0, size=(4, 4))
    Y = rng.uniform(0.0, 1.0, size=(3, 4))
    K = kernel(X, Y)
    assert K.shape == (4, 3)
    assert np.isfinite(K).all()
    # All entries non-negative and bounded above by σ² + ε.
    assert (K >= 0.0).all()
    assert (K <= kernel.sigma ** 2 + kernel.epsilon + 1e-9).all()


def test_kernel_eval_gradient_raises() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    kernel = riemannian_pullback_kernel(stage2_model=stage2, metric=metric)
    X = np.zeros((2, 4))
    with pytest.raises(NotImplementedError, match="externally"):
        kernel(X, eval_gradient=True)


# ---------------------------------------------------------------------------
# Acceptance criterion 1: PD-after-jitter on synthetic data
# ---------------------------------------------------------------------------


def test_kernel_gram_is_pd_with_alpha_jitter_score_mode() -> None:
    """K + α·I admits a Cholesky factorization on a non-trivial Gram."""
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=5)
    kernel = riemannian_pullback_kernel(
        stage2_model=stage2, metric=metric, sigma=1.0, epsilon=1e-3
    )
    rng = np.random.default_rng(7)
    X = rng.uniform(0.0, 1.0, size=(20, 5))
    K = kernel(X) + 1e-2 * np.eye(20)
    np.linalg.cholesky(K)  # raises on non-PD


def test_kernel_gram_is_pd_with_alpha_jitter_jacobian_mode() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    kernel = riemannian_pullback_kernel(
        stage2_model=stage2,
        metric=metric,
        mode="jacobian",
        sigma=1.0,
        epsilon=1e-3,
    )
    rng = np.random.default_rng(8)
    X = rng.uniform(0.0, 1.0, size=(15, 4))
    K = kernel(X) + 1e-2 * np.eye(15)
    np.linalg.cholesky(K)


# ---------------------------------------------------------------------------
# Acceptance criterion 2: high-curvature ranking
# ---------------------------------------------------------------------------


def test_high_curvature_metric_yields_larger_posterior_std_off_training() -> None:
    """The point of the riemannian-pullback kernel is that high curvature
    in score space stretches the kernel's effective bandwidth: distinct
    embeddings whose score-space images differ get pushed further apart
    under a high-`g` metric than under a low-`g` metric. The downstream
    consequence in active learning is that a test point well separated
    from training in score space gets a larger GP posterior std under
    the high-curvature kernel — so candidate-target proposers see it as
    more uncertain and (when near the boundary) rank it higher.

    We verify the underlying mechanism: same training set, same Stage-2
    head, same kernel hyperparameters; only the metric scale changes.
    The high-scale metric yields strictly larger posterior std at a
    held-out boundary point than the low-scale metric.
    """
    d_emb = 4
    k = 2
    stage2 = _make_identity_stage2(k=k, d_emb=d_emb)

    # Spread training embeddings across [0, 1]^d.
    rng = np.random.default_rng(13)
    n_train = 18
    X_train = rng.uniform(0.0, 1.0, size=(n_train, d_emb))
    # y_train: Stage-2 aggregate (mean of first k coordinates) + noise.
    y_train = X_train[:, :k].mean(axis=1) + rng.normal(0.0, 0.02, size=n_train)

    metric_low = _make_constant_g_metric(k=k, scale=0.5, seed=0)
    metric_high = _make_constant_g_metric(k=k, scale=20.0, seed=0)

    kernel_low = riemannian_pullback_kernel(
        stage2_model=stage2,
        metric=metric_low,
        sigma=1.0,
        length_scale=1.0,
        epsilon=1e-3,
    )
    kernel_high = riemannian_pullback_kernel(
        stage2_model=stage2,
        metric=metric_high,
        sigma=1.0,
        length_scale=1.0,
        epsilon=1e-3,
    )

    gp_low = _fit_gp_arrays(X_train, y_train, kernel=kernel_low)
    gp_high = _fit_gp_arrays(X_train, y_train, kernel=kernel_high)

    # Probe at a held-out point near the boundary in score space.
    X_test = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=float)
    _, std_low = gp_low.predict(X_test, return_std=True)
    _, std_high = gp_high.predict(X_test, return_std=True)

    # High-curvature metric → larger Mahalanobis distance to training
    # points → less effective coverage → larger posterior std.
    assert std_high[0] > std_low[0], (
        f"expected std_high ({std_high[0]:.4f}) > std_low "
        f"({std_low[0]:.4f}) under higher-scale metric"
    )


# ---------------------------------------------------------------------------
# `_fit_gp_arrays` integration
# ---------------------------------------------------------------------------


def test_fit_gp_arrays_string_kernel_requires_stage2_and_metric() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(10, 4))
    y = rng.uniform(0.2, 0.8, size=10)
    with pytest.raises(ValueError, match="requires both `stage2_model`"):
        _fit_gp_arrays(X, y, kernel="riemannian_pullback")


def test_fit_gp_arrays_riemannian_pullback_string_path() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(12, 4))
    y = X[:, :2].mean(axis=1) + rng.normal(0.0, 0.02, size=12)
    gp = _fit_gp_arrays(
        X,
        y,
        kernel="riemannian_pullback",
        stage2_model=stage2,
        metric=metric,
    )
    assert isinstance(gp.kernel_, RiemannianPullbackKernel)
    assert gp.optimizer is None  # external-fit path; sklearn's optimizer disabled


def test_fit_gp_arrays_riemannian_pullback_instance_path() -> None:
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    kernel = riemannian_pullback_kernel(stage2_model=stage2, metric=metric)
    rng = np.random.default_rng(2)
    X = rng.uniform(0.0, 1.0, size=(12, 4))
    y = X[:, :2].mean(axis=1)
    gp = _fit_gp_arrays(X, y, kernel=kernel)
    assert isinstance(gp.kernel_, RiemannianPullbackKernel)
    assert gp.optimizer is None


def test_propose_targets_works_under_riemannian_pullback() -> None:
    """Smoke: posterior `predict()` and `propose_targets()` round-trip without
    requiring scaler/PCA preprocessing — the riemannian path stores raw
    embeddings as `training_embeddings` and skips the dim-reduction path."""
    metric = _make_constant_g_metric(k=2)
    stage2 = _make_identity_stage2(k=2, d_emb=4)
    rng = np.random.default_rng(5)
    X = rng.uniform(0.0, 1.0, size=(15, 4))
    y = X[:, :2].mean(axis=1)
    gp_inner = _fit_gp_arrays(
        X,
        y,
        kernel="riemannian_pullback",
        stage2_model=stage2,
        metric=metric,
    )
    cgp = ComplianceGP(
        policy_id="riem_test",
        trained_at=datetime.now(timezone.utc),
        n_samples=len(X),
        embedding_model="synthetic",
        feature_dim=X.shape[1],
        gp=gp_inner,
        training_embeddings=X,
        training_aggregates=y,
        log_marginal_likelihood=float(gp_inner.log_marginal_likelihood_value_),
        kernel_repr=str(gp_inner.kernel_),
        scaler=None,
        pca=None,
        kernel_kind="riemannian_pullback",
        metric_fit_id=None,
    )
    mean, std = cgp.predict(X[:3])
    assert mean.shape == (3,)
    assert std.shape == (3,)
    assert np.isfinite(mean).all()
    assert np.isfinite(std).all()

    targets = propose_targets(cgp, n_targets=3, candidate_pool_size=20, seed=0)
    assert len(targets) == 3
    for t in targets:
        assert isinstance(t, GPTarget)
        assert len(t.embedding) == X.shape[1]
        assert np.isfinite(t.expected_score)
        assert np.isfinite(t.uncertainty)
        assert t.score >= 0.0


# ---------------------------------------------------------------------------
# Persistence acceptance criterion: kernel_kind + metric_fit_id columns
# ---------------------------------------------------------------------------


def test_record_gp_fit_persists_kernel_kind_and_metric_fit_id(db: str) -> None:
    metric_fit_id = record_metric_fit(
        policy_id="riem_test",
        path="/tmp/fake_metric.npz",
        n_anchors=5,
        n_jacobians=20,
        val_loss=0.01,
        train_loss=0.005,
        hyperparams={"hidden": 8, "depth": 1},
    )
    fit_id = record_gp_fit(
        path="/tmp/fake_gp.pkl",
        policy_id="riem_test",
        n_samples=20,
        kernel_name="RiemannianPullbackKernel(...)",
        log_marginal_likelihood=-3.21,
        embedding_model="synthetic",
        kernel_kind="riemannian_pullback",
        metric_fit_id=metric_fit_id,
    )
    rows = list_gp_fits(policy_id="riem_test")
    assert len(rows) == 1
    row = rows[0]
    assert row.id == fit_id
    assert row.kernel_kind == "riemannian_pullback"
    assert row.metric_fit_id == metric_fit_id


def test_record_gp_fit_defaults_kernel_kind_to_stationary(db: str) -> None:
    """Backward compat: existing call sites that don't pass `kernel_kind`
    still work and the column defaults to 'stationary'."""
    fit_id = record_gp_fit(
        path="/tmp/fake_gp.pkl",
        policy_id="riem_test",
        n_samples=20,
        kernel_name="ConstantKernel(1.0) * RBF(length_scale=1.0)",
        log_marginal_likelihood=-2.5,
        embedding_model="synthetic",
    )
    rows = list_gp_fits(policy_id="riem_test")
    assert len(rows) == 1
    row = rows[0]
    assert row.id == fit_id
    assert row.kernel_kind == "stationary"
    assert row.metric_fit_id is None
