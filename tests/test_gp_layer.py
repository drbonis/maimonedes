"""Tests for the Phase 5 GP layer fit + target sampler."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy, load_policy
from maimonedes.llm.embed_client import EmbedResponse
from maimonedes.monitor.gp_layer import (
    ComplianceGP,
    fit_compliance_gp,
    propose_targets,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.gp_fits import (
    list_gp_fits,
    record_gp_fit,
)
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import (
    get_engine,
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
    db_path = tmp_path / "gp.sqlite"
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


# ---- migration -------------------------------------------------------------


def test_migration_creates_gp_fits_table(db: str) -> None:
    insp = inspect(get_engine())
    assert "gp_fits" in insp.get_table_names()
    indexes = {ix["name"] for ix in insp.get_indexes("gp_fits")}
    assert "ix_gp_fits_policy_id" in indexes


def test_alembic_round_trip_clean(db: str) -> None:
    cfg = _alembic_cfg(db)
    command.downgrade(cfg, "0010_audit_runs")
    insp = inspect(get_engine())
    assert "gp_fits" not in insp.get_table_names()
    command.upgrade(cfg, "head")
    assert "gp_fits" in inspect(get_engine()).get_table_names()


# ---- training pipeline -----------------------------------------------------


def _seed_training_pairs(
    policy: Policy,
    *,
    anchors: tuple[str, ...] = ("A1", "A2", "A3"),
    n_per_anchor: int = 10,
) -> int:
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    payloads: list[tuple[str, int, float]] = []
    with get_session() as session:
        for anchor_id in anchors:
            for i in range(n_per_anchor):
                aggregate = 0.5 + 0.4 * (i % 5 - 2) / 4.0
                aggregate = max(0.0, min(1.0, aggregate))
                text = f"{anchor_id}-output-{i}: " + ("a " * (i + 1))
                llm = LLMCall(
                    backend_name="ollama-supervised",
                    model="llama:test",
                    request_messages_json=json.dumps([]),
                    response_content=text,
                    raw_response_json="{}",
                    prompt_tokens=10,
                    completion_tokens=10,
                    latency_ms=1.0,
                    request_hash=f"gp-hash-{anchor_id}-{i}",
                )
                session.add(llm)
                session.flush()
                payloads.append((anchor_id, llm.id, aggregate))
    for anchor_id, llm_id, aggregate in payloads:
        record_score(
            ComplianceScore(
                anchor_id=anchor_id,
                policy_id=policy.id,
                per_sub_condition={sid: aggregate for sid in sub_ids},
                aggregate=aggregate,
                judge_model="judge:test",
                supervised_model="llama:test",
                llm_call_id=llm_id,
            )
        )
    return len(payloads)


def test_fit_compliance_gp_raises_when_min_samples_unmet(
    db: str, policy: Policy
) -> None:
    fake_embed = FakeEmbedClient(default_dim=8)
    with pytest.raises(ValueError, match="at least"):
        fit_compliance_gp(embed_client=fake_embed, policy=policy, min_samples=20)


def test_fit_compliance_gp_returns_artefact(db: str, policy: Policy) -> None:
    n = _seed_training_pairs(policy)
    fake_embed = FakeEmbedClient(default_dim=16)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
    )
    assert gp.policy_id == policy.id
    assert gp.n_samples == n
    assert gp.feature_dim == 16
    assert gp.training_embeddings.shape == (n, 16)
    assert gp.kernel_repr  # non-empty repr after fit
    assert isinstance(gp.log_marginal_likelihood, float)


def test_fit_compliance_gp_caches_library_anchors(
    db: str, policy: Policy
) -> None:
    _seed_training_pairs(policy)
    fake_embed = FakeEmbedClient(default_dim=8)
    library = {"A1": "scenario A1", "A2": "scenario A2"}
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
        library_anchor_texts=library,
    )
    assert set(gp.library_anchor_embeddings) == {"A1", "A2"}
    for emb in gp.library_anchor_embeddings.values():
        assert len(emb) == 8
    # Texts also persisted on the artefact for dashboard tooltips.
    assert gp.library_anchor_texts == library


def test_fit_compliance_gp_persists_training_texts(
    db: str, policy: Policy
) -> None:
    """training_texts is populated in fit-order for tooltip rendering."""
    n = _seed_training_pairs(policy)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
    )
    assert len(gp.training_texts) == n
    assert gp.training_embeddings.shape[0] == n
    assert all(isinstance(t, str) and t for t in gp.training_texts)


def test_compliance_gp_text_fields_have_empty_defaults() -> None:
    """Old artefacts (pre-tooltip) deserialize with empty defaults so
    the dashboard can fall back to bare-label tooltips."""
    from dataclasses import fields

    from maimonedes.monitor.gp_layer import ComplianceGP

    f_map = {f.name: f for f in fields(ComplianceGP)}
    assert "training_texts" in f_map
    assert "library_anchor_texts" in f_map
    assert f_map["training_texts"].default_factory() == []
    assert f_map["library_anchor_texts"].default_factory() == {}


def test_compliance_gp_save_load_round_trip(
    db: str, policy: Policy, tmp_path: Path
) -> None:
    _seed_training_pairs(policy)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
    )
    path = gp.save(tmp_path / "gp_test.pkl")
    loaded = ComplianceGP.load(path)
    assert loaded.policy_id == gp.policy_id
    assert loaded.n_samples == gp.n_samples


# ---- propose_targets -------------------------------------------------------


def test_propose_targets_orders_by_score_descending(
    db: str, policy: Policy
) -> None:
    _seed_training_pairs(policy)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
    )
    targets = propose_targets(gp, n_targets=5, candidate_pool_size=200, seed=0)
    assert len(targets) == 5
    scores = [t.score for t in targets]
    assert scores == sorted(scores, reverse=True)


def test_propose_targets_returns_empty_for_unfit_gp(
    db: str, policy: Policy
) -> None:
    """Edge case: a synthetic 0-sample GP returns no targets."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    gp = ComplianceGP(
        policy_id=policy.id,
        trained_at=__import__("datetime").datetime.now(),
        n_samples=0,
        embedding_model="fake-embed",
        feature_dim=8,
        gp=GaussianProcessRegressor(),
        training_embeddings=np.zeros((0, 8)),
        training_aggregates=np.zeros((0,)),
        log_marginal_likelihood=0.0,
        kernel_repr="empty",
    )
    assert propose_targets(gp, n_targets=5) == []


def test_propose_targets_score_is_boundary_seeking(
    db: str, policy: Policy
) -> None:
    """score = uncertainty × max(0, 1 − 2·|0.5 − mean|).

    Peaks at mean=0.5 (the violation boundary), falls linearly to 0 at
    mean=0 or mean=1.
    """
    _seed_training_pairs(policy)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
    )
    targets = propose_targets(gp, n_targets=3, candidate_pool_size=50, seed=42)
    for t in targets:
        boundary_weight = max(0.0, 1.0 - 2.0 * abs(0.5 - t.expected_score))
        expected = t.uncertainty * boundary_weight
        assert t.score == pytest.approx(expected, rel=1e-6, abs=1e-9)


def test_boundary_seeking_score_peaks_at_half() -> None:
    from maimonedes.monitor.gp_layer import _boundary_seeking_score

    import numpy as np

    means = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    uncertainty = np.full_like(means, 0.1)
    scores = _boundary_seeking_score(uncertainty, means)
    # Peak at mean=0.5 (full weight), linearly decreasing to 0 at extremes.
    assert scores[2] == pytest.approx(0.1, abs=1e-9)
    assert scores[1] == pytest.approx(0.05, abs=1e-9)
    assert scores[3] == pytest.approx(0.05, abs=1e-9)
    assert scores[0] == pytest.approx(0.0, abs=1e-9)
    assert scores[4] == pytest.approx(0.0, abs=1e-9)


def test_stratified_seeding_distributes_across_quartiles(
    db: str, policy: Policy
) -> None:
    """A continuous score distribution skewed toward compliance: stratified
    seeding boosts the under-represented low-quartile share above the
    uniform-sampling baseline."""
    from maimonedes.monitor.gp_layer import _stratified_seed_indices

    import numpy as np

    rng = np.random.default_rng(0)
    # 100 training points: 10 in [0.1, 0.4], 90 in [0.6, 0.99] — continuous skew.
    aggregates = np.concatenate(
        [
            np.linspace(0.1, 0.4, 10),
            np.linspace(0.6, 0.99, 90),
        ]
    )
    q25 = float(np.quantile(aggregates, 0.25))
    indices = _stratified_seed_indices(aggregates, n_candidates=200, rng=rng)
    sampled_scores = aggregates[indices]
    n_low_stratified = int(np.sum(sampled_scores <= q25))
    # 25% of n_candidates is the floor for the bottom quartile under stratified.
    assert n_low_stratified >= 50  # ~50/200 = 25%, with some slack
    # Compare against uniform sampling — stratified gives strictly more
    # low-quartile representation when the distribution is skewed up.
    uniform_indices = rng.integers(0, len(aggregates), size=200)
    n_low_uniform = int(np.sum(aggregates[uniform_indices] <= q25))
    assert n_low_stratified >= n_low_uniform


def test_stratified_seeding_falls_back_to_uniform_when_few_unique_scores(
    db: str,
) -> None:
    from maimonedes.monitor.gp_layer import _stratified_seed_indices

    import numpy as np

    rng = np.random.default_rng(0)
    # All training scores identical → only one bucket → fall back to uniform.
    aggregates = np.full(100, 0.5)
    indices = _stratified_seed_indices(aggregates, n_candidates=20, rng=rng)
    assert len(indices) == 20
    assert indices.min() >= 0
    assert indices.max() < 100


def test_propose_targets_stratify_flag_affects_seeding(
    db: str, policy: Policy
) -> None:
    """With stratify=False the candidate distribution mirrors the training
    score distribution (uniform sampling); with stratify=True it does not."""
    _seed_training_pairs(policy, n_per_anchor=20)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
    )
    # Same seed, same pool size — only stratification differs.
    uniform = propose_targets(
        gp,
        n_targets=10,
        candidate_pool_size=50,
        seed=0,
        stratify_by_score=False,
    )
    stratified = propose_targets(
        gp,
        n_targets=10,
        candidate_pool_size=50,
        seed=0,
        stratify_by_score=True,
    )
    # Different proposal sets → seeding strategy is doing something.
    uniform_embeds = {tuple(t.embedding) for t in uniform}
    stratified_embeds = {tuple(t.embedding) for t in stratified}
    assert uniform_embeds != stratified_embeds


# ---- registry --------------------------------------------------------------


def test_record_and_list_gp_fits(db: str) -> None:
    a = record_gp_fit(
        path="models/gp_a.pkl",
        policy_id="scope_of_practice",
        n_samples=30,
        kernel_name="1.0 * RBF(length_scale=1.0)",
        log_marginal_likelihood=1.23,
        embedding_model="bio_clinicalbert",
    )
    b = record_gp_fit(
        path="models/gp_b.pkl",
        policy_id="scope_of_practice",
        n_samples=50,
        kernel_name="1.0 * RBF(length_scale=2.0)",
        log_marginal_likelihood=2.34,
        embedding_model="bio_clinicalbert",
    )
    rows = list_gp_fits("scope_of_practice")
    assert [r.id for r in rows] == [b, a]


# ---- CLI -------------------------------------------------------------------


def test_fit_gp_cli_writes_artefact(
    db: str,
    policy: Policy,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _seed_training_pairs(policy)
    fake_embed = FakeEmbedClient(default_dim=8)
    monkeypatch.setattr(cli, "_embed_factory", lambda s: fake_embed)
    output = tmp_path / "gp_cli.pkl"
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
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    rows = list_gp_fits(policy.id)
    assert len(rows) == 1


def test_propose_targets_cli_unknown_fit_exits_4(db: str) -> None:
    result = runner.invoke(app, ["propose-targets", "999"])
    assert result.exit_code == 4


def test_propose_targets_cli_renders_targets(
    db: str,
    policy: Policy,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _seed_training_pairs(policy)
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
    )
    path = tmp_path / "gp_cli.pkl"
    gp.save(path)
    fit_id = record_gp_fit(
        path=str(path),
        policy_id=policy.id,
        n_samples=gp.n_samples,
        kernel_name=gp.kernel_repr,
        log_marginal_likelihood=gp.log_marginal_likelihood,
        embedding_model=gp.embedding_model,
    )
    result = runner.invoke(app, ["propose-targets", str(fit_id), "--n", "3"])
    assert result.exit_code == 0, result.output
    assert "n_targets=3" in result.output
    assert result.output.count("T0") >= 3  # T01, T02, T03
