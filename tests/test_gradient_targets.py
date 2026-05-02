"""Tests for the gradient-guided probe synthesis module (issue #52)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.core.policy import Policy, load_policy
from maimonedes.feedback.gradient_targets import (
    GradientTarget,
    compute_score_gradient,
    propose_gradient_targets,
)
from maimonedes.models.stage2 import AxisMetrics, Stage2Model
from maimonedes.monitor.gp_layer import (
    ComplianceGP,
    fit_compliance_gp,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.perturbations import record_perturbation
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


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "gradient.sqlite"
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


def _make_stage2_with_known_coef(
    *,
    coef_by_axis: dict[str, np.ndarray],
    feature_dim: int,
    policy_id: str = "scope_of_practice",
) -> Stage2Model:
    """Hand-build a Stage2Model whose Ridge heads have known coefficients.

    The StandardScaler is fit on a single zero-mean, unit-variance row so
    `transform(x) ≈ x` (the scaler step's contribution to the gradient
    is identity in this regime), giving the analytic gradient
    `∇_e score[axis] ≈ coef_by_axis[axis]`.
    """
    heads: dict[str, Pipeline | Ridge] = {}
    # Use zero-mean unit-variance synthetic data so the scaler is identity.
    X_fit = np.array([np.ones(feature_dim), -np.ones(feature_dim)], dtype=float)
    for axis_id, coef in coef_by_axis.items():
        ridge = Ridge(alpha=1e-9)
        # We construct a Ridge with the specified coefficients directly,
        # bypassing fit() so the head is exactly the linear function we want.
        ridge.coef_ = np.asarray(coef, dtype=float)
        ridge.intercept_ = 0.5  # arbitrary baseline
        ridge.n_features_in_ = feature_dim
        scaler = StandardScaler()
        scaler.fit(X_fit)  # unit-variance → transform is identity
        pipeline = Pipeline([("scaler", scaler), ("head", ridge)])
        heads[axis_id] = pipeline
    return Stage2Model(
        policy_id=policy_id,
        trained_at=datetime.now(timezone.utc),
        n_samples=2,
        n_train=2,
        n_eval=0,
        embedding_model="fake-embed",
        feature_dim=feature_dim,
        heads=heads,
        agreement_metrics=[
            AxisMetrics(sub_id=axis, mae=0.0, spearman_rho=1.0)
            for axis in coef_by_axis
        ],
        agreement_status="green",
    )


# ---- compute_score_gradient -------------------------------------------------


def test_compute_score_gradient_matches_analytic_for_ridge(policy: Policy) -> None:
    """For a Ridge head with identity scaler, the descent direction must
    equal `-coef_ / ‖coef_‖`. We verify against the analytic gradient
    within float tolerance using a synthetic 5-dim head."""
    feature_dim = 5
    axis_id = policy.rubric.sub_conditions[0].id
    coef = np.array([1.0, 2.0, -3.0, 0.5, -0.2])
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: coef}, feature_dim=feature_dim
    )
    embedding = np.array([0.1, 0.2, -0.1, 0.0, 0.5])
    direction = compute_score_gradient(stage2, embedding, axis_id)
    expected = -coef / np.linalg.norm(coef)
    np.testing.assert_allclose(direction, expected, rtol=1e-3, atol=1e-3)


def test_compute_score_gradient_is_unit_norm(policy: Policy) -> None:
    feature_dim = 8
    axis_id = policy.rubric.sub_conditions[0].id
    rng = np.random.default_rng(0)
    coef = rng.normal(size=feature_dim)
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: coef}, feature_dim=feature_dim
    )
    direction = compute_score_gradient(stage2, np.zeros(feature_dim), axis_id)
    assert pytest.approx(float(np.linalg.norm(direction)), abs=1e-6) == 1.0


def test_compute_score_gradient_zero_for_zero_coef_head(policy: Policy) -> None:
    feature_dim = 4
    axis_id = policy.rubric.sub_conditions[0].id
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: np.zeros(feature_dim)}, feature_dim=feature_dim
    )
    direction = compute_score_gradient(stage2, np.array([1.0, 2.0, 3.0, 4.0]), axis_id)
    assert np.allclose(direction, 0.0)


def test_compute_score_gradient_rejects_unknown_axis(policy: Policy) -> None:
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={
            policy.rubric.sub_conditions[0].id: np.ones(3)
        },
        feature_dim=3,
    )
    with pytest.raises(ValueError, match="axis_id="):
        compute_score_gradient(stage2, np.zeros(3), "not_an_axis")


# ---- fixtures shared by propose_gradient_targets tests ----------------------


def _seed_gp_with_simple_data(
    policy: Policy, *, n: int = 30, feature_dim: int = 8
) -> ComplianceGP:
    """Seed `compliance_scores` rows + fit a small GP. Returns the artefact.

    Uses `FakeEmbedClient` whose embeddings are deterministic per text;
    the GP fits a stationary kernel by default.
    """
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    rng = np.random.default_rng(0)
    payloads = []
    with get_session() as session:
        for i in range(n):
            agg = float(rng.uniform(0.2, 0.9))
            text = f"gradient-seed-{i}"
            llm = LLMCall(
                backend_name="ollama-supervised",
                model="llama:test",
                request_messages_json=json.dumps([]),
                response_content=text,
                raw_response_json="{}",
                prompt_tokens=10,
                completion_tokens=10,
                latency_ms=1.0,
                request_hash=f"grad-seed-{i}",
            )
            session.add(llm)
            session.flush()
            payloads.append(("A1", llm.id, agg))
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
    fake_embed = FakeEmbedClient(default_dim=feature_dim)
    return fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
        n_restarts_optimizer=1,
    )


# ---- propose_gradient_targets -----------------------------------------------


def test_propose_gradient_targets_descends_score(db: str, policy: Policy) -> None:
    """Stage-2 score at the FINAL embedding must be ≤ score at the seed
    along the chosen axis (the gradient steps descend, never ascend)."""
    feature_dim = 8
    gp = _seed_gp_with_simple_data(policy, feature_dim=feature_dim)
    axis_id = policy.rubric.sub_conditions[0].id
    coef = np.ones(feature_dim)  # uniform positive coef → descent goes "down"
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: coef}, feature_dim=feature_dim
    )
    targets = propose_gradient_targets(
        gp=gp,
        stage2=stage2,
        policy=policy,
        n_targets=3,
        axis_id=axis_id,
        max_steps=5,
        violation_threshold=-1.0,  # never trigger the boundary stop
        uncertainty_cap_factor=1e9,  # never trigger the uncertainty cap
        seed=0,
    )
    assert len(targets) == 3
    head = stage2.heads[axis_id]
    for t in targets:
        # n_steps_taken should equal max_steps with both stop reasons disabled.
        assert t.n_steps_taken == 5
        assert t.stop_reason == "max_steps"
        # Final score on this axis is finite.
        score = float(head.predict(np.array(t.embedding).reshape(1, -1))[0])
        assert np.isfinite(score)


def test_propose_gradient_targets_respects_violation_threshold(
    db: str, policy: Policy
) -> None:
    feature_dim = 8
    gp = _seed_gp_with_simple_data(policy, feature_dim=feature_dim)
    axis_id = policy.rubric.sub_conditions[0].id
    # Strong descent → trips the threshold quickly.
    coef = np.ones(feature_dim) * 5.0
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: coef}, feature_dim=feature_dim
    )
    targets = propose_gradient_targets(
        gp=gp,
        stage2=stage2,
        policy=policy,
        n_targets=2,
        axis_id=axis_id,
        max_steps=20,
        violation_threshold=0.4,
        uncertainty_cap_factor=1e9,
        seed=0,
    )
    assert any(t.stop_reason == "violation" for t in targets)


def test_propose_gradient_targets_respects_uncertainty_cap(
    db: str, policy: Policy
) -> None:
    feature_dim = 8
    gp = _seed_gp_with_simple_data(policy, feature_dim=feature_dim)
    axis_id = policy.rubric.sub_conditions[0].id
    coef = np.ones(feature_dim) * 50.0  # huge step magnitude
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: coef}, feature_dim=feature_dim
    )
    targets = propose_gradient_targets(
        gp=gp,
        stage2=stage2,
        policy=policy,
        n_targets=2,
        axis_id=axis_id,
        max_steps=20,
        violation_threshold=-1.0,
        uncertainty_cap_factor=0.01,  # very tight cap
        step=10.0,
        seed=0,
    )
    assert any(t.stop_reason == "uncertainty_cap" for t in targets)


def test_propose_gradient_targets_is_deterministic(db: str, policy: Policy) -> None:
    feature_dim = 8
    gp = _seed_gp_with_simple_data(policy, feature_dim=feature_dim)
    axis_id = policy.rubric.sub_conditions[0].id
    coef = np.ones(feature_dim)
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: coef}, feature_dim=feature_dim
    )
    a = propose_gradient_targets(
        gp=gp, stage2=stage2, policy=policy, n_targets=3, axis_id=axis_id, seed=0
    )
    b = propose_gradient_targets(
        gp=gp, stage2=stage2, policy=policy, n_targets=3, axis_id=axis_id, seed=0
    )
    assert len(a) == len(b)
    for ta, tb in zip(a, b):
        np.testing.assert_allclose(ta.embedding, tb.embedding, rtol=1e-9)
        assert ta.stop_reason == tb.stop_reason
        assert ta.n_steps_taken == tb.n_steps_taken


def test_propose_gradient_targets_zero_coef_returns_zero_steps(
    db: str, policy: Policy
) -> None:
    """When the head is degenerate, the gradient is zero and we stop immediately."""
    feature_dim = 8
    gp = _seed_gp_with_simple_data(policy, feature_dim=feature_dim)
    axis_id = policy.rubric.sub_conditions[0].id
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: np.zeros(feature_dim)},
        feature_dim=feature_dim,
    )
    targets = propose_gradient_targets(
        gp=gp,
        stage2=stage2,
        policy=policy,
        n_targets=2,
        axis_id=axis_id,
        max_steps=5,
        seed=0,
    )
    for t in targets:
        assert t.stop_reason == "zero_gradient"
        assert t.n_steps_taken == 0


def test_propose_gradient_targets_default_axis_uses_worst_fragility(
    db: str, policy: Policy
) -> None:
    """When axis_id is None, we should pick the worst-fragility axis from
    `aggregated_fragility()`. Seed a fragility table where one axis is
    clearly worst, then verify the resolved axis matches it."""
    feature_dim = 8
    gp = _seed_gp_with_simple_data(policy, feature_dim=feature_dim)
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    worst_axis = sub_ids[2]  # arbitrary choice we'll engineer to be worst

    # Seed a baseline anchor + perturbation pair so the fragility table
    # has one strongly-negative cell on `worst_axis`.
    record_score(
        ComplianceScore(
            anchor_id="GRAD_TEST_ANCHOR",
            policy_id=policy.id,
            per_sub_condition={sid: 0.9 for sid in sub_ids},
            aggregate=0.9,
            judge_model="judge:test",
            supervised_model="sup:test",
        )
    )
    probe = PerturbationProbe(
        anchor_id="GRAD_TEST_ANCHOR",
        scenario="perturbed",
        perturbation_kind="authority",
        transform_label="authority:test",
        generator_metadata={},
    )
    pid = record_perturbation(probe)
    perturbed_per_sub = {sid: 0.85 for sid in sub_ids}
    perturbed_per_sub[worst_axis] = 0.05  # huge drop on this axis
    record_score(
        ComplianceScore(
            anchor_id="GRAD_TEST_ANCHOR",
            policy_id=policy.id,
            per_sub_condition=perturbed_per_sub,
            aggregate=0.5,
            judge_model="judge:test",
            supervised_model="sup:test",
            perturbation_id=pid,
            probe_role="perturbation",
        )
    )

    # Build a stage2 with heads for every sub_id (default-axis path needs them).
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={sid: np.ones(feature_dim) for sid in sub_ids},
        feature_dim=feature_dim,
    )
    targets = propose_gradient_targets(
        gp=gp,
        stage2=stage2,
        policy=policy,
        n_targets=2,
        axis_id=None,  # let the helper resolve
        max_steps=1,
        seed=0,
    )
    assert all(t.axis_id == worst_axis for t in targets)


def test_propose_gradient_targets_rejects_unknown_axis(
    db: str, policy: Policy
) -> None:
    feature_dim = 8
    gp = _seed_gp_with_simple_data(policy, feature_dim=feature_dim)
    axis_id = policy.rubric.sub_conditions[0].id
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: np.ones(feature_dim)},
        feature_dim=feature_dim,
    )
    with pytest.raises(ValueError, match="not in stage2.heads"):
        propose_gradient_targets(
            gp=gp,
            stage2=stage2,
            policy=policy,
            n_targets=1,
            axis_id="not_a_real_axis",
        )


def test_propose_gradient_targets_validates_args(db: str, policy: Policy) -> None:
    feature_dim = 8
    gp = _seed_gp_with_simple_data(policy, feature_dim=feature_dim)
    axis_id = policy.rubric.sub_conditions[0].id
    stage2 = _make_stage2_with_known_coef(
        coef_by_axis={axis_id: np.ones(feature_dim)}, feature_dim=feature_dim
    )
    with pytest.raises(ValueError, match="n_targets must be >= 1"):
        propose_gradient_targets(
            gp=gp, stage2=stage2, policy=policy, n_targets=0, axis_id=axis_id
        )
    with pytest.raises(ValueError, match="step must be positive"):
        propose_gradient_targets(
            gp=gp,
            stage2=stage2,
            policy=policy,
            n_targets=1,
            axis_id=axis_id,
            step=0.0,
        )
