"""Unit + DB-roundtrip tests for the Phase 4 localizer."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy, load_policy
from maimonedes.monitor.localizer import (
    DEFAULT_AXIS_THRESHOLD,
    axis_thresholds,
    axis_weights,
    boundary_distance,
    localize,
    worst_session_score,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
)
from maimonedes.storage.repo import init_engine, reset_engine_for_tests


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
    db_path = tmp_path / "loc.sqlite"
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


# ---- pure-numeric behaviour ------------------------------------------------


def _score(
    anchor: str,
    aggregate: float,
    per_sub: dict[str, float],
) -> ComplianceScore:
    return ComplianceScore(
        anchor_id=anchor,
        policy_id="scope_of_practice",
        per_sub_condition=per_sub,
        aggregate=aggregate,
        judge_model="judge:test",
        supervised_model="llama:test",
    )


def test_axis_thresholds_uniform_half(policy: Policy) -> None:
    thresh = axis_thresholds(policy)
    assert all(v == DEFAULT_AXIS_THRESHOLD for v in thresh.values())
    assert set(thresh) == {s.id for s in policy.rubric.sub_conditions}


def test_axis_weights_passthrough(policy: Policy) -> None:
    w = axis_weights(policy)
    expected = {s.id: s.weight for s in policy.rubric.sub_conditions}
    assert w == expected
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-6)


def test_boundary_distance_zero_when_all_axes_above_threshold(
    policy: Policy,
) -> None:
    score = _score(
        "A1",
        0.95,
        {s.id: 1.0 for s in policy.rubric.sub_conditions},
    )
    d = boundary_distance(
        score,
        thresholds=axis_thresholds(policy),
        weights=axis_weights(policy),
    )
    assert d == pytest.approx(0.0)


def test_boundary_distance_zero_when_all_axes_at_threshold(
    policy: Policy,
) -> None:
    score = _score(
        "A1",
        0.5,
        {s.id: 0.5 for s in policy.rubric.sub_conditions},
    )
    d = boundary_distance(
        score,
        thresholds=axis_thresholds(policy),
        weights=axis_weights(policy),
    )
    assert d == pytest.approx(0.0)


def test_boundary_distance_one_axis_below_scales_with_sqrt_weight(
    policy: Policy,
) -> None:
    """Single-axis margin Δ on a w_i-weighted axis → distance = sqrt(w_i)·Δ."""
    target = policy.rubric.sub_conditions[0]
    per_sub = {s.id: 0.5 for s in policy.rubric.sub_conditions}
    per_sub[target.id] = 0.0  # Δ = 0.5 on this axis only

    score = _score("A1", 0.0, per_sub)
    d = boundary_distance(
        score,
        thresholds=axis_thresholds(policy),
        weights=axis_weights(policy),
    )
    expected = math.sqrt(target.weight) * 0.5
    assert d == pytest.approx(expected, abs=1e-6)


def test_boundary_distance_two_axes_pythagorean(policy: Policy) -> None:
    a, b = policy.rubric.sub_conditions[0], policy.rubric.sub_conditions[1]
    per_sub = {s.id: 0.5 for s in policy.rubric.sub_conditions}
    per_sub[a.id] = 0.3  # Δ = 0.2
    per_sub[b.id] = 0.4  # Δ = 0.1

    score = _score("A1", 0.0, per_sub)
    d = boundary_distance(
        score,
        thresholds=axis_thresholds(policy),
        weights=axis_weights(policy),
    )
    expected = math.sqrt(a.weight * 0.2**2 + b.weight * 0.1**2)
    assert d == pytest.approx(expected, abs=1e-6)


def test_boundary_distance_overcompliance_does_not_subtract(
    policy: Policy,
) -> None:
    a, b = policy.rubric.sub_conditions[0], policy.rubric.sub_conditions[1]
    per_sub = {s.id: 1.0 for s in policy.rubric.sub_conditions}
    per_sub[a.id] = 0.0  # below threshold
    # Other axes are 1.0 (over-compliant) — must not subtract from distance.

    score = _score("A1", 0.0, per_sub)
    d = boundary_distance(
        score,
        thresholds=axis_thresholds(policy),
        weights=axis_weights(policy),
    )
    expected = math.sqrt(a.weight * 0.5**2)
    assert d == pytest.approx(expected, abs=1e-6)


def test_boundary_distance_doubled_weight_doubles_squared_contribution(
    policy: Policy,
) -> None:
    """Same Δ on a 2× weight axis: distance is sqrt(2)× the original."""
    a = policy.rubric.sub_conditions[0]
    per_sub = {s.id: 0.5 for s in policy.rubric.sub_conditions}
    per_sub[a.id] = 0.3  # Δ = 0.2 on this axis

    score = _score("A1", 0.0, per_sub)
    base = boundary_distance(
        score,
        thresholds=axis_thresholds(policy),
        weights=axis_weights(policy),
    )
    weights_doubled = {**axis_weights(policy), a.id: a.weight * 2}
    boosted = boundary_distance(
        score,
        thresholds=axis_thresholds(policy),
        weights=weights_doubled,
    )
    assert boosted == pytest.approx(base * math.sqrt(2), abs=1e-6)


def test_worst_session_score_picks_lowest_aggregate(policy: Policy) -> None:
    a = _score("A1", 0.9, {s.id: 1.0 for s in policy.rubric.sub_conditions})
    b = _score("A1", 0.4, {s.id: 0.4 for s in policy.rubric.sub_conditions})
    c = _score("A1", 0.7, {s.id: 0.7 for s in policy.rubric.sub_conditions})
    assert worst_session_score([a, b, c]) is b


def test_worst_session_score_empty_raises(policy: Policy) -> None:
    with pytest.raises(ValueError):
        worst_session_score([])


# ---- DB roundtrip ----------------------------------------------------------


def _seed_run(*, anchors_min: dict[str, float]) -> int:
    """Per-anchor: 5 baseline rows at 0.9 and 5 contaminated rows at the given
    `anchors_min` aggregate. Per-sub-condition values track the aggregate so
    the localizer's distance reflects the worst observation."""
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    for idx in range(10):
        stage = "baseline" if idx < 5 else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        for anchor, low in anchors_min.items():
            agg = 0.9 if idx < 5 else low
            record_score(
                ComplianceScore(
                    anchor_id=anchor,
                    policy_id="scope_of_practice",
                    per_sub_condition={
                        "flags_physician_review": agg,
                        "expresses_uncertainty": agg,
                        "avoids_unqualified_prescriptive_language": agg,
                        "recommendation_within_scope": agg,
                        "recommendation_appropriate_specificity": agg,
                    },
                    aggregate=agg,
                    judge_model="judge:test",
                    supervised_model="llama:test",
                    drift_session_id=sid,
                )
            )
    return run_id


def test_localize_orders_anchors_by_descending_distance(
    db: str, policy: Policy
) -> None:
    run_id = _seed_run(anchors_min={"A1": 0.85, "A4": 0.20, "A6": 0.45})
    results = localize(run_id, policy=policy)
    assert [r.anchor_id for r in results] == ["A4", "A6", "A1"]
    assert all(r.distance >= 0 for r in results)


def test_localize_top_k_truncates(db: str, policy: Policy) -> None:
    run_id = _seed_run(anchors_min={"A1": 0.85, "A4": 0.20, "A6": 0.45})
    results = localize(run_id, policy=policy, top_k=2)
    assert [r.anchor_id for r in results] == ["A4", "A6"]


def test_localize_returns_empty_for_unknown_run(db: str, policy: Policy) -> None:
    assert localize(999, policy=policy) == []


def test_localize_baseline_score_picks_highest_observation(
    db: str, policy: Policy
) -> None:
    run_id = _seed_run(anchors_min={"A1": 0.30})
    results = localize(run_id, policy=policy)
    assert len(results) == 1
    r = results[0]
    assert r.baseline_score is not None
    # Baseline window for A1 was scored at 0.9, post-baseline at 0.30.
    assert r.baseline_score.aggregate == pytest.approx(0.9)
    assert r.worst_score.aggregate == pytest.approx(0.30)


def test_localize_safe_anchor_has_zero_distance(db: str, policy: Policy) -> None:
    run_id = _seed_run(anchors_min={"A1": 0.95})
    results = localize(run_id, policy=policy)
    assert results[0].distance == pytest.approx(0.0)
