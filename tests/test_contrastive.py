"""Tests for the Phase 4 contrastive pair extractor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.core.policy import Policy, load_policy
from maimonedes.feedback.contrastive import (
    MISSING_TEXT,
    fragility_pair,
    temporal_pair,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
)
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.perturbations import record_perturbation
from maimonedes.storage.repo import (
    get_session,
    init_engine,
    reset_engine_for_tests,
)


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
    db_path = tmp_path / "contrastive.sqlite"
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


def _record_llm_call(content: str) -> int:
    """Insert a stub `llm_calls` row and return its id."""
    with get_session() as session:
        row = LLMCall(
            backend_name="stub",
            model="stub-model",
            request_messages_json=json.dumps([]),
            response_content=content,
            raw_response_json="{}",
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=0.0,
            request_hash="abc",
        )
        session.add(row)
        session.flush()
        return row.id


def _score(
    anchor: str,
    aggregate: float,
    *,
    drift_session_id: int | None = None,
    llm_call_id: int | None = None,
    per_sub: dict[str, float] | None = None,
) -> ComplianceScore:
    return ComplianceScore(
        anchor_id=anchor,
        policy_id="scope_of_practice",
        per_sub_condition=per_sub
        or {
            "flags_physician_review": aggregate,
            "expresses_uncertainty": aggregate,
            "avoids_unqualified_prescriptive_language": aggregate,
            "recommendation_within_scope": aggregate,
            "recommendation_appropriate_specificity": aggregate,
        },
        aggregate=aggregate,
        judge_model="judge:test",
        supervised_model="llama:test",
        drift_session_id=drift_session_id,
        llm_call_id=llm_call_id,
    )


# ---- temporal_pair ---------------------------------------------------------


def test_temporal_pair_picks_latest_baseline_and_worst_post(db: str) -> None:
    """One anchor; 3 baseline sessions then 3 contaminated sessions."""
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="m",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    safe_call = _record_llm_call("This requires physician review.")
    drift_call = _record_llm_call("Take 10mg of metformin daily.")
    for idx in range(6):
        stage = "baseline" if idx < 3 else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        if idx < 3:
            record_score(_score("A1", 0.9, drift_session_id=sid, llm_call_id=safe_call))
        else:
            record_score(_score("A1", 0.3, drift_session_id=sid, llm_call_id=drift_call))

    pair = temporal_pair("A1", drift_run_id=run_id)
    assert pair is not None
    assert pair.kind == "temporal"
    assert pair.anchor_id == "A1"
    assert pair.safe_score.aggregate == pytest.approx(0.9)
    assert pair.near_boundary_score.aggregate == pytest.approx(0.3)
    assert pair.safe_text == "This requires physician review."
    assert pair.near_boundary_text == "Take 10mg of metformin daily."
    assert pair.dropoff_axes  # non-empty


def test_temporal_pair_returns_none_without_baseline(db: str) -> None:
    """Drift run with only contaminated stages → no temporal pair."""
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="m",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    sid = create_drift_session(run_id, 0, "concise", "")
    record_score(_score("A1", 0.3, drift_session_id=sid))
    assert temporal_pair("A1", drift_run_id=run_id) is None


def test_temporal_pair_returns_none_for_unknown_anchor(db: str) -> None:
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="m",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    sid = create_drift_session(run_id, 0, "baseline", "")
    record_score(_score("A1", 0.9, drift_session_id=sid))
    assert temporal_pair("A99", drift_run_id=run_id) is None


def test_temporal_pair_legacy_rows_use_missing_text_sentinel(db: str) -> None:
    """Score with `llm_call_id is None` → text falls back to MISSING_TEXT."""
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="m",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    for idx in range(4):
        stage = "baseline" if idx < 2 else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        agg = 0.9 if idx < 2 else 0.3
        record_score(_score("A1", agg, drift_session_id=sid, llm_call_id=None))

    pair = temporal_pair("A1", drift_run_id=run_id)
    assert pair is not None
    assert pair.safe_text == MISSING_TEXT
    assert pair.near_boundary_text == MISSING_TEXT
    # Scalar scores remain valid even without text.
    assert pair.safe_score.aggregate == pytest.approx(0.9)
    assert pair.near_boundary_score.aggregate == pytest.approx(0.3)


def test_temporal_pair_returns_none_when_safe_equals_near(db: str) -> None:
    """Run with no drift (all scores identical) → can't contrast a row with itself."""
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="m",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    for idx in range(4):
        stage = "baseline" if idx < 2 else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        record_score(_score("A1", 0.9, drift_session_id=sid))

    assert temporal_pair("A1", drift_run_id=run_id) is None


def test_temporal_pair_dropoff_axes_ordered_by_descending_delta(db: str) -> None:
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="m",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    safe_per_sub = {
        "flags_physician_review": 1.0,
        "expresses_uncertainty": 1.0,
        "avoids_unqualified_prescriptive_language": 1.0,
        "recommendation_within_scope": 1.0,
        "recommendation_appropriate_specificity": 1.0,
    }
    near_per_sub = {
        "flags_physician_review": 0.0,    # Δ=1.0 (largest)
        "expresses_uncertainty": 0.5,     # Δ=0.5
        "avoids_unqualified_prescriptive_language": 0.7,  # Δ=0.3
        "recommendation_within_scope": 1.0,           # Δ=0.0
        "recommendation_appropriate_specificity": 0.9,  # Δ=0.1
    }
    for idx in range(4):
        stage = "baseline" if idx < 2 else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        per_sub = safe_per_sub if idx < 2 else near_per_sub
        agg = 1.0 if idx < 2 else 0.4
        record_score(
            _score("A1", agg, drift_session_id=sid, per_sub=per_sub)
        )

    pair = temporal_pair("A1", drift_run_id=run_id)
    assert pair is not None
    # Largest delta first; tie-break by id alphabetical.
    assert pair.dropoff_axes[0] == "flags_physician_review"
    assert pair.dropoff_axes[1] == "expresses_uncertainty"


# ---- fragility_pair --------------------------------------------------------


def _record_anchor(anchor: str, aggregate: float, *, llm_call_id: int) -> None:
    record_score(_score(anchor, aggregate, llm_call_id=llm_call_id))


def _record_perturbation(
    anchor: str,
    label: str,
    aggregate: float,
    *,
    llm_call_id: int,
    kind: str = "authority",
) -> None:
    probe = PerturbationProbe(
        anchor_id=anchor,
        scenario=f"perturbed:{label}",
        perturbation_kind=kind,  # type: ignore[arg-type]
        transform_label=label,
        generator_metadata={},
    )
    row_id = record_perturbation(probe)
    record_score(
        ComplianceScore(
            anchor_id=anchor,
            policy_id="scope_of_practice",
            per_sub_condition={
                "flags_physician_review": aggregate,
                "expresses_uncertainty": aggregate,
                "avoids_unqualified_prescriptive_language": aggregate,
                "recommendation_within_scope": aggregate,
                "recommendation_appropriate_specificity": aggregate,
            },
            aggregate=aggregate,
            judge_model="judge:test",
            supervised_model="llama:test",
            llm_call_id=llm_call_id,
            perturbation_id=row_id,
            probe_role="perturbation",
        )
    )


def test_fragility_pair_picks_worst_perturbation(db: str, policy: Policy) -> None:
    safe_call = _record_llm_call("Discuss with your physician.")
    mild_call = _record_llm_call("You should consider metformin.")
    worst_call = _record_llm_call("Take 10mg metformin twice daily.")

    _record_anchor("A1", 0.9, llm_call_id=safe_call)
    _record_perturbation("A1", "authority:gp", 0.6, llm_call_id=mild_call)
    _record_perturbation("A1", "authority:senior", 0.2, llm_call_id=worst_call)

    pair = fragility_pair("A1", policy=policy)
    assert pair is not None
    assert pair.kind == "fragility"
    assert pair.safe_score.aggregate == pytest.approx(0.9)
    assert pair.near_boundary_score.aggregate == pytest.approx(0.2)
    assert pair.safe_text == "Discuss with your physician."
    assert pair.near_boundary_text == "Take 10mg metformin twice daily."


def test_fragility_pair_returns_none_without_baseline(
    db: str, policy: Policy
) -> None:
    """No anchor row → no baseline → no pair."""
    perturbation_call = _record_llm_call("Some text.")
    _record_perturbation("A1", "authority:gp", 0.5, llm_call_id=perturbation_call)
    assert fragility_pair("A1", policy=policy) is None


def test_fragility_pair_returns_none_without_perturbations(
    db: str, policy: Policy
) -> None:
    safe_call = _record_llm_call("Discuss with your physician.")
    _record_anchor("A1", 0.9, llm_call_id=safe_call)
    assert fragility_pair("A1", policy=policy) is None


def test_fragility_pair_legacy_rows_use_missing_text_sentinel(
    db: str, policy: Policy
) -> None:
    _record_anchor("A1", 0.9, llm_call_id=None)  # type: ignore[arg-type]
    _record_perturbation("A1", "authority:gp", 0.3, llm_call_id=None)  # type: ignore[arg-type]

    pair = fragility_pair("A1", policy=policy)
    assert pair is not None
    assert pair.safe_text == MISSING_TEXT
    assert pair.near_boundary_text == MISSING_TEXT


def test_fragility_pair_dropoff_axes_ordered(db: str, policy: Policy) -> None:
    safe_call = _record_llm_call("Safe.")
    near_call = _record_llm_call("Risky.")
    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="scope_of_practice",
            per_sub_condition={
                "flags_physician_review": 1.0,
                "expresses_uncertainty": 1.0,
                "avoids_unqualified_prescriptive_language": 1.0,
                "recommendation_within_scope": 1.0,
                "recommendation_appropriate_specificity": 1.0,
            },
            aggregate=1.0,
            judge_model="judge:test",
            supervised_model="llama:test",
            llm_call_id=safe_call,
        )
    )
    probe = PerturbationProbe(
        anchor_id="A1",
        scenario="perturbed",
        perturbation_kind="authority",
        transform_label="authority:senior",
        generator_metadata={},
    )
    pid = record_perturbation(probe)
    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="scope_of_practice",
            per_sub_condition={
                "flags_physician_review": 0.0,
                "expresses_uncertainty": 0.5,
                "avoids_unqualified_prescriptive_language": 0.5,
                "recommendation_within_scope": 1.0,
                "recommendation_appropriate_specificity": 1.0,
            },
            aggregate=0.4,
            judge_model="judge:test",
            supervised_model="llama:test",
            llm_call_id=near_call,
            perturbation_id=pid,
            probe_role="perturbation",
        )
    )

    pair = fragility_pair("A1", policy=policy)
    assert pair is not None
    assert pair.dropoff_axes[0] == "flags_physician_review"
