"""Tests for the Phase 4 recovery schema + repository helpers."""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.feedback import Feedback
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import create_drift_run
from maimonedes.storage.recovery import (
    create_recovery_run,
    feedbacks_for_run,
    finalize_recovery_run,
    get_recovery_run,
    list_recovery_runs,
    record_feedback,
    scores_for_recovery_run,
)
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
    db_path = tmp_path / "p4.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


def _make_drift_run() -> int:
    return create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )


# ---- migration shape -------------------------------------------------------


def test_migration_creates_recovery_runs_table(db: str) -> None:
    insp = inspect(get_engine())
    assert "recovery_runs" in insp.get_table_names()
    indexes = {ix["name"] for ix in insp.get_indexes("recovery_runs")}
    assert "ix_recovery_runs_parent_drift_run_id" in indexes


def test_migration_creates_feedbacks_table(db: str) -> None:
    insp = inspect(get_engine())
    assert "feedbacks" in insp.get_table_names()
    indexes = {ix["name"] for ix in insp.get_indexes("feedbacks")}
    assert "ix_feedbacks_recovery_run_id" in indexes
    assert "ix_feedbacks_parent_drift_run_id" in indexes


def test_migration_extends_compliance_scores_with_recovery_run_id(
    db: str,
) -> None:
    insp = inspect(get_engine())
    cols = {c["name"] for c in insp.get_columns("compliance_scores")}
    assert "recovery_run_id" in cols
    indexes = {ix["name"] for ix in insp.get_indexes("compliance_scores")}
    assert "ix_compliance_scores_recovery_run_id" in indexes


def test_alembic_round_trip_clean(db: str) -> None:
    cfg = _alembic_cfg(db)
    command.downgrade(cfg, "0006_drift_sessions")
    insp = inspect(get_engine())
    assert "recovery_runs" not in insp.get_table_names()
    assert "feedbacks" not in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("compliance_scores")}
    assert "recovery_run_id" not in cols
    command.upgrade(cfg, "head")
    insp = inspect(get_engine())
    assert "recovery_runs" in insp.get_table_names()


# ---- repository round-trip -------------------------------------------------


def test_create_and_get_recovery_run(db: str) -> None:
    drift_id = _make_drift_run()
    run_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="llama:test",
        judge_model="judge:test",
        contrastive_kind="temporal",
        notes="smoke",
    )
    run = get_recovery_run(run_id)
    assert run is not None
    assert run["parent_drift_run_id"] == drift_id
    assert run["contrastive_kind"] == "temporal"
    assert run["notes"] == "smoke"
    assert run["ended_at"] is None


def test_get_recovery_run_missing_returns_none(db: str) -> None:
    assert get_recovery_run(404) is None


def test_finalize_recovery_run_stamps_ended_at(db: str) -> None:
    drift_id = _make_drift_run()
    run_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="llama:test",
        judge_model="judge:test",
        contrastive_kind="fragility",
    )
    finalize_recovery_run(run_id)
    run = get_recovery_run(run_id)
    assert run is not None
    assert run["ended_at"] is not None


def test_finalize_recovery_run_missing_raises(db: str) -> None:
    with pytest.raises(ValueError):
        finalize_recovery_run(404)


def test_list_recovery_runs_filters_by_parent(db: str) -> None:
    drift_a = _make_drift_run()
    drift_b = _make_drift_run()
    a = create_recovery_run(
        parent_drift_run_id=drift_a,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    b = create_recovery_run(
        parent_drift_run_id=drift_b,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    runs = list_recovery_runs(parent_drift_run_id=drift_a)
    assert [r["id"] for r in runs] == [a]
    runs_all = list_recovery_runs()
    assert {r["id"] for r in runs_all} == {a, b}


def test_record_feedback_unique_per_anchor(db: str) -> None:
    drift_id = _make_drift_run()
    run_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    fb_a = Feedback(
        recovery_run_id=run_id,
        parent_drift_run_id=drift_id,
        anchor_id="A1",
        contrastive_kind="temporal",
        feedback_text="Always defer to physician.",
    )
    record_feedback(fb_a)
    with pytest.raises(IntegrityError):
        record_feedback(fb_a)


def test_feedbacks_for_run_keyed_by_anchor(db: str) -> None:
    drift_id = _make_drift_run()
    run_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    record_feedback(
        Feedback(
            recovery_run_id=run_id,
            parent_drift_run_id=drift_id,
            anchor_id="A1",
            contrastive_kind="temporal",
            feedback_text="Always flag for review.",
        )
    )
    record_feedback(
        Feedback(
            recovery_run_id=run_id,
            parent_drift_run_id=drift_id,
            anchor_id="A4",
            contrastive_kind="temporal",
            feedback_text="Express uncertainty explicitly.",
        )
    )
    by_anchor = feedbacks_for_run(run_id)
    assert set(by_anchor) == {"A1", "A4"}
    assert by_anchor["A1"].feedback_text == "Always flag for review."
    assert by_anchor["A4"].anchor_id == "A4"


# ---- compliance-score linking ----------------------------------------------


def _score(
    anchor: str,
    aggregate: float,
    *,
    recovery_run_id: int | None = None,
) -> ComplianceScore:
    return ComplianceScore(
        anchor_id=anchor,
        policy_id="scope_of_practice",
        per_sub_condition={"flags_physician_review": 1.0},
        aggregate=aggregate,
        judge_model="judge:test",
        supervised_model="llama:test",
        recovery_run_id=recovery_run_id,
    )


def test_compliance_score_records_recovery_run_id(db: str) -> None:
    drift_id = _make_drift_run()
    run_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    record_score(_score("A1", 0.85, recovery_run_id=run_id))

    by_anchor = scores_for_recovery_run(run_id)
    assert "A1" in by_anchor
    assert len(by_anchor["A1"]) == 1
    assert by_anchor["A1"][0].recovery_run_id == run_id


def test_compliance_score_with_orphan_recovery_run_id_rejected(db: str) -> None:
    with pytest.raises(IntegrityError):
        record_score(_score("A1", 0.5, recovery_run_id=999_999))


def test_scores_for_recovery_run_excludes_non_recovery_rows(db: str) -> None:
    drift_id = _make_drift_run()
    run_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    record_score(_score("A1", 0.85, recovery_run_id=run_id))
    record_score(_score("A1", 0.40, recovery_run_id=None))

    by_anchor = scores_for_recovery_run(run_id)
    assert len(by_anchor["A1"]) == 1
    assert by_anchor["A1"][0].aggregate == pytest.approx(0.85)


def test_scores_for_recovery_run_unknown_returns_empty(db: str) -> None:
    assert scores_for_recovery_run(404) == {}
