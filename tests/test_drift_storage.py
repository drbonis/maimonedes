"""Tests for the Phase 3 drift schema + repository helpers."""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.drift import STAGE_LABELS
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
    finalize_drift_run,
    finalize_drift_session,
    get_drift_run,
    list_drift_runs,
    list_drift_sessions,
    scores_for_run,
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
    db_path = tmp_path / "p3.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


# ---- migration shape -------------------------------------------------------


def test_stage_labels_match_schedule_vocabulary() -> None:
    assert STAGE_LABELS == (
        "baseline",
        "concise",
        "actionable",
        "no_caveats",
        "trust",
    )


def test_migration_creates_drift_tables_with_indexes(db: str) -> None:
    insp = inspect(get_engine())
    assert "drift_runs" in insp.get_table_names()
    assert "drift_sessions" in insp.get_table_names()
    indexes = {ix["name"] for ix in insp.get_indexes("drift_sessions")}
    assert "ix_drift_sessions_drift_run_id" in indexes


def test_migration_extends_compliance_scores_with_drift_session_id(db: str) -> None:
    insp = inspect(get_engine())
    cols = {c["name"] for c in insp.get_columns("compliance_scores")}
    assert "drift_session_id" in cols
    indexes = {ix["name"] for ix in insp.get_indexes("compliance_scores")}
    assert "ix_compliance_scores_drift_session_id" in indexes


def test_alembic_round_trip_clean(db: str) -> None:
    cfg = _alembic_cfg(db)
    command.downgrade(cfg, "0005_compliance_scores_perturbation_id")
    insp = inspect(get_engine())
    assert "drift_runs" not in insp.get_table_names()
    assert "drift_sessions" not in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("compliance_scores")}
    assert "drift_session_id" not in cols
    # Re-upgrade to head — must succeed cleanly.
    command.upgrade(cfg, "head")
    insp = inspect(get_engine())
    assert "drift_runs" in insp.get_table_names()


# ---- repository round-trip -------------------------------------------------


def _create_run(notes: str | None = None, k: float = 4.0) -> int:
    return create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
        notes=notes,
        k_threshold=k,
    )


def test_create_drift_run_persists_fields(db: str) -> None:
    run_id = _create_run(notes="smoke test", k=3.5)
    run = get_drift_run(run_id)
    assert run is not None
    assert run["policy_id"] == "scope_of_practice"
    assert run["supervised_model"] == "llama:test"
    assert run["judge_model"] == "judge:test"
    assert run["schedule_path"] == "config/drift/scope_of_practice_v1.yaml"
    assert run["notes"] == "smoke test"
    assert run["k_threshold"] == 3.5
    assert run["ended_at"] is None


def test_get_drift_run_missing_returns_none(db: str) -> None:
    assert get_drift_run(404) is None


def test_list_drift_runs_orders_most_recent_first(db: str) -> None:
    a = _create_run(notes="a")
    b = _create_run(notes="b")
    runs = list_drift_runs()
    assert [r["id"] for r in runs[:2]] == [b, a]


def test_create_drift_session_unique_constraint(db: str) -> None:
    run_id = _create_run()
    create_drift_session(run_id, 0, "baseline", "")
    with pytest.raises(IntegrityError):
        create_drift_session(run_id, 0, "concise", "Be concise and direct.")


def test_finalize_drift_session_stamps_ended_at(db: str) -> None:
    run_id = _create_run()
    sid = create_drift_session(run_id, 0, "baseline", "")
    finalize_drift_session(sid)
    sessions = list_drift_sessions(run_id)
    assert len(sessions) == 1
    assert sessions[0].ended_at is not None


def test_finalize_drift_run_stamps_ended_at(db: str) -> None:
    run_id = _create_run()
    finalize_drift_run(run_id)
    run = get_drift_run(run_id)
    assert run is not None
    assert run["ended_at"] is not None


def test_finalize_drift_session_missing_raises(db: str) -> None:
    with pytest.raises(ValueError):
        finalize_drift_session(404)


def test_finalize_drift_run_missing_raises(db: str) -> None:
    with pytest.raises(ValueError):
        finalize_drift_run(404)


def test_list_drift_sessions_orders_by_session_index(db: str) -> None:
    run_id = _create_run()
    # Insert out of order; helper must sort.
    create_drift_session(run_id, 2, "actionable", "...actionable...")
    create_drift_session(run_id, 0, "baseline", "")
    create_drift_session(run_id, 1, "concise", "Be concise and direct.")
    sessions = list_drift_sessions(run_id)
    assert [s.session_index for s in sessions] == [0, 1, 2]


# ---- compliance-score linking ----------------------------------------------


def _score(
    anchor: str,
    aggregate: float,
    drift_session_id: int | None = None,
) -> ComplianceScore:
    return ComplianceScore(
        anchor_id=anchor,
        policy_id="scope_of_practice",
        per_sub_condition={"flags_physician_review": 1.0},
        aggregate=aggregate,
        judge_model="judge:test",
        supervised_model="llama:test",
        drift_session_id=drift_session_id,
    )


def test_compliance_score_records_drift_session_id(db: str) -> None:
    run_id = _create_run()
    sid = create_drift_session(run_id, 0, "baseline", "")
    record_score(_score("A1", 0.9, drift_session_id=sid))

    by_anchor = scores_for_run(run_id)
    assert "A1" in by_anchor
    assert len(by_anchor["A1"]) == 1
    assert by_anchor["A1"][0].drift_session_id == sid
    assert by_anchor["A1"][0].aggregate == pytest.approx(0.9)


def test_compliance_score_with_orphan_drift_session_id_rejected_by_fk(
    db: str,
) -> None:
    with pytest.raises(IntegrityError):
        record_score(_score("A1", 0.5, drift_session_id=999_999))


def test_scores_for_run_orders_by_session_index_per_anchor(db: str) -> None:
    run_id = _create_run()
    s0 = create_drift_session(run_id, 0, "baseline", "")
    s1 = create_drift_session(run_id, 1, "baseline", "")
    s2 = create_drift_session(run_id, 2, "concise", "Be concise and direct.")
    # Insert out of order to ensure ordering comes from session_index, not insert order.
    record_score(_score("A1", 0.5, drift_session_id=s2))
    record_score(_score("A1", 0.95, drift_session_id=s0))
    record_score(_score("A2", 0.7, drift_session_id=s1))
    record_score(_score("A1", 0.9, drift_session_id=s1))

    by_anchor = scores_for_run(run_id)
    assert [s.aggregate for s in by_anchor["A1"]] == pytest.approx([0.95, 0.9, 0.5])
    assert [s.aggregate for s in by_anchor["A2"]] == pytest.approx([0.7])


def test_scores_for_run_excludes_non_drift_rows(db: str) -> None:
    run_id = _create_run()
    sid = create_drift_session(run_id, 0, "baseline", "")
    record_score(_score("A1", 0.9, drift_session_id=sid))
    # An ordinary anchor score (no drift_session_id) must not leak in.
    record_score(_score("A1", 0.5, drift_session_id=None))

    by_anchor = scores_for_run(run_id)
    assert len(by_anchor["A1"]) == 1
    assert by_anchor["A1"][0].drift_session_id == sid


def test_scores_for_run_unknown_run_returns_empty(db: str) -> None:
    assert scores_for_run(404) == {}
