"""Tests for the Phase 4 `recovery-report` CLI + builder helper."""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.feedback import Feedback
from maimonedes.monitor.recovery_report import (
    NoRecoveryDataError,
    OrphanRecoveryRunError,
    build_report,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
)
from maimonedes.storage.recovery import (
    create_recovery_run,
    record_feedback,
)
from maimonedes.storage.repo import init_engine, reset_engine_for_tests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

runner = CliRunner()


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "report.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


def _make_score(
    anchor: str,
    aggregate: float,
    *,
    drift_session_id: int | None = None,
    recovery_run_id: int | None = None,
) -> ComplianceScore:
    return ComplianceScore(
        anchor_id=anchor,
        policy_id="scope_of_practice",
        per_sub_condition={"flags_physician_review": aggregate},
        aggregate=aggregate,
        judge_model="judge:test",
        supervised_model="llama:test",
        drift_session_id=drift_session_id,
        recovery_run_id=recovery_run_id,
    )


def _seed_full_recovery(
    *,
    pre_aggregate: float = 0.3,
    post_aggregate: float = 0.85,
    contrastive_kind: str = "temporal",
) -> tuple[int, int]:
    """Seed a parent drift run + recovery run + feedback + scores."""
    drift_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    sid_baseline = create_drift_session(drift_id, 0, "baseline", "")
    sid_drift = create_drift_session(drift_id, 5, "concise", "x")
    record_score(_make_score("A1", 0.9, drift_session_id=sid_baseline))
    record_score(_make_score("A1", pre_aggregate, drift_session_id=sid_drift))

    recovery_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="llama:test",
        judge_model="judge:test",
        contrastive_kind=contrastive_kind,  # type: ignore[arg-type]
    )
    record_feedback(
        Feedback(
            recovery_run_id=recovery_id,
            parent_drift_run_id=drift_id,
            anchor_id="A1",
            contrastive_kind=contrastive_kind,  # type: ignore[arg-type]
            feedback_text="Always defer prescribing decisions to physician.",
        )
    )
    record_score(
        _make_score("A1", post_aggregate, recovery_run_id=recovery_id)
    )
    return drift_id, recovery_id


# ---- builder ---------------------------------------------------------------


def test_build_report_full_path_yields_closed_loop_verdict(db: str) -> None:
    _, recovery_id = _seed_full_recovery(pre_aggregate=0.3, post_aggregate=0.85)
    report = build_report(recovery_id)
    row = report.rows[0]
    assert row.anchor_id == "A1"
    assert row.pre_worst_aggregate == pytest.approx(0.3)
    assert row.pre_worst_session == 5
    assert row.post_aggregate == pytest.approx(0.85)
    assert row.delta_toward_baseline == pytest.approx(0.55)
    assert row.recovered is True
    assert report.recovered_count == 1
    assert report.verdict == "closed_loop"


def test_build_report_partial_when_delta_positive_but_below_threshold(
    db: str,
) -> None:
    _, recovery_id = _seed_full_recovery(pre_aggregate=0.2, post_aggregate=0.4)
    report = build_report(recovery_id)
    assert report.verdict == "partial"
    assert report.recovered_count == 0


def test_build_report_failed_when_delta_non_positive(db: str) -> None:
    _, recovery_id = _seed_full_recovery(pre_aggregate=0.3, post_aggregate=0.2)
    report = build_report(recovery_id)
    assert report.verdict == "failed"


def test_build_report_unknown_recovery_run_raises(db: str) -> None:
    with pytest.raises(NoRecoveryDataError):
        build_report(999)


def _drop_drift_run_bypassing_fk(drift_id: int) -> None:
    """Force-delete a drift_run row to simulate an orphaned recovery run.

    SQLite enforces FKs via a per-connection PRAGMA we enable in
    `storage.repo`; toggling it off scoped to one connection lets us
    delete the parent without cascading to children.
    """
    from sqlalchemy import text

    from maimonedes.storage.repo import get_engine

    with get_engine().connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = OFF"))
        conn.execute(text("DELETE FROM drift_runs WHERE id = :id"), {"id": drift_id})
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()


def test_build_report_orphan_parent_raises(db: str) -> None:
    """Recovery run whose parent_drift_run_id points at a deleted drift run."""
    drift_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="m",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    recovery_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    record_feedback(
        Feedback(
            recovery_run_id=recovery_id,
            parent_drift_run_id=drift_id,
            anchor_id="A1",
            contrastive_kind="temporal",
            feedback_text="x",
        )
    )
    _drop_drift_run_bypassing_fk(drift_id)

    with pytest.raises(OrphanRecoveryRunError):
        build_report(recovery_id)


def test_build_report_no_scored_anchors_raises(db: str) -> None:
    """Recovery run created but no feedbacks or scores."""
    drift_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="m",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    sid = create_drift_session(drift_id, 0, "baseline", "")
    record_score(_make_score("A1", 0.9, drift_session_id=sid))

    recovery_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    with pytest.raises(NoRecoveryDataError):
        build_report(recovery_id)


# ---- CLI -------------------------------------------------------------------


def test_recovery_report_cli_renders_table(db: str) -> None:
    _, recovery_id = _seed_full_recovery()
    result = runner.invoke(app, ["recovery-report", str(recovery_id)])
    assert result.exit_code == 0, result.output
    assert "anchor" in result.output
    assert "A1" in result.output
    assert "verdict: closed_loop" in result.output
    assert "mean_delta_toward_baseline=" in result.output


def test_recovery_report_cli_show_feedback(db: str) -> None:
    _, recovery_id = _seed_full_recovery()
    result = runner.invoke(
        app, ["recovery-report", str(recovery_id), "--show-feedback"]
    )
    assert result.exit_code == 0, result.output
    assert "[A1 contrastive=temporal]" in result.output
    assert "> Always defer prescribing" in result.output


def test_recovery_report_cli_unknown_run_exits_2(db: str) -> None:
    result = runner.invoke(app, ["recovery-report", "999"])
    assert result.exit_code == 2


def test_recovery_report_cli_orphan_exits_4(db: str) -> None:
    drift_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="m",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    recovery_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    record_feedback(
        Feedback(
            recovery_run_id=recovery_id,
            parent_drift_run_id=drift_id,
            anchor_id="A1",
            contrastive_kind="temporal",
            feedback_text="x",
        )
    )
    _drop_drift_run_bypassing_fk(drift_id)

    result = runner.invoke(app, ["recovery-report", str(recovery_id)])
    assert result.exit_code == 4
