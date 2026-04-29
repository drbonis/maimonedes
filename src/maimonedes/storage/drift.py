"""ORM mapping + repository helpers for drift runs and sessions.

A `DriftRun` is one execution of the synthetic drift schedule: it
fixes the policy, supervised + judge models, schedule path, and the
CUSUM threshold tuner constant `k`. A `DriftSession` is one step
inside that run — a fixed `(session_index, stage_label, suffix_text)`
under which the orchestrator evaluates every anchor.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.drift import StageLabel
from maimonedes.storage.compliance import ComplianceScoreRow, row_to_score
from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DriftRunRow(Base):
    __tablename__ = "drift_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    supervised_model: Mapped[str] = mapped_column(String(128), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(128), nullable=False)
    schedule_path: Mapped[str] = mapped_column(String(256), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    k_threshold: Mapped[float] = mapped_column(
        Float, nullable=False, default=4.0, server_default="4.0"
    )


class DriftSessionRow(Base):
    __tablename__ = "drift_sessions"
    __table_args__ = (
        UniqueConstraint(
            "drift_run_id",
            "session_index",
            name="uq_drift_sessions_drift_run_id_session_index",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    drift_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drift_runs.id"), nullable=False, index=True
    )
    session_index: Mapped[int] = mapped_column(Integer, nullable=False)
    stage_label: Mapped[str] = mapped_column(String(32), nullable=False)
    suffix_text: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


def _row_to_run(row: DriftRunRow) -> dict[str, object]:
    started = row.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    ended = row.ended_at
    if ended is not None and ended.tzinfo is None:
        ended = ended.replace(tzinfo=timezone.utc)
    return {
        "id": row.id,
        "started_at": started,
        "ended_at": ended,
        "policy_id": row.policy_id,
        "supervised_model": row.supervised_model,
        "judge_model": row.judge_model,
        "schedule_path": row.schedule_path,
        "notes": row.notes,
        "k_threshold": row.k_threshold,
    }


def create_drift_run(
    *,
    policy_id: str,
    supervised_model: str,
    judge_model: str,
    schedule_path: str,
    notes: str | None = None,
    k_threshold: float = 4.0,
) -> int:
    """Create a new drift run row and return its id."""
    with get_session() as session:
        row = DriftRunRow(
            policy_id=policy_id,
            supervised_model=supervised_model,
            judge_model=judge_model,
            schedule_path=schedule_path,
            notes=notes,
            k_threshold=k_threshold,
        )
        session.add(row)
        session.flush()
        return row.id


def create_drift_session(
    drift_run_id: int,
    session_index: int,
    stage_label: StageLabel,
    suffix_text: str,
) -> int:
    """Create a new drift session row and return its id."""
    with get_session() as session:
        row = DriftSessionRow(
            drift_run_id=drift_run_id,
            session_index=session_index,
            stage_label=stage_label,
            suffix_text=suffix_text,
        )
        session.add(row)
        session.flush()
        return row.id


def finalize_drift_session(session_id: int) -> None:
    """Stamp `ended_at` on a drift session."""
    with get_session() as session:
        row = session.get(DriftSessionRow, session_id)
        if row is None:
            raise ValueError(f"drift_session {session_id} not found")
        row.ended_at = _utcnow()


def finalize_drift_run(run_id: int) -> None:
    """Stamp `ended_at` on a drift run."""
    with get_session() as session:
        row = session.get(DriftRunRow, run_id)
        if row is None:
            raise ValueError(f"drift_run {run_id} not found")
        row.ended_at = _utcnow()


def get_drift_run(run_id: int) -> dict[str, object] | None:
    """Fetch one drift run as a plain dict, or None if missing."""
    with get_session() as session:
        row = session.get(DriftRunRow, run_id)
        return _row_to_run(row) if row is not None else None


def list_drift_runs() -> list[dict[str, object]]:
    """List all drift runs, most recent first."""
    with get_session() as session:
        rows = session.execute(
            select(DriftRunRow).order_by(
                DriftRunRow.started_at.desc(), DriftRunRow.id.desc()
            )
        ).scalars().all()
        return [_row_to_run(r) for r in rows]


def list_drift_sessions(run_id: int) -> list[DriftSessionRow]:
    """All `drift_sessions` for a run, ordered by `session_index`."""
    with get_session() as session:
        rows = session.execute(
            select(DriftSessionRow)
            .where(DriftSessionRow.drift_run_id == run_id)
            .order_by(DriftSessionRow.session_index.asc())
        ).scalars().all()
        # Detach so callers can read attributes outside the session.
        for r in rows:
            session.expunge(r)
        return list(rows)


def scores_for_run(run_id: int) -> dict[str, list[ComplianceScore]]:
    """Per-anchor compliance scores for a drift run, ordered by session_index.

    The returned mapping is keyed by `anchor_id`; each list is ordered
    by ascending `drift_sessions.session_index`. Anchors that have no
    score in the run are absent from the mapping.
    """
    with get_session() as session:
        stmt = (
            select(ComplianceScoreRow, DriftSessionRow.session_index)
            .join(
                DriftSessionRow,
                ComplianceScoreRow.drift_session_id == DriftSessionRow.id,
            )
            .where(DriftSessionRow.drift_run_id == run_id)
            .order_by(
                ComplianceScoreRow.anchor_id.asc(),
                DriftSessionRow.session_index.asc(),
                ComplianceScoreRow.id.asc(),
            )
        )
        out: dict[str, list[ComplianceScore]] = {}
        for score_row, _idx in session.execute(stmt).all():
            score = row_to_score(score_row)
            out.setdefault(score.anchor_id, []).append(score)
        return out


__all__ = [
    "DriftRunRow",
    "DriftSessionRow",
    "create_drift_run",
    "create_drift_session",
    "finalize_drift_run",
    "finalize_drift_session",
    "get_drift_run",
    "list_drift_runs",
    "list_drift_sessions",
    "scores_for_run",
]
