"""ORM mapping + repository helpers for recovery runs and feedbacks.

A `RecoveryRun` is one execution of the closed-loop step on top of a
parent drift run. A `Feedback` is one synthesized recommendation
within that recovery run, scoped to a single anchor; the unique
constraint on `(recovery_run_id, anchor_id)` enforces the per-anchor
delivery model locked in the planning thread.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.feedback import ContrastiveKind, Feedback
from maimonedes.storage.compliance import ComplianceScoreRow, row_to_score
from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RecoveryRunRow(Base):
    __tablename__ = "recovery_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_drift_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drift_runs.id"), nullable=False, index=True
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    supervised_model: Mapped[str] = mapped_column(String(128), nullable=False)
    judge_model: Mapped[str] = mapped_column(String(128), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    contrastive_kind: Mapped[str] = mapped_column(String(16), nullable=False)


class FeedbackRow(Base):
    __tablename__ = "feedbacks"
    __table_args__ = (
        UniqueConstraint(
            "recovery_run_id",
            "anchor_id",
            name="uq_feedbacks_recovery_run_id_anchor_id",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recovery_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("recovery_runs.id"), nullable=False, index=True
    )
    parent_drift_run_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("drift_runs.id"), nullable=False, index=True
    )
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    contrastive_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    feedback_text: Mapped[str] = mapped_column(Text, nullable=False)
    llm_call_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("llm_calls.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


def _row_to_run(row: RecoveryRunRow) -> dict[str, object]:
    started = row.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    ended = row.ended_at
    if ended is not None and ended.tzinfo is None:
        ended = ended.replace(tzinfo=timezone.utc)
    return {
        "id": row.id,
        "parent_drift_run_id": row.parent_drift_run_id,
        "started_at": started,
        "ended_at": ended,
        "supervised_model": row.supervised_model,
        "judge_model": row.judge_model,
        "notes": row.notes,
        "contrastive_kind": row.contrastive_kind,
    }


def _row_to_feedback(row: FeedbackRow) -> Feedback:
    created = row.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return Feedback(
        id=row.id,
        recovery_run_id=row.recovery_run_id,
        parent_drift_run_id=row.parent_drift_run_id,
        anchor_id=row.anchor_id,
        contrastive_kind=row.contrastive_kind,  # type: ignore[arg-type]
        feedback_text=row.feedback_text,
        llm_call_id=row.llm_call_id,
        created_at=created,
    )


def create_recovery_run(
    *,
    parent_drift_run_id: int,
    supervised_model: str,
    judge_model: str,
    contrastive_kind: ContrastiveKind,
    notes: str | None = None,
) -> int:
    """Create a recovery run row and return its id."""
    with get_session() as session:
        row = RecoveryRunRow(
            parent_drift_run_id=parent_drift_run_id,
            supervised_model=supervised_model,
            judge_model=judge_model,
            contrastive_kind=contrastive_kind,
            notes=notes,
        )
        session.add(row)
        session.flush()
        return row.id


def finalize_recovery_run(run_id: int) -> None:
    """Stamp `ended_at` on a recovery run."""
    with get_session() as session:
        row = session.get(RecoveryRunRow, run_id)
        if row is None:
            raise ValueError(f"recovery_run {run_id} not found")
        row.ended_at = _utcnow()


def get_recovery_run(run_id: int) -> dict[str, object] | None:
    with get_session() as session:
        row = session.get(RecoveryRunRow, run_id)
        return _row_to_run(row) if row is not None else None


def list_recovery_runs(
    parent_drift_run_id: int | None = None,
) -> list[dict[str, object]]:
    """Most-recent first; optionally filtered by parent drift run."""
    with get_session() as session:
        stmt = select(RecoveryRunRow).order_by(
            RecoveryRunRow.started_at.desc(), RecoveryRunRow.id.desc()
        )
        if parent_drift_run_id is not None:
            stmt = stmt.where(
                RecoveryRunRow.parent_drift_run_id == parent_drift_run_id
            )
        rows = session.execute(stmt).scalars().all()
        return [_row_to_run(r) for r in rows]


def record_feedback(feedback: Feedback) -> int:
    """Insert a `Feedback` and return the new row id."""
    with get_session() as session:
        row = FeedbackRow(
            recovery_run_id=feedback.recovery_run_id,
            parent_drift_run_id=feedback.parent_drift_run_id,
            anchor_id=feedback.anchor_id,
            contrastive_kind=feedback.contrastive_kind,
            feedback_text=feedback.feedback_text,
            llm_call_id=feedback.llm_call_id,
        )
        session.add(row)
        session.flush()
        return row.id


def feedbacks_for_run(run_id: int) -> dict[str, Feedback]:
    """Per-anchor feedback for a recovery run; keyed by `anchor_id`."""
    with get_session() as session:
        rows = session.execute(
            select(FeedbackRow)
            .where(FeedbackRow.recovery_run_id == run_id)
            .order_by(FeedbackRow.id.asc())
        ).scalars().all()
        return {r.anchor_id: _row_to_feedback(r) for r in rows}


def scores_for_recovery_run(run_id: int) -> dict[str, list[ComplianceScore]]:
    """Per-anchor compliance scores for a recovery run, ordered by `scored_at`.

    Includes both anchor (`probe_role="anchor"`) and perturbation
    (`probe_role="perturbation"`) rows tagged with this `recovery_run_id`,
    so the fragility-scenario re-evaluations land in the same table
    the Phase 2 pipeline already consumes.
    """
    with get_session() as session:
        rows = session.execute(
            select(ComplianceScoreRow)
            .where(ComplianceScoreRow.recovery_run_id == run_id)
            .order_by(
                ComplianceScoreRow.anchor_id.asc(),
                ComplianceScoreRow.scored_at.asc(),
                ComplianceScoreRow.id.asc(),
            )
        ).scalars().all()
        out: dict[str, list[ComplianceScore]] = {}
        for r in rows:
            score = row_to_score(r)
            out.setdefault(score.anchor_id, []).append(score)
        return out


__all__ = [
    "FeedbackRow",
    "RecoveryRunRow",
    "create_recovery_run",
    "feedbacks_for_run",
    "finalize_recovery_run",
    "get_recovery_run",
    "list_recovery_runs",
    "record_feedback",
    "scores_for_recovery_run",
]
