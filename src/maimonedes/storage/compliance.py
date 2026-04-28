"""ORM mapping + repository helpers for compliance scores.

Lives in its own module so #5's `LLMCall` and this one can both
extend the shared `Base` without circular imports.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.core.compliance import ComplianceScore
from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ComplianceScoreRow(Base):
    __tablename__ = "compliance_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    per_sub_condition_json: Mapped[str] = mapped_column(Text, nullable=False)
    aggregate: Mapped[float] = mapped_column(Float, nullable=False)
    judge_model: Mapped[str] = mapped_column(String(128), nullable=False)
    supervised_model: Mapped[str] = mapped_column(String(128), nullable=False)
    llm_call_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("llm_calls.id"), nullable=True
    )
    scored_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


def _row_to_score(row: ComplianceScoreRow) -> ComplianceScore:
    try:
        per_sub = json.loads(row.per_sub_condition_json)
    except json.JSONDecodeError:
        per_sub = {}
    if not isinstance(per_sub, dict):
        per_sub = {}
    scored_at = row.scored_at
    if scored_at.tzinfo is None:
        # SQLite drops tzinfo on read; restamp as UTC.
        scored_at = scored_at.replace(tzinfo=timezone.utc)
    return ComplianceScore(
        anchor_id=row.anchor_id,
        policy_id=row.policy_id,
        per_sub_condition={k: float(v) for k, v in per_sub.items()},
        aggregate=row.aggregate,
        judge_model=row.judge_model,
        supervised_model=row.supervised_model,
        llm_call_id=row.llm_call_id,
        scored_at=scored_at,
    )


def record_score(score: ComplianceScore) -> int:
    """Insert a `ComplianceScore` and return the row id."""
    payload = json.dumps(score.per_sub_condition, sort_keys=True)
    with get_session() as session:
        row = ComplianceScoreRow(
            anchor_id=score.anchor_id,
            policy_id=score.policy_id,
            per_sub_condition_json=payload,
            aggregate=score.aggregate,
            judge_model=score.judge_model,
            supervised_model=score.supervised_model,
            llm_call_id=score.llm_call_id,
            scored_at=score.scored_at,
        )
        session.add(row)
        session.flush()
        return row.id


def recent_scores(anchor_id: str, *, limit: int = 50) -> list[ComplianceScore]:
    """Return the most-recent scores for an anchor, newest first."""
    with get_session() as session:
        stmt = (
            select(ComplianceScoreRow)
            .where(ComplianceScoreRow.anchor_id == anchor_id)
            .order_by(ComplianceScoreRow.scored_at.desc(), ComplianceScoreRow.id.desc())
            .limit(limit)
        )
        rows = session.execute(stmt).scalars().all()
        return [_row_to_score(r) for r in rows]


def latest_score_per_anchor() -> dict[str, ComplianceScore]:
    """Most-recent score for every anchor that has at least one row.

    Used by the Phase 1 dashboard. Not the world's tightest query
    plan, but the table has at most a few hundred thousand rows
    even after the full drift study, and the `(anchor_id, scored_at)`
    index makes the per-anchor scan cheap.
    """
    with get_session() as session:
        anchor_ids = session.execute(
            select(ComplianceScoreRow.anchor_id).distinct()
        ).scalars().all()
        out: dict[str, ComplianceScore] = {}
        for aid in anchor_ids:
            row = session.execute(
                select(ComplianceScoreRow)
                .where(ComplianceScoreRow.anchor_id == aid)
                .order_by(
                    ComplianceScoreRow.scored_at.desc(),
                    ComplianceScoreRow.id.desc(),
                )
                .limit(1)
            ).scalar_one_or_none()
            if row is not None:
                out[aid] = _row_to_score(row)
        return out


__all__ = [
    "ComplianceScoreRow",
    "latest_score_per_anchor",
    "record_score",
    "recent_scores",
]
