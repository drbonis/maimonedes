"""ORM model for the recorded-LLM-calls table.

Lives in its own module so #5's migration and #5's RecordingClient can
import the model without dragging in unrelated domain models.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.core.compliance import ComplianceScore
from maimonedes.storage.compliance import ComplianceScoreRow, row_to_score
from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )
    backend_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    request_messages_json: Mapped[str] = mapped_column(Text, nullable=False)
    response_content: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response_json: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


# ---- supervised ↔ compliance_scores pairing -------------------------------


SUPERVISED_TO_SCORE_MAX_SECONDS = 600  # 10 min — judge timeout is 60s by default


@dataclass(frozen=True)
class SupervisedScorePair:
    """One paired supervised text + compliance_score for training."""

    anchor_id: str
    supervised_text: str
    score: ComplianceScore


def _aware(ts: datetime) -> datetime:
    """SQLite drops tzinfo on read; restamp as UTC for safe comparison."""
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def pair_supervised_with_scores(
    policy_id: str,
    *,
    supervised_backend_prefix: str = "ollama-supervised",
    max_delta_seconds: float = SUPERVISED_TO_SCORE_MAX_SECONDS,
) -> list[SupervisedScorePair]:
    """Pair supervised LLM calls to compliance_scores by chronological order.

    The Phase 1+ orchestrators do not wire `compliance_scores.llm_call_id`
    to the supervised call's row, so a FK join returns nothing on
    historical data. Each supervised call is followed within
    milliseconds by exactly one score row in sequence; pair them by
    walking both lists in order. Orphaned sup calls (failed judge mid-
    pair) are skipped via the `max_delta_seconds` window.
    """
    with get_session() as session:
        score_rows = list(
            session.execute(
                select(ComplianceScoreRow)
                .where(ComplianceScoreRow.policy_id == policy_id)
                .order_by(
                    ComplianceScoreRow.scored_at.asc(),
                    ComplianceScoreRow.id.asc(),
                )
            )
            .scalars()
            .all()
        )
        sup_rows = list(
            session.execute(
                select(LLMCall)
                .where(LLMCall.backend_name.like(f"{supervised_backend_prefix}%"))
                .order_by(LLMCall.timestamp.asc(), LLMCall.id.asc())
            )
            .scalars()
            .all()
        )
        for r in score_rows:
            session.expunge(r)
        for r in sup_rows:
            session.expunge(r)

    pairs: list[SupervisedScorePair] = []
    sup_idx = 0
    for score_row in score_rows:
        scored_at = _aware(score_row.scored_at)
        while sup_idx < len(sup_rows):
            sup_ts = _aware(sup_rows[sup_idx].timestamp)
            if sup_ts > scored_at:
                break
            delta = (scored_at - sup_ts).total_seconds()
            if delta > max_delta_seconds:
                # Orphaned (judge failure / hiccup); drop and try the next sup.
                sup_idx += 1
                continue
            pairs.append(
                SupervisedScorePair(
                    anchor_id=score_row.anchor_id,
                    supervised_text=sup_rows[sup_idx].response_content,
                    score=row_to_score(score_row),
                )
            )
            sup_idx += 1
            break
    return pairs


__all__ = [
    "LLMCall",
    "SUPERVISED_TO_SCORE_MAX_SECONDS",
    "SupervisedScorePair",
    "pair_supervised_with_scores",
]
