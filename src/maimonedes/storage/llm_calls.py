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
    prefer_fk_when_set: bool = True,
) -> list[SupervisedScorePair]:
    """Pair supervised LLM calls to compliance_scores.

    Two paths, in order of preference:

    1. **FK path (post-#44 default).** When a `compliance_scores`
       row has `llm_call_id` set, look up the matching supervised
       row and pair directly — no chronological guesswork.
    2. **Heuristic path (legacy).** Older rows (and any future
       hand-loaded data without the FK) fall back to walking
       supervised + score lists in chronological order, pairing
       1-to-1 within `max_delta_seconds`.

    Pass `prefer_fk_when_set=False` to force every row through the
    chronological heuristic — useful for testing the heuristic
    on a DB that already has FKs.
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

    sup_by_id = {row.id: row for row in sup_rows}
    pairs: list[SupervisedScorePair] = []
    pending_score_rows: list[ComplianceScoreRow] = []

    for score_row in score_rows:
        if (
            prefer_fk_when_set
            and score_row.llm_call_id is not None
            and score_row.llm_call_id in sup_by_id
        ):
            sup_row = sup_by_id[score_row.llm_call_id]
            pairs.append(
                SupervisedScorePair(
                    anchor_id=score_row.anchor_id,
                    supervised_text=sup_row.response_content,
                    score=row_to_score(score_row),
                )
            )
        else:
            pending_score_rows.append(score_row)

    if pending_score_rows:
        # Don't reuse supervised rows that were already FK-paired —
        # pair the remainder against the *unused* supervised pool.
        used_ids: set[int] = {p.score.llm_call_id for p in pairs if p.score.llm_call_id is not None}
        remaining_sup = [r for r in sup_rows if r.id not in used_ids]
        sup_idx = 0
        for score_row in pending_score_rows:
            scored_at = _aware(score_row.scored_at)
            while sup_idx < len(remaining_sup):
                sup_ts = _aware(remaining_sup[sup_idx].timestamp)
                if sup_ts > scored_at:
                    break
                delta = (scored_at - sup_ts).total_seconds()
                if delta > max_delta_seconds:
                    # Orphaned (judge failure / hiccup); skip ahead.
                    sup_idx += 1
                    continue
                pairs.append(
                    SupervisedScorePair(
                        anchor_id=score_row.anchor_id,
                        supervised_text=remaining_sup[sup_idx].response_content,
                        score=row_to_score(score_row),
                    )
                )
                sup_idx += 1
                break

    # Re-sort to match the input scoring order (FK rows can appear
    # anywhere in the timeline; heuristic-only callers used to see
    # a chronological list).
    pairs.sort(key=lambda p: (p.score.scored_at, p.score.llm_call_id or 0))
    return pairs


def backfill_llm_call_ids(
    *,
    supervised_backend_prefix: str = "ollama-supervised",
    max_delta_seconds: float = SUPERVISED_TO_SCORE_MAX_SECONDS,
    dry_run: bool = False,
) -> int:
    """Walk all `compliance_scores` rows where `llm_call_id IS NULL`,
    chronologically pair them with supervised LLM calls, and update
    the FK column.

    Returns the number of rows that were (would be) updated. In
    `dry_run=True` mode no writes happen; the function still
    returns the count it would have written. Idempotent — already-
    set rows are skipped.
    """
    with get_session() as session:
        null_rows = list(
            session.execute(
                select(ComplianceScoreRow)
                .where(ComplianceScoreRow.llm_call_id.is_(None))
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
        # Exclude supervised rows already claimed by an FK on some
        # other compliance_scores row — never double-pair.
        used_sup_ids = set(
            session.execute(
                select(ComplianceScoreRow.llm_call_id).where(
                    ComplianceScoreRow.llm_call_id.is_not(None)
                )
            ).scalars().all()
        )
        remaining_sup = [r for r in sup_rows if r.id not in used_sup_ids]
        for r in null_rows:
            session.expunge(r)
        for r in remaining_sup:
            session.expunge(r)

    updates: list[tuple[int, int]] = []  # (compliance_score_id, llm_call_id)
    sup_idx = 0
    for score_row in null_rows:
        scored_at = _aware(score_row.scored_at)
        while sup_idx < len(remaining_sup):
            sup_ts = _aware(remaining_sup[sup_idx].timestamp)
            if sup_ts > scored_at:
                break
            delta = (scored_at - sup_ts).total_seconds()
            if delta > max_delta_seconds:
                sup_idx += 1
                continue
            updates.append((score_row.id, remaining_sup[sup_idx].id))
            sup_idx += 1
            break

    if dry_run or not updates:
        return len(updates)

    with get_session() as session:
        for cs_id, sup_id in updates:
            session.execute(
                ComplianceScoreRow.__table__.update()
                .where(ComplianceScoreRow.id == cs_id)
                .values(llm_call_id=sup_id)
            )
    return len(updates)


__all__ = [
    "LLMCall",
    "SUPERVISED_TO_SCORE_MAX_SECONDS",
    "SupervisedScorePair",
    "backfill_llm_call_ids",
    "pair_supervised_with_scores",
]
