"""Feedback / recovery primitives.

Phase 4 closes the loop on a Phase 3 drift run by synthesizing a
short natural-language recommendation per affected anchor and
prepending it to the supervised system's prompt for a re-evaluation.

A `Feedback` row records one such recommendation: the parent drift
run it was motivated by, the anchor it targets, the contrastive
scenario the synthesizer used (`temporal` vs `fragility`), the
synthesized text, and the audit-trail id of the LLM call that
produced it. The `ContrastivePair` model is the wire shape between
the contrastive extractor (#30) and the synthesizer (#31).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from maimonedes.core.compliance import ComplianceScore


ContrastiveKind = Literal["temporal", "fragility"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Feedback(BaseModel):
    """One synthesized recommendation, ready to deliver as a system prompt."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    recovery_run_id: int
    parent_drift_run_id: int
    anchor_id: str = Field(min_length=1)
    contrastive_kind: ContrastiveKind
    feedback_text: str = Field(min_length=1)
    llm_call_id: int | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class ContrastivePair(BaseModel):
    """Safe + near-boundary outputs paired up for the synthesizer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    anchor_id: str = Field(min_length=1)
    kind: ContrastiveKind
    safe_text: str
    safe_score: ComplianceScore
    near_boundary_text: str
    near_boundary_score: ComplianceScore
    # Sub-condition ids ranked by descending |safe_score - near_boundary_score|
    # on each axis. Synthesizer renders this as a natural-language hint.
    dropoff_axes: list[str] = Field(default_factory=list)


__all__ = ["ContrastiveKind", "ContrastivePair", "Feedback"]
