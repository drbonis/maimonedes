"""Compliance score data model.

A `ComplianceScore` is the result of running the Stage-1 LLM-as-Judge
on one supervised output: per-sub-condition scalars (boolean → 0.0/1.0,
0–3 → val/3.0) plus the rubric's weighted aggregate in [0, 1].

The model is the wire shape between the judge (#12) and the
orchestrator + dashboard. ORM mapping lives in
`maimonedes.storage.compliance`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ProbeRole = Literal["anchor", "perturbation"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ComplianceScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    anchor_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    per_sub_condition: dict[str, float] = Field(default_factory=dict)
    aggregate: float = Field(ge=0.0, le=1.0)
    judge_model: str = Field(min_length=1)
    supervised_model: str = Field(min_length=1)
    llm_call_id: int | None = None
    # Phase 2: scores can be evaluated on either the parent anchor
    # (probe_role="anchor", perturbation_id=None) or a generated
    # perturbation (probe_role="perturbation", perturbation_id set).
    # `anchor_id` always names the parent anchor regardless of role.
    perturbation_id: int | None = None
    probe_role: ProbeRole = "anchor"
    # Phase 3: anchor scores collected during a synthetic drift run
    # carry the FK to their `drift_session`; non-drift rows leave it None.
    drift_session_id: int | None = None
    scored_at: datetime = Field(default_factory=_utcnow)


def to_payload(score: ComplianceScore) -> dict[str, Any]:
    """Convenience for log lines / CLI output without dragging pydantic into format strings."""
    return score.model_dump(mode="json")


__all__ = ["ComplianceScore", "ProbeRole", "to_payload"]
