"""Phase 5 synthesized-probe primitives.

Generated probes — produced by the K-NN exemplar synthesizer (#42)
from GP-proposed embedding targets — are kept distinct from the
curated `anchors_v1.yaml` library so the v1 baseline data never gets
conflated with LLM-synthesized scenarios. They DO produce
`compliance_scores` rows (with `probe_role="anchor"` since they ARE
anchor evaluations of the synthesized scenario), but those rows
carry a new `synthesized_probe_id` FK alongside the existing
perturbation_id / drift_session_id / recovery_run_id slots.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


GenerationMethod = Literal["knn_exemplar"]
QualityStatus = Literal["approved", "rejected", "pending"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SynthesizedProbe(BaseModel):
    """One probe synthesized from a GP-proposed embedding target."""

    model_config = ConfigDict(extra="forbid")

    id: int | None = None
    policy_id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    generation_method: GenerationMethod
    target_embedding: list[float] = Field(min_length=1)
    achieved_embedding: list[float] = Field(min_length=1)
    tau_distance: float  # cosine similarity in [-1, 1]; v1 requires ≥ 0.7
    parent_anchor_ids: list[str] = Field(default_factory=list)
    synthesizer_llm_call_id: int | None = None
    validator_llm_call_id: int | None = None
    quality_status: QualityStatus
    quality_reason: str | None = None
    gp_fit_id: int | None = None
    created_at: datetime = Field(default_factory=_utcnow)


__all__ = [
    "GenerationMethod",
    "QualityStatus",
    "SynthesizedProbe",
]
