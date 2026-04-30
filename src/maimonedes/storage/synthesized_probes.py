"""ORM model + repository helpers for synthesized_probes."""
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
from maimonedes.core.synthesized_probe import SynthesizedProbe
from maimonedes.storage.compliance import ComplianceScoreRow, row_to_score
from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SynthesizedProbeRow(Base):
    __tablename__ = "synthesized_probes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    scenario: Mapped[str] = mapped_column(Text, nullable=False)
    generation_method: Mapped[str] = mapped_column(String(32), nullable=False)
    target_embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    achieved_embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    tau_distance: Mapped[float] = mapped_column(Float, nullable=False)
    parent_anchor_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    synthesizer_llm_call_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("llm_calls.id"), nullable=True
    )
    validator_llm_call_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("llm_calls.id"), nullable=True
    )
    quality_status: Mapped[str] = mapped_column(String(16), nullable=False)
    quality_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    gp_fit_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("gp_fits.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


def _row_to_probe(row: SynthesizedProbeRow) -> SynthesizedProbe:
    try:
        target_emb = json.loads(row.target_embedding_json)
    except json.JSONDecodeError:
        target_emb = []
    try:
        achieved_emb = json.loads(row.achieved_embedding_json)
    except json.JSONDecodeError:
        achieved_emb = []
    try:
        parents = json.loads(row.parent_anchor_ids_json)
    except json.JSONDecodeError:
        parents = []
    if not isinstance(parents, list):
        parents = []
    created = row.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return SynthesizedProbe(
        id=row.id,
        policy_id=row.policy_id,
        scenario=row.scenario,
        generation_method=row.generation_method,  # type: ignore[arg-type]
        target_embedding=[float(v) for v in target_emb],
        achieved_embedding=[float(v) for v in achieved_emb],
        tau_distance=float(row.tau_distance),
        parent_anchor_ids=[str(x) for x in parents],
        synthesizer_llm_call_id=row.synthesizer_llm_call_id,
        validator_llm_call_id=row.validator_llm_call_id,
        quality_status=row.quality_status,  # type: ignore[arg-type]
        quality_reason=row.quality_reason,
        gp_fit_id=row.gp_fit_id,
        created_at=created,
    )


def record_synthesized_probe(probe: SynthesizedProbe) -> int:
    with get_session() as session:
        row = SynthesizedProbeRow(
            policy_id=probe.policy_id,
            scenario=probe.scenario,
            generation_method=probe.generation_method,
            target_embedding_json=json.dumps(probe.target_embedding),
            achieved_embedding_json=json.dumps(probe.achieved_embedding),
            tau_distance=probe.tau_distance,
            parent_anchor_ids_json=json.dumps(probe.parent_anchor_ids),
            synthesizer_llm_call_id=probe.synthesizer_llm_call_id,
            validator_llm_call_id=probe.validator_llm_call_id,
            quality_status=probe.quality_status,
            quality_reason=probe.quality_reason,
            gp_fit_id=probe.gp_fit_id,
        )
        session.add(row)
        session.flush()
        return row.id


def get_synthesized_probe(probe_id: int) -> SynthesizedProbe | None:
    with get_session() as session:
        row = session.get(SynthesizedProbeRow, probe_id)
        return _row_to_probe(row) if row is not None else None


def list_synthesized_probes(
    policy_id: str | None = None,
    quality_status: str | None = None,
) -> list[SynthesizedProbe]:
    with get_session() as session:
        stmt = select(SynthesizedProbeRow).order_by(
            SynthesizedProbeRow.created_at.desc(),
            SynthesizedProbeRow.id.desc(),
        )
        if policy_id is not None:
            stmt = stmt.where(SynthesizedProbeRow.policy_id == policy_id)
        if quality_status is not None:
            stmt = stmt.where(SynthesizedProbeRow.quality_status == quality_status)
        rows = list(session.execute(stmt).scalars().all())
        return [_row_to_probe(r) for r in rows]


def scores_for_synthesized_probe(probe_id: int) -> list[ComplianceScore]:
    with get_session() as session:
        rows = list(
            session.execute(
                select(ComplianceScoreRow)
                .where(ComplianceScoreRow.synthesized_probe_id == probe_id)
                .order_by(
                    ComplianceScoreRow.scored_at.asc(),
                    ComplianceScoreRow.id.asc(),
                )
            )
            .scalars()
            .all()
        )
        return [row_to_score(r) for r in rows]


__all__ = [
    "SynthesizedProbeRow",
    "get_synthesized_probe",
    "list_synthesized_probes",
    "record_synthesized_probe",
    "scores_for_synthesized_probe",
]
