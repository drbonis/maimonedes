"""ORM mapping + repository helpers for perturbation probes."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.core.perturbation import PerturbationKind, PerturbationProbe
from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class PerturbationProbeRow(Base):
    __tablename__ = "perturbation_probes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    perturbation_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    transform_label: Mapped[str] = mapped_column(String(128), nullable=False)
    scenario: Mapped[str] = mapped_column(Text, nullable=False)
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    generator_metadata_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )
    synthesized_probe_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("synthesized_probes.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


def _row_to_probe(row: PerturbationProbeRow) -> PerturbationProbe:
    try:
        metadata = json.loads(row.generator_metadata_json)
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return PerturbationProbe(
        id=f"{row.anchor_id}#{row.transform_label}",
        anchor_id=row.anchor_id,
        scenario=row.scenario,
        policy_id=row.policy_id,
        perturbation_kind=row.perturbation_kind,  # type: ignore[arg-type]
        transform_label=row.transform_label,
        generator_metadata=metadata,
        synthesized_probe_id=row.synthesized_probe_id,
    )


def record_perturbation(probe: PerturbationProbe) -> int:
    """Insert a `PerturbationProbe` and return the new row id."""
    payload = json.dumps(probe.generator_metadata, sort_keys=True, default=str)
    with get_session() as session:
        row = PerturbationProbeRow(
            anchor_id=probe.anchor_id,
            perturbation_kind=probe.perturbation_kind,
            transform_label=probe.transform_label,
            scenario=probe.scenario,
            policy_id=probe.policy_id,
            generator_metadata_json=payload,
            synthesized_probe_id=probe.synthesized_probe_id,
        )
        session.add(row)
        session.flush()
        return row.id


def recent_perturbations(
    anchor_id: str,
    *,
    limit: int = 20,
    perturbation_kind: PerturbationKind | None = None,
) -> list[PerturbationProbe]:
    """Most-recent perturbations for an anchor, newest-first."""
    with get_session() as session:
        stmt = (
            select(PerturbationProbeRow)
            .where(PerturbationProbeRow.anchor_id == anchor_id)
            .order_by(
                PerturbationProbeRow.created_at.desc(),
                PerturbationProbeRow.id.desc(),
            )
            .limit(limit)
        )
        if perturbation_kind is not None:
            stmt = stmt.where(
                PerturbationProbeRow.perturbation_kind == perturbation_kind
            )
        rows = session.execute(stmt).scalars().all()
        return [_row_to_probe(r) for r in rows]


def get_perturbation_row_id(
    anchor_id: str, transform_label: str
) -> int | None:
    """Find the most-recent row id for a (anchor, transform_label) pair."""
    with get_session() as session:
        row = session.execute(
            select(PerturbationProbeRow.id)
            .where(
                PerturbationProbeRow.anchor_id == anchor_id,
                PerturbationProbeRow.transform_label == transform_label,
            )
            .order_by(PerturbationProbeRow.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        return row


__all__ = [
    "PerturbationProbeRow",
    "get_perturbation_row_id",
    "record_perturbation",
    "recent_perturbations",
]
