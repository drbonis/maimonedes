"""ORM mapping + repository helpers for structural alert signals (issue #51).

A `structural_signal` row is one fired alert from the decoupling or
curvature detectors. The schema is intentionally narrow — `signal_type`
discriminates the two flavours, `metric_value` stores the scalar
above `threshold`, and `evidence_json` carries the per-detector blob
that explains why the alert fired (flipped covariance pairs,
condition-number ratio, etc.).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


SIGNAL_TYPES = ("decoupling", "curvature")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class StructuralSignalRow(Base):
    __tablename__ = "structural_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    anchor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    signal_type: Mapped[str] = mapped_column(String(16), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    fired_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    evidence_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )


def _row_to_dict(row: StructuralSignalRow) -> dict[str, object]:
    fired = row.fired_at
    if fired.tzinfo is None:
        fired = fired.replace(tzinfo=timezone.utc)
    try:
        evidence = json.loads(row.evidence_json)
    except json.JSONDecodeError:
        evidence = {}
    return {
        "id": row.id,
        "anchor_id": row.anchor_id,
        "signal_type": row.signal_type,
        "metric_value": row.metric_value,
        "threshold": row.threshold,
        "fired_at": fired,
        "evidence": evidence,
    }


def record_structural_signal(
    *,
    anchor_id: str,
    signal_type: str,
    metric_value: float,
    threshold: float,
    evidence: dict[str, object] | None = None,
) -> int:
    """Insert one fired signal; raises on unknown signal_type values."""
    if signal_type not in SIGNAL_TYPES:
        raise ValueError(
            f"unknown signal_type {signal_type!r}; expected one of {SIGNAL_TYPES}"
        )
    payload = json.dumps(evidence or {}, sort_keys=True, default=str)
    with get_session() as session:
        row = StructuralSignalRow(
            anchor_id=anchor_id,
            signal_type=signal_type,
            metric_value=metric_value,
            threshold=threshold,
            evidence_json=payload,
        )
        session.add(row)
        session.flush()
        return row.id


def signals_for_anchor(
    anchor_id: str, *, signal_type: str | None = None, limit: int = 50
) -> list[dict[str, object]]:
    """Most-recent first; optionally filter by signal_type."""
    with get_session() as session:
        stmt = (
            select(StructuralSignalRow)
            .where(StructuralSignalRow.anchor_id == anchor_id)
            .order_by(
                StructuralSignalRow.fired_at.desc(),
                StructuralSignalRow.id.desc(),
            )
            .limit(limit)
        )
        if signal_type is not None:
            stmt = stmt.where(StructuralSignalRow.signal_type == signal_type)
        rows = session.execute(stmt).scalars().all()
        return [_row_to_dict(r) for r in rows]


def list_structural_signals(
    *, signal_type: str | None = None, limit: int = 100
) -> list[dict[str, object]]:
    """Cross-anchor signal feed — useful for the dashboard timeline."""
    with get_session() as session:
        stmt = (
            select(StructuralSignalRow)
            .order_by(
                StructuralSignalRow.fired_at.desc(),
                StructuralSignalRow.id.desc(),
            )
            .limit(limit)
        )
        if signal_type is not None:
            stmt = stmt.where(StructuralSignalRow.signal_type == signal_type)
        rows = session.execute(stmt).scalars().all()
        return [_row_to_dict(r) for r in rows]


__all__ = [
    "SIGNAL_TYPES",
    "StructuralSignalRow",
    "list_structural_signals",
    "record_structural_signal",
    "signals_for_anchor",
]
