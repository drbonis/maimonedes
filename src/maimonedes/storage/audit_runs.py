"""ORM model + repository helpers for the audit_runs table."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AuditRunRow(Base):
    __tablename__ = "audit_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    stage2_model_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("stage2_models.id"),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    mae_per_axis_json: Mapped[str] = mapped_column(Text, nullable=False)
    spearman_per_axis_json: Mapped[str] = mapped_column(Text, nullable=False)
    agreement_status: Mapped[str] = mapped_column(String(16), nullable=False)
    trigger_reason: Mapped[str] = mapped_column(String(32), nullable=False)


def list_audit_runs(stage2_model_id: int) -> list[AuditRunRow]:
    with get_session() as session:
        rows = list(
            session.execute(
                select(AuditRunRow)
                .where(AuditRunRow.stage2_model_id == stage2_model_id)
                .order_by(AuditRunRow.id.asc())
            )
            .scalars()
            .all()
        )
        for r in rows:
            session.expunge(r)
        return rows


def get_audit_run(audit_run_id: int) -> AuditRunRow | None:
    with get_session() as session:
        row = session.get(AuditRunRow, audit_run_id)
        if row is None:
            return None
        session.expunge(row)
        return row


__all__ = ["AuditRunRow", "get_audit_run", "list_audit_runs"]
