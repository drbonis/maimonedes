"""ORM model + repository helpers for the stage2_models registry."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Stage2ModelRow(Base):
    __tablename__ = "stage2_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(256), nullable=False)
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    mae_per_axis_json: Mapped[str] = mapped_column(Text, nullable=False)
    spearman_per_axis_json: Mapped[str] = mapped_column(Text, nullable=False)
    agreement_status: Mapped[str] = mapped_column(String(16), nullable=False)


def record_stage2_model(
    *,
    path: str,
    policy_id: str,
    n_samples: int,
    embedding_model: str,
    mae_per_axis: dict[str, float],
    spearman_per_axis: dict[str, float],
    agreement_status: str,
) -> int:
    with get_session() as session:
        row = Stage2ModelRow(
            path=path,
            policy_id=policy_id,
            n_samples=n_samples,
            embedding_model=embedding_model,
            mae_per_axis_json=json.dumps(mae_per_axis, sort_keys=True),
            spearman_per_axis_json=json.dumps(spearman_per_axis, sort_keys=True),
            agreement_status=agreement_status,
        )
        session.add(row)
        session.flush()
        return row.id


def get_stage2_model(model_id: int) -> Stage2ModelRow | None:
    with get_session() as session:
        row = session.get(Stage2ModelRow, model_id)
        if row is None:
            return None
        session.expunge(row)
        return row


def list_stage2_models(policy_id: str | None = None) -> list[Stage2ModelRow]:
    with get_session() as session:
        stmt = select(Stage2ModelRow).order_by(
            Stage2ModelRow.trained_at.desc(), Stage2ModelRow.id.desc()
        )
        if policy_id is not None:
            stmt = stmt.where(Stage2ModelRow.policy_id == policy_id)
        rows = list(session.execute(stmt).scalars().all())
        for r in rows:
            session.expunge(r)
        return rows


def latest_stage2_model_row(policy_id: str) -> Stage2ModelRow | None:
    rows = list_stage2_models(policy_id=policy_id)
    return rows[0] if rows else None


__all__ = [
    "Stage2ModelRow",
    "get_stage2_model",
    "latest_stage2_model_row",
    "list_stage2_models",
    "record_stage2_model",
]
