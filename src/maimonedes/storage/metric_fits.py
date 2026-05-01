"""ORM mapping + repository helpers for Riemannian metric fits.

A `metric_fit` row records one trained metric — the on-disk path of
the `.npz` model file plus enough provenance (policy_id, n_anchors,
n_jacobians, val_loss) to pick the right metric for a CLI command
without re-fitting. Mirrors the shape of `recovery_runs` (a record
of a fit/run + a pointer to its artefacts).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MetricFitRow(Base):
    __tablename__ = "metric_fits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    n_anchors: Mapped[int] = mapped_column(Integer, nullable=False)
    n_jacobians: Mapped[int] = mapped_column(Integer, nullable=False)
    val_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    train_loss: Mapped[float | None] = mapped_column(Float, nullable=True)
    hyperparams_json: Mapped[str] = mapped_column(
        Text, nullable=False, default="{}"
    )


def _row_to_dict(row: MetricFitRow) -> dict[str, object]:
    trained = row.trained_at
    if trained.tzinfo is None:
        trained = trained.replace(tzinfo=timezone.utc)
    try:
        hp = json.loads(row.hyperparams_json)
    except json.JSONDecodeError:
        hp = {}
    return {
        "id": row.id,
        "policy_id": row.policy_id,
        "path": row.path,
        "trained_at": trained,
        "n_anchors": row.n_anchors,
        "n_jacobians": row.n_jacobians,
        "val_loss": row.val_loss,
        "train_loss": row.train_loss,
        "hyperparams": hp,
    }


def record_metric_fit(
    *,
    policy_id: str,
    path: str,
    n_anchors: int,
    n_jacobians: int,
    val_loss: float | None,
    train_loss: float | None,
    hyperparams: dict[str, object] | None = None,
) -> int:
    """Insert a `metric_fit` row and return its id."""
    payload = json.dumps(hyperparams or {}, sort_keys=True, default=str)
    with get_session() as session:
        row = MetricFitRow(
            policy_id=policy_id,
            path=path,
            n_anchors=n_anchors,
            n_jacobians=n_jacobians,
            val_loss=val_loss,
            train_loss=train_loss,
            hyperparams_json=payload,
        )
        session.add(row)
        session.flush()
        return row.id


def get_metric_fit(fit_id: int) -> dict[str, object] | None:
    with get_session() as session:
        row = session.get(MetricFitRow, fit_id)
        return _row_to_dict(row) if row is not None else None


def latest_metric_fit_for_policy(policy_id: str) -> dict[str, object] | None:
    """Return the newest fit for `policy_id`, or None if none exist."""
    with get_session() as session:
        row = session.execute(
            select(MetricFitRow)
            .where(MetricFitRow.policy_id == policy_id)
            .order_by(
                MetricFitRow.trained_at.desc(),
                MetricFitRow.id.desc(),
            )
            .limit(1)
        ).scalar_one_or_none()
        return _row_to_dict(row) if row is not None else None


def list_metric_fits(
    policy_id: str | None = None, *, limit: int = 50
) -> list[dict[str, object]]:
    """Most-recent first; optionally scoped to a policy."""
    with get_session() as session:
        stmt = select(MetricFitRow).order_by(
            MetricFitRow.trained_at.desc(), MetricFitRow.id.desc()
        ).limit(limit)
        if policy_id is not None:
            stmt = stmt.where(MetricFitRow.policy_id == policy_id)
        rows = session.execute(stmt).scalars().all()
        return [_row_to_dict(r) for r in rows]


__all__ = [
    "MetricFitRow",
    "get_metric_fit",
    "latest_metric_fit_for_policy",
    "list_metric_fits",
    "record_metric_fit",
]
