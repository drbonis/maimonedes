"""ORM model + helpers for the gp_fits registry."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GPFitRow(Base):
    __tablename__ = "gp_fits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String(256), nullable=False)
    policy_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    n_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    kernel_name: Mapped[str] = mapped_column(String(256), nullable=False)
    log_marginal_likelihood: Mapped[float] = mapped_column(Float, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    # Issue #62: normalized kernel kind ("stationary" | "non_stationary" |
    # "riemannian_pullback") + FK to the consumed metric_fits row (riemannian
    # pullback only). Older rows default to "stationary" / NULL via migration
    # 0018.
    kernel_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="stationary"
    )
    metric_fit_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("metric_fits.id", ondelete="SET NULL"), nullable=True
    )


def record_gp_fit(
    *,
    path: str,
    policy_id: str,
    n_samples: int,
    kernel_name: str,
    log_marginal_likelihood: float,
    embedding_model: str,
    kernel_kind: str = "stationary",
    metric_fit_id: int | None = None,
) -> int:
    with get_session() as session:
        row = GPFitRow(
            path=path,
            policy_id=policy_id,
            n_samples=n_samples,
            kernel_name=kernel_name,
            log_marginal_likelihood=log_marginal_likelihood,
            embedding_model=embedding_model,
            kernel_kind=kernel_kind,
            metric_fit_id=metric_fit_id,
        )
        session.add(row)
        session.flush()
        return row.id


def get_gp_fit(fit_id: int) -> GPFitRow | None:
    with get_session() as session:
        row = session.get(GPFitRow, fit_id)
        if row is None:
            return None
        session.expunge(row)
        return row


def list_gp_fits(policy_id: str | None = None) -> list[GPFitRow]:
    with get_session() as session:
        stmt = select(GPFitRow).order_by(
            GPFitRow.trained_at.desc(), GPFitRow.id.desc()
        )
        if policy_id is not None:
            stmt = stmt.where(GPFitRow.policy_id == policy_id)
        rows = list(session.execute(stmt).scalars().all())
        for r in rows:
            session.expunge(r)
        return rows


def latest_gp_fit_row(policy_id: str) -> GPFitRow | None:
    rows = list_gp_fits(policy_id=policy_id)
    return rows[0] if rows else None


__all__ = [
    "GPFitRow",
    "get_gp_fit",
    "latest_gp_fit_row",
    "list_gp_fits",
    "record_gp_fit",
]
