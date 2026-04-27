"""SQLAlchemy ORM models.

Phase 0 keeps this minimal: only the declarative base and a
`SchemaVersion` placeholder so the framework has at least one table
to migrate against. Domain tables (probes, scores, llm_calls,
perturbations, ...) are introduced by later issues' migrations.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base shared by every model in the project."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SchemaVersion(Base):
    """Sentinel row recording when a baseline schema was applied.

    Alembic itself maintains `alembic_version`; this table is a
    user-visible record we can inspect and extend with metadata
    without colliding with Alembic's bookkeeping.
    """

    __tablename__ = "schema_version"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    label: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
