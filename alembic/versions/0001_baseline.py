"""baseline schema

Revision ID: 0001_baseline
Revises:
Create Date: 2026-04-27 00:00:00

Creates the placeholder `schema_version` table so subsequent migrations
have a known starting point. Domain tables are added by later
migrations as their owning issues land.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "schema_version",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("label", sa.String(length=64), nullable=False),
        sa.Column(
            "applied_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.UniqueConstraint("label", name="uq_schema_version_label"),
    )


def downgrade() -> None:
    op.drop_table("schema_version")
