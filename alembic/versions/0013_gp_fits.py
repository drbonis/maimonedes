"""gp_fits table

Revision ID: 0013_gp_fits
Revises: 0012_audit_runs
Create Date: 2026-04-30 00:00:03

Phase 5 GP-layer registry. Tracks each fit on disk + its kernel +
log-marginal-likelihood so the dashboard can pick a recent fit
without scanning the filesystem.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013_gp_fits"
down_revision: str | None = "0012_audit_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "gp_fits",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("path", sa.String(length=256), nullable=False),
        sa.Column("policy_id", sa.String(length=64), nullable=False),
        sa.Column(
            "trained_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column("kernel_name", sa.String(length=256), nullable=False),
        sa.Column("log_marginal_likelihood", sa.Float(), nullable=False),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
    )
    op.create_index("ix_gp_fits_policy_id", "gp_fits", ["policy_id"])


def downgrade() -> None:
    op.drop_index("ix_gp_fits_policy_id", table_name="gp_fits")
    op.drop_table("gp_fits")
