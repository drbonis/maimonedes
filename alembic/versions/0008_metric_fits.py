"""metric_fits table — Riemannian metric fit registry

Revision ID: 0008_metric_fits
Revises: 0007_recovery_runs
Create Date: 2026-05-01 00:00:00

Phase 5's storage substrate for the Riemannian metric learner. Each
row is one trained metric: a pointer to the on-disk `.npz` artefact
plus enough provenance (policy_id, n_anchors, n_jacobians, val_loss,
hyperparams) for downstream commands to pick the right metric without
re-fitting. Issue #50 spec called this `0012_metric_fits` assuming
intermediate Phase-5 issues would land first; we file under the next
free slot so the chain stays linear.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_metric_fits"
down_revision: str | None = "0007_recovery_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "metric_fits",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("policy_id", sa.String(length=64), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column(
            "trained_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column("n_anchors", sa.Integer(), nullable=False),
        sa.Column("n_jacobians", sa.Integer(), nullable=False),
        sa.Column("val_loss", sa.Float(), nullable=True),
        sa.Column("train_loss", sa.Float(), nullable=True),
        sa.Column(
            "hyperparams_json",
            sa.Text(),
            nullable=False,
            server_default="{}",
        ),
    )
    op.create_index(
        "ix_metric_fits_policy_id",
        "metric_fits",
        ["policy_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_metric_fits_policy_id", table_name="metric_fits")
    op.drop_table("metric_fits")
