"""stage2_models table

Revision ID: 0011_stage2_models
Revises: 0010_embed_calls
Create Date: 2026-04-30 00:00:01

Phase 5 Stage-2 classifier registry. Each row tracks one trained
artefact on disk (path) plus its per-axis agreement metrics vs the
Stage-1 LLM judge so the audit detector can find the latest model
without scanning the filesystem.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011_stage2_models"
down_revision: str | None = "0010_embed_calls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "stage2_models",
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
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("mae_per_axis_json", sa.Text(), nullable=False),
        sa.Column("spearman_per_axis_json", sa.Text(), nullable=False),
        sa.Column("agreement_status", sa.String(length=16), nullable=False),
    )
    op.create_index(
        "ix_stage2_models_policy_id",
        "stage2_models",
        ["policy_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_stage2_models_policy_id", table_name="stage2_models")
    op.drop_table("stage2_models")
