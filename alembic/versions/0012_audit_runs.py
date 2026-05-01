"""audit_runs table

Revision ID: 0012_audit_runs
Revises: 0011_stage2_models
Create Date: 2026-04-30 00:00:02

Phase 5 Stage-2 audit log. Each row records one re-routing of
Stage-2 outputs through the Stage-1 LLM judge — what the audit's
hybrid trigger fired on, how many samples it covered, and the
resulting per-axis MAE + Spearman ρ. The detector uses the latest
ended_at as the timer reset for the next "K hours elapsed" check.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012_audit_runs"
down_revision: str | None = "0011_stage2_models"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "stage2_model_id",
            sa.Integer(),
            sa.ForeignKey(
                "stage2_models.id",
                name="fk_audit_runs_stage2_model_id_stage2_models",
            ),
            nullable=False,
        ),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("n_samples", sa.Integer(), nullable=False),
        sa.Column("mae_per_axis_json", sa.Text(), nullable=False),
        sa.Column("spearman_per_axis_json", sa.Text(), nullable=False),
        sa.Column("agreement_status", sa.String(length=16), nullable=False),
        sa.Column("trigger_reason", sa.String(length=32), nullable=False),
    )
    op.create_index(
        "ix_audit_runs_stage2_model_id",
        "audit_runs",
        ["stage2_model_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_audit_runs_stage2_model_id", table_name="audit_runs")
    op.drop_table("audit_runs")
