"""compliance_scores table

Revision ID: 0003_compliance_scores
Revises: 0002_llm_calls
Create Date: 2026-04-27 00:00:02

Persists the result of running the Stage-1 LLM-as-Judge on one
supervised output. Composite indexes on (anchor_id, scored_at) and
(policy_id, scored_at) make the queries that drive Phase 3's CUSUM
monitor and Phase 1's dashboard cheap.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_compliance_scores"
down_revision: str | None = "0002_llm_calls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "compliance_scores",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("anchor_id", sa.String(length=64), nullable=False),
        sa.Column("policy_id", sa.String(length=64), nullable=False),
        sa.Column("per_sub_condition_json", sa.Text(), nullable=False),
        sa.Column("aggregate", sa.Float(), nullable=False),
        sa.Column("judge_model", sa.String(length=128), nullable=False),
        sa.Column("supervised_model", sa.String(length=128), nullable=False),
        sa.Column(
            "llm_call_id",
            sa.Integer(),
            sa.ForeignKey("llm_calls.id"),
            nullable=True,
        ),
        sa.Column(
            "scored_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_index(
        "ix_compliance_scores_anchor_scored_at",
        "compliance_scores",
        ["anchor_id", "scored_at"],
    )
    op.create_index(
        "ix_compliance_scores_policy_scored_at",
        "compliance_scores",
        ["policy_id", "scored_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_compliance_scores_policy_scored_at",
        table_name="compliance_scores",
    )
    op.drop_index(
        "ix_compliance_scores_anchor_scored_at",
        table_name="compliance_scores",
    )
    op.drop_table("compliance_scores")
