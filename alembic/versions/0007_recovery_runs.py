"""recovery_runs + feedbacks + compliance_scores.recovery_run_id

Revision ID: 0007_recovery_runs
Revises: 0006_drift_sessions
Create Date: 2026-04-29 12:00:00

Phase 4's storage substrate. A `recovery_run` is one execution of
the closed-loop step (localize → contrastive → synthesize → re-run)
on top of a parent drift run; one `feedback` row per (recovery_run,
anchor) carries the synthesized recommendation. `compliance_scores`
gains a nullable FK to `recovery_runs.id` so anchor and perturbation
re-evaluations under feedback land in the same table the rest of
the codebase already reads.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_recovery_runs"
down_revision: str | None = "0006_drift_sessions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Same naming convention used by 0005 / 0006 — needed for SQLite
# `batch_alter_table` to reflect the existing FKs on
# `compliance_scores` without colliding.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def upgrade() -> None:
    op.create_table(
        "recovery_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "parent_drift_run_id",
            sa.Integer(),
            sa.ForeignKey(
                "drift_runs.id",
                name="fk_recovery_runs_parent_drift_run_id_drift_runs",
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
        sa.Column("supervised_model", sa.String(length=128), nullable=False),
        sa.Column("judge_model", sa.String(length=128), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("contrastive_kind", sa.String(length=16), nullable=False),
    )
    op.create_index(
        "ix_recovery_runs_parent_drift_run_id",
        "recovery_runs",
        ["parent_drift_run_id"],
    )

    op.create_table(
        "feedbacks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "recovery_run_id",
            sa.Integer(),
            sa.ForeignKey(
                "recovery_runs.id",
                name="fk_feedbacks_recovery_run_id_recovery_runs",
            ),
            nullable=False,
        ),
        sa.Column(
            "parent_drift_run_id",
            sa.Integer(),
            sa.ForeignKey(
                "drift_runs.id",
                name="fk_feedbacks_parent_drift_run_id_drift_runs",
            ),
            nullable=False,
        ),
        sa.Column("anchor_id", sa.String(length=64), nullable=False),
        sa.Column("contrastive_kind", sa.String(length=16), nullable=False),
        sa.Column("feedback_text", sa.Text(), nullable=False),
        sa.Column(
            "llm_call_id",
            sa.Integer(),
            sa.ForeignKey(
                "llm_calls.id",
                name="fk_feedbacks_llm_call_id_llm_calls",
            ),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.UniqueConstraint(
            "recovery_run_id",
            "anchor_id",
            name="uq_feedbacks_recovery_run_id_anchor_id",
        ),
    )
    op.create_index(
        "ix_feedbacks_recovery_run_id",
        "feedbacks",
        ["recovery_run_id"],
    )
    op.create_index(
        "ix_feedbacks_parent_drift_run_id",
        "feedbacks",
        ["parent_drift_run_id"],
    )

    with op.batch_alter_table(
        "compliance_scores", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "recovery_run_id",
                sa.Integer(),
                sa.ForeignKey(
                    "recovery_runs.id",
                    name="fk_compliance_scores_recovery_run_id_recovery_runs",
                ),
                nullable=True,
            )
        )

    op.create_index(
        "ix_compliance_scores_recovery_run_id",
        "compliance_scores",
        ["recovery_run_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_compliance_scores_recovery_run_id",
        table_name="compliance_scores",
    )
    with op.batch_alter_table(
        "compliance_scores", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_column("recovery_run_id")

    op.drop_index("ix_feedbacks_parent_drift_run_id", table_name="feedbacks")
    op.drop_index("ix_feedbacks_recovery_run_id", table_name="feedbacks")
    op.drop_table("feedbacks")

    op.drop_index(
        "ix_recovery_runs_parent_drift_run_id",
        table_name="recovery_runs",
    )
    op.drop_table("recovery_runs")
