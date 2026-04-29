"""drift_runs + drift_sessions + compliance_scores.drift_session_id

Revision ID: 0006_drift_sessions
Revises: 0005_compliance_scores_perturbation_id
Create Date: 2026-04-29 00:00:00

Phase 3's storage substrate. A `drift_run` represents one execution
of the synthetic drift schedule (50 sessions across 5 stages by
default); `drift_session` rows record per-step metadata (session
index, stage label, suffix text) so every compliance score collected
under the contaminated prompt traces back to the exact step that
produced it. `compliance_scores` gains a nullable FK to
`drift_sessions.id` — pre-existing anchor and perturbation rows keep
NULL there.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_drift_sessions"
down_revision: str | None = "0005_compliance_scores_perturbation_id"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Reuse the 0005 naming convention so SQLite batch_alter_table can
# reflect the existing `compliance_scores` FKs without colliding.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def upgrade() -> None:
    op.create_table(
        "drift_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("policy_id", sa.String(length=64), nullable=False),
        sa.Column("supervised_model", sa.String(length=128), nullable=False),
        sa.Column("judge_model", sa.String(length=128), nullable=False),
        sa.Column("schedule_path", sa.String(length=256), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "k_threshold",
            sa.Float(),
            nullable=False,
            server_default=sa.text("4.0"),
        ),
    )

    op.create_table(
        "drift_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "drift_run_id",
            sa.Integer(),
            sa.ForeignKey(
                "drift_runs.id",
                name="fk_drift_sessions_drift_run_id_drift_runs",
            ),
            nullable=False,
        ),
        sa.Column("session_index", sa.Integer(), nullable=False),
        sa.Column("stage_label", sa.String(length=32), nullable=False),
        sa.Column("suffix_text", sa.Text(), nullable=False),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint(
            "drift_run_id",
            "session_index",
            name="uq_drift_sessions_drift_run_id_session_index",
        ),
    )
    op.create_index(
        "ix_drift_sessions_drift_run_id",
        "drift_sessions",
        ["drift_run_id"],
    )

    with op.batch_alter_table(
        "compliance_scores", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "drift_session_id",
                sa.Integer(),
                sa.ForeignKey(
                    "drift_sessions.id",
                    name="fk_compliance_scores_drift_session_id_drift_sessions",
                ),
                nullable=True,
            )
        )

    op.create_index(
        "ix_compliance_scores_drift_session_id",
        "compliance_scores",
        ["drift_session_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_compliance_scores_drift_session_id",
        table_name="compliance_scores",
    )
    with op.batch_alter_table(
        "compliance_scores", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_column("drift_session_id")

    op.drop_index(
        "ix_drift_sessions_drift_run_id",
        table_name="drift_sessions",
    )
    op.drop_table("drift_sessions")
    op.drop_table("drift_runs")
