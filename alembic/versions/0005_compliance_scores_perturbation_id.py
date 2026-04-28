"""compliance_scores: perturbation_id + probe_role

Revision ID: 0005_compliance_scores_perturbation_id
Revises: 0004_perturbation_probes
Create Date: 2026-04-28 00:00:01

Adds `perturbation_id` (nullable FK to perturbation_probes.id) and
`probe_role` (default "anchor") so a single `compliance_scores` row
can record either an anchor evaluation or a perturbation evaluation.
Existing rows backfill to `probe_role = "anchor"` via the
`server_default`; FK column starts NULL for those rows.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_compliance_scores_perturbation_id"
down_revision: str | None = "0004_perturbation_probes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Auto-naming convention so SQLite batch_alter_table doesn't trip over
# the existing unnamed `llm_call_id` FK when it reflects the table.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def upgrade() -> None:
    with op.batch_alter_table(
        "compliance_scores", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "perturbation_id",
                sa.Integer(),
                sa.ForeignKey(
                    "perturbation_probes.id",
                    name="fk_compliance_scores_perturbation_id_perturbation_probes",
                ),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "probe_role",
                sa.String(length=16),
                nullable=False,
                server_default="anchor",
            )
        )

    op.create_index(
        "ix_compliance_scores_perturbation_id",
        "compliance_scores",
        ["perturbation_id"],
    )
    op.create_index(
        "ix_compliance_scores_role_anchor",
        "compliance_scores",
        ["probe_role", "anchor_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_compliance_scores_role_anchor", table_name="compliance_scores"
    )
    op.drop_index(
        "ix_compliance_scores_perturbation_id", table_name="compliance_scores"
    )
    with op.batch_alter_table(
        "compliance_scores", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_column("probe_role")
        batch_op.drop_column("perturbation_id")
