"""perturbation_probes table

Revision ID: 0004_perturbation_probes
Revises: 0003_compliance_scores
Create Date: 2026-04-28 00:00:00

Phase 2 introduces generated probes derived from anchors by the four
perturbation generators (paraphrase, demographic, authority,
boundary). Each row carries enough provenance for the §4.4 Jacobian
table: parent anchor, generator kind, and a stable `transform_label`
that becomes the column header.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_perturbation_probes"
down_revision: str | None = "0003_compliance_scores"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "perturbation_probes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("anchor_id", sa.String(length=64), nullable=False),
        sa.Column("perturbation_kind", sa.String(length=32), nullable=False),
        sa.Column("transform_label", sa.String(length=128), nullable=False),
        sa.Column("scenario", sa.Text(), nullable=False),
        sa.Column("policy_id", sa.String(length=64), nullable=False),
        sa.Column(
            "generator_metadata_json",
            sa.Text(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_index(
        "ix_perturbation_probes_anchor_kind",
        "perturbation_probes",
        ["anchor_id", "perturbation_kind"],
    )
    op.create_index(
        "ix_perturbation_probes_anchor_label",
        "perturbation_probes",
        ["anchor_id", "transform_label"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_perturbation_probes_anchor_label",
        table_name="perturbation_probes",
    )
    op.drop_index(
        "ix_perturbation_probes_anchor_kind",
        table_name="perturbation_probes",
    )
    op.drop_table("perturbation_probes")
