"""synthesized_probes table + compliance_scores.synthesized_probe_id

Revision ID: 0014_synthesized_probes
Revises: 0013_gp_fits
Create Date: 2026-04-30 00:00:04

Phase 5 probe synthesis (#42 / #43) records every LLM-generated
probe along with its provenance — target embedding, achieved
embedding, parent anchor exemplars, generation method, validator
verdict — distinct from the curated anchors_v1.yaml. The
compliance_scores FK lets generated probes' scores live alongside
curated probes' scores in the same table the rest of the codebase
already consumes.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014_synthesized_probes"
down_revision: str | None = "0013_gp_fits"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def upgrade() -> None:
    op.create_table(
        "synthesized_probes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("policy_id", sa.String(length=64), nullable=False),
        sa.Column("scenario", sa.Text(), nullable=False),
        sa.Column("generation_method", sa.String(length=32), nullable=False),
        sa.Column("target_embedding_json", sa.Text(), nullable=False),
        sa.Column("achieved_embedding_json", sa.Text(), nullable=False),
        sa.Column("tau_distance", sa.Float(), nullable=False),
        sa.Column("parent_anchor_ids_json", sa.Text(), nullable=False),
        sa.Column(
            "synthesizer_llm_call_id",
            sa.Integer(),
            sa.ForeignKey(
                "llm_calls.id",
                name="fk_synthesized_probes_synthesizer_llm_call_id_llm_calls",
            ),
            nullable=True,
        ),
        sa.Column(
            "validator_llm_call_id",
            sa.Integer(),
            sa.ForeignKey(
                "llm_calls.id",
                name="fk_synthesized_probes_validator_llm_call_id_llm_calls",
            ),
            nullable=True,
        ),
        sa.Column("quality_status", sa.String(length=16), nullable=False),
        sa.Column("quality_reason", sa.Text(), nullable=True),
        sa.Column(
            "gp_fit_id",
            sa.Integer(),
            sa.ForeignKey(
                "gp_fits.id",
                name="fk_synthesized_probes_gp_fit_id_gp_fits",
            ),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
    )
    op.create_index(
        "ix_synthesized_probes_policy_id_created_at",
        "synthesized_probes",
        ["policy_id", "created_at"],
    )
    op.create_index(
        "ix_synthesized_probes_quality_status",
        "synthesized_probes",
        ["quality_status"],
    )
    op.create_index(
        "ix_synthesized_probes_gp_fit_id",
        "synthesized_probes",
        ["gp_fit_id"],
    )

    with op.batch_alter_table(
        "compliance_scores", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "synthesized_probe_id",
                sa.Integer(),
                sa.ForeignKey(
                    "synthesized_probes.id",
                    name="fk_compliance_scores_synthesized_probe_id_synthesized_probes",
                ),
                nullable=True,
            )
        )

    op.create_index(
        "ix_compliance_scores_synthesized_probe_id",
        "compliance_scores",
        ["synthesized_probe_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_compliance_scores_synthesized_probe_id",
        table_name="compliance_scores",
    )
    with op.batch_alter_table(
        "compliance_scores", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_column("synthesized_probe_id")

    op.drop_index(
        "ix_synthesized_probes_gp_fit_id", table_name="synthesized_probes"
    )
    op.drop_index(
        "ix_synthesized_probes_quality_status",
        table_name="synthesized_probes",
    )
    op.drop_index(
        "ix_synthesized_probes_policy_id_created_at",
        table_name="synthesized_probes",
    )
    op.drop_table("synthesized_probes")
