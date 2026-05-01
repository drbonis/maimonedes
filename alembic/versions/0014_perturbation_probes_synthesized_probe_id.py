"""perturbation_probes.synthesized_probe_id

Revision ID: 0014_perturbation_probes_synthesized_probe_id
Revises: 0013_stage2_head_kind
Create Date: 2026-05-01 00:00:01

Phase 5 #53 — `maimonedes perturb --synthesized <id>` extends the
Phase 2 perturbation cloud to GP-driven synthesized probes. Adds a
nullable `synthesized_probe_id` FK on `perturbation_probes` plus an
index. Soft invariant (not enforced via CHECK because SQLite's
multi-column CHECK is rough): for any row, exactly one of
`anchor_id` / `synthesized_probe_id` is the parent — `anchor_id`
remains required (it carries the namespaced id like
"synth-1#authority:..." for synthesized parents), but
`synthesized_probe_id` is the explicit FK back to
`synthesized_probes.id`.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014_perturbation_probes_synthesized_probe_id"
down_revision: str | None = "0013_stage2_head_kind"
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
    with op.batch_alter_table(
        "perturbation_probes", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "synthesized_probe_id",
                sa.Integer(),
                sa.ForeignKey(
                    "synthesized_probes.id",
                    name="fk_perturbation_probes_synthesized_probe_id_synthesized_probes",
                ),
                nullable=True,
            )
        )
    op.create_index(
        "ix_perturbation_probes_synthesized_probe_id",
        "perturbation_probes",
        ["synthesized_probe_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_perturbation_probes_synthesized_probe_id",
        table_name="perturbation_probes",
    )
    with op.batch_alter_table(
        "perturbation_probes", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_column("synthesized_probe_id")
