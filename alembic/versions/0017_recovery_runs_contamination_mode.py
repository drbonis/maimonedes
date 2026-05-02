"""recovery_runs.contamination_mode + .contamination_stage_label

Revision ID: 0017_recovery_runs_contamination_mode
Revises: 0016_perturbation_probes_synthesized_probe_id
Create Date: 2026-05-01 00:00:02

Phase 4 #35 — `maimonedes apply-feedback --under-contamination` re-evaluates
each anchor with both the synthesized feedback AND the parent drift's
contamination suffix as system messages. The recovery_run row records
which mode was used so the report and dashboard can read the verdict
unambiguously: a +0.6 Δ-toward-baseline under "clean" is a much weaker
claim than the same Δ under "per_anchor_worst".

`contamination_mode` values:
  - "clean"                   — feedback only (default, pre-#35 behaviour)
  - "per_anchor_worst"        — feedback + the suffix from each anchor's
                                worst-aggregate session in the parent run
  - "fixed_stage:<label>"     — feedback + a single stage's suffix applied
                                uniformly across anchors

`contamination_stage_label` is non-NULL only in the fixed-stage mode and
holds the stage label (e.g. "concise") for quick filtering.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0017_recovery_runs_contamination_mode"
down_revision: str | None = "0016_perturbation_probes_synthesized_probe_id"
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
        "recovery_runs", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "contamination_mode",
                sa.String(32),
                nullable=False,
                server_default="clean",
            )
        )
        batch_op.add_column(
            sa.Column(
                "contamination_stage_label",
                sa.String(32),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "recovery_runs", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_column("contamination_stage_label")
        batch_op.drop_column("contamination_mode")
