"""stage2_models.head_kind

Revision ID: 0015_stage2_head_kind
Revises: 0014_synthesized_probes
Create Date: 2026-04-30 00:00:05

Phase 5 #45 — Stage-2 optionally fits MLP heads (sklearn
MLPRegressor) instead of Ridge per axis. The head architecture
becomes a property of the model artefact, recorded on the registry
row so an operator querying `stage2_models` can tell at a glance
which heads each row's pickle uses.

Default `'ridge'` for back-compat with rows persisted by the
pre-#45 trainer.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015_stage2_head_kind"
down_revision: str | None = "0014_synthesized_probes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("stage2_models") as batch_op:
        batch_op.add_column(
            sa.Column(
                "head_kind",
                sa.String(length=16),
                nullable=False,
                server_default="ridge",
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("stage2_models") as batch_op:
        batch_op.drop_column("head_kind")
