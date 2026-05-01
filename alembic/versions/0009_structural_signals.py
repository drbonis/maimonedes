"""structural_signals — fired-alerts feed for decoupling + curvature signals

Revision ID: 0009_structural_signals
Revises: 0008_metric_fits
Create Date: 2026-05-01 00:01:00

Phase 5's storage substrate for the structural early-warning signals
defined in §6.4: decoupling (covariance flip between policy axes) and
curvature (steeper local metric at the anchor's location). Each row
is one fired alert with its scalar metric value, the threshold it
crossed, and a JSON evidence blob explaining the fire (flipped pairs
for decoupling, condition-number ratio for curvature). Issue #51's
spec called this `0013_structural_signals`; we file it under the next
free slot (0009) so the chain stays linear with `0008_metric_fits`.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009_structural_signals"
down_revision: str | None = "0008_metric_fits"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "structural_signals",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("anchor_id", sa.String(length=64), nullable=False),
        sa.Column("signal_type", sa.String(length=16), nullable=False),
        sa.Column("metric_value", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column(
            "fired_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column(
            "evidence_json",
            sa.Text(),
            nullable=False,
            server_default="{}",
        ),
    )
    op.create_index(
        "ix_structural_signals_anchor_id",
        "structural_signals",
        ["anchor_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_structural_signals_anchor_id",
        table_name="structural_signals",
    )
    op.drop_table("structural_signals")
