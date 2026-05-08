"""gp_fits.kernel_kind + gp_fits.metric_fit_id

Revision ID: 0018_gp_fits_kernel_kind
Revises: 0017_recovery_runs_contamination_mode
Create Date: 2026-05-07 00:00:00

Issue #62 / patent disclosure §4.5.3, §6.8 — adds the
`riemannian_pullback` kernel to the GP layer. We need to record (a)
which of the three kernel kinds a fit used, and (b) which metric_fits
row it consumed (only meaningful for the riemannian path).

`kernel_kind`:
  - "stationary"          (RBF, default)
  - "non_stationary"      (Gibbs kernel, issue #49)
  - "riemannian_pullback" (issue #62)

`metric_fit_id` is non-NULL only for `riemannian_pullback` fits.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0018_gp_fits_kernel_kind"
down_revision: str | None = "0017_recovery_runs_contamination_mode"
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
        "gp_fits", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.add_column(
            sa.Column(
                "kernel_kind",
                sa.String(32),
                nullable=False,
                server_default="stationary",
            )
        )
        batch_op.add_column(
            sa.Column(
                "metric_fit_id",
                sa.Integer(),
                nullable=True,
            )
        )
        batch_op.create_foreign_key(
            "fk_gp_fits_metric_fit_id_metric_fits",
            "metric_fits",
            ["metric_fit_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    with op.batch_alter_table(
        "gp_fits", naming_convention=NAMING_CONVENTION
    ) as batch_op:
        batch_op.drop_constraint(
            "fk_gp_fits_metric_fit_id_metric_fits", type_="foreignkey"
        )
        batch_op.drop_column("metric_fit_id")
        batch_op.drop_column("kernel_kind")
