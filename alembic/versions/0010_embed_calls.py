"""embed_calls table

Revision ID: 0010_embed_calls
Revises: 0009_structural_signals
Create Date: 2026-04-30 00:00:00

Phase 5 introduces the Bio_ClinicalBERT embedding service as a first-
class component. Mirrors `llm_calls`: every embed call is recorded
for replay + audit. Embeddings (~5 KB per row at 768-dim float JSON)
are stored inline rather than in a sibling blob table — at the
projected scale (tens of thousands of rows max) the storage cost is
trivial and inline storage keeps replay queries simple.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010_embed_calls"
down_revision: str | None = "0009_structural_signals"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "embed_calls",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column("backend_name", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("request_text", sa.Text(), nullable=False),
        sa.Column("embedding_json", sa.Text(), nullable=False),
        sa.Column("raw_response_json", sa.Text(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
    )
    op.create_index(
        "ix_embed_calls_timestamp", "embed_calls", ["timestamp"]
    )
    op.create_index("ix_embed_calls_model", "embed_calls", ["model"])
    op.create_index(
        "ix_embed_calls_request_hash", "embed_calls", ["request_hash"]
    )


def downgrade() -> None:
    op.drop_index("ix_embed_calls_request_hash", table_name="embed_calls")
    op.drop_index("ix_embed_calls_model", table_name="embed_calls")
    op.drop_index("ix_embed_calls_timestamp", table_name="embed_calls")
    op.drop_table("embed_calls")
