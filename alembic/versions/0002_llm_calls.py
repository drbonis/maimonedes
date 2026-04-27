"""llm_calls table

Revision ID: 0002_llm_calls
Revises: 0001_baseline
Create Date: 2026-04-27 00:00:01

Adds the table that records every LLM round-trip made by
RecordingClient. Indexed on `request_hash`, `timestamp`, `model` so
replay lookups and time-window queries are fast.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_llm_calls"
down_revision: str | None = "0001_baseline"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "llm_calls",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.current_timestamp(),
        ),
        sa.Column("backend_name", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("request_messages_json", sa.Text(), nullable=False),
        sa.Column("response_content", sa.Text(), nullable=False),
        sa.Column("raw_response_json", sa.Text(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("request_hash", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_llm_calls_timestamp", "llm_calls", ["timestamp"])
    op.create_index("ix_llm_calls_model", "llm_calls", ["model"])
    op.create_index("ix_llm_calls_request_hash", "llm_calls", ["request_hash"])


def downgrade() -> None:
    op.drop_index("ix_llm_calls_request_hash", table_name="llm_calls")
    op.drop_index("ix_llm_calls_model", table_name="llm_calls")
    op.drop_index("ix_llm_calls_timestamp", table_name="llm_calls")
    op.drop_table("llm_calls")
