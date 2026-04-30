"""ORM model + repository helpers for the recorded-embed-calls table."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text, select
from sqlalchemy.orm import Mapped, mapped_column

from maimonedes.llm.embed_client import EmbedResponse
from maimonedes.storage.models import Base
from maimonedes.storage.repo import get_session


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EmbedCall(Base):
    __tablename__ = "embed_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, index=True
    )
    backend_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    request_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    raw_response_json: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)


def record_embed_call(
    *,
    backend_name: str,
    model: str,
    request_text: str,
    embedding: list[float],
    raw_response: dict[str, Any],
    latency_ms: float,
    request_hash: str,
) -> int:
    """Insert one embed_calls row and return the id."""
    embedding_json = json.dumps(embedding)
    raw_json = json.dumps(raw_response, sort_keys=True, default=str, ensure_ascii=False)
    with get_session() as session:
        row = EmbedCall(
            backend_name=backend_name,
            model=model,
            request_text=request_text,
            embedding_json=embedding_json,
            raw_response_json=raw_json,
            latency_ms=latency_ms,
            request_hash=request_hash,
        )
        session.add(row)
        session.flush()
        return row.id


def cached_embed(
    *,
    backend_name: str,
    model: str,
    request_hash: str,
) -> EmbedResponse | None:
    """Return the most-recent cached `EmbedResponse` matching the key, or None."""
    with get_session() as session:
        row = (
            session.execute(
                select(EmbedCall)
                .where(
                    EmbedCall.backend_name == backend_name,
                    EmbedCall.model == model,
                    EmbedCall.request_hash == request_hash,
                )
                .order_by(EmbedCall.id.desc())
                .limit(1)
            )
            .scalar_one_or_none()
        )
        if row is None:
            return None
        try:
            embedding = json.loads(row.embedding_json)
            raw = json.loads(row.raw_response_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(embedding, list) or not embedding:
            return None
        return EmbedResponse(
            embedding=[float(v) for v in embedding],
            model=row.model,
            latency_ms=row.latency_ms,
            raw=raw if isinstance(raw, dict) else {},
        )


def latest_embed_for_text(
    *,
    backend_name: str,
    model: str,
    text: str,
) -> EmbedResponse | None:
    """Convenience: lookup-by-text without caller computing the hash."""
    from maimonedes.llm.recording_embed_client import compute_request_hash

    return cached_embed(
        backend_name=backend_name,
        model=model,
        request_hash=compute_request_hash(model=model, text=text),
    )


__all__ = [
    "EmbedCall",
    "cached_embed",
    "latest_embed_for_text",
    "record_embed_call",
]
