"""Unified embedding-client contract.

Mirrors `LLMClient`: a `Protocol` so backends + recording wrappers
duck-type the same surface. Embedding-batch is a default-implemented
helper so callers (Stage-2 trainer, GP fitter) get one-call ergonomics
even when the backend doesn't natively batch.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class EmbedResponse(BaseModel):
    """Backend-agnostic shape returned by every `EmbedClient.embed`."""

    model_config = ConfigDict(extra="forbid")

    embedding: list[float] = Field(min_length=1)
    model: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    raw: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class EmbedClient(Protocol):
    """Structural contract every embedding backend (and decorator) implements."""

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        **extra: Any,
    ) -> EmbedResponse: ...

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **extra: Any,
    ) -> list[EmbedResponse]:
        """Default loop fallback. Backends may override for performance."""
        ...


def loop_embed_batch(
    client: EmbedClient,
    texts: Sequence[str],
    *,
    model: str | None = None,
    **extra: Any,
) -> list[EmbedResponse]:
    """Reusable default for `embed_batch` — call `embed` in a loop.

    Backends inherit this by importing it and assigning to their own
    `embed_batch` attribute, which keeps the Protocol's runtime check
    happy without forcing inheritance.
    """
    return [client.embed(text, model=model, **extra) for text in texts]


__all__ = [
    "EmbedClient",
    "EmbedResponse",
    "loop_embed_batch",
]
