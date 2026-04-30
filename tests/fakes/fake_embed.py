"""In-memory `EmbedClient` for tests.

Returns canned embeddings so unit tests can exercise the embedding
pipeline without HTTP. Records every call so tests can assert on
dispatch behaviour (cache hits, batch ordering, etc.).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from maimonedes.llm.embed_client import EmbedResponse, loop_embed_batch


@dataclass
class EmbedCallRecord:
    text: str
    model: str | None
    extra: dict[str, Any] = field(default_factory=dict)


class FakeEmbedClient:
    """Deterministic test double conforming to the `EmbedClient` Protocol."""

    default_model = "fake-embed"

    def __init__(
        self,
        responses: Iterable[EmbedResponse | list[float]] | None = None,
        *,
        default_dim: int = 8,
    ) -> None:
        self._queue: list[EmbedResponse] = [
            self._coerce(r, "fake-embed") for r in (responses or [])
        ]
        self._default_dim = default_dim
        self.calls: list[EmbedCallRecord] = []

    @staticmethod
    def _coerce(item: EmbedResponse | list[float], model: str) -> EmbedResponse:
        if isinstance(item, EmbedResponse):
            return item
        return EmbedResponse(
            embedding=[float(v) for v in item],
            model=model,
            latency_ms=0.0,
            raw={},
        )

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        **extra: Any,
    ) -> EmbedResponse:
        self.calls.append(
            EmbedCallRecord(text=text, model=model, extra=dict(extra))
        )
        if self._queue:
            return self._queue.pop(0)
        # Deterministic synthetic embedding: one-hot at the hash mod dim.
        dim = self._default_dim
        seed = sum(ord(c) for c in text) % dim
        vec = [0.0] * dim
        vec[seed] = 1.0
        return EmbedResponse(
            embedding=vec,
            model=model or self.default_model,
            latency_ms=0.0,
            raw={"synthetic": True},
        )

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **extra: Any,
    ) -> list[EmbedResponse]:
        return loop_embed_batch(self, texts, model=model, **extra)

    def queue(self, response: EmbedResponse | list[float]) -> None:
        self._queue.append(self._coerce(response, "fake-embed"))


__all__ = ["EmbedCallRecord", "FakeEmbedClient"]
