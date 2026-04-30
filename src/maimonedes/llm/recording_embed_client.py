"""Persisting / replaying decorator over any `EmbedClient`.

Mirrors `RecordingClient` for the embedding side. Embeddings are
high-dimensional but small per-row (~5 KB JSON for 768-dim float32),
so we store the full vector — replay then becomes free.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

from maimonedes.llm.embed_client import EmbedClient, EmbedResponse
from maimonedes.storage.embed_calls import (
    EmbedCall,
    cached_embed,
    record_embed_call,
)


log = logging.getLogger(__name__)


def compute_request_hash(*, model: str, text: str) -> str:
    """SHA-256 over normalised (model, text). Identical requests collide."""
    payload = {"model": model, "text": text}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RecordingEmbedClient:
    """Wrap any `EmbedClient`; persist every call; optionally replay."""

    def __init__(
        self,
        wrapped: EmbedClient,
        *,
        backend_name: str,
        replay: bool = False,
    ) -> None:
        self._wrapped = wrapped
        self._backend_name = backend_name
        self._replay = replay

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        **extra: Any,
    ) -> EmbedResponse:
        chosen_model = model or getattr(
            self._wrapped, "default_model", self._backend_name
        )
        request_hash = compute_request_hash(model=chosen_model, text=text)

        if self._replay:
            cached = cached_embed(
                backend_name=self._backend_name,
                model=chosen_model,
                request_hash=request_hash,
            )
            if cached is not None:
                log.info(
                    "recording_embed_client.cache_hit",
                    extra={
                        "backend": self._backend_name,
                        "model": chosen_model,
                        "request_hash": request_hash,
                    },
                )
                return cached
            log.info(
                "recording_embed_client.cache_miss",
                extra={
                    "backend": self._backend_name,
                    "model": chosen_model,
                    "request_hash": request_hash,
                },
            )

        start = time.perf_counter()
        response = self._wrapped.embed(text, model=model, **extra)
        if response.latency_ms == 0.0:
            response = response.model_copy(
                update={"latency_ms": (time.perf_counter() - start) * 1000.0}
            )

        record_embed_call(
            backend_name=self._backend_name,
            model=chosen_model,
            request_text=text,
            embedding=response.embedding,
            raw_response=response.raw,
            latency_ms=response.latency_ms,
            request_hash=request_hash,
        )
        return response

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **extra: Any,
    ) -> list[EmbedResponse]:
        return [self.embed(t, model=model, **extra) for t in texts]


__all__ = ["RecordingEmbedClient", "compute_request_hash"]
