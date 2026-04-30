"""HTTP client for the Bio_ClinicalBERT embedding service.

The service runs on the GPU laptop alongside Ollama at
`http://192.168.1.30:8000/embed` and accepts `{"text": "..."}`. The
exact response key is decided by the service implementation; this
backend tries the obvious names (`embedding`, `embeddings`, `vector`)
in order and falls back to scanning for the first list-of-floats so
the client stays robust to upstream renames.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from maimonedes.llm.client import (
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
)
from maimonedes.llm.embed_client import EmbedResponse, loop_embed_batch


DEFAULT_BASE_URL = "http://192.168.1.30:8000"
DEFAULT_MODEL = "bio_clinicalbert"
DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 0.5
DEFAULT_BACKOFF_FACTOR = 2.0

EMBED_KEYS_TRIED = ("embedding", "embeddings", "vector", "embed", "data")


def _extract_embedding(payload: dict[str, Any]) -> list[float] | None:
    """Find the embedding vector in a service response without assumptions."""
    for key in EMBED_KEYS_TRIED:
        if key in payload:
            value = payload[key]
            extracted = _coerce_to_float_list(value)
            if extracted is not None:
                return extracted
    # Last-ditch: scan top-level values for the first list-of-floats.
    for value in payload.values():
        extracted = _coerce_to_float_list(value)
        if extracted is not None:
            return extracted
    return None


def _coerce_to_float_list(value: Any) -> list[float] | None:
    """Accept `[float, ...]`, `[[float, ...]]` (batched 1-of-n), or nested."""
    if isinstance(value, list):
        if not value:
            return None
        if all(isinstance(v, (int, float)) for v in value):
            return [float(v) for v in value]
        if isinstance(value[0], list) and all(
            isinstance(v, (int, float)) for v in value[0]
        ):
            return [float(v) for v in value[0]]
    return None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or 500 <= status < 600
    if isinstance(exc, httpx.TransportError):
        return True
    return False


def _translate(exc: BaseException) -> LLMError:
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError(str(exc))
    if isinstance(exc, httpx.HTTPStatusError):
        return LLMResponseError(str(exc), status_code=exc.response.status_code)
    if isinstance(exc, httpx.TransportError):
        return LLMConnectionError(str(exc))
    return LLMResponseError(str(exc))


class ClinicalBertBackend:
    """HTTP client targeting the `/embed` endpoint."""

    backend_name = "clinicalbert"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_model = model
        self._client = client or httpx.Client(timeout=request_timeout_s)
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_factor = backoff_factor
        self._sleeper = sleeper
        self._rng = rng or random.Random()

    @property
    def default_model(self) -> str:
        return self._default_model

    def embed(
        self,
        text: str,
        *,
        model: str | None = None,
        **extra: Any,
    ) -> EmbedResponse:
        if not text:
            raise LLMResponseError("embed: text must be non-empty")
        chosen_model = model or self._default_model
        payload: dict[str, Any] = {"text": text}
        payload.update(extra)

        response_json, latency_ms = self._call_with_retry(payload)
        embedding = _extract_embedding(response_json)
        if embedding is None:
            raise LLMResponseError(
                f"embed: could not locate embedding vector in response keys "
                f"{sorted(response_json.keys())}"
            )

        return EmbedResponse(
            embedding=embedding,
            model=chosen_model,
            latency_ms=latency_ms,
            raw=response_json,
        )

    def embed_batch(
        self,
        texts: Sequence[str],
        *,
        model: str | None = None,
        **extra: Any,
    ) -> list[EmbedResponse]:
        return loop_embed_batch(self, texts, model=model, **extra)

    # ---- internals ---------------------------------------------------------

    def _call_with_retry(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], float]:
        attempts_allowed = self._max_retries + 1
        last_exc: BaseException | None = None
        url = f"{self._base_url}/embed"

        for attempt_idx in range(attempts_allowed):
            start = time.perf_counter()
            try:
                response = self._client.post(url, json=payload)
                response.raise_for_status()
            except Exception as exc:
                last_exc = exc
                attempts_left = attempts_allowed - attempt_idx - 1
                if attempts_left <= 0 or not _is_retryable(exc):
                    raise _translate(exc) from exc
                self._sleep_with_backoff(attempt_idx)
                continue
            latency_ms = (time.perf_counter() - start) * 1000.0
            try:
                body = response.json()
            except ValueError as exc:
                raise LLMResponseError(
                    f"embed: response was not valid JSON: {exc}"
                ) from exc
            if not isinstance(body, dict):
                raise LLMResponseError(
                    f"embed: expected JSON object, got {type(body).__name__}"
                )
            return body, latency_ms

        assert last_exc is not None
        raise _translate(last_exc) from last_exc

    def _sleep_with_backoff(self, attempt_idx: int) -> None:
        base_delay = self._backoff_base_s * (self._backoff_factor ** attempt_idx)
        delay = base_delay * self._rng.uniform(0.5, 1.5)
        self._sleeper(delay)


__all__ = ["ClinicalBertBackend", "DEFAULT_BASE_URL", "DEFAULT_MODEL"]
