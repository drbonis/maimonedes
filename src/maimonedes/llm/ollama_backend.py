"""OpenAI-compatible client pointed at Ollama's `/v1` endpoint.

Ollama exposes an OpenAI-compatible chat-completions API, so this
backend wraps the official `openai` Python SDK with `base_url`
overridden. Both the supervised system and the judge are reached
through the same backend instance — only the `model` argument differs.

The SDK's own retry layer is disabled (`max_retries=0`) so this class
owns the retry policy: jittered exponential backoff over a configurable
attempt budget, retrying only on transient failures (network /
timeout / 5xx / 429). 4xx other than 429 are surfaced immediately
because they signal a request-shape bug, not a flake.
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from maimonedes.llm.client import (
    ChatResponse,
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    Message,
)

DEFAULT_BASE_URL = "http://192.168.1.30:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_REQUEST_TIMEOUT_S = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE_S = 0.5
DEFAULT_BACKOFF_FACTOR = 2.0


def _is_retryable(exc: BaseException) -> bool:
    """Network and server-side flakes are retryable; client-side bugs are not."""
    if isinstance(exc, (APIConnectionError, APITimeoutError, httpx.TransportError)):
        return True
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        return status == 429 or (status is not None and 500 <= status < 600)
    return False


def _translate(exc: BaseException) -> LLMError:
    if isinstance(exc, APITimeoutError):
        return LLMTimeoutError(str(exc))
    if isinstance(exc, APIConnectionError):
        return LLMConnectionError(str(exc))
    if isinstance(exc, APIStatusError):
        return LLMResponseError(str(exc), status_code=exc.status_code)
    if isinstance(exc, httpx.TimeoutException):
        return LLMTimeoutError(str(exc))
    if isinstance(exc, httpx.HTTPError):
        return LLMConnectionError(str(exc))
    return LLMResponseError(str(exc))


class OllamaBackend:
    """Stateless OpenAI-compatible chat client pointed at an Ollama server.

    Conforms structurally to `LLMClient`; importing the Protocol is not
    required because consumers rely on duck typing.
    """

    backend_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_s: float = DEFAULT_BACKOFF_BASE_S,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        client: OpenAI | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self._base_url = base_url
        # Disable SDK-level retries; this class owns retry semantics so they
        # can't compound or smuggle un-translated exceptions through.
        self._client = client or OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout_s,
            max_retries=0,
        )
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_factor = backoff_factor
        self._sleeper = sleeper
        self._rng = rng or random.Random()

    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload.update(extra)

        completion, latency_ms = self._call_with_retry(payload)

        if not completion.choices:
            raise LLMResponseError("backend returned no choices")

        choice = completion.choices[0]
        content = (choice.message.content or "") if choice.message else ""

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        try:
            raw = completion.model_dump()
        except AttributeError:
            raw = {"content": content}

        return ChatResponse(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw=raw,
        )

    # ---- internals ---------------------------------------------------------

    def _call_with_retry(self, payload: dict[str, Any]) -> tuple[Any, float]:
        attempts_allowed = self._max_retries + 1
        last_exc: BaseException | None = None

        for attempt_idx in range(attempts_allowed):
            start = time.perf_counter()
            try:
                completion = self._client.chat.completions.create(**payload)
            except Exception as exc:
                last_exc = exc
                attempts_left = attempts_allowed - attempt_idx - 1
                if attempts_left <= 0 or not _is_retryable(exc):
                    raise _translate(exc) from exc
                self._sleep_with_backoff(attempt_idx)
                continue
            latency_ms = (time.perf_counter() - start) * 1000.0
            return completion, latency_ms

        # Defensive: the loop always raises or returns above.
        assert last_exc is not None
        raise _translate(last_exc) from last_exc

    def _sleep_with_backoff(self, attempt_idx: int) -> None:
        # Backoff schedule on attempt_idx (0-based) is base * factor**idx,
        # jittered by uniform multiplier in [0.5, 1.5).
        base_delay = self._backoff_base_s * (self._backoff_factor ** attempt_idx)
        delay = base_delay * self._rng.uniform(0.5, 1.5)
        self._sleeper(delay)


__all__ = ["OllamaBackend"]
