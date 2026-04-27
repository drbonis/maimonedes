"""Minimal OpenAI-compatible client pointed at Ollama's `/v1` endpoint.

This is the just-enough implementation needed to wire up the Phase 0
`maimonedes ping` smoke command. Retries, structured timeouts, model
listing, and other production niceties are tracked in their own
issues; the surface here is intentionally small.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAI

from maimonedes.llm.client import (
    ChatResponse,
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
    Message,
)


class OllamaBackend:
    """OpenAI-compatible chat client pointed at an Ollama server.

    Conforms structurally to `LLMClient`; importing the Protocol is not
    required because consumers rely on duck typing.
    """

    backend_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "ollama",
        timeout_s: float = 60.0,
        client: OpenAI | None = None,
    ) -> None:
        self._base_url = base_url
        self._client = client or OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_s,
        )

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

        start = time.perf_counter()
        try:
            completion = self._client.chat.completions.create(**payload)
        except APITimeoutError as exc:
            raise LLMTimeoutError(str(exc)) from exc
        except APIConnectionError as exc:
            raise LLMConnectionError(str(exc)) from exc
        except APIStatusError as exc:
            raise LLMResponseError(str(exc), status_code=exc.status_code) from exc
        except httpx.HTTPError as exc:
            raise LLMConnectionError(str(exc)) from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        if not completion.choices:
            raise LLMResponseError("backend returned no choices")

        choice = completion.choices[0]
        content = (choice.message.content or "") if choice.message else ""

        usage = getattr(completion, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

        # `model_dump` works for pydantic-modelled SDK responses; fall back
        # otherwise so we always have something replayable.
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


__all__ = ["OllamaBackend"]
