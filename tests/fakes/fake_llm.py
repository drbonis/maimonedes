"""In-memory `LLMClient` for tests.

Returns canned responses so unit tests can exercise the rest of the
framework without HTTP. Records every call it received so tests can
assert on dispatch behaviour (cache hits, retries, etc.).
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from maimonedes.llm.client import ChatResponse, Message


@dataclass
class CallRecord:
    messages: list[Message]
    model: str
    temperature: float | None
    max_tokens: int | None
    extra: dict[str, Any]


class FakeLLMClient:
    """Deterministic test double that satisfies the `LLMClient` Protocol."""

    def __init__(
        self,
        responses: Iterable[ChatResponse | str] | None = None,
        *,
        default_content: str = "PONG",
        default_model: str = "fake-model",
    ) -> None:
        self._queue: list[ChatResponse] = [
            self._coerce(r, default_model) for r in (responses or [])
        ]
        self._default_content = default_content
        self._default_model = default_model
        self.calls: list[CallRecord] = []

    @staticmethod
    def _coerce(item: ChatResponse | str, default_model: str) -> ChatResponse:
        if isinstance(item, ChatResponse):
            return item
        return ChatResponse(
            content=item,
            model=default_model,
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=0.0,
            raw={},
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
        self.calls.append(
            CallRecord(
                messages=list(messages),
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=dict(extra),
            )
        )
        if self._queue:
            return self._queue.pop(0)
        return ChatResponse(
            content=self._default_content,
            model=model or self._default_model,
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=0.0,
            raw={},
        )

    def queue(self, response: ChatResponse | str) -> None:
        self._queue.append(self._coerce(response, self._default_model))


__all__ = ["CallRecord", "FakeLLMClient"]
