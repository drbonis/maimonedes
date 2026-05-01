"""Unified LLM client contract.

A `Protocol` is used over an ABC: consumers (RecordingClient,
scorers, CLI) only need structural conformance — they never inherit
from `LLMClient`. This keeps the framework decoupled from the
concrete backend (Ollama today, anything else tomorrow) and lets the
recording / mocking layers wrap a single interface.

Streaming is intentionally out of scope for v1.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class ChatResponse(BaseModel):
    """Backend-agnostic shape returned by every `LLMClient.chat_completion`.

    `raw` carries the full backend response so debugging and replay are
    not lossy; downstream code should prefer the structured fields.

    `llm_call_id` is set by `RecordingClient` after the row has been
    inserted into `llm_calls`. Callers that need to wire the FK on a
    derived `ComplianceScore` (or `Feedback`, `SynthesizedProbe`, ...)
    read it from the response instead of re-querying the DB. Plain
    backend clients leave it `None`.
    """

    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = Field(ge=0.0)
    raw: dict[str, Any] = Field(default_factory=dict)
    llm_call_id: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Structural contract every backend (and decorator) implements."""

    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> ChatResponse: ...


# ---- Error hierarchy -------------------------------------------------------


class LLMError(Exception):
    """Base class for any failure originating in an `LLMClient` call."""


class LLMConnectionError(LLMError):
    """Network-layer failure: DNS, TCP, TLS, refused connection, etc."""


class LLMTimeoutError(LLMError):
    """Backend did not respond within the configured deadline."""


class LLMResponseError(LLMError):
    """Backend responded but the response was malformed or non-2xx.

    `status_code` is optional because non-HTTP backends may surface
    structured errors without one.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


__all__ = [
    "ChatResponse",
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMResponseError",
    "LLMTimeoutError",
    "Message",
    "Role",
]
