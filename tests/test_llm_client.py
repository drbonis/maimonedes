"""Contract tests for the LLMClient Protocol."""
from __future__ import annotations

import pytest

from maimonedes.llm.client import (
    ChatResponse,
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    Message,
)
from tests.fakes import FakeLLMClient


def test_message_is_immutable() -> None:
    m = Message(role="user", content="hello")
    with pytest.raises(Exception):
        m.role = "assistant"  # type: ignore[misc]


def test_chat_response_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        ChatResponse(content="x", model="m", latency_ms=0.0, surprise=1)  # type: ignore[call-arg]


def test_chat_response_latency_must_be_non_negative() -> None:
    with pytest.raises(Exception):
        ChatResponse(content="x", model="m", latency_ms=-1.0)


def test_fake_satisfies_protocol_runtime_check() -> None:
    fake = FakeLLMClient()
    assert isinstance(fake, LLMClient)


def test_fake_returns_default_response() -> None:
    fake = FakeLLMClient()
    resp = fake.chat_completion(
        [Message(role="user", content="hi")],
        model="fake-model",
        temperature=0.2,
    )
    assert isinstance(resp, ChatResponse)
    assert resp.content == "PONG"
    assert resp.model == "fake-model"

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call.model == "fake-model"
    assert call.temperature == 0.2
    assert call.messages[0].content == "hi"


def test_fake_drains_queued_responses_in_order() -> None:
    fake = FakeLLMClient(responses=["one", "two"])
    a = fake.chat_completion([Message(role="user", content="?")], model="m")
    b = fake.chat_completion([Message(role="user", content="?")], model="m")
    c = fake.chat_completion([Message(role="user", content="?")], model="m")
    assert a.content == "one"
    assert b.content == "two"
    assert c.content == "PONG"


def test_error_hierarchy_inherits_from_llm_error() -> None:
    assert issubclass(LLMConnectionError, LLMError)
    assert issubclass(LLMTimeoutError, LLMError)
    assert issubclass(LLMResponseError, LLMError)


def test_response_error_carries_status_code() -> None:
    err = LLMResponseError("bad gateway", status_code=502)
    assert err.status_code == 502
    assert "bad gateway" in str(err)
