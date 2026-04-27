"""Unit + integration tests for `OllamaBackend`.

Unit tests run the openai SDK against `httpx.MockTransport`, which
keeps coverage of request shape, error translation, and retry policy
honest without touching the network. The integration test (gated by
`@pytest.mark.integration`) hits the real endpoint and is skipped by
default — run `pytest -m integration` to opt in.
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Callable

import httpx
import pytest
from openai import OpenAI

from maimonedes.llm.client import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
    Message,
)
from maimonedes.llm.ollama_backend import OllamaBackend
from maimonedes.settings import get_settings

BASE_URL = "http://ollama.test/v1"


# ---- helpers ---------------------------------------------------------------


def _build_completion_body(
    *,
    content: str = "PONG",
    prompt_tokens: int | None = 7,
    completion_tokens: int | None = 1,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if prompt_tokens is not None and completion_tokens is not None:
        body["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    return body


def _make_backend(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    sleeps: list[float] | None = None,
    max_retries: int = 3,
) -> OllamaBackend:
    transport = httpx.MockTransport(handler)
    http = httpx.Client(transport=transport, base_url=BASE_URL)
    client = OpenAI(base_url=BASE_URL, api_key="ollama", http_client=http, max_retries=0)

    def fake_sleep(delay: float) -> None:
        if sleeps is not None:
            sleeps.append(delay)

    return OllamaBackend(
        base_url=BASE_URL,
        client=client,
        max_retries=max_retries,
        sleeper=fake_sleep,
        rng=random.Random(0xDEADBEEF),
    )


def _msg(role: str = "user", content: str = "Reply with the single word PONG.") -> Message:
    return Message(role=role, content=content)  # type: ignore[arg-type]


# ---- request shape ---------------------------------------------------------


def test_request_payload_contains_model_messages_and_optional_args() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_build_completion_body())

    backend = _make_backend(handler)
    backend.chat_completion(
        [_msg(content="hi")],
        model="llama3.1:8b-instruct-q4_K_M",
        temperature=0.2,
        max_tokens=16,
    )

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/chat/completions")
    body = captured["body"]
    assert body["model"] == "llama3.1:8b-instruct-q4_K_M"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 16


def test_optional_args_omitted_when_not_provided() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=_build_completion_body())

    backend = _make_backend(handler)
    backend.chat_completion([_msg()], model="m")

    body = captured["body"]
    assert "temperature" not in body
    assert "max_tokens" not in body


# ---- response parsing ------------------------------------------------------


def test_successful_response_populates_usage_and_latency() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_build_completion_body(content="PONG", prompt_tokens=11, completion_tokens=1),
        )

    backend = _make_backend(handler)
    resp = backend.chat_completion([_msg()], model="m")

    assert resp.content == "PONG"
    assert resp.model == "m"
    assert resp.prompt_tokens == 11
    assert resp.completion_tokens == 1
    assert resp.latency_ms > 0
    assert resp.raw["choices"][0]["message"]["content"] == "PONG"


def test_response_without_usage_yields_none_token_counts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_build_completion_body(prompt_tokens=None, completion_tokens=None),
        )

    backend = _make_backend(handler)
    resp = backend.chat_completion([_msg()], model="m")

    assert resp.prompt_tokens is None
    assert resp.completion_tokens is None


def test_empty_choices_raises_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _build_completion_body()
        body["choices"] = []
        return httpx.Response(200, json=body)

    backend = _make_backend(handler)
    with pytest.raises(LLMResponseError):
        backend.chat_completion([_msg()], model="m")


# ---- error translation -----------------------------------------------------


def test_4xx_other_than_429_is_not_retried() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    backend = _make_backend(handler, sleeps=sleeps, max_retries=3)

    with pytest.raises(LLMResponseError) as info:
        backend.chat_completion([_msg()], model="m")
    assert info.value.status_code == 400
    assert calls["n"] == 1
    assert sleeps == [], "4xx (non-429) must not retry"


def test_5xx_is_retried_until_success() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= 2:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=_build_completion_body())

    backend = _make_backend(handler, sleeps=sleeps, max_retries=3)
    resp = backend.chat_completion([_msg()], model="m")

    assert resp.content == "PONG"
    assert calls["n"] == 3
    assert len(sleeps) == 2, "two transient failures -> two backoff sleeps"
    assert all(s > 0 for s in sleeps)


def test_5xx_exhausts_retries_and_surfaces_error() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    backend = _make_backend(handler, sleeps=sleeps, max_retries=3)

    with pytest.raises(LLMResponseError) as info:
        backend.chat_completion([_msg()], model="m")
    assert info.value.status_code == 503
    # max_retries=3 -> 4 attempts total, 3 sleeps between them.
    assert calls["n"] == 4
    assert len(sleeps) == 3


def test_429_is_retryable() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=_build_completion_body())

    backend = _make_backend(handler, sleeps=sleeps, max_retries=3)
    resp = backend.chat_completion([_msg()], model="m")

    assert resp.content == "PONG"
    assert calls["n"] == 2
    assert len(sleeps) == 1


def test_connection_error_is_retried_then_translated() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("connection refused")

    backend = _make_backend(handler, sleeps=sleeps, max_retries=2)

    with pytest.raises(LLMConnectionError):
        backend.chat_completion([_msg()], model="m")
    assert calls["n"] == 3  # 2 retries + 1 initial attempt
    assert len(sleeps) == 2


def test_timeout_is_retried_then_translated() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("read timeout", request=request)

    backend = _make_backend(handler, sleeps=sleeps, max_retries=1)

    with pytest.raises(LLMTimeoutError):
        backend.chat_completion([_msg()], model="m")
    assert calls["n"] == 2


def test_backoff_grows_geometrically_within_jitter_band() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"message": "down"}})

    backend = _make_backend(handler, sleeps=sleeps, max_retries=3)
    with pytest.raises(LLMResponseError):
        backend.chat_completion([_msg()], model="m")

    # base=0.5, factor=2 -> nominal 0.5, 1.0, 2.0, jittered by [0.5, 1.5).
    assert len(sleeps) == 3
    assert 0.25 <= sleeps[0] < 0.75
    assert 0.5 <= sleeps[1] < 1.5
    assert 1.0 <= sleeps[2] < 3.0


# ---- integration -----------------------------------------------------------


@pytest.mark.integration
def test_real_ollama_returns_non_empty_response() -> None:
    """Hit the configured endpoint with a one-token prompt.

    Requires `OLLAMA_BASE_URL` set and the supervised model pulled.
    """
    if not os.environ.get("OLLAMA_BASE_URL"):
        pytest.skip("OLLAMA_BASE_URL not set")

    settings = get_settings()
    backend = OllamaBackend(
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
        request_timeout_s=settings.ollama_request_timeout_s,
        max_retries=settings.ollama_max_retries,
    )
    resp = backend.chat_completion(
        [Message(role="user", content="Reply with the single word PONG.")],
        model=settings.ollama_supervised_model,
        temperature=0.0,
        max_tokens=8,
    )
    assert resp.content.strip() != ""
    assert resp.latency_ms > 0
