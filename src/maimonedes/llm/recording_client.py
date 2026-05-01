"""Persisting / replaying decorator over any `LLMClient`.

Drift studies replay the same probes many times across debugging
sessions. Recording each request/response pair lets us re-score
offline, debug calibration drift, and produce reproducible analyses.
Replay mode is opt-in — when enabled, identical requests (matched by
`request_hash`) are served from the cache without a live call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from typing import Any

from maimonedes.llm.client import ChatResponse, LLMClient, Message
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import get_session

log = logging.getLogger(__name__)


def compute_request_hash(
    messages: Sequence[Message],
    *,
    model: str,
    temperature: float | None,
) -> str:
    """SHA-256 over normalised (messages, model, temperature).

    `json.dumps(..., sort_keys=True)` makes the hash stable across
    Python runs and key-insertion orders. Identical requests collide.
    """
    payload = {
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "model": model,
        "temperature": temperature,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RecordingClient:
    """Wrap any `LLMClient`; persist every call; optionally replay."""

    def __init__(
        self,
        wrapped: LLMClient,
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

    def chat_completion(
        self,
        messages: Sequence[Message],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> ChatResponse:
        request_hash = compute_request_hash(
            messages, model=model, temperature=temperature
        )

        if self._replay:
            cached = self._lookup_cached(request_hash)
            if cached is not None:
                log.info(
                    "recording_client.cache_hit",
                    extra={
                        "backend": self._backend_name,
                        "model": model,
                        "request_hash": request_hash,
                    },
                )
                return cached
            log.info(
                "recording_client.cache_miss",
                extra={
                    "backend": self._backend_name,
                    "model": model,
                    "request_hash": request_hash,
                },
            )

        start = time.perf_counter()
        response = self._wrapped.chat_completion(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )

        # Honour the wrapped backend's measured latency if it set one;
        # otherwise stamp our own. Avoids double-counting wall time when
        # the backend already tracks it.
        if response.latency_ms == 0.0:
            response = response.model_copy(
                update={"latency_ms": (time.perf_counter() - start) * 1000.0}
            )

        llm_call_id = self._persist(messages, model, response, request_hash)
        return response.model_copy(update={"llm_call_id": llm_call_id})

    # ---- internals ---------------------------------------------------------

    def _persist(
        self,
        messages: Sequence[Message],
        model: str,
        response: ChatResponse,
        request_hash: str,
    ) -> int:
        request_json = json.dumps(
            [{"role": m.role, "content": m.content} for m in messages],
            sort_keys=True,
            ensure_ascii=False,
        )
        raw_json = json.dumps(response.raw, sort_keys=True, default=str, ensure_ascii=False)
        with get_session() as session:
            row = LLMCall(
                backend_name=self._backend_name,
                model=model,
                request_messages_json=request_json,
                response_content=response.content,
                raw_response_json=raw_json,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                latency_ms=response.latency_ms,
                request_hash=request_hash,
            )
            session.add(row)
            session.flush()
            return int(row.id)

    def _lookup_cached(self, request_hash: str) -> ChatResponse | None:
        with get_session() as session:
            row = (
                session.query(LLMCall)
                .filter(LLMCall.request_hash == request_hash)
                .order_by(LLMCall.id.desc())
                .first()
            )
            if row is None:
                return None
            try:
                raw = json.loads(row.raw_response_json)
            except json.JSONDecodeError:
                raw = {}
            return ChatResponse(
                content=row.response_content,
                model=row.model,
                prompt_tokens=row.prompt_tokens,
                completion_tokens=row.completion_tokens,
                latency_ms=row.latency_ms,
                raw=raw if isinstance(raw, dict) else {},
                llm_call_id=int(row.id),
            )


__all__ = ["RecordingClient", "compute_request_hash"]
