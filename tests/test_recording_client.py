"""Tests for the persisting / replaying decorator."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from maimonedes.llm.client import ChatResponse, Message
from maimonedes.llm.recording_client import RecordingClient, compute_request_hash
from maimonedes.settings import Settings
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import (
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from tests.fakes import FakeLLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "rec.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


def _msg(s: str) -> Message:
    return Message(role="user", content=s)


# ---- request_hash stability ------------------------------------------------


def test_request_hash_is_stable_across_runs() -> None:
    h1 = compute_request_hash([_msg("hello")], model="m", temperature=0.5)
    h2 = compute_request_hash([_msg("hello")], model="m", temperature=0.5)
    assert h1 == h2
    assert len(h1) == 64


def test_request_hash_distinguishes_inputs() -> None:
    base = compute_request_hash([_msg("a")], model="m", temperature=0.5)
    assert base != compute_request_hash([_msg("b")], model="m", temperature=0.5)
    assert base != compute_request_hash([_msg("a")], model="m2", temperature=0.5)
    assert base != compute_request_hash([_msg("a")], model="m", temperature=0.0)
    assert base != compute_request_hash([_msg("a")], model="m", temperature=None)


# ---- pass-through (replay disabled) ----------------------------------------


def test_pass_through_persists_call_and_does_not_consult_cache(db: str) -> None:
    fake = FakeLLMClient(responses=["first", "second"])
    rc = RecordingClient(fake, backend_name="fake", replay=False)

    a = rc.chat_completion([_msg("hi")], model="m", temperature=0.0)
    b = rc.chat_completion([_msg("hi")], model="m", temperature=0.0)

    # No replay -> wrapped client got called twice and returned both queue items
    assert a.content == "first"
    assert b.content == "second"
    assert len(fake.calls) == 2

    with get_session() as session:
        rows = session.query(LLMCall).order_by(LLMCall.id).all()
        assert len(rows) == 2
        for row in rows:
            assert row.backend_name == "fake"
            assert row.model == "m"
            assert json.loads(row.request_messages_json) == [
                {"role": "user", "content": "hi"}
            ]
            assert row.request_hash == compute_request_hash(
                [_msg("hi")], model="m", temperature=0.0
            )


# ---- replay hit ------------------------------------------------------------


def test_replay_hit_returns_cached_without_calling_wrapped(
    db: str, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeLLMClient(responses=["live"])
    # First call (non-replay) seeds the cache.
    seeder = RecordingClient(fake, backend_name="fake", replay=False)
    seeder.chat_completion([_msg("ping")], model="m", temperature=0.0)
    assert len(fake.calls) == 1

    # Second call with replay=True must hit the cache.
    rc = RecordingClient(fake, backend_name="fake", replay=True)
    caplog.set_level(logging.INFO, logger="maimonedes.llm.recording_client")
    cached = rc.chat_completion([_msg("ping")], model="m", temperature=0.0)

    assert cached.content == "live"
    assert len(fake.calls) == 1, "wrapped client must NOT be called on cache hit"
    cache_msgs = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "maimonedes.llm.recording_client"
    ]
    assert any("cache_hit" in m for m in cache_msgs), cache_msgs


# ---- replay miss falls back to live call -----------------------------------


def test_replay_miss_falls_back_to_live_call(
    db: str, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeLLMClient(responses=["fresh"])
    rc = RecordingClient(fake, backend_name="fake", replay=True)

    caplog.set_level(logging.INFO, logger="maimonedes.llm.recording_client")
    resp = rc.chat_completion([_msg("brand-new")], model="m", temperature=0.0)

    assert resp.content == "fresh"
    assert len(fake.calls) == 1
    cache_msgs = [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "maimonedes.llm.recording_client"
    ]
    assert any("cache_miss" in m for m in cache_msgs), cache_msgs

    # Live result was persisted, so a follow-up replay should now hit.
    rc2 = RecordingClient(fake, backend_name="fake", replay=True)
    second = rc2.chat_completion([_msg("brand-new")], model="m", temperature=0.0)
    assert second.content == "fresh"
    assert len(fake.calls) == 1


# ---- raw payload survives the round-trip -----------------------------------


def test_pass_through_returns_response_with_llm_call_id_set(db: str) -> None:
    """#44: live calls must return a ChatResponse whose llm_call_id is the row id."""
    fake = FakeLLMClient(responses=["one"])
    rc = RecordingClient(fake, backend_name="fake", replay=False)
    resp = rc.chat_completion([_msg("hi")], model="m", temperature=0.0)
    assert resp.llm_call_id is not None
    with get_session() as session:
        row_id = session.query(LLMCall.id).scalar()
    assert resp.llm_call_id == row_id


def test_replay_hit_response_carries_llm_call_id(db: str) -> None:
    fake = FakeLLMClient(responses=["live"])
    seeder = RecordingClient(fake, backend_name="fake", replay=False)
    seeded = seeder.chat_completion([_msg("ping")], model="m", temperature=0.0)
    rc = RecordingClient(fake, backend_name="fake", replay=True)
    cached = rc.chat_completion([_msg("ping")], model="m", temperature=0.0)
    assert cached.llm_call_id == seeded.llm_call_id


def test_raw_payload_round_trip(db: str) -> None:
    canned = ChatResponse(
        content="hello",
        model="m",
        prompt_tokens=3,
        completion_tokens=2,
        latency_ms=12.5,
        raw={"id": "abc", "choices": [{"index": 0}]},
    )
    fake = FakeLLMClient(responses=[canned])
    rc = RecordingClient(fake, backend_name="fake", replay=False)
    rc.chat_completion([_msg("x")], model="m", temperature=0.0)

    rc2 = RecordingClient(fake, backend_name="fake", replay=True)
    cached = rc2.chat_completion([_msg("x")], model="m", temperature=0.0)
    assert cached.raw == {"id": "abc", "choices": [{"index": 0}]}
    assert cached.prompt_tokens == 3
    assert cached.completion_tokens == 2
    assert cached.latency_ms == 12.5
