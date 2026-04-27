"""CLI tests for `maimonedes ping`.

Unit tests use FakeLLMClient via the backend factory hook so they
don't need HTTP. The integration test (marked) runs against a real
Ollama instance and is skipped by default; run with
`pytest -m integration` to opt in.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.llm.client import ChatResponse
from maimonedes.settings import Settings, get_settings
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import (
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from tests.fakes import FakeLLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

runner = CliRunner()


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "cli.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeLLMClient:
    fake = FakeLLMClient(default_content="PONG")
    monkeypatch.setenv("SUPERVISED_MODEL", "supervised:test")
    monkeypatch.setenv("JUDGE_MODEL", "judge:test")
    cli.set_backend_factory(lambda settings: fake)
    yield fake
    cli.reset_backend_factory()


# ---- help / usage ----------------------------------------------------------


def test_top_level_help_lists_ping() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ping" in result.output


def test_ping_help_documents_options() -> None:
    result = runner.invoke(app, ["ping", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--all" in result.output


# ---- happy path ------------------------------------------------------------


def test_ping_single_model_persists_one_row(db: str, fake_backend: FakeLLMClient) -> None:
    fake_backend.queue(
        ChatResponse(
            content="PONG",
            model="m",
            prompt_tokens=4,
            completion_tokens=1,
            latency_ms=12.0,
            raw={"id": "x"},
        )
    )

    result = runner.invoke(app, ["ping", "--model", "llama3.1:8b-instruct"])

    assert result.exit_code == 0, result.output
    assert "PONG" in result.output
    assert "latency_ms=" in result.output
    assert len(fake_backend.calls) == 1
    assert fake_backend.calls[0].model == "llama3.1:8b-instruct"

    with get_session() as session:
        rows = session.query(LLMCall).all()
        assert len(rows) == 1
        assert rows[0].model == "llama3.1:8b-instruct"


def test_ping_all_exercises_both_models(db: str, fake_backend: FakeLLMClient) -> None:
    result = runner.invoke(app, ["ping", "--all"])

    assert result.exit_code == 0, result.output
    assert len(fake_backend.calls) == 2
    assert {c.model for c in fake_backend.calls} == {"supervised:test", "judge:test"}

    with get_session() as session:
        rows = session.query(LLMCall).all()
        assert len(rows) == 2


# ---- error paths -----------------------------------------------------------


def test_ping_without_model_or_all_exits_nonzero(db: str, fake_backend: FakeLLMClient) -> None:
    result = runner.invoke(app, ["ping"])
    assert result.exit_code != 0
    # Typer prints usage / error to either stdout or stderr depending on version
    assert (
        "Provide --model" in result.output
        or "Missing" in result.output
        or "Usage" in result.output
    )


def test_ping_surfaces_backend_failure_with_useful_message(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from maimonedes.llm.client import LLMConnectionError

    class ExplodingClient:
        def chat_completion(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise LLMConnectionError("connection refused at 192.168.1.30:11434")

    cli.set_backend_factory(lambda settings: ExplodingClient())
    try:
        result = runner.invoke(app, ["ping", "--model", "llama3.1:8b-instruct"])
    finally:
        cli.reset_backend_factory()

    assert result.exit_code == 2
    assert "connection refused" in (result.output + (result.stderr or ""))


# ---- integration -----------------------------------------------------------


@pytest.mark.integration
def test_ping_all_against_real_ollama(db: str) -> None:
    """End-to-end: hit the configured Ollama, expect llm_calls to grow.

    Skipped unless explicitly selected with `-m integration`. Requires
    OLLAMA_BASE_URL to point at a reachable server with both
    SUPERVISED_MODEL and JUDGE_MODEL pulled.
    """
    if not os.environ.get("OLLAMA_BASE_URL"):
        pytest.skip("OLLAMA_BASE_URL not set")

    settings = get_settings()
    with get_session() as session:
        before = session.query(LLMCall).count()

    result = runner.invoke(app, ["ping", "--all"])
    assert result.exit_code == 0, result.output

    with get_session() as session:
        after = session.query(LLMCall).count()
    assert after - before >= 2, f"expected at least 2 new rows, saw {after - before}"
    _ = settings  # keep reference for clarity
