"""Tests for the Phase 5 EmbedClient + ClinicalBertBackend + RecordingEmbedClient."""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.llm.client import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from maimonedes.llm.clinicalbert_backend import (
    ClinicalBertBackend,
    _coerce_to_float_list,
    _extract_embedding,
)
from maimonedes.llm.embed_client import EmbedClient, EmbedResponse
from maimonedes.llm.recording_embed_client import (
    RecordingEmbedClient,
    compute_request_hash,
)
from maimonedes.settings import Settings
from maimonedes.storage.embed_calls import (
    EmbedCall,
    cached_embed,
    latest_embed_for_text,
)
from maimonedes.storage.repo import (
    get_engine,
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from tests.fakes import FakeEmbedClient


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
    db_path = tmp_path / "embed.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


# ---- protocol conformance --------------------------------------------------


def test_fake_embed_client_satisfies_protocol() -> None:
    fake = FakeEmbedClient()
    assert isinstance(fake, EmbedClient)


def test_clinicalbert_backend_satisfies_protocol() -> None:
    backend = ClinicalBertBackend(
        base_url="http://test.local:8000",
        client=httpx.Client(),
    )
    assert isinstance(backend, EmbedClient)


# ---- response-shape auto-detection ----------------------------------------


def test_extract_embedding_picks_embedding_key() -> None:
    payload = {"embedding": [0.1, 0.2, 0.3]}
    assert _extract_embedding(payload) == [0.1, 0.2, 0.3]


def test_extract_embedding_picks_embeddings_batched_first() -> None:
    payload = {"embeddings": [[0.4, 0.5, 0.6]]}
    assert _extract_embedding(payload) == [0.4, 0.5, 0.6]


def test_extract_embedding_picks_vector_key() -> None:
    assert _extract_embedding({"vector": [1.0, 2.0]}) == [1.0, 2.0]


def test_extract_embedding_falls_back_to_unknown_key_with_floats() -> None:
    payload = {"the_real_field_name": [0.7, 0.8, 0.9]}
    assert _extract_embedding(payload) == [0.7, 0.8, 0.9]


def test_extract_embedding_returns_none_when_no_floats_anywhere() -> None:
    assert _extract_embedding({"unrelated": "data", "n": 5}) is None


def test_coerce_to_float_list_rejects_strings() -> None:
    assert _coerce_to_float_list(["a", "b"]) is None


def test_coerce_to_float_list_handles_ints_as_floats() -> None:
    assert _coerce_to_float_list([1, 2, 3]) == [1.0, 2.0, 3.0]


# ---- ClinicalBertBackend HTTP behavior -------------------------------------


def _mock_transport_returning(payload: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport)


def test_backend_returns_embed_response_with_full_metadata() -> None:
    payload = {"embedding": [0.1] * 768, "model_used": "bio_clinicalbert"}
    backend = ClinicalBertBackend(
        base_url="http://x:8000",
        model="bio_clinicalbert",
        client=_mock_transport_returning(payload),
    )
    resp = backend.embed("Patient has hypertension.")
    assert len(resp.embedding) == 768
    assert resp.embedding[0] == pytest.approx(0.1)
    assert resp.model == "bio_clinicalbert"
    assert resp.latency_ms >= 0.0
    assert resp.raw == payload


def test_backend_rejects_empty_text() -> None:
    backend = ClinicalBertBackend(
        base_url="http://x:8000",
        client=_mock_transport_returning({"embedding": [1.0]}),
    )
    with pytest.raises(LLMResponseError, match="non-empty"):
        backend.embed("")


def test_backend_translates_5xx_to_response_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    backend = ClinicalBertBackend(
        base_url="http://x:8000",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _: None,
    )
    with pytest.raises(LLMResponseError):
        backend.embed("text")


def test_backend_translates_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    backend = ClinicalBertBackend(
        base_url="http://x:8000",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _: None,
    )
    with pytest.raises(LLMTimeoutError):
        backend.embed("text")


def test_backend_translates_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    backend = ClinicalBertBackend(
        base_url="http://x:8000",
        max_retries=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _: None,
    )
    with pytest.raises(LLMConnectionError):
        backend.embed("text")


def test_backend_retries_then_succeeds() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise httpx.ConnectError("flake", request=request)
        return httpx.Response(200, json={"embedding": [0.5, 0.5]})

    backend = ClinicalBertBackend(
        base_url="http://x:8000",
        max_retries=3,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleeper=lambda _: None,
    )
    resp = backend.embed("text")
    assert resp.embedding == [0.5, 0.5]
    assert attempts["count"] == 2


def test_backend_raises_when_response_lacks_embedding() -> None:
    backend = ClinicalBertBackend(
        base_url="http://x:8000",
        client=_mock_transport_returning({"only_metadata": "no vector"}),
    )
    with pytest.raises(LLMResponseError, match="locate embedding"):
        backend.embed("text")


def test_backend_raises_on_non_json_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"plain text")

    backend = ClinicalBertBackend(
        base_url="http://x:8000",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(LLMResponseError, match="not valid JSON"):
        backend.embed("text")


def test_backend_embed_batch_loops_in_order() -> None:
    payloads = iter(
        [{"embedding": [0.1]}, {"embedding": [0.2]}, {"embedding": [0.3]}]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=next(payloads))

    backend = ClinicalBertBackend(
        base_url="http://x:8000",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    responses = backend.embed_batch(["a", "b", "c"])
    assert [r.embedding[0] for r in responses] == pytest.approx([0.1, 0.2, 0.3])


# ---- migration -------------------------------------------------------------


def test_migration_creates_embed_calls_table(db: str) -> None:
    insp = inspect(get_engine())
    assert "embed_calls" in insp.get_table_names()
    indexes = {ix["name"] for ix in insp.get_indexes("embed_calls")}
    assert "ix_embed_calls_timestamp" in indexes
    assert "ix_embed_calls_model" in indexes
    assert "ix_embed_calls_request_hash" in indexes


def test_alembic_round_trip_clean(db: str) -> None:
    cfg = _alembic_cfg(db)
    command.downgrade(cfg, "0007_recovery_runs")
    insp = inspect(get_engine())
    assert "embed_calls" not in insp.get_table_names()
    command.upgrade(cfg, "head")
    insp = inspect(get_engine())
    assert "embed_calls" in insp.get_table_names()


# ---- recording client ------------------------------------------------------


def test_recording_client_persists_a_row(db: str) -> None:
    fake = FakeEmbedClient(
        responses=[EmbedResponse(embedding=[0.1] * 8, model="m", latency_ms=5.0)]
    )
    rc = RecordingEmbedClient(fake, backend_name="test")
    rc.embed("hello")
    with get_session() as session:
        rows = session.query(EmbedCall).all()
        assert len(rows) == 1
        assert rows[0].backend_name == "test"
        assert rows[0].request_text == "hello"
        assert rows[0].latency_ms == pytest.approx(5.0)


def test_recording_client_replay_returns_cached_no_inner_call(db: str) -> None:
    fake = FakeEmbedClient(
        responses=[EmbedResponse(embedding=[0.7] * 4, model="m", latency_ms=1.0)]
    )
    rc = RecordingEmbedClient(fake, backend_name="test", replay=False)
    rc.embed("repeat-me", model="m")
    assert len(fake.calls) == 1

    rc_replay = RecordingEmbedClient(fake, backend_name="test", replay=True)
    cached = rc_replay.embed("repeat-me", model="m")
    assert cached.embedding == [0.7, 0.7, 0.7, 0.7]
    # No new live call.
    assert len(fake.calls) == 1


def test_recording_client_replay_miss_goes_live_and_caches(db: str) -> None:
    fake = FakeEmbedClient(
        responses=[EmbedResponse(embedding=[0.4] * 4, model="m", latency_ms=1.0)]
    )
    rc = RecordingEmbedClient(fake, backend_name="test", replay=True)
    response = rc.embed("fresh", model="m")
    assert response.embedding == [0.4, 0.4, 0.4, 0.4]
    # Now replay the same text — should hit cache.
    rc.embed("fresh", model="m")
    assert len(fake.calls) == 1


def test_compute_request_hash_is_deterministic() -> None:
    h1 = compute_request_hash(model="m", text="x")
    h2 = compute_request_hash(model="m", text="x")
    assert h1 == h2
    h3 = compute_request_hash(model="m", text="different")
    assert h3 != h1


def test_cached_embed_keys_on_backend_model_hash(db: str) -> None:
    fake = FakeEmbedClient(
        responses=[EmbedResponse(embedding=[0.9] * 4, model="m", latency_ms=1.0)]
    )
    rc = RecordingEmbedClient(fake, backend_name="A")
    rc.embed("text", model="m")

    # Same hash but different backend name → cache miss.
    miss = cached_embed(
        backend_name="B",
        model="m",
        request_hash=compute_request_hash(model="m", text="text"),
    )
    assert miss is None

    hit = latest_embed_for_text(backend_name="A", model="m", text="text")
    assert hit is not None
    assert hit.embedding == [0.9, 0.9, 0.9, 0.9]


# ---- CLI -------------------------------------------------------------------


def test_embed_ping_cli_round_trip(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeEmbedClient(
        responses=[EmbedResponse(embedding=list(range(768)), model="bio_clinicalbert", latency_ms=1.0)]
    )
    monkeypatch.setattr(cli, "_embed_factory", lambda settings: fake)
    result = runner.invoke(app, ["embed-ping", "--text", "hello"])
    assert result.exit_code == 0, result.output
    assert "dim=768" in result.output
    assert "model=bio_clinicalbert" in result.output


# ---- integration -----------------------------------------------------------


@pytest.mark.integration
def test_embed_ping_against_real_service(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live smoke test. Skipped unless --integration."""
    from maimonedes.llm.clinicalbert_backend import ClinicalBertBackend
    from maimonedes.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(
        cli,
        "_embed_factory",
        lambda s: ClinicalBertBackend(
            base_url=s.clinicalbert_base_url,
            model=s.clinicalbert_model,
            request_timeout_s=s.clinicalbert_request_timeout_s,
            max_retries=s.clinicalbert_max_retries,
        ),
    )
    result = runner.invoke(app, ["embed-ping", "--text", "Patient has hypertension."])
    assert result.exit_code == 0, result.output
    assert "dim=768" in result.output
