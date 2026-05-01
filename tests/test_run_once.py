"""Tests for `maimonedes run-once` and the underlying orchestrator."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.core.policy import Policy, load_policy
from maimonedes.core.probe import load_anchors
from maimonedes.experiments.run_session import run_once
from maimonedes.llm.client import ChatResponse
from maimonedes.settings import Settings, get_settings
from maimonedes.storage.compliance import ComplianceScoreRow, recent_scores
from maimonedes.storage.repo import (
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from tests.fakes import FakeLLMClient, compliant_response_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"
PROBES_PATH = PROJECT_ROOT / "config" / "probes" / "anchors_v1.yaml"

runner = CliRunner()


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "run.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


@pytest.fixture
def policy() -> Policy:
    return load_policy(POLICY_PATH, RUBRIC_PATH)


def _stitched_fake(policy: Policy) -> FakeLLMClient:
    """Single backend that serves both supervised and judge calls.

    The orchestrator issues exactly two calls in this order:
      1. supervised system: gets the anchor scenario back
      2. judge: gets a well-formed all-compliant JSON response
    """
    supervised_resp = ChatResponse(
        content="I encourage lifestyle changes; this is general guidance, "
        "and you should review specific changes with your physician.",
        model="supervised:test",
        latency_ms=10.0,
    )
    judge_resp = ChatResponse(
        content=compliant_response_json(policy),
        model="judge:test",
        latency_ms=12.0,
    )
    return FakeLLMClient(responses=[supervised_resp, judge_resp])


# ---- orchestrator ----------------------------------------------------------


def test_run_once_persists_score_and_returns_compliance_score(
    db: str, policy: Policy
) -> None:
    fake = _stitched_fake(policy)
    anchors = load_anchors(PROBES_PATH)

    score = run_once(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
    )
    assert score.anchor_id == "A1"
    assert score.aggregate == pytest.approx(1.0)

    rows = recent_scores("A1")
    assert len(rows) == 1
    assert rows[0].aggregate == pytest.approx(1.0)


def test_run_once_wires_supervised_llm_call_id_on_score(
    db: str, policy: Policy
) -> None:
    """#44: persisted compliance_scores row carries the supervised FK."""
    from maimonedes.storage.llm_calls import LLMCall

    fake = _stitched_fake(policy)
    anchors = load_anchors(PROBES_PATH)
    run_once(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
    )

    with get_session() as session:
        sup_id = (
            session.query(LLMCall.id)
            .filter(LLMCall.backend_name == "ollama-supervised")
            .scalar()
        )
        score_llm_call_id = (
            session.query(ComplianceScoreRow.llm_call_id).scalar()
        )
    assert score_llm_call_id == sup_id, (
        f"expected score.llm_call_id={sup_id}, got {score_llm_call_id}"
    )


def test_run_once_records_two_llm_calls_with_distinct_backend_names(
    db: str, policy: Policy
) -> None:
    fake = _stitched_fake(policy)
    anchors = load_anchors(PROBES_PATH)

    run_once(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
    )

    from maimonedes.storage.llm_calls import LLMCall

    with get_session() as session:
        rows = session.query(LLMCall).order_by(LLMCall.id).all()
        backend_names = {r.backend_name for r in rows}
        models = {r.model for r in rows}
    assert len(rows) == 2
    assert backend_names == {"ollama-supervised", "ollama-judge"}
    assert models == {"supervised:test", "judge:test"}


def test_run_once_unknown_anchor_raises_keyerror(db: str, policy: Policy) -> None:
    fake = _stitched_fake(policy)
    anchors = load_anchors(PROBES_PATH)
    with pytest.raises(KeyError):
        run_once(
            "A99",
            policy=policy,
            anchors=anchors,
            supervised_client=fake,
            judge_client=fake,
            supervised_model="supervised:test",
            judge_model="judge:test",
        )


def test_run_once_anchor_policy_mismatch_raises_valueerror(
    db: str, policy: Policy
) -> None:
    fake = _stitched_fake(policy)
    anchors = load_anchors(PROBES_PATH)
    # Build a divergent policy (id != scope_of_practice) but reuse rubric
    divergent = policy.model_copy(update={"id": "different_policy"})
    with pytest.raises(ValueError, match="declares policy"):
        run_once(
            "A1",
            policy=divergent,
            anchors=anchors,
            supervised_client=fake,
            judge_client=fake,
            supervised_model="supervised:test",
            judge_model="judge:test",
        )


def test_run_once_persist_false_skips_db_write(db: str, policy: Policy) -> None:
    fake = _stitched_fake(policy)
    anchors = load_anchors(PROBES_PATH)
    run_once(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        persist=False,
    )
    assert recent_scores("A1") == []


# ---- CLI -------------------------------------------------------------------


@pytest.fixture
def cli_backend(monkeypatch: pytest.MonkeyPatch, policy: Policy) -> FakeLLMClient:
    fake = _stitched_fake(policy)
    monkeypatch.setenv("OLLAMA_SUPERVISED_MODEL", "supervised:test")
    monkeypatch.setenv("OLLAMA_JUDGE_MODEL", "judge:test")
    cli.set_backend_factory(lambda settings: fake)
    yield fake
    cli.reset_backend_factory()


def test_cli_run_once_prints_aggregate_and_persists(
    db: str, policy: Policy, cli_backend: FakeLLMClient
) -> None:
    result = runner.invoke(
        app,
        [
            "run-once",
            "A1",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "anchor=A1" in result.output
    assert "aggregate=1.000" in result.output
    for s in policy.rubric.sub_conditions:
        assert s.id in result.output

    rows = recent_scores("A1")
    assert len(rows) == 1


def test_cli_unknown_anchor_exits_nonzero(
    db: str, cli_backend: FakeLLMClient
) -> None:
    result = runner.invoke(
        app,
        [
            "run-once",
            "A99",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
        ],
    )
    assert result.exit_code != 0
    assert "unknown anchor" in (result.output + (result.stderr or "")).lower()


def test_cli_missing_config_file_exits_nonzero(
    db: str, cli_backend: FakeLLMClient, tmp_path: Path
) -> None:
    missing = tmp_path / "nope.yaml"
    result = runner.invoke(
        app,
        [
            "run-once",
            "A1",
            "--policy",
            str(missing),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
        ],
    )
    assert result.exit_code != 0


def test_cli_help_lists_run_once_subcommand() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-once" in result.output


# ---- integration -----------------------------------------------------------


@pytest.mark.integration
def test_run_once_against_real_ollama(db: str) -> None:
    if not os.environ.get("OLLAMA_BASE_URL"):
        pytest.skip("OLLAMA_BASE_URL not set")
    settings = get_settings()
    result = runner.invoke(
        app,
        [
            "run-once",
            "A1",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = recent_scores("A1")
    assert len(rows) >= 1
    _ = settings
