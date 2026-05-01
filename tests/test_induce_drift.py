"""Tests for the Phase 3 drift orchestrator + `maimonedes induce-drift` CLI."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.drift import DriftSchedule
from maimonedes.core.policy import Policy, load_policy
from maimonedes.core.probe import load_anchors
from maimonedes.experiments.induce_drift import run_drift
from maimonedes.llm.client import ChatResponse
from maimonedes.settings import Settings
from maimonedes.storage.compliance import ComplianceScoreRow
from maimonedes.storage.drift import (
    get_drift_run,
    list_drift_sessions,
    scores_for_run,
)
from maimonedes.storage.repo import (
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from tests.fakes import FakeLLMClient, compliant_response_json, payload_json_for_target

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"
PROBES_PATH = PROJECT_ROOT / "config" / "probes" / "anchors_v1.yaml"
DEFAULT_SCHEDULE_PATH = PROJECT_ROOT / "config" / "drift" / "scope_of_practice_v1.yaml"
SHORT_SCHEDULE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "drift_short.yaml"

runner = CliRunner()


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "drift.sqlite"
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


# ---- DriftSchedule ---------------------------------------------------------


def test_default_schedule_has_50_sessions_in_5_stages() -> None:
    schedule = DriftSchedule.from_yaml(DEFAULT_SCHEDULE_PATH)
    assert schedule.total_sessions == 50
    assert len(schedule.stages) == 5
    sessions = list(schedule.iter_sessions())
    assert len(sessions) == 50
    assert sessions[0] == (0, "baseline", "")
    assert sessions[10][1] == "concise"
    assert sessions[40][1] == "trust"
    # Indices are dense and ascending.
    assert [s[0] for s in sessions] == list(range(50))


def test_drift_schedule_rejects_unknown_label(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({"stages": [{"label": "wat", "sessions": 1, "suffix": ""}]})
    )
    with pytest.raises(ValueError, match="not in"):
        DriftSchedule.from_yaml(bad)


def test_drift_schedule_rejects_zero_sessions(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {"stages": [{"label": "baseline", "sessions": 0, "suffix": ""}]}
        )
    )
    with pytest.raises(ValueError, match="sessions must be"):
        DriftSchedule.from_yaml(bad)


def test_drift_schedule_short_fixture_loads() -> None:
    schedule = DriftSchedule.from_yaml(SHORT_SCHEDULE_PATH)
    assert schedule.total_sessions == 5
    sessions = list(schedule.iter_sessions())
    assert sessions[0] == (0, "baseline", "")
    assert sessions[3][1] == "concise"
    assert sessions[3][2] == "Be concise and direct."


# ---- run_drift orchestrator (FakeLLMClient) --------------------------------


def _supervised_response(text: str = "Discuss with your physician.") -> ChatResponse:
    return ChatResponse(content=text, model="supervised:test", latency_ms=1.0)


def _judge_response(policy: Policy) -> ChatResponse:
    return ChatResponse(
        content=compliant_response_json(policy),
        model="judge:test",
        latency_ms=1.0,
    )


def _queue_for(policy: Policy, n_pairs: int) -> list[ChatResponse]:
    """`n_pairs` × (supervised, judge). Drift orchestrator consumes them in
    that order: supervised then judge per (session, anchor)."""
    out: list[ChatResponse] = []
    for _ in range(n_pairs):
        out.append(_supervised_response())
        out.append(_judge_response(policy))
    return out


def test_run_drift_persists_one_score_per_session_anchor(
    db: str, policy: Policy
) -> None:
    schedule = DriftSchedule.from_yaml(SHORT_SCHEDULE_PATH)
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id in {"A1", "A2"}]
    n_pairs = schedule.total_sessions * len(anchors)
    fake = FakeLLMClient(responses=_queue_for(policy, n_pairs))

    run_id, summary = run_drift(
        schedule,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        schedule_path=str(SHORT_SCHEDULE_PATH),
    )

    assert run_id > 0
    assert summary.drift_run_id == run_id
    assert summary.total_scored == n_pairs
    assert summary.failure_count == 0

    sessions = list_drift_sessions(run_id)
    assert len(sessions) == 5  # 3 baseline + 2 concise

    by_anchor = scores_for_run(run_id)
    assert set(by_anchor) == {"A1", "A2"}
    for anchor_id, scores in by_anchor.items():
        assert len(scores) == schedule.total_sessions
        # Drift-session ids resolve to sessions of this run.
        assert all(
            s.drift_session_id in {sess.id for sess in sessions} for s in scores
        )


def test_run_drift_wires_supervised_llm_call_id_on_each_score(
    db: str, policy: Policy
) -> None:
    """#44: every drift score row carries a supervised FK."""
    from maimonedes.storage.compliance import ComplianceScoreRow
    from maimonedes.storage.llm_calls import LLMCall

    schedule = DriftSchedule.from_yaml(SHORT_SCHEDULE_PATH)
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id == "A1"]
    n_pairs = schedule.total_sessions
    fake = FakeLLMClient(responses=_queue_for(policy, n_pairs))

    run_drift(
        schedule,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        schedule_path=str(SHORT_SCHEDULE_PATH),
    )

    with get_session() as session:
        score_data = [
            (sr.id, sr.llm_call_id)
            for sr in session.query(ComplianceScoreRow).all()
        ]
        sup_ids = {
            r.id
            for r in session.query(LLMCall)
            .filter(LLMCall.backend_name == "ollama-supervised")
            .all()
        }
    assert len(score_data) > 0
    for sr_id, llm_call_id in score_data:
        assert llm_call_id is not None, sr_id
        assert llm_call_id in sup_ids


def test_run_drift_prepends_suffix_as_system_message(
    db: str, policy: Policy
) -> None:
    schedule = DriftSchedule.from_yaml(SHORT_SCHEDULE_PATH)
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id == "A1"]
    n_pairs = schedule.total_sessions
    fake = FakeLLMClient(responses=_queue_for(policy, n_pairs))

    run_drift(
        schedule,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        schedule_path=str(SHORT_SCHEDULE_PATH),
    )

    # Each (session, anchor) is 2 calls (supervised, judge); supervised is even-indexed.
    supervised_calls = [c for i, c in enumerate(fake.calls) if i % 2 == 0]
    # Sessions 0,1,2 are baseline (no suffix → no system message); sessions 3,4 are concise.
    assert all(m.role != "system" for m in supervised_calls[0].messages)
    assert any(
        m.role == "system" and m.content == "Be concise and direct."
        for m in supervised_calls[3].messages
    )


def test_run_drift_isolated_anchor_failure_does_not_poison_others(
    db: str, policy: Policy
) -> None:
    """One judge response is malformed; only that one (session, anchor) pair fails."""
    schedule = DriftSchedule.from_yaml(SHORT_SCHEDULE_PATH)
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id in {"A1", "A2"}]
    n_pairs = schedule.total_sessions * len(anchors)
    queue = _queue_for(policy, n_pairs)
    # Replace the second judge response (call index 3) with garbage JSON.
    queue[3] = ChatResponse(
        content="not-json", model="judge:test", latency_ms=1.0
    )
    fake = FakeLLMClient(responses=queue)

    _, summary = run_drift(
        schedule,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        schedule_path=str(SHORT_SCHEDULE_PATH),
    )

    assert summary.failure_count == 1
    assert summary.total_scored == n_pairs - 1


def test_run_drift_baseline_mean_excludes_post_baseline(
    db: str, policy: Policy
) -> None:
    """`mean_baseline_aggregate` averages only baseline-stage scores."""
    schedule = DriftSchedule.from_yaml(SHORT_SCHEDULE_PATH)
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id == "A1"]
    queue: list[ChatResponse] = []
    for session_idx in range(schedule.total_sessions):
        queue.append(_supervised_response())
        # baseline sessions 0..2 score 1.0; concise sessions 3..4 score 0.0.
        target = 1.0 if session_idx < 3 else 0.0
        queue.append(
            ChatResponse(
                content=payload_json_for_target(policy, target),
                model="judge:test",
                latency_ms=1.0,
            )
        )
    fake = FakeLLMClient(responses=queue)

    _, summary = run_drift(
        schedule,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        schedule_path=str(SHORT_SCHEDULE_PATH),
    )

    assert summary.mean_baseline_aggregate is not None
    assert summary.mean_baseline_aggregate == pytest.approx(1.0)


def test_run_drift_persists_run_metadata(db: str, policy: Policy) -> None:
    schedule = DriftSchedule.from_yaml(SHORT_SCHEDULE_PATH)
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id == "A1"]
    fake = FakeLLMClient(responses=_queue_for(policy, schedule.total_sessions))

    run_id, _ = run_drift(
        schedule,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        schedule_path=str(SHORT_SCHEDULE_PATH),
        run_notes="smoke",
        k_threshold=3.5,
    )

    run = get_drift_run(run_id)
    assert run is not None
    assert run["notes"] == "smoke"
    assert run["k_threshold"] == 3.5
    assert run["policy_id"] == policy.id
    assert run["supervised_model"] == "supervised:test"
    assert run["judge_model"] == "judge:test"
    assert run["ended_at"] is not None  # finalized on completion


def test_run_drift_rejects_anchor_without_matching_policy(
    db: str, policy: Policy
) -> None:
    schedule = DriftSchedule.from_yaml(SHORT_SCHEDULE_PATH)
    fake = FakeLLMClient(responses=[])
    # No anchors match an arbitrary policy id.
    bogus = type(policy)(**{**policy.model_dump(), "id": "no_such_policy"})
    with pytest.raises(ValueError, match="no anchors"):
        run_drift(
            schedule,
            policy=bogus,
            anchors=load_anchors(PROBES_PATH),
            supervised_client=fake,
            judge_client=fake,
            supervised_model="supervised:test",
            judge_model="judge:test",
            schedule_path=str(SHORT_SCHEDULE_PATH),
        )


# ---- CLI subcommand --------------------------------------------------------


def test_induce_drift_dry_run_makes_no_llm_calls(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLLMClient(responses=[])
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake)

    result = runner.invoke(
        app,
        [
            "induce-drift",
            "--schedule",
            str(SHORT_SCHEDULE_PATH),
            "--anchors",
            "A1",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run: no LLM calls" in result.output
    assert fake.calls == []


def test_induce_drift_runs_short_schedule(
    db: str, policy: Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule = DriftSchedule.from_yaml(SHORT_SCHEDULE_PATH)
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id in {"A1", "A2"}]
    n_pairs = schedule.total_sessions * len(anchors)
    fake = FakeLLMClient(responses=_queue_for(policy, n_pairs))
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake)

    result = runner.invoke(
        app,
        [
            "induce-drift",
            "--schedule",
            str(SHORT_SCHEDULE_PATH),
            "--anchors",
            "A1,A2",
            "--notes",
            "cli smoke",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "drift_run_id=" in result.output
    assert "scored=10" in result.output
    # Score rows persisted via DB.
    with get_session() as session:
        n = session.query(ComplianceScoreRow).filter(
            ComplianceScoreRow.drift_session_id.is_not(None)
        ).count()
        assert n == n_pairs


def test_induce_drift_unknown_anchor_id_fails_cleanly(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLLMClient(responses=[])
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake)

    result = runner.invoke(
        app,
        [
            "induce-drift",
            "--schedule",
            str(SHORT_SCHEDULE_PATH),
            "--anchors",
            "A99",
        ],
    )
    assert result.exit_code == 4
    assert "unknown anchor" in result.output


def test_induce_drift_rejects_non_positive_k(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLLMClient(responses=[])
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake)

    result = runner.invoke(
        app,
        [
            "induce-drift",
            "--schedule",
            str(SHORT_SCHEDULE_PATH),
            "--anchors",
            "A1",
            "--k",
            "0",
            "--dry-run",
        ],
    )
    assert result.exit_code != 0
    assert "--k must be > 0" in result.output


# ---- Integration test (real LLM backend) -----------------------------------


@pytest.mark.integration
def test_induce_drift_short_schedule_against_ollama(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live-LLM smoke. Skipped unless `--integration` is enabled."""
    from maimonedes.llm.ollama_backend import OllamaBackend
    from maimonedes.settings import get_settings

    settings = get_settings()
    monkeypatch.setattr(
        cli,
        "_backend_factory",
        lambda s: OllamaBackend(
            base_url=s.ollama_base_url,
            api_key=s.ollama_api_key,
            request_timeout_s=s.ollama_request_timeout_s,
            max_retries=s.ollama_max_retries,
        ),
    )

    result = runner.invoke(
        app,
        [
            "induce-drift",
            "--schedule",
            str(SHORT_SCHEDULE_PATH),
            "--anchors",
            "A1,A2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "drift_run_id=" in result.output
