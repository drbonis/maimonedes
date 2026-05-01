"""Tests for the Phase 4 apply-feedback orchestrator + CLI."""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.core.policy import Policy, load_policy
from maimonedes.core.probe import load_anchors
from maimonedes.experiments.apply_feedback import apply_feedback
from maimonedes.llm.client import ChatResponse
from maimonedes.settings import Settings
from maimonedes.storage.compliance import (
    ComplianceScoreRow,
    record_score,
)
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
)
from maimonedes.storage.perturbations import record_perturbation
from maimonedes.storage.recovery import (
    feedbacks_for_run,
    get_recovery_run,
    list_recovery_runs,
    scores_for_recovery_run,
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

runner = CliRunner()


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "apply.sqlite"
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


def _seed_drift_run(
    *,
    anchor_ids: tuple[str, ...] = ("A1", "A4"),
    baseline_aggregate: float = 0.9,
    drift_aggregate: float = 0.3,
) -> int:
    """Synthesise a drift run with hand-built scores so the localizer can rank."""
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    sub_ids = (
        "flags_physician_review",
        "expresses_uncertainty",
        "avoids_unqualified_prescriptive_language",
        "recommendation_within_scope",
        "recommendation_appropriate_specificity",
    )
    for idx in range(6):
        stage = "baseline" if idx < 3 else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        for anchor_id in anchor_ids:
            agg = baseline_aggregate if idx < 3 else drift_aggregate
            record_score(
                ComplianceScore(
                    anchor_id=anchor_id,
                    policy_id="scope_of_practice",
                    per_sub_condition={s: agg for s in sub_ids},
                    aggregate=agg,
                    judge_model="judge:test",
                    supervised_model="llama:test",
                    drift_session_id=sid,
                )
            )
    return run_id


def _supervised_response(text: str = "Discuss with your physician.") -> ChatResponse:
    return ChatResponse(content=text, model="supervised:test", latency_ms=1.0)


def _judge_response(policy: Policy, target: float = 1.0) -> ChatResponse:
    return ChatResponse(
        content=payload_json_for_target(policy, target),
        model="judge:test",
        latency_ms=1.0,
    )


def _feedback_response(text: str = "Always defer to physician.") -> ChatResponse:
    return ChatResponse(content=text, model="judge:test", latency_ms=1.0)


def _compliant_judge(policy: Policy) -> ChatResponse:
    return ChatResponse(
        content=compliant_response_json(policy),
        model="judge:test",
        latency_ms=1.0,
    )


# ---- orchestrator (FakeLLMClient) ------------------------------------------


def test_apply_feedback_temporal_creates_recovery_run_and_feedbacks(
    db: str, policy: Policy
) -> None:
    parent = _seed_drift_run(anchor_ids=("A1",))
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id == "A1"]

    # Order of calls per anchor: synthesizer (feedback), supervised, judge.
    fake = FakeLLMClient(
        responses=[
            _feedback_response("Always defer to the supervising physician."),
            _supervised_response(),
            _compliant_judge(policy),
        ]
    )

    run_id, summary = apply_feedback(
        parent,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        contrastive_kind="temporal",
        top_k=1,
    )

    run = get_recovery_run(run_id)
    assert run is not None
    assert run["parent_drift_run_id"] == parent
    assert run["contrastive_kind"] == "temporal"
    assert run["ended_at"] is not None

    feedbacks = feedbacks_for_run(run_id)
    assert "A1" in feedbacks
    assert feedbacks["A1"].feedback_text == "Always defer to the supervising physician."

    scores = scores_for_recovery_run(run_id)
    assert "A1" in scores
    assert len(scores["A1"]) == 1  # one anchor re-evaluation, no perturbations

    # Per-anchor delta = post (1.0 from compliant judge) - pre (0.3 worst).
    assert summary.anchor_count == 1
    assert summary.failure_count == 0
    assert summary.mean_delta_toward_baseline is not None
    assert summary.mean_delta_toward_baseline == pytest.approx(0.7)


def test_apply_feedback_wires_supervised_llm_call_id_on_recovery_score(
    db: str, policy: Policy
) -> None:
    """#44: recovery score row carries the supervised FK."""
    from maimonedes.storage.compliance import ComplianceScoreRow
    from maimonedes.storage.llm_calls import LLMCall

    parent = _seed_drift_run(anchor_ids=("A1",))
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id == "A1"]
    fake = FakeLLMClient(
        responses=[
            _feedback_response("Defer to the supervising physician."),
            _supervised_response(),
            _compliant_judge(policy),
        ]
    )
    run_id, _ = apply_feedback(
        parent,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        contrastive_kind="temporal",
        top_k=1,
    )

    with get_session() as session:
        recovery_data = [
            (sr.id, sr.llm_call_id)
            for sr in session.query(ComplianceScoreRow)
            .filter(ComplianceScoreRow.recovery_run_id == run_id)
            .all()
        ]
        sup_ids = {
            r.id
            for r in session.query(LLMCall)
            .filter(LLMCall.backend_name == "ollama-supervised")
            .all()
        }
    assert len(recovery_data) >= 1
    for sr_id, llm_call_id in recovery_data:
        assert llm_call_id is not None
        assert llm_call_id in sup_ids


def test_apply_feedback_per_anchor_feedback_unique_constraint(
    db: str, policy: Policy
) -> None:
    """Two affected anchors → two distinct feedback rows under one recovery run."""
    parent = _seed_drift_run(anchor_ids=("A1", "A4"))
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id in {"A1", "A4"}]

    fake = FakeLLMClient(
        responses=[
            _feedback_response("Defer to physician for A1."),
            _supervised_response(),
            _compliant_judge(policy),
            _feedback_response("Defer to physician for A4."),
            _supervised_response(),
            _compliant_judge(policy),
        ]
    )

    run_id, summary = apply_feedback(
        parent,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        contrastive_kind="temporal",
        top_k=2,
    )

    feedbacks = feedbacks_for_run(run_id)
    assert set(feedbacks) == {"A1", "A4"}
    assert feedbacks["A1"].feedback_text != feedbacks["A4"].feedback_text
    assert summary.anchor_count == 2


def test_apply_feedback_selected_anchor_ids_override_top_k(
    db: str, policy: Policy
) -> None:
    parent = _seed_drift_run(anchor_ids=("A1", "A4"))
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id in {"A1", "A4"}]
    fake = FakeLLMClient(
        responses=[
            _feedback_response(),
            _supervised_response(),
            _compliant_judge(policy),
        ]
    )
    run_id, summary = apply_feedback(
        parent,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="supervised:test",
        judge_model="judge:test",
        contrastive_kind="temporal",
        top_k=99,
        selected_anchor_ids=["A1"],
    )
    feedbacks = feedbacks_for_run(run_id)
    assert set(feedbacks) == {"A1"}
    assert summary.anchor_count == 1


def test_apply_feedback_unknown_drift_run_raises(db: str, policy: Policy) -> None:
    fake = FakeLLMClient(responses=[])
    with pytest.raises(ValueError, match="not found"):
        apply_feedback(
            999,
            policy=policy,
            anchors=load_anchors(PROBES_PATH),
            supervised_client=fake,
            judge_client=fake,
            supervised_model="m",
            judge_model="j",
        )


def test_apply_feedback_synthesize_failure_does_not_poison_others(
    db: str, policy: Policy
) -> None:
    """One synthesizer response is too long → that anchor fails, others succeed."""
    parent = _seed_drift_run(anchor_ids=("A1", "A4"))
    anchors = [a for a in load_anchors(PROBES_PATH) if a.id in {"A1", "A4"}]
    overlong = "Defer. " * 200  # >600 chars
    fake = FakeLLMClient(
        responses=[
            _feedback_response(overlong),  # A4 (worst-affected first by distance)
            _feedback_response("Defer to physician."),  # A1
            _supervised_response(),
            _compliant_judge(policy),
        ]
    )
    run_id, summary = apply_feedback(
        parent,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
        top_k=2,
    )
    assert summary.anchor_count == 2
    assert summary.failure_count == 1
    feedbacks = feedbacks_for_run(run_id)
    # Only the surviving anchor produced a feedback row.
    assert len(feedbacks) == 1


def test_apply_feedback_fragility_re_runs_perturbations(
    db: str, policy: Policy
) -> None:
    """Fragility scenario: after the anchor re-eval, every existing
    perturbation is re-run under the synthesized feedback."""
    parent = _seed_drift_run(anchor_ids=("A1",))
    # Seed two perturbations for A1.
    for label, kind, agg in (
        ("authority:gp", "authority", 0.4),
        ("authority:senior", "authority", 0.3),
    ):
        probe = PerturbationProbe(
            anchor_id="A1",
            scenario=f"perturbed:{label}",
            perturbation_kind=kind,  # type: ignore[arg-type]
            transform_label=label,
            generator_metadata={},
        )
        pid = record_perturbation(probe)
        record_score(
            ComplianceScore(
                anchor_id="A1",
                policy_id="scope_of_practice",
                per_sub_condition={
                    "flags_physician_review": agg,
                    "expresses_uncertainty": agg,
                    "avoids_unqualified_prescriptive_language": agg,
                    "recommendation_within_scope": agg,
                    "recommendation_appropriate_specificity": agg,
                },
                aggregate=agg,
                judge_model="judge:test",
                supervised_model="llama:test",
                perturbation_id=pid,
                probe_role="perturbation",
            )
        )

    anchors = [a for a in load_anchors(PROBES_PATH) if a.id == "A1"]
    fake = FakeLLMClient(
        responses=[
            _feedback_response(),       # synthesizer call
            _supervised_response(),     # anchor re-eval (supervised)
            _compliant_judge(policy),   # anchor re-eval (judge)
            _supervised_response(),     # perturbation 1 (supervised)
            _compliant_judge(policy),   # perturbation 1 (judge)
            _supervised_response(),     # perturbation 2 (supervised)
            _compliant_judge(policy),   # perturbation 2 (judge)
        ]
    )
    run_id, summary = apply_feedback(
        parent,
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="fragility",
        top_k=1,
    )
    scores = scores_for_recovery_run(run_id)
    # 1 anchor re-eval + 2 perturbation re-evals.
    assert len(scores["A1"]) == 3
    perturbation_scores = [s for s in scores["A1"] if s.probe_role == "perturbation"]
    assert len(perturbation_scores) == 2
    assert summary.failure_count == 0


# ---- CLI -------------------------------------------------------------------


def test_apply_feedback_cli_dry_run_makes_no_llm_calls(
    db: str, policy: Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _seed_drift_run()
    fake = FakeLLMClient(responses=[])
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake)
    result = runner.invoke(
        app,
        ["apply-feedback", str(parent), "--top-k", "1", "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run: no LLM calls issued" in result.output
    assert fake.calls == []


def test_apply_feedback_cli_runs_temporal_path(
    db: str, policy: Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = _seed_drift_run(anchor_ids=("A1",))
    fake = FakeLLMClient(
        responses=[
            _feedback_response(),
            _supervised_response(),
            _compliant_judge(policy),
        ]
    )
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake)
    result = runner.invoke(
        app,
        ["apply-feedback", str(parent), "--top-k", "1", "--anchors", "A1"],
    )
    assert result.exit_code == 0, result.output
    assert "recovery_run_id=" in result.output
    assert "anchors=1" in result.output

    # DB has the recovery row.
    runs = list_recovery_runs(parent_drift_run_id=parent)
    assert len(runs) == 1


def test_apply_feedback_cli_unknown_drift_run_exits_4(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLLMClient(responses=[])
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake)
    result = runner.invoke(app, ["apply-feedback", "999", "--top-k", "1"])
    assert result.exit_code == 4
    assert "unknown drift_run_id" in result.output


def test_apply_feedback_cli_rejects_invalid_kind(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLLMClient(responses=[])
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake)
    parent = _seed_drift_run()
    result = runner.invoke(
        app,
        ["apply-feedback", str(parent), "--kind", "bogus", "--dry-run"],
    )
    assert result.exit_code != 0


def test_apply_feedback_cli_unknown_anchor_id_exits_4(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLLMClient(responses=[])
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake)
    parent = _seed_drift_run()
    result = runner.invoke(
        app,
        ["apply-feedback", str(parent), "--anchors", "A99", "--dry-run"],
    )
    assert result.exit_code == 4
    assert "unknown anchor" in result.output


# ---- integration -----------------------------------------------------------


@pytest.mark.integration
def test_apply_feedback_temporal_against_ollama(
    db: str, policy: Policy, monkeypatch: pytest.MonkeyPatch
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

    parent = _seed_drift_run(anchor_ids=("A1",))
    result = runner.invoke(
        app,
        [
            "apply-feedback",
            str(parent),
            "--top-k",
            "1",
            "--anchors",
            "A1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "recovery_run_id=" in result.output
