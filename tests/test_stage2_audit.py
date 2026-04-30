"""Tests for the Stage-2 online scorer + hybrid audit detector."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy, load_policy
from maimonedes.core.probe import AnchorProbe, load_anchors
from maimonedes.llm.client import ChatResponse
from maimonedes.models.stage2 import Stage2Model, train_stage2
from maimonedes.monitor.stage2_audit import Stage2AuditDetector
from maimonedes.scorer.stage2 import Stage2Scorer
from maimonedes.settings import Settings
from maimonedes.storage.audit_runs import (
    AuditRunRow,
    list_audit_runs,
)
from maimonedes.storage.compliance import (
    ComplianceScoreRow,
    record_score,
)
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import (
    get_engine,
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from maimonedes.storage.stage2_models import record_stage2_model
from tests.fakes import (
    FakeEmbedClient,
    FakeLLMClient,
    payload_json_for_target,
)


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
    db_path = tmp_path / "audit.sqlite"
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


@pytest.fixture
def anchors() -> list[AnchorProbe]:
    return load_anchors(PROBES_PATH)


# ---- migration -------------------------------------------------------------


def test_migration_creates_audit_runs_table(db: str) -> None:
    insp = inspect(get_engine())
    assert "audit_runs" in insp.get_table_names()
    indexes = {ix["name"] for ix in insp.get_indexes("audit_runs")}
    assert "ix_audit_runs_stage2_model_id" in indexes


def test_alembic_round_trip_clean(db: str) -> None:
    cfg = _alembic_cfg(db)
    command.downgrade(cfg, "0009_stage2_models")
    insp = inspect(get_engine())
    assert "audit_runs" not in insp.get_table_names()
    command.upgrade(cfg, "head")
    assert "audit_runs" in inspect(get_engine()).get_table_names()


# ---- Stage2Scorer ----------------------------------------------------------


def _seed_training_pairs(
    policy: Policy,
    *,
    anchors: tuple[str, ...] = ("A1", "A2"),
    n_per_anchor: int = 30,
) -> None:
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    payloads: list[tuple[str, str, float]] = []
    with get_session() as session:
        for anchor_id in anchors:
            for i in range(n_per_anchor):
                aggregate = 0.5 + 0.4 * (i % 5 - 2) / 4.0
                aggregate = max(0.0, min(1.0, aggregate))
                text = f"{anchor_id}-output-{i}: " + ("a " * (i + 1))
                llm = LLMCall(
                    backend_name="ollama-supervised",
                    model="llama:test",
                    request_messages_json=json.dumps([]),
                    response_content=text,
                    raw_response_json="{}",
                    prompt_tokens=10,
                    completion_tokens=10,
                    latency_ms=1.0,
                    request_hash=f"hash-{anchor_id}-{i}",
                )
                session.add(llm)
                session.flush()
                payloads.append((anchor_id, str(llm.id), aggregate))
    for anchor_id, llm_id, aggregate in payloads:
        record_score(
            ComplianceScore(
                anchor_id=anchor_id,
                policy_id=policy.id,
                per_sub_condition={sid: aggregate for sid in sub_ids},
                aggregate=aggregate,
                judge_model="judge:test",
                supervised_model="llama:test",
                llm_call_id=int(llm_id),
            )
        )


def _make_stage2_model(policy: Policy) -> Stage2Model:
    fake_embed = FakeEmbedClient(default_dim=16)
    return train_stage2(
        policy=policy, embed_client=fake_embed, min_samples=50, seed=0
    )


def test_stage2_scorer_round_trip(
    db: str, policy: Policy, anchors: list[AnchorProbe]
) -> None:
    _seed_training_pairs(policy)
    model = _make_stage2_model(policy)
    fake_embed = FakeEmbedClient(default_dim=16)
    scorer = Stage2Scorer(model, fake_embed, policy, model_path="models/stage2_x.pkl")

    a1 = next(a for a in anchors if a.id == "A1")
    score = scorer.score(a1, "Discuss with your physician.")
    assert score.anchor_id == "A1"
    assert score.judge_model == "stage2:stage2_x.pkl"
    assert 0.0 <= score.aggregate <= 1.0
    assert set(score.per_sub_condition) == {
        s.id for s in policy.rubric.sub_conditions
    }


def test_stage2_scorer_rejects_policy_mismatch(
    db: str, policy: Policy, anchors: list[AnchorProbe]
) -> None:
    _seed_training_pairs(policy)
    model = _make_stage2_model(policy)
    fake_embed = FakeEmbedClient(default_dim=16)
    scorer = Stage2Scorer(model, fake_embed, policy)
    with pytest.raises(ValueError, match="anchor"):
        bogus_anchor = anchors[0].model_copy(update={"policy_id": "other"})
        scorer.score(bogus_anchor, "text")


# ---- audit detector --------------------------------------------------------


def test_should_audit_first_ever_returns_true(db: str) -> None:
    detector = Stage2AuditDetector(stage2_model_id=1, n_scores=10, hours=24.0)
    decision, reason = detector.should_audit(
        now=datetime.now(timezone.utc),
        scores_since_last=0,
        last_ended_at=None,
    )
    assert decision is True
    assert reason == "first_ever"


def test_should_audit_n_scores_threshold(db: str) -> None:
    detector = Stage2AuditDetector(stage2_model_id=1, n_scores=10, hours=24.0)
    last = datetime.now(timezone.utc) - timedelta(hours=1)
    decision, reason = detector.should_audit(
        now=datetime.now(timezone.utc),
        scores_since_last=12,
        last_ended_at=last,
    )
    assert decision is True
    assert reason == "n_scores"


def test_should_audit_hours_threshold(db: str) -> None:
    detector = Stage2AuditDetector(stage2_model_id=1, n_scores=10, hours=24.0)
    last = datetime.now(timezone.utc) - timedelta(hours=25)
    decision, reason = detector.should_audit(
        now=datetime.now(timezone.utc),
        scores_since_last=2,
        last_ended_at=last,
    )
    assert decision is True
    assert reason == "hours_elapsed"


def test_should_audit_below_thresholds_returns_false(db: str) -> None:
    detector = Stage2AuditDetector(stage2_model_id=1, n_scores=10, hours=24.0)
    last = datetime.now(timezone.utc) - timedelta(hours=1)
    decision, reason = detector.should_audit(
        now=datetime.now(timezone.utc),
        scores_since_last=2,
        last_ended_at=last,
    )
    assert decision is False
    assert reason == "below_thresholds"


def test_should_audit_picks_n_scores_when_both_trip(db: str) -> None:
    """Trigger reason picks N-scores first when both thresholds tripped."""
    detector = Stage2AuditDetector(stage2_model_id=1, n_scores=5, hours=1.0)
    last = datetime.now(timezone.utc) - timedelta(hours=2)
    decision, reason = detector.should_audit(
        now=datetime.now(timezone.utc),
        scores_since_last=10,
        last_ended_at=last,
    )
    assert decision is True
    assert reason == "n_scores"


# ---- run_audit -------------------------------------------------------------


def _seed_stage2_score_rows(
    *,
    n: int,
    model_tag: str,
    anchor_id: str = "A1",
    aggregate: float = 0.85,
) -> int:
    """Seed n compliance_scores rows tagged as stage2 outputs with linked llm_calls."""
    inserted = 0
    payloads: list[tuple[int, float]] = []
    with get_session() as session:
        for i in range(n):
            llm = LLMCall(
                backend_name="ollama-supervised",
                model="llama:test",
                request_messages_json=json.dumps([]),
                response_content=f"Stage2 input {i}",
                raw_response_json="{}",
                prompt_tokens=10,
                completion_tokens=10,
                latency_ms=1.0,
                request_hash=f"stage2-input-{i}",
            )
            session.add(llm)
            session.flush()
            payloads.append((llm.id, aggregate))
    for llm_id, agg in payloads:
        record_score(
            ComplianceScore(
                anchor_id=anchor_id,
                policy_id="scope_of_practice",
                per_sub_condition={
                    "flags_physician_review": agg,
                    "expresses_uncertainty": agg,
                    "avoids_unqualified_prescriptive_language": agg,
                    "recommendation_within_scope": agg,
                    "recommendation_appropriate_specificity": agg,
                },
                aggregate=agg,
                judge_model=model_tag,
                supervised_model="llama:test",
                llm_call_id=llm_id,
            )
        )
        inserted += 1
    return inserted


def test_run_audit_writes_audit_run_row(
    db: str, policy: Policy, anchors: list[AnchorProbe]
) -> None:
    _seed_training_pairs(policy)
    stage2_model = _make_stage2_model(policy)
    model_row_id = record_stage2_model(
        path="models/stage2_x.pkl",
        policy_id=policy.id,
        n_samples=stage2_model.n_samples,
        embedding_model=stage2_model.embedding_model,
        mae_per_axis={m.sub_id: m.mae for m in stage2_model.agreement_metrics},
        spearman_per_axis={
            m.sub_id: m.spearman_rho for m in stage2_model.agreement_metrics
        },
        agreement_status=stage2_model.agreement_status,
    )

    model_tag = "stage2:stage2_x.pkl"
    _seed_stage2_score_rows(n=5, model_tag=model_tag, anchor_id="A1")

    fake_judge = FakeLLMClient(
        responses=[
            ChatResponse(
                content=payload_json_for_target(policy, 0.85),
                model="judge:test",
                latency_ms=1.0,
            )
            for _ in range(5)
        ]
    )

    detector = Stage2AuditDetector(
        stage2_model_id=model_row_id, n_scores=10, hours=24.0
    )
    rows = detector.scores_since_last(model_tag, last_ended_at=None)
    report = detector.run_audit(
        stage2_model=stage2_model,
        policy=policy,
        anchors=anchors,
        judge_client=fake_judge,
        judge_model="judge:test",
        supervised_model="llama:test",
        score_rows=rows,
        trigger_reason="first_ever",
    )
    assert report.audit_run_id > 0
    assert report.n_samples == 5
    assert report.agreement_status in {"green", "amber", "red"}
    persisted = list_audit_runs(model_row_id)
    assert len(persisted) == 1
    assert persisted[0].agreement_status == report.agreement_status


def test_run_audit_stores_metrics_in_persisted_row(
    db: str, policy: Policy, anchors: list[AnchorProbe]
) -> None:
    _seed_training_pairs(policy)
    stage2_model = _make_stage2_model(policy)
    model_row_id = record_stage2_model(
        path="models/stage2_x.pkl",
        policy_id=policy.id,
        n_samples=60,
        embedding_model="fake-embed",
        mae_per_axis={},
        spearman_per_axis={},
        agreement_status="amber",
    )
    model_tag = "stage2:stage2_x.pkl"
    _seed_stage2_score_rows(n=3, model_tag=model_tag)
    fake_judge = FakeLLMClient(
        responses=[
            ChatResponse(
                content=payload_json_for_target(policy, 0.85),
                model="judge:test",
                latency_ms=1.0,
            )
            for _ in range(3)
        ]
    )
    detector = Stage2AuditDetector(stage2_model_id=model_row_id)
    rows = detector.scores_since_last(model_tag, last_ended_at=None)
    detector.run_audit(
        stage2_model=stage2_model,
        policy=policy,
        anchors=anchors,
        judge_client=fake_judge,
        judge_model="judge:test",
        supervised_model="llama:test",
        score_rows=rows,
        trigger_reason="first_ever",
    )
    persisted = list_audit_runs(model_row_id)
    assert persisted[0].n_samples == 3
    metrics = json.loads(persisted[0].mae_per_axis_json)
    assert set(metrics) == {s.id for s in policy.rubric.sub_conditions}


# ---- CLI -------------------------------------------------------------------


def test_audit_stage2_cli_runs_force_path(
    db: str,
    policy: Policy,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _seed_training_pairs(policy)
    stage2_model = _make_stage2_model(policy)
    model_path = tmp_path / "stage2_x.pkl"
    stage2_model.save(model_path)
    model_row_id = record_stage2_model(
        path=str(model_path),
        policy_id=policy.id,
        n_samples=stage2_model.n_samples,
        embedding_model=stage2_model.embedding_model,
        mae_per_axis={
            m.sub_id: m.mae for m in stage2_model.agreement_metrics
        },
        spearman_per_axis={
            m.sub_id: m.spearman_rho for m in stage2_model.agreement_metrics
        },
        agreement_status=stage2_model.agreement_status,
    )
    model_tag = f"stage2:{model_path.name}"
    _seed_stage2_score_rows(n=3, model_tag=model_tag)

    fake_judge = FakeLLMClient(
        responses=[
            ChatResponse(
                content=payload_json_for_target(policy, 0.85),
                model="judge:test",
                latency_ms=1.0,
            )
            for _ in range(5)
        ]
    )
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: fake_judge)

    result = runner.invoke(
        app,
        [
            "audit-stage2",
            str(model_row_id),
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "audit_run_id=" in result.output
    assert f"stage2_model_id={model_row_id}" in result.output


def test_audit_stage2_cli_unknown_model_exits_4(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_backend_factory", lambda settings: FakeLLMClient())
    result = runner.invoke(
        app,
        [
            "audit-stage2",
            "999",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
        ],
    )
    assert result.exit_code == 4


def test_score_stage2_cli_with_text_override(
    db: str,
    policy: Policy,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _seed_training_pairs(policy)
    stage2_model = _make_stage2_model(policy)
    model_path = tmp_path / "stage2_x.pkl"
    stage2_model.save(model_path)
    model_row_id = record_stage2_model(
        path=str(model_path),
        policy_id=policy.id,
        n_samples=stage2_model.n_samples,
        embedding_model=stage2_model.embedding_model,
        mae_per_axis={
            m.sub_id: m.mae for m in stage2_model.agreement_metrics
        },
        spearman_per_axis={
            m.sub_id: m.spearman_rho for m in stage2_model.agreement_metrics
        },
        agreement_status=stage2_model.agreement_status,
    )
    fake_embed = FakeEmbedClient(default_dim=16)
    monkeypatch.setattr(cli, "_embed_factory", lambda s: fake_embed)

    result = runner.invoke(
        app,
        [
            "score-stage2",
            "A1",
            "--model",
            str(model_row_id),
            "--text",
            "Discuss with your physician.",
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
    assert "stage2:" in result.output
