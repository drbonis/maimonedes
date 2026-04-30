"""Tests for Phase 5 Stage-2 offline trainer."""
from __future__ import annotations

import json
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
from maimonedes.llm.embed_client import EmbedResponse
from maimonedes.models.stage2 import (
    AxisMetrics,
    Stage2Model,
    _grade_agreement,
    _stratified_split,
    train_stage2,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import (
    get_engine,
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from maimonedes.storage.stage2_models import (
    Stage2ModelRow,
    latest_stage2_model_row,
    list_stage2_models,
    record_stage2_model,
)
from tests.fakes import FakeEmbedClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"

runner = CliRunner()


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "stage2.sqlite"
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


# ---- migration -------------------------------------------------------------


def test_migration_creates_stage2_models_table(db: str) -> None:
    insp = inspect(get_engine())
    assert "stage2_models" in insp.get_table_names()
    indexes = {ix["name"] for ix in insp.get_indexes("stage2_models")}
    assert "ix_stage2_models_policy_id" in indexes


def test_alembic_round_trip_clean(db: str) -> None:
    cfg = _alembic_cfg(db)
    command.downgrade(cfg, "0008_embed_calls")
    insp = inspect(get_engine())
    assert "stage2_models" not in insp.get_table_names()
    command.upgrade(cfg, "head")
    assert "stage2_models" in inspect(get_engine()).get_table_names()


# ---- helpers ---------------------------------------------------------------


def test_grade_agreement_green_when_all_axes_pass() -> None:
    metrics = [
        AxisMetrics(sub_id="a", mae=0.10, spearman_rho=0.7),
        AxisMetrics(sub_id="b", mae=0.12, spearman_rho=0.65),
    ]
    assert _grade_agreement(metrics) == "green"


def test_grade_agreement_amber_when_one_axis_falls_to_amber() -> None:
    metrics = [
        AxisMetrics(sub_id="a", mae=0.10, spearman_rho=0.7),
        AxisMetrics(sub_id="b", mae=0.20, spearman_rho=0.5),
    ]
    assert _grade_agreement(metrics) == "amber"


def test_grade_agreement_red_when_any_axis_fails_both() -> None:
    metrics = [
        AxisMetrics(sub_id="a", mae=0.10, spearman_rho=0.7),
        AxisMetrics(sub_id="b", mae=0.30, spearman_rho=0.2),
    ]
    assert _grade_agreement(metrics) == "red"


def test_grade_agreement_empty_is_red() -> None:
    assert _grade_agreement([]) == "red"


# ---- stratified split ------------------------------------------------------


def _example(anchor_id: str, aggregate: float, text: str = "x"):
    from maimonedes.models.stage2 import TrainingExample

    return TrainingExample(
        anchor_id=anchor_id,
        text=text,
        score=ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": aggregate},
            aggregate=aggregate,
            judge_model="judge:test",
            supervised_model="llama:test",
        ),
    )


def test_stratified_split_keeps_singletons_in_train() -> None:
    examples = [_example("A1", 0.9), _example("A2", 0.5)]
    train, evalset = _stratified_split(examples, eval_fraction=0.2, seed=0)
    assert len(evalset) == 0
    assert len(train) == 2


def test_stratified_split_distributes_within_anchor() -> None:
    examples = [_example("A1", v) for v in (0.9, 0.8, 0.7, 0.6, 0.5)]
    examples += [_example("A2", v) for v in (0.4, 0.3, 0.2, 0.1, 0.0)]
    train, evalset = _stratified_split(examples, eval_fraction=0.2, seed=0)
    assert len(evalset) == 2
    train_anchors = {e.anchor_id for e in train}
    eval_anchors = {e.anchor_id for e in evalset}
    assert train_anchors == {"A1", "A2"}
    assert eval_anchors.issubset({"A1", "A2"})


# ---- training pipeline -----------------------------------------------------


def _seed_training_data(
    policy: Policy, *, n_per_anchor: int = 20, anchors: tuple[str, ...] = ("A1", "A2")
) -> int:
    """Insert (llm_calls, compliance_scores) pairs for trainer pickup."""
    n_inserted = 0
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    # Insert llm_calls first so we can pass real ids into record_score
    # (record_score opens its own session; nesting causes SQLite to lock).
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
    for anchor_id, llm_id_str, aggregate in payloads:
        per_sub = {sid: aggregate for sid in sub_ids}
        record_score(
            ComplianceScore(
                anchor_id=anchor_id,
                policy_id=policy.id,
                per_sub_condition=per_sub,
                aggregate=aggregate,
                judge_model="judge:test",
                supervised_model="llama:test",
                llm_call_id=int(llm_id_str),
            )
        )
        n_inserted += 1
    return n_inserted


def test_train_stage2_raises_when_min_samples_unmet(
    db: str, policy: Policy
) -> None:
    fake = FakeEmbedClient(default_dim=8)
    with pytest.raises(ValueError, match="at least"):
        train_stage2(
            policy=policy,
            embed_client=fake,
            min_samples=50,
        )


def test_train_stage2_returns_model_with_metrics(
    db: str, policy: Policy
) -> None:
    n = _seed_training_data(policy, n_per_anchor=30, anchors=("A1", "A2"))
    assert n == 60
    fake = FakeEmbedClient(default_dim=16)
    model = train_stage2(
        policy=policy,
        embed_client=fake,
        embedding_model="fake-embed",
        min_samples=50,
        seed=0,
    )
    assert model.policy_id == policy.id
    assert model.n_samples == 60
    assert model.n_train + model.n_eval == 60
    assert model.feature_dim == 16
    assert set(model.heads) == {s.id for s in policy.rubric.sub_conditions}
    assert len(model.agreement_metrics) == len(policy.rubric.sub_conditions)
    assert model.agreement_status in {"green", "amber", "red"}


def test_train_stage2_model_predict_round_trips(
    db: str, policy: Policy, tmp_path: Path
) -> None:
    _seed_training_data(policy, n_per_anchor=30, anchors=("A1", "A2"))
    fake = FakeEmbedClient(default_dim=16)
    model = train_stage2(
        policy=policy,
        embed_client=fake,
        min_samples=50,
        seed=42,
    )

    embedding = [0.5] * 16
    per_sub = model.predict_per_axis(embedding)
    assert set(per_sub) == {s.id for s in policy.rubric.sub_conditions}
    assert all(0.0 <= v <= 1.0 for v in per_sub.values())

    aggregate, per_sub = model.predict_aggregate(embedding, policy=policy)
    assert 0.0 <= aggregate <= 1.0


def test_stage2_model_save_load_round_trip(
    db: str, policy: Policy, tmp_path: Path
) -> None:
    _seed_training_data(policy, n_per_anchor=30, anchors=("A1", "A2"))
    fake = FakeEmbedClient(default_dim=16)
    model = train_stage2(
        policy=policy,
        embed_client=fake,
        min_samples=50,
        seed=0,
    )
    path = model.save(tmp_path / "stage2_test.pkl")
    loaded = Stage2Model.load(path)
    assert loaded.policy_id == model.policy_id
    assert set(loaded.heads) == set(model.heads)
    a = loaded.predict_per_axis([0.3] * 16)
    b = model.predict_per_axis([0.3] * 16)
    assert a == pytest.approx(b)


def test_stage2_model_load_rejects_non_model_pickle(tmp_path: Path) -> None:
    import pickle

    path = tmp_path / "junk.pkl"
    with path.open("wb") as fh:
        pickle.dump({"not": "a model"}, fh)
    with pytest.raises(ValueError, match="not contain a Stage2Model"):
        Stage2Model.load(path)


# ---- registry --------------------------------------------------------------


def test_record_and_list_stage2_models(db: str) -> None:
    a = record_stage2_model(
        path="models/stage2_a.pkl",
        policy_id="scope_of_practice",
        n_samples=60,
        embedding_model="bio_clinicalbert",
        mae_per_axis={"x": 0.1},
        spearman_per_axis={"x": 0.7},
        agreement_status="green",
    )
    b = record_stage2_model(
        path="models/stage2_b.pkl",
        policy_id="scope_of_practice",
        n_samples=80,
        embedding_model="bio_clinicalbert",
        mae_per_axis={"x": 0.2},
        spearman_per_axis={"x": 0.55},
        agreement_status="amber",
    )
    rows = list_stage2_models(policy_id="scope_of_practice")
    assert [r.id for r in rows] == [b, a]
    latest = latest_stage2_model_row("scope_of_practice")
    assert latest is not None
    assert latest.id == b


# ---- CLI -------------------------------------------------------------------


def test_train_stage2_cli_writes_artefact_and_registers(
    db: str, policy: Policy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_training_data(policy, n_per_anchor=30, anchors=("A1", "A2"))
    fake = FakeEmbedClient(default_dim=16)
    monkeypatch.setattr(cli, "_embed_factory", lambda settings: fake)

    output = tmp_path / "stage2_cli.pkl"
    result = runner.invoke(
        app,
        [
            "train-stage2",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--output",
            str(output),
            "--min-samples",
            "50",
        ],
    )
    assert result.exit_code in (0, 6), result.output  # 6 if red, 0 otherwise
    assert output.exists()
    rows = list_stage2_models(policy_id=policy.id)
    assert len(rows) == 1
    assert rows[0].path == str(output)


def test_train_stage2_cli_exits_5_on_short_dataset(
    db: str, policy: Policy, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = FakeEmbedClient(default_dim=8)
    monkeypatch.setattr(cli, "_embed_factory", lambda settings: fake)

    result = runner.invoke(
        app,
        [
            "train-stage2",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--output",
            str(tmp_path / "x.pkl"),
            "--min-samples",
            "50",
        ],
    )
    assert result.exit_code == 5
    assert "training failed" in result.output


@pytest.mark.integration
def test_train_stage2_cli_against_real_embed_service(
    db: str, policy: Policy, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Live smoke. Skipped unless --integration."""
    from maimonedes.llm.clinicalbert_backend import ClinicalBertBackend
    from maimonedes.settings import get_settings

    _seed_training_data(policy, n_per_anchor=30, anchors=("A1", "A2"))
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
    output = tmp_path / "stage2_real.pkl"
    result = runner.invoke(
        app,
        [
            "train-stage2",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--output",
            str(output),
            "--min-samples",
            "50",
        ],
    )
    assert result.exit_code in (0, 6), result.output
    assert output.exists()
