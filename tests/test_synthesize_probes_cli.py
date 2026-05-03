"""Tests for the Phase 5 synthesize-probes orchestrator + CLI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy, load_policy
from maimonedes.core.probe import load_anchors
from maimonedes.experiments.synthesize_probes import synthesize_probes
from maimonedes.llm.client import ChatResponse
from maimonedes.llm.embed_client import EmbedResponse
from maimonedes.models.stage2 import train_stage2
from maimonedes.monitor.gp_layer import (
    ComplianceGP,
    fit_compliance_gp,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.gp_fits import (
    record_gp_fit,
)
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import (
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from maimonedes.storage.stage2_models import record_stage2_model
from maimonedes.storage.synthesized_probes import (
    list_synthesized_probes,
    scores_for_synthesized_probe,
)
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
    db_path = tmp_path / "synth_cli.sqlite"
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
def anchors():
    return load_anchors(PROBES_PATH)


def _seed_supervised_pairs(
    policy: Policy,
    *,
    anchors_subset: tuple[str, ...] = ("A1", "A2", "A3"),
    n_per_anchor: int = 8,
) -> int:
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    payloads = []
    with get_session() as session:
        for anchor_id in anchors_subset:
            for i in range(n_per_anchor):
                aggregate = 0.5 + 0.4 * (i % 5 - 2) / 4.0
                aggregate = max(0.0, min(1.0, aggregate))
                text = f"{anchor_id}-output-{i}"
                llm = LLMCall(
                    backend_name="ollama-supervised",
                    model="llama:test",
                    request_messages_json=json.dumps([]),
                    response_content=text,
                    raw_response_json="{}",
                    prompt_tokens=10,
                    completion_tokens=10,
                    latency_ms=1.0,
                    request_hash=f"synth-cli-seed-{anchor_id}-{i}",
                )
                session.add(llm)
                session.flush()
                payloads.append((anchor_id, llm.id, aggregate))
    for anchor_id, llm_id, aggregate in payloads:
        record_score(
            ComplianceScore(
                anchor_id=anchor_id,
                policy_id=policy.id,
                per_sub_condition={sid: aggregate for sid in sub_ids},
                aggregate=aggregate,
                judge_model="judge:test",
                supervised_model="llama:test",
                llm_call_id=llm_id,
            )
        )
    return len(payloads)


def _seed_gp_fit(policy: Policy, anchors, tmp_path: Path) -> tuple[int, ComplianceGP]:
    """Fit a GP and persist the artefact + registry row; return (id, gp)."""
    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
        library_anchor_texts={a.id: a.scenario for a in anchors[:5]},
    )
    path = tmp_path / "gp_synth_cli.pkl"
    gp.save(path)
    fit_id = record_gp_fit(
        path=str(path),
        policy_id=policy.id,
        n_samples=gp.n_samples,
        kernel_name=gp.kernel_repr,
        log_marginal_likelihood=gp.log_marginal_likelihood,
        embedding_model=gp.embedding_model,
    )
    return fit_id, gp


# ---- orchestrator (in-process) --------------------------------------------


def test_synthesize_probes_happy_path_judge(
    db: str, policy: Policy, anchors, tmp_path: Path
) -> None:
    _seed_supervised_pairs(policy)
    fit_id, gp = _seed_gp_fit(policy, anchors, tmp_path)

    target_dim = 8
    matching = [1.0] + [0.0] * (target_dim - 1)
    n = 2
    # The synthesizer embeds every library anchor at construction; the
    # orchestrator passes all anchors (8 from the v1 library). Queue
    # uniform non-zero embeddings for the library (so KNN search has
    # defined cosines), then `n` matching embeds for the synthesize
    # re-embed step (which is what cos-vs-target compares against).
    n_library = sum(1 for a in anchors if a.policy_id == policy.id)
    embed_fake = FakeEmbedClient(default_dim=target_dim)
    for _ in range(n_library):
        embed_fake.queue(
            EmbedResponse(
                embedding=[0.1] * target_dim,
                model="fake-embed",
                latency_ms=0.0,
            )
        )
    for _ in range(n):
        embed_fake.queue(
            EmbedResponse(embedding=matching, model="fake-embed", latency_ms=0.0)
        )

    chat_responses = []
    for _ in range(n):
        chat_responses.extend(
            [
                ChatResponse(
                    content="Hi, I'm John, a 60-year-old man working as a teacher. Generated scenario.", model="gen:test", latency_ms=0.0
                ),
                ChatResponse(
                    content="yes: realistic.", model="val:test", latency_ms=0.0
                ),
                ChatResponse(
                    content="Supervised says: discuss with physician.",
                    model="sup:test",
                    latency_ms=0.0,
                ),
                ChatResponse(
                    content=payload_json_for_target(policy, 0.85),
                    model="judge:test",
                    latency_ms=0.0,
                ),
            ]
        )
    fake_llm = FakeLLMClient(responses=chat_responses)

    summary = synthesize_probes(
        gp_fit=gp,
        n_targets=n,
        embed_client=embed_fake,
        generator_client=fake_llm,
        validator_client=fake_llm,
        supervised_client=fake_llm,
        judge_client=fake_llm,
        embedding_model="fake-embed",
        generator_model="gen:test",
        validator_model="val:test",
        supervised_model="sup:test",
        judge_model="judge:test",
        policy=policy,
        anchors=anchors,
        scorer="judge",
        gp_fit_id=fit_id,
        tau=-1.0,  # disable the gate; this test exercises the orchestrator flow, not the synthesizer's tau check
        max_retries=0,
    )

    assert summary.n_targets == n
    assert summary.n_approved == n
    assert summary.n_scored == n
    assert summary.mean_aggregate is not None
    persisted = list_synthesized_probes(policy_id=policy.id)
    assert len(persisted) == n
    for probe in persisted:
        assert probe.quality_status == "approved"
        scores = scores_for_synthesized_probe(probe.id)  # type: ignore[arg-type]
        assert len(scores) == 1


def test_synthesize_probes_classifier_path(
    db: str, policy: Policy, anchors, tmp_path: Path
) -> None:
    _seed_supervised_pairs(policy)
    fit_id, gp = _seed_gp_fit(policy, anchors, tmp_path)
    fake_embed_for_train = FakeEmbedClient(default_dim=8)
    stage2 = train_stage2(
        policy=policy,
        embed_client=fake_embed_for_train,
        embedding_model="fake-embed",
        min_samples=10,
    )

    target_dim = 8
    matching = [1.0] + [0.0] * (target_dim - 1)
    embed_fake = FakeEmbedClient(default_dim=target_dim)
    n_library = sum(1 for a in anchors if a.policy_id == policy.id)
    for _ in range(n_library):
        embed_fake.queue(
            EmbedResponse(
                embedding=[0.1] * target_dim,
                model="fake-embed",
                latency_ms=0.0,
            )
        )
    embed_fake.queue(
        EmbedResponse(embedding=matching, model="fake-embed", latency_ms=0.0)
    )
    # Stage-2 scoring also embeds the supervised output once.
    embed_fake.queue(
        EmbedResponse(
            embedding=[0.5] * target_dim, model="fake-embed", latency_ms=0.0
        )
    )

    fake_llm = FakeLLMClient(
        responses=[
            ChatResponse(content="Hi, I'm John, a 60-year-old man working as a teacher. Generated scenario.", model="gen:test", latency_ms=0.0),
            ChatResponse(content="yes: realistic.", model="val:test", latency_ms=0.0),
            ChatResponse(
                content="Supervised says: discuss with physician.",
                model="sup:test",
                latency_ms=0.0,
            ),
        ]
    )

    summary = synthesize_probes(
        gp_fit=gp,
        n_targets=1,
        embed_client=embed_fake,
        generator_client=fake_llm,
        validator_client=fake_llm,
        supervised_client=fake_llm,
        judge_client=fake_llm,
        embedding_model="fake-embed",
        generator_model="gen:test",
        validator_model="val:test",
        supervised_model="sup:test",
        judge_model="judge:test",
        policy=policy,
        anchors=anchors,
        scorer="classifier",
        stage2_model=stage2,
        gp_fit_id=fit_id,
        tau=-1.0,
        max_retries=0,
    )

    assert summary.n_approved == 1
    assert summary.n_scored == 1
    persisted = list_synthesized_probes(policy_id=policy.id)
    assert len(persisted) == 1
    scores = scores_for_synthesized_probe(persisted[0].id)  # type: ignore[arg-type]
    assert scores
    assert scores[0].judge_model.startswith("stage2")


def test_synthesize_probes_classifier_without_stage2_model_raises(
    db: str, policy: Policy, anchors, tmp_path: Path
) -> None:
    _seed_supervised_pairs(policy)
    fit_id, gp = _seed_gp_fit(policy, anchors, tmp_path)
    embed_fake = FakeEmbedClient(default_dim=8)
    fake_llm = FakeLLMClient()
    with pytest.raises(ValueError, match="requires a Stage2Model"):
        synthesize_probes(
            gp_fit=gp,
            n_targets=1,
            embed_client=embed_fake,
            generator_client=fake_llm,
            validator_client=fake_llm,
            supervised_client=fake_llm,
            judge_client=fake_llm,
            embedding_model="fake-embed",
            generator_model="gen:test",
            validator_model="val:test",
            supervised_model="sup:test",
            judge_model="judge:test",
            policy=policy,
            anchors=anchors,
            scorer="classifier",
            stage2_model=None,
            gp_fit_id=fit_id,
        )


# ---- CLI -------------------------------------------------------------------


def test_synthesize_probes_cli_dry_run(
    db: str, policy: Policy, anchors, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_supervised_pairs(policy)
    fit_id, _gp = _seed_gp_fit(policy, anchors, tmp_path)

    fake_embed = FakeEmbedClient(default_dim=8)
    fake_chat = FakeLLMClient()
    monkeypatch.setattr(cli, "_embed_factory", lambda s: fake_embed)
    monkeypatch.setattr(cli, "_backend_factory", lambda s: fake_chat)

    result = runner.invoke(
        app,
        [
            "synthesize-probes",
            str(fit_id),
            "--n",
            "3",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert "n_targets=3" in result.output
    # No LLM calls should have happened.
    assert fake_chat.calls == []


def test_synthesize_probes_cli_unknown_gp_fit_exits_4(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_embed_factory", lambda s: FakeEmbedClient())
    monkeypatch.setattr(cli, "_backend_factory", lambda s: FakeLLMClient())
    result = runner.invoke(app, ["synthesize-probes", "999", "--dry-run"])
    assert result.exit_code == 4


def test_synthesize_probes_cli_classifier_without_stage2_id_exits_5(
    db: str, policy: Policy, anchors, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_supervised_pairs(policy)
    fit_id, _gp = _seed_gp_fit(policy, anchors, tmp_path)
    monkeypatch.setattr(cli, "_embed_factory", lambda s: FakeEmbedClient())
    monkeypatch.setattr(cli, "_backend_factory", lambda s: FakeLLMClient())
    result = runner.invoke(
        app,
        [
            "synthesize-probes",
            str(fit_id),
            "--scorer",
            "classifier",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
        ],
    )
    assert result.exit_code == 5
    assert "stage2-model" in result.output


def test_synthesize_probes_cli_invalid_scorer_rejected(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_embed_factory", lambda s: FakeEmbedClient())
    monkeypatch.setattr(cli, "_backend_factory", lambda s: FakeLLMClient())
    result = runner.invoke(
        app,
        ["synthesize-probes", "1", "--scorer", "bogus", "--dry-run"],
    )
    assert result.exit_code != 0


# ---- #52: gradient strategy ------------------------------------------------


def _stage2_with_unit_coef(policy: Policy, *, feature_dim: int = 8) -> "Stage2Model":  # noqa: F821
    """Synthetic Stage-2 with one ridge head per axis, all coefs ones."""
    from datetime import datetime, timezone

    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from maimonedes.models.stage2 import AxisMetrics, Stage2Model

    heads: dict = {}
    X_fit = np.array(
        [np.ones(feature_dim), -np.ones(feature_dim)], dtype=float
    )
    for sub in policy.rubric.sub_conditions:
        ridge = Ridge(alpha=1e-9)
        ridge.coef_ = np.ones(feature_dim, dtype=float)
        ridge.intercept_ = 0.5
        ridge.n_features_in_ = feature_dim
        scaler = StandardScaler()
        scaler.fit(X_fit)
        heads[sub.id] = Pipeline([("scaler", scaler), ("head", ridge)])
    return Stage2Model(
        policy_id=policy.id,
        trained_at=datetime.now(timezone.utc),
        n_samples=2,
        n_train=2,
        n_eval=0,
        embedding_model="fake-embed",
        feature_dim=feature_dim,
        heads=heads,
        agreement_metrics=[
            AxisMetrics(sub_id=s.id, mae=0.0, spearman_rho=1.0)
            for s in policy.rubric.sub_conditions
        ],
        agreement_status="green",
    )


def test_synthesize_probes_strategy_gradient_persists_generation_method(
    db: str, policy: Policy, anchors, tmp_path: Path
) -> None:
    """Issue #52: strategy='gradient' results in probes recorded as 'gradient_v1'."""
    _seed_supervised_pairs(policy)
    fit_id, gp = _seed_gp_fit(policy, anchors, tmp_path)
    stage2 = _stage2_with_unit_coef(policy)

    target_dim = 8
    matching = [1.0] + [0.0] * (target_dim - 1)
    n = 1
    n_library = sum(1 for a in anchors if a.policy_id == policy.id)
    embed_fake = FakeEmbedClient(default_dim=target_dim)
    for _ in range(n_library):
        embed_fake.queue(
            EmbedResponse(
                embedding=[0.1] * target_dim,
                model="fake-embed",
                latency_ms=0.0,
            )
        )
    for _ in range(n):
        embed_fake.queue(
            EmbedResponse(embedding=matching, model="fake-embed", latency_ms=0.0)
        )

    chat_responses = []
    for _ in range(n):
        chat_responses.extend(
            [
                ChatResponse(content="Hi, I'm John, a 60-year-old man working as a teacher. Generated.", model="gen:test", latency_ms=0.0),
                ChatResponse(content="yes: realistic.", model="val:test", latency_ms=0.0),
                ChatResponse(content="Reply.", model="sup:test", latency_ms=0.0),
                ChatResponse(
                    content=payload_json_for_target(policy, 0.85),
                    model="judge:test",
                    latency_ms=0.0,
                ),
            ]
        )
    fake_llm = FakeLLMClient(responses=chat_responses)

    summary = synthesize_probes(
        gp_fit=gp,
        n_targets=n,
        embed_client=embed_fake,
        generator_client=fake_llm,
        validator_client=fake_llm,
        supervised_client=fake_llm,
        judge_client=fake_llm,
        embedding_model="fake-embed",
        generator_model="gen:test",
        validator_model="val:test",
        supervised_model="sup:test",
        judge_model="judge:test",
        policy=policy,
        anchors=anchors,
        scorer="judge",
        stage2_model=stage2,
        gp_fit_id=fit_id,
        tau=-1.0,
        max_retries=0,
        strategy="gradient",
        gradient_axis_id=policy.rubric.sub_conditions[0].id,
    )
    assert summary.n_targets == n
    assert summary.generation_method == "gradient_v1"
    persisted = list_synthesized_probes(policy_id=policy.id)
    assert len(persisted) == n
    for probe in persisted:
        assert probe.generation_method == "gradient_v1"


def test_synthesize_probes_strategy_gradient_without_stage2_raises(
    db: str, policy: Policy, anchors, tmp_path: Path
) -> None:
    _seed_supervised_pairs(policy)
    fit_id, gp = _seed_gp_fit(policy, anchors, tmp_path)
    embed_fake = FakeEmbedClient(default_dim=8)
    fake_llm = FakeLLMClient()
    with pytest.raises(ValueError, match="strategy='gradient' requires a Stage2Model"):
        synthesize_probes(
            gp_fit=gp,
            n_targets=1,
            embed_client=embed_fake,
            generator_client=fake_llm,
            validator_client=fake_llm,
            supervised_client=fake_llm,
            judge_client=fake_llm,
            embedding_model="fake-embed",
            generator_model="gen:test",
            validator_model="val:test",
            supervised_model="sup:test",
            judge_model="judge:test",
            policy=policy,
            anchors=anchors,
            stage2_model=None,
            gp_fit_id=fit_id,
            strategy="gradient",
        )


def test_synthesize_probes_unknown_strategy_raises(
    db: str, policy: Policy, anchors, tmp_path: Path
) -> None:
    _seed_supervised_pairs(policy)
    fit_id, gp = _seed_gp_fit(policy, anchors, tmp_path)
    embed_fake = FakeEmbedClient(default_dim=8)
    fake_llm = FakeLLMClient()
    with pytest.raises(ValueError, match="unknown strategy"):
        synthesize_probes(
            gp_fit=gp,
            n_targets=1,
            embed_client=embed_fake,
            generator_client=fake_llm,
            validator_client=fake_llm,
            supervised_client=fake_llm,
            judge_client=fake_llm,
            embedding_model="fake-embed",
            generator_model="gen:test",
            validator_model="val:test",
            supervised_model="sup:test",
            judge_model="judge:test",
            policy=policy,
            anchors=anchors,
            gp_fit_id=fit_id,
            strategy="bogus",  # type: ignore[arg-type]
        )


def test_synthesize_probes_cli_strategy_gradient_dry_run(
    db: str, policy: Policy, anchors, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_supervised_pairs(policy)
    fit_id, _gp = _seed_gp_fit(policy, anchors, tmp_path)
    stage2 = _stage2_with_unit_coef(policy)
    stage2_path = tmp_path / "stage2_grad.pkl"
    stage2.save(stage2_path)
    stage2_id = record_stage2_model(
        path=str(stage2_path),
        policy_id=policy.id,
        n_samples=stage2.n_samples,
        embedding_model=stage2.embedding_model,
        mae_per_axis={m.sub_id: m.mae for m in stage2.agreement_metrics},
        spearman_per_axis={m.sub_id: m.spearman_rho for m in stage2.agreement_metrics},
        agreement_status=stage2.agreement_status,
        head_kind=stage2.head_kind,
    )

    fake_embed = FakeEmbedClient(default_dim=8)
    fake_chat = FakeLLMClient()
    monkeypatch.setattr(cli, "_embed_factory", lambda s: fake_embed)
    monkeypatch.setattr(cli, "_backend_factory", lambda s: fake_chat)

    result = runner.invoke(
        app,
        [
            "synthesize-probes",
            str(fit_id),
            "--n",
            "2",
            "--strategy",
            "gradient",
            "--stage2-model",
            str(stage2_id),
            "--gradient-axis",
            policy.rubric.sub_conditions[0].id,
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "strategy=gradient" in result.output
    assert "n_targets=2" in result.output
    # No chat calls in dry-run.
    assert fake_chat.calls == []


def test_synthesize_probes_cli_strategy_gradient_without_stage2_id_exits_5(
    db: str, policy: Policy, anchors, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_supervised_pairs(policy)
    fit_id, _gp = _seed_gp_fit(policy, anchors, tmp_path)
    monkeypatch.setattr(cli, "_embed_factory", lambda s: FakeEmbedClient())
    monkeypatch.setattr(cli, "_backend_factory", lambda s: FakeLLMClient())
    result = runner.invoke(
        app,
        [
            "synthesize-probes",
            str(fit_id),
            "--strategy",
            "gradient",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
        ],
    )
    assert result.exit_code == 5
    assert "stage2-model" in result.output


def test_synthesize_probes_cli_invalid_strategy_rejected(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "_embed_factory", lambda s: FakeEmbedClient())
    monkeypatch.setattr(cli, "_backend_factory", lambda s: FakeLLMClient())
    result = runner.invoke(
        app,
        ["synthesize-probes", "1", "--strategy", "wrong", "--dry-run"],
    )
    assert result.exit_code != 0
