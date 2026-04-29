"""Tests for `maimonedes perturb` orchestrator and CLI subcommand."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.core.perturbation_generators import (
    AuthorityGenerator,
    BoundaryGenerator,
    DemographicGenerator,
)
from maimonedes.core.policy import Policy, load_policy
from maimonedes.core.probe import load_anchors
from maimonedes.experiments.perturbation_session import run_perturbations
from maimonedes.llm.client import ChatResponse
from maimonedes.settings import Settings, get_settings
from maimonedes.storage.compliance import (
    ComplianceScoreRow,
    record_score,
    recent_scores,
)
from maimonedes.storage.perturbations import recent_perturbations
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
DEMO_PATH = PROJECT_ROOT / "config" / "perturbations" / "demographic_v1.yaml"
AUTH_PATH = PROJECT_ROOT / "config" / "perturbations" / "authority_v1.yaml"
BOUND_PATH = PROJECT_ROOT / "config" / "perturbations" / "boundary_v1.yaml"

runner = CliRunner()


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "perturb.sqlite"
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


def _supervised_response(text: str = "I encourage discussing with your physician.") -> ChatResponse:
    return ChatResponse(content=text, model="supervised:test", latency_ms=10.0)


def _judge_response(policy: Policy) -> ChatResponse:
    return ChatResponse(
        content=compliant_response_json(policy),
        model="judge:test",
        latency_ms=10.0,
    )


# ---- orchestrator: rule-based-only path (no LLM in generators) ------------


def test_run_perturbations_authority_only_persists_probes_and_scores(
    db: str, policy: Policy
) -> None:
    """Authority generator emits 4 probes; orchestrator runs each through
    supervised + judge → 4 perturbation_probes rows + 4 compliance_scores rows
    (probe_role=perturbation, perturbation_id set).
    """
    # 4 authority prefixes → 4 supervised + 4 judge calls = 8 fake responses.
    responses: list[ChatResponse] = []
    for _ in range(4):
        responses.append(_supervised_response())
        responses.append(_judge_response(policy))
    fake = FakeLLMClient(responses=responses)
    anchors = load_anchors(PROBES_PATH)

    outcomes = run_perturbations(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        generators=[AuthorityGenerator(AUTH_PATH)],
        supervised_model="supervised:test",
        judge_model="judge:test",
    )
    assert len(outcomes) == 4
    assert all(o.score is not None for o in outcomes)
    assert {o.probe.transform_label for o in outcomes} == {
        "authority:senior_cardiologist",
        "authority:on_call_attending",
        "authority:surgical_team",
        "authority:gp",
    }

    persisted = recent_perturbations("A1")
    assert len(persisted) == 4

    # All compliance_scores written by the orchestrator must be perturbations
    # — no anchor row was seeded by this test, so dashboard query stays empty.
    with get_session() as session:
        rows = session.query(ComplianceScoreRow).all()
        roles = {r.probe_role for r in rows}
        perturbation_ids = {r.perturbation_id for r in rows}
    assert len(rows) == 4
    assert roles == {"perturbation"}
    assert all(pid is not None for pid in perturbation_ids)


def test_run_perturbations_with_baseline_computes_delta(
    db: str, policy: Policy
) -> None:
    # Seed an anchor baseline at aggregate=0.6
    from maimonedes.core.compliance import ComplianceScore

    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": 1.0},
            aggregate=0.6,
            judge_model="judge:test",
            supervised_model="supervised:test",
        )
    )
    responses: list[ChatResponse] = []
    for _ in range(4):
        responses.append(_supervised_response())
        responses.append(_judge_response(policy))
    fake = FakeLLMClient(responses=responses)
    anchors = load_anchors(PROBES_PATH)

    outcomes = run_perturbations(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        generators=[AuthorityGenerator(AUTH_PATH)],
        supervised_model="supervised:test",
        judge_model="judge:test",
    )
    # The judge fake returns all-compliant → aggregate 1.0; baseline 0.6 → Δ = +0.4
    for o in outcomes:
        assert o.delta_aggregate == pytest.approx(0.4)


def test_run_perturbations_continues_through_judge_failures(
    db: str, policy: Policy
) -> None:
    # 4 authority probes; first judge response is malformed (will raise),
    # second supervised+judge is clean, then we run out of responses for
    # the 3rd and 4th — those should fail cleanly too.
    responses: list[ChatResponse] = [
        _supervised_response(),
        ChatResponse(content="not-json", model="judge:test", latency_ms=1.0),
        _supervised_response(),
        _judge_response(policy),
    ]
    fake = FakeLLMClient(responses=responses)
    anchors = load_anchors(PROBES_PATH)

    outcomes = run_perturbations(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        generators=[AuthorityGenerator(AUTH_PATH)],
        supervised_model="supervised:test",
        judge_model="judge:test",
    )
    # 4 probes total; one full success, one judge-failure, two depletion-failures
    assert len(outcomes) == 4
    successes = [o for o in outcomes if o.score is not None]
    failures = [o for o in outcomes if o.score is None]
    assert len(successes) == 1
    assert len(failures) == 3
    assert all(f.error for f in failures)


def test_anchor_baseline_unaffected_by_perturbation_run(
    db: str, policy: Policy
) -> None:
    """`latest_score_per_anchor` must continue to ignore perturbation rows."""
    from maimonedes.core.compliance import ComplianceScore
    from maimonedes.storage.compliance import latest_score_per_anchor

    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": 1.0},
            aggregate=0.7,
            judge_model="judge:test",
            supervised_model="supervised:test",
        )
    )
    responses: list[ChatResponse] = []
    for _ in range(4):
        responses.append(_supervised_response())
        responses.append(_judge_response(policy))
    fake = FakeLLMClient(responses=responses)
    anchors = load_anchors(PROBES_PATH)
    run_perturbations(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        generators=[AuthorityGenerator(AUTH_PATH)],
        supervised_model="supervised:test",
        judge_model="judge:test",
    )

    latest = latest_score_per_anchor()
    # The seeded anchor score (0.7) remains the dashboard's view of A1
    # even though four perturbation rows now exist for it.
    assert latest["A1"].aggregate == pytest.approx(0.7)
    assert latest["A1"].probe_role == "anchor"


def test_run_perturbations_streams_progress_callback(
    db: str, policy: Policy
) -> None:
    """`on_outcome` fires once per probe with index 1..N and a stable total."""
    from maimonedes.experiments.perturbation_session import PerturbationProgress

    responses: list[ChatResponse] = []
    for _ in range(4):
        responses.append(_supervised_response())
        responses.append(_judge_response(policy))
    fake = FakeLLMClient(responses=responses)
    anchors = load_anchors(PROBES_PATH)
    events: list[PerturbationProgress] = []

    run_perturbations(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        generators=[AuthorityGenerator(AUTH_PATH)],
        supervised_model="supervised:test",
        judge_model="judge:test",
        on_outcome=events.append,
    )

    assert len(events) == 4
    assert [e.index for e in events] == [1, 2, 3, 4]
    assert all(e.total == 4 for e in events)
    assert all(e.anchor_id == "A1" for e in events)
    # The outcome on the progress event matches what the function returns.
    assert {e.outcome.probe.transform_label for e in events} == {
        "authority:senior_cardiologist",
        "authority:on_call_attending",
        "authority:surgical_team",
        "authority:gp",
    }


def test_run_perturbations_emits_generation_event_per_generator(
    db: str, policy: Policy
) -> None:
    """`on_generation` fires once per generator with the probe count."""
    responses: list[ChatResponse] = []
    for _ in range(8):  # 4 authority + 4-ish boundary
        responses.append(_supervised_response())
        responses.append(_judge_response(policy))
    fake = FakeLLMClient(responses=responses)
    anchors = load_anchors(PROBES_PATH)
    seen: list[tuple[str, int]] = []

    run_perturbations(
        "A1",
        policy=policy,
        anchors=anchors,
        supervised_client=fake,
        judge_client=fake,
        generators=[AuthorityGenerator(AUTH_PATH), BoundaryGenerator(BOUND_PATH)],
        supervised_model="supervised:test",
        judge_model="judge:test",
        on_generation=lambda name, n: seen.append((name, n)),
    )

    names = [name for name, _ in seen]
    assert names == ["AuthorityGenerator", "BoundaryGenerator"]
    # Authority always emits 4; boundary emits whatever the v1 templates
    # match against A1's scenario (we don't pin the exact count here —
    # just that the event fired with a non-negative integer).
    counts = [n for _, n in seen]
    assert counts[0] == 4
    assert counts[1] >= 0


def test_run_perturbations_unknown_anchor_raises(db: str, policy: Policy) -> None:
    fake = FakeLLMClient()
    anchors = load_anchors(PROBES_PATH)
    with pytest.raises(KeyError):
        run_perturbations(
            "A99",
            policy=policy,
            anchors=anchors,
            supervised_client=fake,
            judge_client=fake,
            generators=[AuthorityGenerator(AUTH_PATH)],
            supervised_model="supervised:test",
            judge_model="judge:test",
        )


# ---- CLI -------------------------------------------------------------------


@pytest.fixture
def cli_backend(monkeypatch: pytest.MonkeyPatch, policy: Policy) -> FakeLLMClient:
    # 4 boundary probes ("should I" only matches one anchor; let's pick A4)
    # × 2 calls per probe = 8 responses minimum. Be generous.
    responses: list[ChatResponse] = []
    for _ in range(20):
        responses.append(_supervised_response())
        responses.append(_judge_response(policy))
    fake = FakeLLMClient(responses=responses)
    monkeypatch.setenv("OLLAMA_SUPERVISED_MODEL", "supervised:test")
    monkeypatch.setenv("OLLAMA_JUDGE_MODEL", "judge:test")
    cli.set_backend_factory(lambda settings: fake)
    yield fake
    cli.reset_backend_factory()


def test_cli_perturb_authority_only_persists_rows(
    db: str, cli_backend: FakeLLMClient
) -> None:
    result = runner.invoke(
        app,
        [
            "perturb",
            "A1",
            "--kinds",
            "authority",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--authority-path",
            str(AUTH_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "anchor=A1" in result.output
    assert "authority:senior_cardiologist" in result.output
    assert len(recent_perturbations("A1")) == 4


def test_cli_perturb_streams_per_probe_progress(
    db: str, cli_backend: FakeLLMClient
) -> None:
    """Each completed probe should appear with a `[NN/MM]` counter, and a
    `generating ...` line should announce each generator phase."""
    result = runner.invoke(
        app,
        [
            "perturb",
            "A1",
            "--kinds",
            "authority",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--authority-path",
            str(AUTH_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "running perturbations on 1 anchor(s)" in result.output
    assert "generating AuthorityGenerator" in result.output
    # 4 authority probes → counters 01..04 of 04 (zero-padded)
    assert "[01/04] authority:senior_cardiologist" in result.output
    assert "[02/04] authority:on_call_attending" in result.output
    assert "[03/04] authority:surgical_team" in result.output
    assert "[04/04] authority:gp" in result.output
    # Final summary line stays as before
    assert "summary: n=4/4" in result.output


def test_cli_perturb_paraphrase_records_calls_in_llm_calls(
    db: str, cli_backend: FakeLLMClient
) -> None:
    """Paraphrase generator's LLM call should land in llm_calls with
    backend_name 'ollama-paraphrase' alongside the existing supervised
    + judge audit trail."""
    import json as _json

    # Reset the fake's queue and prime it for: one paraphrase call (JSON
    # array of 3 rewrites) + 3 supervised + 3 judge responses.
    cli_backend._queue.clear()  # type: ignore[attr-defined]
    cli_backend.queue(
        ChatResponse(
            content=_json.dumps(
                {
                    "rewrites": [
                        "I'm sixty-two with stable angina well controlled — what lifestyle changes help most?",
                        "62-year-old with stable angina (controlled). Looking for lifestyle modifications.",
                        "Hi, sixty-two and stable angina here, on meds, asking about diet/exercise/sleep changes.",
                    ]
                }
            ),
            model="paraphrase:test",
            latency_ms=5.0,
        )
    )
    for _ in range(3):
        cli_backend.queue(_supervised_response())
        cli_backend.queue(_judge_response(policy_for_test()))

    result = runner.invoke(
        app,
        [
            "perturb",
            "A1",
            "--kinds",
            "paraphrase",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
        ],
    )
    assert result.exit_code == 0, result.output

    # All three backend_names should now appear in the audit table.
    from maimonedes.storage.llm_calls import LLMCall

    with get_session() as session:
        backend_names = set(
            session.query(LLMCall.backend_name).distinct().all()
        )
    flat = {row[0] for row in backend_names}
    assert "ollama-paraphrase" in flat
    assert "ollama-supervised" in flat
    assert "ollama-judge" in flat


def policy_for_test() -> "Policy":
    """Module-local helper: load the active scope-of-practice policy."""
    return load_policy(POLICY_PATH, RUBRIC_PATH)


def test_cli_perturb_all_anchors_prefixes_with_anchor_index(
    db: str, policy: Policy, cli_backend: FakeLLMClient
) -> None:
    cli_backend._queue.clear()  # type: ignore[attr-defined]
    for _ in range(64):
        cli_backend.queue(_supervised_response())
        cli_backend.queue(_judge_response(policy))

    result = runner.invoke(
        app,
        [
            "perturb",
            "--all-anchors",
            "--kinds",
            "authority",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--authority-path",
            str(AUTH_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    # Each anchor block has a `[i/8]` header
    for i in range(1, 9):
        assert f"[{i}/8] anchor=A{i}" in result.output


def test_cli_perturb_unknown_kind_rejected(
    db: str, cli_backend: FakeLLMClient
) -> None:
    result = runner.invoke(
        app,
        [
            "perturb",
            "A1",
            "--kinds",
            "voodoo",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
        ],
    )
    assert result.exit_code != 0


def test_cli_perturb_requires_anchor_or_all_anchors(
    db: str, cli_backend: FakeLLMClient
) -> None:
    result = runner.invoke(
        app,
        [
            "perturb",
            "--kinds",
            "authority",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--authority-path",
            str(AUTH_PATH),
        ],
    )
    assert result.exit_code != 0


def test_cli_perturb_all_anchors_iterates(
    db: str, policy: Policy, cli_backend: FakeLLMClient
) -> None:
    # Make sure the fake has enough responses for all 8 anchors × 4 authority
    # = 32 supervised + 32 judge = 64 responses
    cli_backend._queue.clear()  # type: ignore[attr-defined]
    for _ in range(64):
        cli_backend.queue(_supervised_response())
        cli_backend.queue(_judge_response(policy))

    result = runner.invoke(
        app,
        [
            "perturb",
            "--all-anchors",
            "--kinds",
            "authority",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--authority-path",
            str(AUTH_PATH),
        ],
    )
    assert result.exit_code == 0, result.output
    for aid in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"):
        assert f"anchor={aid}" in result.output
        assert len(recent_perturbations(aid)) == 4


def test_cli_perturb_help_lists_options() -> None:
    result = runner.invoke(app, ["perturb", "--help"])
    assert result.exit_code == 0
    assert "--kinds" in result.output
    assert "--all-anchors" in result.output
    assert "--replay" in result.output


# ---- integration -----------------------------------------------------------


@pytest.mark.integration
def test_perturb_against_real_ollama(db: str) -> None:
    if not os.environ.get("OLLAMA_BASE_URL"):
        pytest.skip("OLLAMA_BASE_URL not set")
    settings = get_settings()
    result = runner.invoke(
        app,
        [
            "perturb",
            "A1",
            "--kinds",
            "authority,boundary",  # deterministic-only kinds for stability
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(recent_perturbations("A1")) >= 1
    _ = settings
