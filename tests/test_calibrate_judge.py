"""Tests for the mini-calibration harness."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maimonedes import cli
from maimonedes.cli import app
from maimonedes.core.policy import Policy, load_policy
from maimonedes.core.probe import load_anchors
from maimonedes.experiments.calibrate_judge import (
    Reference,
    load_references,
    run_calibration,
)
from maimonedes.llm.client import ChatResponse
from maimonedes.settings import get_settings
from tests.fakes import FakeLLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"
PROBES_PATH = PROJECT_ROOT / "config" / "probes" / "anchors_v1.yaml"
REFERENCES_PATH = PROJECT_ROOT / "config" / "calibration" / "references_v1.yaml"

runner = CliRunner()


@pytest.fixture
def policy() -> Policy:
    return load_policy(POLICY_PATH, RUBRIC_PATH)


def _judge_payload(policy: Policy, target: float) -> str:
    """Return a JSON payload whose aggregate, after weighting, is `target`.

    Trick: set every sub-condition's normalised value to `target` directly.
      - boolean → `target >= 0.5` flips to true (1.0); else false (0.0).
      - 0-3 → round target * 3 to the nearest int 0..3, then judge_payload
        maps that back to val/3.0.

    For coarse targets in {0, 1/3, 2/3, 1} the math is exact; for
    intermediate values it's the closest representable score on the
    given scale.
    """
    scores: dict[str, bool | int] = {}
    for s in policy.rubric.sub_conditions:
        if s.scale == "boolean":
            scores[s.id] = target >= 0.5
        else:
            scores[s.id] = round(max(0.0, min(1.0, target)) * 3)
    return json.dumps({"scores": scores})


# ---- reference loader ------------------------------------------------------


def test_references_v1_loads_20_entries() -> None:
    refs = load_references(REFERENCES_PATH)
    assert len(refs) == 20
    aggregates = sorted(r.hand_aggregate for r in refs)
    # Coverage check: at least one in the bottom decile and at least one
    # near the top, so the corpus actually spans [0, 1].
    assert aggregates[0] < 0.2
    assert aggregates[-1] > 0.9


def test_references_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    bad = tmp_path / "refs.yaml"
    bad.write_text(
        "references:\n"
        "  - id: r1\n    anchor_id: A1\n    hand_aggregate: 0.5\n"
        "    supervised_output: x\n"
        "  - id: r1\n    anchor_id: A1\n    hand_aggregate: 0.6\n"
        "    supervised_output: y\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate reference ids"):
        load_references(bad)


def test_references_loader_rejects_empty_corpus(tmp_path: Path) -> None:
    bad = tmp_path / "refs.yaml"
    bad.write_text("references: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_references(bad)


# ---- run_calibration -------------------------------------------------------


def _references_for_test(targets: list[float]) -> list[Reference]:
    return [
        Reference(
            id=f"r{i}",
            anchor_id="A1",
            hand_aggregate=t,
            supervised_output=f"output {i}",
        )
        for i, t in enumerate(targets)
    ]


def test_run_calibration_perfect_judge_is_green(
    policy: Policy, tmp_path: Path
) -> None:
    targets = [0.0, 1 / 3, 2 / 3, 1.0, 1.0, 0.0, 1 / 3, 2 / 3]
    refs = _references_for_test(targets)
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content=_judge_payload(policy, t),
                model="judge:test",
                latency_ms=1.0,
            )
            for t in targets
        ]
    )
    anchors = load_anchors(PROBES_PATH)

    report = run_calibration(
        refs,
        policy=policy,
        anchors=anchors,
        judge_client=fake,
        judge_model="judge:test",
        supervised_model="llama:test",
        output_dir=tmp_path,
        timestamp="20260427T000000Z",
    )
    assert report.n == len(targets)
    assert report.status == "green"
    assert report.spearman > 0.9
    # `mae` is bounded by the discretisation: 0-3 sub-conditions can't
    # represent every hand value exactly. Allow a generous slack.
    assert report.mae <= 0.10
    assert report.report_path.exists()


def test_run_calibration_anti_correlated_judge_is_red(
    policy: Policy, tmp_path: Path
) -> None:
    hand = [0.0, 1 / 3, 2 / 3, 1.0, 1.0, 0.0, 1 / 3, 2 / 3]
    # Judge predicts the OPPOSITE of hand → strong negative Spearman → red.
    pred = [1.0 - h for h in hand]
    refs = _references_for_test(hand)
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content=_judge_payload(policy, p),
                model="judge:test",
                latency_ms=1.0,
            )
            for p in pred
        ]
    )
    anchors = load_anchors(PROBES_PATH)

    report = run_calibration(
        refs,
        policy=policy,
        anchors=anchors,
        judge_client=fake,
        judge_model="judge:test",
        supervised_model="llama:test",
        output_dir=tmp_path,
    )
    assert report.status == "red"
    assert report.spearman < 0.0


def test_run_calibration_writes_csv_with_one_row_per_reference(
    policy: Policy, tmp_path: Path
) -> None:
    targets = [0.0, 1.0, 0.0, 1.0, 0.0]
    refs = _references_for_test(targets)
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content=_judge_payload(policy, t),
                model="judge:test",
                latency_ms=1.0,
            )
            for t in targets
        ]
    )
    anchors = load_anchors(PROBES_PATH)
    report = run_calibration(
        refs,
        policy=policy,
        anchors=anchors,
        judge_client=fake,
        judge_model="judge:test",
        supervised_model="llama:test",
        output_dir=tmp_path,
        timestamp="t",
    )
    rows = list(csv.reader(report.report_path.open(encoding="utf-8")))
    # Find the data section start (after the summary block + blank row)
    header_idx = next(
        i for i, row in enumerate(rows) if row and row[0] == "reference_id"
    )
    data_rows = rows[header_idx + 1 :]
    assert len(data_rows) == len(targets)
    assert data_rows[0][:2] == ["r0", "A1"]


def test_run_calibration_handles_constant_predictions(
    policy: Policy, tmp_path: Path
) -> None:
    refs = _references_for_test([0.1, 0.5, 0.9])
    # Judge always returns the same payload → constant predictions.
    payload = _judge_payload(policy, 0.5)
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content=payload, model="judge:test", latency_ms=1.0)
            for _ in refs
        ]
    )
    anchors = load_anchors(PROBES_PATH)
    report = run_calibration(
        refs,
        policy=policy,
        anchors=anchors,
        judge_client=fake,
        judge_model="judge:test",
        supervised_model="llama:test",
        output_dir=tmp_path,
    )
    assert report.spearman == 0.0
    assert report.status == "red"


# ---- CLI -------------------------------------------------------------------


@pytest.fixture
def cli_backend(monkeypatch: pytest.MonkeyPatch, policy: Policy) -> FakeLLMClient:
    targets = [0.0, 1 / 3, 2 / 3, 1.0]
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content=_judge_payload(policy, t),
                model="judge:test",
                latency_ms=1.0,
            )
            for t in targets
        ]
    )
    monkeypatch.setenv("OLLAMA_JUDGE_MODEL", "judge:test")
    monkeypatch.setenv("OLLAMA_SUPERVISED_MODEL", "llama:test")
    cli.set_backend_factory(lambda settings: fake)
    yield fake
    cli.reset_backend_factory()


def test_cli_calibrate_runs_against_synthetic_corpus(
    tmp_path: Path, cli_backend: FakeLLMClient
) -> None:
    refs_yaml = tmp_path / "refs.yaml"
    refs_yaml.write_text(
        "references:\n"
        "  - id: r0\n    anchor_id: A1\n    hand_aggregate: 0.0\n"
        "    supervised_output: out0\n"
        "  - id: r1\n    anchor_id: A1\n    hand_aggregate: 0.333\n"
        "    supervised_output: out1\n"
        "  - id: r2\n    anchor_id: A1\n    hand_aggregate: 0.667\n"
        "    supervised_output: out2\n"
        "  - id: r3\n    anchor_id: A1\n    hand_aggregate: 1.0\n"
        "    supervised_output: out3\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--references",
            str(refs_yaml),
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--probes",
            str(PROBES_PATH),
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "spearman=" in result.output
    assert "status=green" in result.output
    csvs = list(out_dir.glob("calibration_*.csv"))
    assert len(csvs) == 1


def test_cli_calibrate_help_lists_options() -> None:
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    assert "--references" in result.output
    assert "--output-dir" in result.output


# ---- integration -----------------------------------------------------------


@pytest.mark.integration
def test_calibrate_against_real_ollama(tmp_path: Path) -> None:
    if not os.environ.get("OLLAMA_BASE_URL"):
        pytest.skip("OLLAMA_BASE_URL not set")
    settings = get_settings()
    out_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--references",
            str(REFERENCES_PATH),
            "--output-dir",
            str(out_dir),
        ],
    )
    # red-band judge exits 6, but a successful run is the intended path.
    assert result.exit_code in (0, 6), result.output
    assert any(out_dir.glob("calibration_*.csv"))
    _ = settings
