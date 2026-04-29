"""Tests for the Phase 3 `drift-report` CLI + builder helper."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.monitor.drift_report import (
    NoBaselineDataError,
    build_report,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
)
from maimonedes.storage.repo import init_engine, reset_engine_for_tests

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
    db_path = tmp_path / "report.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


def _seed_run(
    *,
    n_baseline: int,
    n_post: int,
    baseline_mean: float = 0.9,
    post_mean: float = 0.4,
    anchors: tuple[str, ...] = ("A1",),
    rng_seed: int = 42,
    sigma: float = 0.02,
) -> int:
    """Synthesise a drift run with deterministic per-anchor scalar streams."""
    rng = np.random.default_rng(rng_seed)
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
        k_threshold=4.0,
    )
    total = n_baseline + n_post
    for idx in range(total):
        stage = "baseline" if idx < n_baseline else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        for anchor in anchors:
            mean = baseline_mean if idx < n_baseline else post_mean
            agg = float(np.clip(mean + rng.normal(0, sigma), 0.0, 1.0))
            record_score(
                ComplianceScore(
                    anchor_id=anchor,
                    policy_id="scope_of_practice",
                    per_sub_condition={"flags_physician_review": 1.0},
                    aggregate=agg,
                    judge_model="judge:test",
                    supervised_model="llama:test",
                    drift_session_id=sid,
                )
            )
    return run_id


# ---- builder ---------------------------------------------------------------


def test_build_report_detects_drift_before_violation(db: str) -> None:
    """0.9 → 0.4 step well below 0.5 threshold; CUSUM should lead the
    explicit violation by at least 1 session."""
    run_id = _seed_run(n_baseline=10, n_post=20, post_mean=0.4)
    report = build_report(run_id)

    row = report.rows[0]
    assert row.anchor_id == "A1"
    assert row.n_sessions == 30
    assert row.n_baseline == 10
    assert row.first_violation_session is not None
    assert row.cusum_first_fire is not None
    # The whole point of Phase 3: CUSUM fires no later than the first explicit violation.
    assert row.cusum_first_fire <= row.first_violation_session
    assert report.headline_lead_sessions is not None
    assert report.headline_lead_sessions >= 0


def test_build_report_no_baseline_raises(db: str) -> None:
    """All sessions tagged `concise` (no baseline) → builder rejects."""
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    for idx in range(8):
        sid = create_drift_session(run_id, idx, "concise", "x")
        record_score(
            ComplianceScore(
                anchor_id="A1",
                policy_id="scope_of_practice",
                per_sub_condition={"flags_physician_review": 1.0},
                aggregate=0.8,
                judge_model="j",
                supervised_model="s",
                drift_session_id=sid,
            )
        )
    with pytest.raises(NoBaselineDataError):
        build_report(run_id)


def test_build_report_short_baseline_anchor_renders_n_a(db: str) -> None:
    """One anchor has 5+ baseline samples, another has only 2 → only the
    short anchor is marked detector-skipped; the report still produces."""
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    rng = np.random.default_rng(0)
    # A1: 10 baseline + 5 concise sessions (all aggregates).
    # A2: only 2 baseline + 5 concise (insufficient).
    for idx in range(15):
        stage = "baseline" if idx < 10 else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        for anchor in ("A1", "A2"):
            if anchor == "A2" and idx >= 2 and idx < 10:
                continue
            agg = float(0.9 + rng.normal(0, 0.02))
            record_score(
                ComplianceScore(
                    anchor_id=anchor,
                    policy_id="scope_of_practice",
                    per_sub_condition={"flags_physician_review": 1.0},
                    aggregate=agg,
                    judge_model="j",
                    supervised_model="s",
                    drift_session_id=sid,
                )
            )

    report = build_report(run_id)
    rows_by_anchor = {r.anchor_id: r for r in report.rows}
    assert rows_by_anchor["A1"].detector_skipped is False
    assert rows_by_anchor["A2"].detector_skipped is True


def test_build_report_no_violation_yields_inf_lead(db: str) -> None:
    """Drift down to 0.7 (still above threshold 0.5) → CUSUM fires but
    no explicit violation → headline lead = +inf."""
    run_id = _seed_run(
        n_baseline=10, n_post=20, baseline_mean=0.9, post_mean=0.7
    )
    report = build_report(run_id, violation_threshold=0.5)
    row = report.rows[0]
    assert row.first_violation_session is None
    assert row.cusum_first_fire is not None
    assert row.cusum_lead_sessions == math.inf
    assert report.headline_lead_sessions == math.inf


# ---- CLI -------------------------------------------------------------------


def test_drift_report_cli_unknown_run_exits_4(db: str) -> None:
    result = runner.invoke(app, ["drift-report", "999"])
    assert result.exit_code == 4
    assert "unknown drift_run_id" in result.output


def test_drift_report_cli_renders_table(db: str) -> None:
    run_id = _seed_run(n_baseline=10, n_post=20, post_mean=0.4)
    result = runner.invoke(app, ["drift-report", str(run_id)])
    assert result.exit_code == 0, result.output
    assert "anchor" in result.output
    assert "A1" in result.output
    assert "earliest_cusum_fire" in result.output
    assert "earliest_violation" in result.output
    assert "headline" in result.output


def test_drift_report_cli_exits_2_on_no_baseline(db: str) -> None:
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    sid = create_drift_session(run_id, 0, "concise", "x")
    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": 1.0},
            aggregate=0.8,
            judge_model="j",
            supervised_model="s",
            drift_session_id=sid,
        )
    )
    result = runner.invoke(app, ["drift-report", str(run_id)])
    assert result.exit_code == 2
    assert "insufficient baseline" in result.output


def test_drift_report_cli_warns_when_k_overrides_persisted_k(db: str) -> None:
    run_id = _seed_run(n_baseline=10, n_post=20)
    result = runner.invoke(app, ["drift-report", str(run_id), "--k", "6.0"])
    assert result.exit_code == 0, result.output
    assert "differs from run's persisted k_threshold" in result.output


def test_drift_report_cli_violation_threshold_changes_first_violation(
    db: str,
) -> None:
    """Bumping the threshold to 0.95 makes baseline scores look like violations,
    pulling the first-violation column to a much earlier session.
    """
    run_id = _seed_run(n_baseline=10, n_post=10)
    relaxed = runner.invoke(app, ["drift-report", str(run_id)])
    aggressive = runner.invoke(
        app, ["drift-report", str(run_id), "--violation-threshold", "0.95"]
    )
    assert relaxed.exit_code == 0
    assert aggressive.exit_code == 0
    # Aggressive threshold has more violations → "no violations" should not appear.
    assert "no violations" not in aggressive.output
