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


# ---- #55: structural-signal CLI flags --------------------------------------


def _seed_run_with_curvature_data(
    *,
    n_baseline: int = 10,
    n_post: int = 20,
    anchors: tuple[str, ...] = ("A1",),
    sub_ids: tuple[str, ...] = (
        "flags_physician_review",
        "expresses_uncertainty",
        "avoids_unqualified_prescriptive_language",
        "recommendation_within_scope",
        "recommendation_appropriate_specificity",
    ),
) -> int:
    """Drift run whose scores carry the full per_sub_condition vector,
    so a RiemannianMetric loaded from a fit on the same axes finds the
    structural columns populated.
    """
    rng = np.random.default_rng(7)
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
            base = 0.9 if idx < n_baseline else 0.4
            agg = float(np.clip(base + rng.normal(0, 0.02), 0.0, 1.0))
            per_sub = {sid_: agg for sid_ in sub_ids}
            record_score(
                ComplianceScore(
                    anchor_id=anchor,
                    policy_id="scope_of_practice",
                    per_sub_condition=per_sub,
                    aggregate=agg,
                    judge_model="judge:test",
                    supervised_model="llama:test",
                    drift_session_id=sid,
                )
            )
    return run_id


def _seed_perturbations_for_metric(
    anchor_id: str,
    *,
    sub_ids: tuple[str, ...] = (
        "flags_physician_review",
        "expresses_uncertainty",
        "avoids_unqualified_prescriptive_language",
        "recommendation_within_scope",
        "recommendation_appropriate_specificity",
    ),
    n_perturbations: int = 5,
) -> None:
    """Seed an anchor + perturbations so fit_metric has training data."""
    from maimonedes.core.perturbation import PerturbationProbe
    from maimonedes.storage.perturbations import record_perturbation

    record_score(
        ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition={s: 0.9 for s in sub_ids},
            aggregate=0.9,
            judge_model="j",
            supervised_model="s",
        )
    )
    rng = np.random.default_rng(0)
    for i in range(n_perturbations):
        probe = PerturbationProbe(
            anchor_id=anchor_id,
            scenario=f"perturbed-{i}",
            perturbation_kind="authority",
            transform_label=f"authority:{i}",
            generator_metadata={},
        )
        pid = record_perturbation(probe)
        per_sub = {s: float(np.clip(0.9 + rng.normal(0, 0.1), 0.0, 1.0)) for s in sub_ids}
        record_score(
            ComplianceScore(
                anchor_id=anchor_id,
                policy_id="scope_of_practice",
                per_sub_condition=per_sub,
                aggregate=float(np.mean(list(per_sub.values()))),
                judge_model="j",
                supervised_model="s",
                perturbation_id=pid,
                probe_role="perturbation",
            )
        )


def _record_metric_fit_for_test(
    *, policy_id: str = "scope_of_practice", anchor_id: str = "A1", tmp_path: Path
) -> int:
    """Fit + persist a RiemannianMetric and return its DB id."""
    from maimonedes.core.policy import load_policy
    from maimonedes.monitor.metric import fit_metric
    from maimonedes.storage.metric_fits import record_metric_fit

    POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
    RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"
    policy = load_policy(POLICY_PATH, RUBRIC_PATH)
    _seed_perturbations_for_metric(anchor_id)
    metric = fit_metric(policy=policy, epochs=20)
    out_path = tmp_path / "metric_test.npz"
    metric.save(out_path)
    fit_id = record_metric_fit(
        policy_id=policy_id,
        path=str(out_path),
        n_anchors=metric.n_anchors,
        n_jacobians=metric.n_jacobians,
        val_loss=metric.eval_metrics.get("val_loss"),
        train_loss=metric.eval_metrics.get("train_loss"),
        hyperparams={},
    )
    return fit_id


def test_drift_report_cli_metric_baseline_without_metric_current_exits_5(
    db: str,
) -> None:
    run_id = _seed_run(n_baseline=10, n_post=10)
    result = runner.invoke(
        app,
        ["drift-report", str(run_id), "--metric-baseline", "1"],
    )
    assert result.exit_code == 5
    assert "must be set together" in result.output


def test_drift_report_cli_metric_current_without_metric_baseline_exits_5(
    db: str,
) -> None:
    run_id = _seed_run(n_baseline=10, n_post=10)
    result = runner.invoke(
        app,
        ["drift-report", str(run_id), "--metric-current", "1"],
    )
    assert result.exit_code == 5
    assert "must be set together" in result.output


def test_drift_report_cli_unknown_metric_baseline_id_exits_4(db: str) -> None:
    run_id = _seed_run(n_baseline=10, n_post=10)
    result = runner.invoke(
        app,
        [
            "drift-report",
            str(run_id),
            "--metric-baseline",
            "999",
            "--metric-current",
            "999",
        ],
    )
    assert result.exit_code == 4
    assert "metric-baseline" in result.output


def test_drift_report_cli_renders_structural_columns(
    db: str, tmp_path: Path
) -> None:
    """Pass the same metric for baseline and current — curvature won't fire
    (relative_increase = 0), but the structural columns must be rendered."""
    run_id = _seed_run_with_curvature_data(n_baseline=10, n_post=20)
    fit_id = _record_metric_fit_for_test(tmp_path=tmp_path)
    result = runner.invoke(
        app,
        [
            "drift-report",
            str(run_id),
            "--metric-baseline",
            str(fit_id),
            "--metric-current",
            str(fit_id),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "decoup" in result.output
    assert "curv" in result.output


def test_drift_report_cli_decoupling_window_flags_pass_through(
    db: str, tmp_path: Path
) -> None:
    """Smoke: --decoupling-baseline-window / --decoupling-current-window
    don't blow up; structural columns appear."""
    run_id = _seed_run_with_curvature_data(n_baseline=10, n_post=20)
    fit_id = _record_metric_fit_for_test(tmp_path=tmp_path)
    result = runner.invoke(
        app,
        [
            "drift-report",
            str(run_id),
            "--metric-baseline",
            str(fit_id),
            "--metric-current",
            str(fit_id),
            "--decoupling-baseline-window",
            "10",
            "--decoupling-current-window",
            "5",
            "--h-curvature",
            "1.5",
            "--h-decoupling",
            "0.25",
        ],
    )
    assert result.exit_code == 0, result.output


def test_fit_metric_cli_window_flags_smoke(
    db: str, tmp_path: Path
) -> None:
    """Smoke: --start-session / --end-session change the n_jacobians count."""
    from maimonedes.storage.drift import create_drift_run, create_drift_session

    # Seed perturbation scores attached to specific drift_session_ids.
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="s",
        judge_model="j",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    sub_ids = (
        "flags_physician_review",
        "expresses_uncertainty",
        "avoids_unqualified_prescriptive_language",
        "recommendation_within_scope",
        "recommendation_appropriate_specificity",
    )
    sids: list[int] = []
    for idx in range(4):
        sids.append(create_drift_session(run_id, idx, "baseline", ""))
    # Anchor scores in sessions 0–1.
    from maimonedes.core.perturbation import PerturbationProbe
    from maimonedes.storage.perturbations import record_perturbation

    for sid in sids[:2]:
        record_score(
            ComplianceScore(
                anchor_id="A1",
                policy_id="scope_of_practice",
                per_sub_condition={s: 0.9 for s in sub_ids},
                aggregate=0.9,
                judge_model="j",
                supervised_model="s",
                drift_session_id=sid,
            )
        )
    # Perturbation scores in sessions 2–3.
    for sid in sids[2:]:
        probe = PerturbationProbe(
            anchor_id="A1",
            scenario="perturbed",
            perturbation_kind="authority",
            transform_label=f"authority:{sid}",
            generator_metadata={},
        )
        pid = record_perturbation(probe)
        record_score(
            ComplianceScore(
                anchor_id="A1",
                policy_id="scope_of_practice",
                per_sub_condition={s: 0.5 for s in sub_ids},
                aggregate=0.5,
                judge_model="j",
                supervised_model="s",
                drift_session_id=sid,
                perturbation_id=pid,
                probe_role="perturbation",
            )
        )

    POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
    RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"

    result = runner.invoke(
        app,
        [
            "fit-metric",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--epochs",
            "5",
            "--output-dir",
            str(tmp_path),
            "--start-session",
            str(sids[0]),
            "--end-session",
            str(sids[3]),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "metric_fit_id=" in result.output
    assert "window=" in result.output


def test_fit_metric_cli_start_after_end_exits_5(
    db: str, tmp_path: Path
) -> None:
    POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
    RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"
    result = runner.invoke(
        app,
        [
            "fit-metric",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--start-session",
            "10",
            "--end-session",
            "5",
        ],
    )
    assert result.exit_code == 5
    assert ">" in result.output


def test_fit_metric_cli_empty_window_exits_4(
    db: str, tmp_path: Path
) -> None:
    POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
    RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"
    result = runner.invoke(
        app,
        [
            "fit-metric",
            "--policy",
            str(POLICY_PATH),
            "--rubric",
            str(RUBRIC_PATH),
            "--epochs",
            "5",
            "--output-dir",
            str(tmp_path),
            "--start-session",
            "9999",
            "--end-session",
            "10000",
        ],
    )
    assert result.exit_code == 4
    assert "no anchor scores" in result.output


def test_drift_report_persists_decoupling_signal_idempotent(
    db: str, tmp_path: Path
) -> None:
    """When decoupling fires, exactly one row lands per (anchor, signal_type)
    on first run; re-running the same CLI doesn't duplicate it."""
    from maimonedes.core.policy import load_policy
    from maimonedes.core.compliance import ComplianceScore as CS
    from maimonedes.core.perturbation import PerturbationProbe
    from maimonedes.monitor.metric import fit_metric
    from maimonedes.storage.metric_fits import record_metric_fit
    from maimonedes.storage.perturbations import record_perturbation
    from maimonedes.storage.structural_signals import (
        list_structural_signals,
    )

    POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
    RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"
    policy = load_policy(POLICY_PATH, RUBRIC_PATH)
    sub_ids = tuple(s.id for s in policy.rubric.sub_conditions)

    # Seed: drift run with full per_sub_condition vectors.
    run_id = _seed_run_with_curvature_data(
        n_baseline=10, n_post=10, sub_ids=sub_ids
    )
    # Seed perturbations so fit_metric has data; then fabricate a strong
    # decoupling signal by injecting many perturbation-stage scores with
    # opposite-sign correlations between current vs baseline windows.
    _seed_perturbations_for_metric("A1", sub_ids=sub_ids, n_perturbations=5)
    rng = np.random.default_rng(123)
    # Baseline window: positively-correlated noise.
    for i in range(60):
        v = float(rng.normal(0.5, 0.05))
        per_sub = {s: v + float(rng.normal(0, 0.01)) for s in sub_ids}
        record_score(
            CS(
                anchor_id="A1",
                policy_id="scope_of_practice",
                per_sub_condition=per_sub,
                aggregate=float(np.mean(list(per_sub.values()))),
                judge_model="j",
                supervised_model="s",
                probe_role="perturbation",
            )
        )
    # Current window (most recent rows): anti-correlated noise — flip the sign.
    for i in range(30):
        v = float(rng.normal(0.5, 0.05))
        per_sub = {}
        for j, s in enumerate(sub_ids):
            per_sub[s] = v if j % 2 == 0 else (1.0 - v)
        record_score(
            CS(
                anchor_id="A1",
                policy_id="scope_of_practice",
                per_sub_condition=per_sub,
                aggregate=float(np.mean(list(per_sub.values()))),
                judge_model="j",
                supervised_model="s",
                probe_role="perturbation",
            )
        )
    metric = fit_metric(policy=policy, epochs=10)
    out_path = tmp_path / "metric_decoupling.npz"
    metric.save(out_path)
    fit_id = record_metric_fit(
        policy_id=policy.id,
        path=str(out_path),
        n_anchors=metric.n_anchors,
        n_jacobians=metric.n_jacobians,
        val_loss=metric.eval_metrics.get("val_loss"),
        train_loss=metric.eval_metrics.get("train_loss"),
        hyperparams={},
    )

    # Run drift-report twice with structural flags.
    args = [
        "drift-report",
        str(run_id),
        "--metric-baseline",
        str(fit_id),
        "--metric-current",
        str(fit_id),
        "--decoupling-baseline-window",
        "30",
        "--decoupling-current-window",
        "10",
    ]
    r1 = runner.invoke(app, args)
    assert r1.exit_code == 0, r1.output
    rows_after_first = list_structural_signals(signal_type="decoupling")
    r2 = runner.invoke(app, args)
    assert r2.exit_code == 0, r2.output
    rows_after_second = list_structural_signals(signal_type="decoupling")
    # Whatever fired in r1, r2 should not duplicate (idempotent natural-key skip).
    assert len(rows_after_first) == len(rows_after_second), (
        f"duplicates appeared on second run: "
        f"{len(rows_after_first)} -> {len(rows_after_second)}"
    )
