"""Tests for the Phase 5 `label-distribution` diagnostic CLI."""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy, load_policy
from maimonedes.monitor.label_distribution import (
    NEAR_UNIFORM_THRESHOLD,
    compute_distribution,
    format_report,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.repo import init_engine, reset_engine_for_tests

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
    db_path = tmp_path / "label_dist.sqlite"
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


def _record_uniform(
    policy: Policy,
    *,
    sub_id: str,
    value: float,
    n: int,
    judge_model: str = "judge:test",
) -> None:
    """Insert n compliance_scores rows; sub_id pinned to value, others zero."""
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    for i in range(n):
        per_sub = {sid: 0.0 for sid in sub_ids}
        per_sub[sub_id] = value
        record_score(
            ComplianceScore(
                anchor_id=f"A{i % 4 + 1}",
                policy_id=policy.id,
                per_sub_condition=per_sub,
                aggregate=value,
                judge_model=judge_model,
                supervised_model="llama:test",
            )
        )


def _record_distributed(
    policy: Policy,
    *,
    sub_id: str,
    values_with_counts: list[tuple[float, int]],
) -> None:
    """Insert (value × count) pairs for one axis; pad zeros for others."""
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    idx = 0
    for value, count in values_with_counts:
        for _ in range(count):
            per_sub = {sid: 0.0 for sid in sub_ids}
            per_sub[sub_id] = value
            record_score(
                ComplianceScore(
                    anchor_id=f"A{idx % 4 + 1}",
                    policy_id=policy.id,
                    per_sub_condition=per_sub,
                    aggregate=value,
                    judge_model="judge:test",
                    supervised_model="llama:test",
                )
            )
            idx += 1


def test_all_values_identical_flags_near_uniform(
    db: str, policy: Policy
) -> None:
    """If 100% of rows have the same value, dominant_share=1.0 and ⚠ fires."""
    target_axis = policy.rubric.sub_conditions[0].id
    _record_uniform(policy, sub_id=target_axis, value=1.0, n=10)

    report = compute_distribution(policy)
    axis = next(a for a in report.axes if a.sub_id == target_axis)
    assert axis.dominant_share == pytest.approx(1.0)
    assert axis.near_uniform is True
    assert axis.entropy_bits == pytest.approx(0.0)


def test_uniform_distribution_across_levels_no_flag(
    db: str, policy: Policy
) -> None:
    """4-level uniform → entropy_normalised ≈ 1.0, no flag."""
    target_axis = policy.rubric.sub_conditions[0].id
    _record_distributed(
        policy,
        sub_id=target_axis,
        values_with_counts=[(0.0, 25), (0.25, 25), (0.5, 25), (1.0, 25)],
    )

    report = compute_distribution(policy)
    axis = next(a for a in report.axes if a.sub_id == target_axis)
    assert axis.dominant_share == pytest.approx(0.25)
    assert axis.near_uniform is False
    # 4 buckets each at 25% → entropy = log2(4) = 2.0 bits
    assert axis.entropy_bits == pytest.approx(2.0, abs=1e-3)


def test_dominant_with_tail_flags_near_uniform(
    db: str, policy: Policy
) -> None:
    """90% at one value → ⚠ flag fires (above 0.85 threshold)."""
    target_axis = policy.rubric.sub_conditions[0].id
    _record_distributed(
        policy,
        sub_id=target_axis,
        values_with_counts=[(1.0, 90), (0.5, 7), (0.0, 3)],
    )

    report = compute_distribution(policy)
    axis = next(a for a in report.axes if a.sub_id == target_axis)
    assert axis.dominant_share == pytest.approx(0.90)
    assert axis.near_uniform is True
    assert axis.entropy_bits > 0.0


def test_filter_sql_scopes_query(db: str, policy: Policy) -> None:
    """The --filter parameter must scope the query."""
    target_axis = policy.rubric.sub_conditions[0].id
    # Two judge models; one gives uniform 1.0, the other uniform 0.0.
    _record_uniform(
        policy, sub_id=target_axis, value=1.0, n=5, judge_model="judge:v1"
    )
    _record_uniform(
        policy, sub_id=target_axis, value=0.0, n=5, judge_model="other:v2"
    )

    # Without filter — bimodal split → dominant_share = 0.5.
    full = compute_distribution(policy)
    full_axis = next(a for a in full.axes if a.sub_id == target_axis)
    assert full_axis.n_rows == 10
    assert full_axis.dominant_share == pytest.approx(0.5)

    # With filter to only judge:v1 — uniform 1.0 → dominant_share = 1.0.
    filtered = compute_distribution(
        policy, filter_sql="judge_model = 'judge:v1'"
    )
    filt_axis = next(a for a in filtered.axes if a.sub_id == target_axis)
    assert filt_axis.n_rows == 5
    assert filt_axis.dominant_share == pytest.approx(1.0)


def test_axes_sorted_by_dominant_share_descending(
    db: str, policy: Policy
) -> None:
    """Most-uniform axis lands at the top."""
    sub_a, sub_b = (s.id for s in policy.rubric.sub_conditions[:2])
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    # axis A: 100% at 1.0 (dominant=1.0); axis B: 50/50 split (dominant=0.5).
    for i in range(10):
        per_sub = {sid: 0.0 for sid in sub_ids}
        per_sub[sub_a] = 1.0
        per_sub[sub_b] = 0.0 if i < 5 else 1.0
        record_score(
            ComplianceScore(
                anchor_id="A1",
                policy_id=policy.id,
                per_sub_condition=per_sub,
                aggregate=0.5,
                judge_model="judge:test",
                supervised_model="llama:test",
            )
        )

    report = compute_distribution(policy)
    # Top axis must be `sub_a` (more dominant share).
    assert report.axes[0].sub_id == sub_a
    # Among the first two axes, sub_b must follow.
    sub_b_index = next(
        i for i, a in enumerate(report.axes) if a.sub_id == sub_b
    )
    sub_a_index = next(
        i for i, a in enumerate(report.axes) if a.sub_id == sub_a
    )
    assert sub_a_index < sub_b_index


def test_format_report_includes_warning_for_flagged_axes(
    db: str, policy: Policy
) -> None:
    target_axis = policy.rubric.sub_conditions[0].id
    _record_uniform(policy, sub_id=target_axis, value=1.0, n=10)

    report = compute_distribution(policy)
    text = "\n".join(format_report(report))
    assert "⚠ near-uniform" in text
    assert f"axis: {target_axis}" in text
    assert "entropy=0.000" in text


def test_empty_db_returns_zero_rows(db: str, policy: Policy) -> None:
    report = compute_distribution(policy)
    assert report.n_rows_total == 0
    text = "\n".join(format_report(report))
    assert "no compliance_scores rows" in text


def test_cli_label_distribution_runs_against_seeded_db(
    db: str, policy: Policy, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_axis = policy.rubric.sub_conditions[0].id
    _record_uniform(policy, sub_id=target_axis, value=1.0, n=20)
    monkeypatch.setenv("DATABASE_URL", db)

    result = runner.invoke(app, ["label-distribution"])
    assert result.exit_code == 0, result.output
    assert "policy=scope_of_practice" in result.output
    assert f"axis: {target_axis}" in result.output
    assert "⚠ near-uniform" in result.output


def test_threshold_value_is_85_percent() -> None:
    """Pin the heuristic threshold so accidental drift is caught."""
    assert NEAR_UNIFORM_THRESHOLD == 0.85
