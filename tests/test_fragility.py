"""Tests for the empirical Jacobian + aggregated fragility table."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.monitor.fragility import (
    AGGREGATE_COLUMN,
    aggregated_fragility,
    all_jacobians,
    jacobian_for_anchor,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.perturbations import record_perturbation
from maimonedes.storage.repo import (
    init_engine,
    reset_engine_for_tests,
)

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
    db_path = tmp_path / "frag.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


SUB_IDS = ("flags_physician_review", "expresses_uncertainty")


def _seed_anchor_baseline(
    anchor_id: str,
    aggregate: float,
    sub_values: dict[str, float] | None = None,
) -> None:
    record_score(
        ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition=sub_values or {SUB_IDS[0]: 1.0, SUB_IDS[1]: 1.0},
            aggregate=aggregate,
            judge_model="judge:test",
            supervised_model="supervised:test",
            probe_role="anchor",
        )
    )


def _seed_perturbation(
    anchor_id: str,
    transform_label: str,
    kind: str,
    aggregate: float,
    sub_values: dict[str, float] | None = None,
    run_id: str | None = None,
    replicate_index: int | None = None,
) -> int:
    metadata: dict[str, object] = {}
    if run_id is not None:
        metadata["run_id"] = run_id
    if replicate_index is not None:
        metadata["replicate_index"] = replicate_index
    probe = PerturbationProbe(
        anchor_id=anchor_id,
        scenario=f"perturbed by {transform_label}",
        perturbation_kind=kind,  # type: ignore[arg-type]
        transform_label=transform_label,
        generator_metadata=metadata,
    )
    row_id = record_perturbation(probe)
    record_score(
        ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition=sub_values or {SUB_IDS[0]: 1.0, SUB_IDS[1]: 1.0},
            aggregate=aggregate,
            judge_model="judge:test",
            supervised_model="supervised:test",
            perturbation_id=row_id,
            probe_role="perturbation",
        )
    )
    return row_id


# ---- Jacobian per anchor ---------------------------------------------------


def test_jacobian_returns_none_when_baseline_is_missing(db: str) -> None:
    # Only a perturbation, no anchor baseline → Jacobian undefined.
    _seed_perturbation("A1", "authority:gp", "authority", 0.4)
    assert jacobian_for_anchor("A1") is None


def test_jacobian_columns_include_aggregate_and_every_sub_id(db: str) -> None:
    _seed_anchor_baseline("A1", 0.8)
    _seed_perturbation("A1", "authority:gp", "authority", 0.4)

    jac = jacobian_for_anchor("A1")
    assert jac is not None
    assert jac.columns[0] == AGGREGATE_COLUMN
    assert set(jac.columns[1:]) == set(SUB_IDS)
    assert jac.baseline_aggregate == pytest.approx(0.8)


def test_jacobian_delta_signs_and_values_match_hand_calculation(db: str) -> None:
    _seed_anchor_baseline(
        "A1",
        aggregate=0.8,
        sub_values={SUB_IDS[0]: 1.0, SUB_IDS[1]: 0.6},
    )
    _seed_perturbation(
        "A1",
        "authority:senior_cardiologist",
        "authority",
        aggregate=0.3,
        sub_values={SUB_IDS[0]: 0.0, SUB_IDS[1]: 0.6},
    )

    jac = jacobian_for_anchor("A1")
    assert jac is not None
    assert len(jac.rows) == 1
    row = jac.rows[0]
    assert row.deltas[AGGREGATE_COLUMN] == pytest.approx(-0.5)
    assert row.deltas[SUB_IDS[0]] == pytest.approx(-1.0)  # 0.0 - 1.0
    assert row.deltas[SUB_IDS[1]] == pytest.approx(0.0)


def test_jacobian_rows_sorted_by_total_abs_delta_descending(db: str) -> None:
    _seed_anchor_baseline("A1", aggregate=0.8)
    _seed_perturbation("A1", "demographic:mild", "demographic", aggregate=0.79)
    _seed_perturbation(
        "A1",
        "authority:senior_cardiologist",
        "authority",
        aggregate=0.20,
    )
    _seed_perturbation("A1", "boundary:should_to_will", "boundary", aggregate=0.50)

    jac = jacobian_for_anchor("A1")
    assert jac is not None
    labels = [r.transform_label for r in jac.rows]
    # authority:senior_cardiologist drops aggregate by 0.6 → biggest |Δ|
    # boundary:should_to_will drops 0.3 → mid
    # demographic:mild drops 0.01 → smallest |Δ|
    assert labels == [
        "authority:senior_cardiologist",
        "boundary:should_to_will",
        "demographic:mild",
    ]


def test_jacobian_aggregates_replicates_with_mean_and_std(db: str) -> None:
    """Replicates from one run are kept and aggregated as mean ± std."""
    _seed_anchor_baseline("A1", aggregate=0.8)
    # 3 replicates of the same (anchor, transform_label), all sharing
    # one run_id, with aggregate values 0.4, 0.5, 0.6 → mean Δ -0.30,
    # std 0.10 (sample std, ddof=1).
    for i, agg in enumerate([0.4, 0.5, 0.6]):
        _seed_perturbation(
            "A1",
            "authority:gp",
            "authority",
            aggregate=agg,
            run_id="run_abc",
            replicate_index=i,
        )

    jac = jacobian_for_anchor("A1")
    assert jac is not None
    assert len(jac.rows) == 1
    row = jac.rows[0]
    assert row.transform_label == "authority:gp"
    assert row.n_replicates == 3
    assert row.deltas[AGGREGATE_COLUMN] == pytest.approx(-0.30, abs=1e-9)
    assert row.std_deltas[AGGREGATE_COLUMN] == pytest.approx(0.10, abs=1e-9)


def test_jacobian_replicates_from_different_runs_disambiguated_by_run_id(
    db: str,
) -> None:
    """Re-running perturb with a new run creates new probes with a new
    run_id; the Jacobian uses ONLY the latest run for each label."""
    _seed_anchor_baseline("A1", aggregate=0.8)
    # Old run: 2 replicates with aggregate 0.1 (mean Δ -0.7 — should be ignored)
    for i in range(2):
        _seed_perturbation(
            "A1", "authority:gp", "authority",
            aggregate=0.1, run_id="old_run", replicate_index=i,
        )
    # New run: 3 replicates with aggregate 0.5 (mean Δ -0.3 — should be used)
    for i in range(3):
        _seed_perturbation(
            "A1", "authority:gp", "authority",
            aggregate=0.5, run_id="new_run", replicate_index=i,
        )

    jac = jacobian_for_anchor("A1")
    assert jac is not None
    assert len(jac.rows) == 1
    row = jac.rows[0]
    assert row.n_replicates == 3
    assert row.deltas[AGGREGATE_COLUMN] == pytest.approx(-0.30, abs=1e-9)
    assert row.std_deltas[AGGREGATE_COLUMN] == pytest.approx(0.0, abs=1e-9)


def test_jacobian_legacy_probes_without_run_id_still_dedupe_to_latest(
    db: str,
) -> None:
    """Probes without `run_id` (legacy schema) keep the prior dedup-to-
    latest behaviour. Three legacy probes → only the highest-id one
    contributes; n_replicates = 1."""
    _seed_anchor_baseline("A1", aggregate=0.8)
    _seed_perturbation("A1", "authority:gp", "authority", aggregate=0.4)
    _seed_perturbation("A1", "authority:gp", "authority", aggregate=0.5)
    _seed_perturbation("A1", "authority:gp", "authority", aggregate=0.6)

    jac = jacobian_for_anchor("A1")
    assert jac is not None
    assert len(jac.rows) == 1
    row = jac.rows[0]
    assert row.n_replicates == 1
    # Latest probe = 0.6 → Δ = -0.20
    assert row.deltas[AGGREGATE_COLUMN] == pytest.approx(-0.20, abs=1e-9)
    assert row.std_deltas[AGGREGATE_COLUMN] == pytest.approx(0.0, abs=1e-9)


def test_aggregated_fragility_has_std_column_with_replicates(db: str) -> None:
    """Across-anchor std is non-zero when multiple anchors contribute
    different mean Δs for the same (kind, column) cell."""
    _seed_anchor_baseline("A1", aggregate=0.8)
    _seed_anchor_baseline("A2", aggregate=0.8)
    for i in range(3):
        _seed_perturbation(
            "A1", "authority:gp", "authority",
            aggregate=0.4, run_id="r1", replicate_index=i,
        )
    for i in range(3):
        _seed_perturbation(
            "A2", "authority:gp", "authority",
            aggregate=0.6, run_id="r2", replicate_index=i,
        )

    table = aggregated_fragility()
    cell = table.cell("authority", AGGREGATE_COLUMN)
    assert cell is not None
    # Mean of two anchor means (-0.4 and -0.2) → -0.30
    assert cell.mean_delta == pytest.approx(-0.30, abs=1e-9)
    # Between-anchor std of [-0.4, -0.2] with ddof=1 → ≈ 0.1414
    assert cell.std_delta == pytest.approx(0.1414, abs=1e-3)
    assert cell.count == 2


def test_jacobian_dedupes_repeated_perturb_runs(db: str) -> None:
    """Re-running `maimonedes perturb` creates new probe rows with the
    same transform_label. The Jacobian must collapse those into a
    single row keyed by the transform_label, otherwise pandas Styler
    refuses to render the heatmap."""
    _seed_anchor_baseline("A1", aggregate=0.8)
    # Three independent probe rows, all with the same transform_label.
    _seed_perturbation("A1", "authority:gp", "authority", aggregate=0.4)
    _seed_perturbation("A1", "authority:gp", "authority", aggregate=0.5)
    _seed_perturbation("A1", "authority:gp", "authority", aggregate=0.6)

    jac = jacobian_for_anchor("A1")
    assert jac is not None
    labels = [r.transform_label for r in jac.rows]
    # One row, not three.
    assert labels == ["authority:gp"]
    # And it's the LATEST (highest-id) probe's score that wins.
    assert jac.rows[0].deltas[AGGREGATE_COLUMN] == pytest.approx(0.6 - 0.8)


def test_jacobian_uses_most_recent_perturbation_score(db: str) -> None:
    """If a perturbation is scored twice, Jacobian uses the most-recent."""
    import time

    _seed_anchor_baseline("A1", aggregate=0.8)
    probe_id = _seed_perturbation(
        "A1", "authority:gp", "authority", aggregate=0.4
    )
    time.sleep(0.01)
    # Re-score the SAME perturbation row with a different aggregate.
    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="scope_of_practice",
            per_sub_condition={SUB_IDS[0]: 1.0, SUB_IDS[1]: 1.0},
            aggregate=0.9,  # new score
            judge_model="judge:test",
            supervised_model="supervised:test",
            perturbation_id=probe_id,
            probe_role="perturbation",
        )
    )

    jac = jacobian_for_anchor("A1")
    assert jac is not None
    assert jac.rows[0].deltas[AGGREGATE_COLUMN] == pytest.approx(0.9 - 0.8)


# ---- aggregated fragility --------------------------------------------------


def test_aggregated_fragility_averages_across_anchors(db: str) -> None:
    # Anchor A1 baseline 1.0; authority drops to 0.0 → Δ = -1.0
    _seed_anchor_baseline("A1", aggregate=1.0)
    _seed_perturbation("A1", "authority:gp", "authority", aggregate=0.0)
    # Anchor A2 baseline 0.9; authority drops to 0.5 → Δ = -0.4
    _seed_anchor_baseline("A2", aggregate=0.9)
    _seed_perturbation("A2", "authority:gp", "authority", aggregate=0.5)

    table = aggregated_fragility()
    auth_agg = table.cell("authority", AGGREGATE_COLUMN)
    assert auth_agg is not None
    assert auth_agg.mean_delta == pytest.approx(((-1.0) + (-0.4)) / 2)
    assert auth_agg.count == 2


def test_aggregated_fragility_skips_anchors_without_baseline(
    db: str, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    # A1 has both baseline + perturbation; A2 has only a perturbation.
    _seed_anchor_baseline("A1", aggregate=1.0)
    _seed_perturbation("A1", "authority:gp", "authority", aggregate=0.0)
    _seed_perturbation("A2", "authority:gp", "authority", aggregate=0.0)

    caplog.set_level(logging.INFO)
    table = aggregated_fragility()
    cell = table.cell("authority", AGGREGATE_COLUMN)
    assert cell is not None
    # Only A1 contributes — A2 is skipped because it has no baseline.
    assert cell.count == 1


def test_aggregated_fragility_empty_when_no_data(db: str) -> None:
    table = aggregated_fragility()
    assert table.cells == []
    assert table.perturbation_kinds == []


def test_all_jacobians_returns_one_per_anchor_with_baseline(db: str) -> None:
    _seed_anchor_baseline("A1", aggregate=0.8)
    _seed_anchor_baseline("A2", aggregate=0.7)
    _seed_perturbation("A1", "authority:gp", "authority", aggregate=0.5)
    _seed_perturbation("A2", "authority:gp", "authority", aggregate=0.4)

    jacs = all_jacobians()
    assert set(jacs.keys()) == {"A1", "A2"}


# ---- CLI -------------------------------------------------------------------


def test_fragility_report_cli_writes_csv(db: str, tmp_path: Path) -> None:
    _seed_anchor_baseline("A1", aggregate=0.8)
    _seed_perturbation(
        "A1", "authority:senior_cardiologist", "authority", aggregate=0.3
    )
    _seed_perturbation(
        "A1", "boundary:should_to_will", "boundary", aggregate=0.6
    )
    out_dir = tmp_path / "reports"

    result = runner.invoke(
        app, ["fragility-report", "--output-dir", str(out_dir)]
    )
    assert result.exit_code == 0, result.output
    csvs = list(out_dir.glob("fragility_*.csv"))
    assert len(csvs) == 1

    rows = list(csv.reader(csvs[0].open(encoding="utf-8")))
    flat = [",".join(r) for r in rows]
    # aggregated section header + rows for both kinds
    assert any("# aggregated fragility" in line for line in flat)
    assert any(line.startswith("authority,") for line in flat)
    assert any(line.startswith("boundary,") for line in flat)
    # per-anchor section
    assert any("## A1" in line for line in flat)


def test_fragility_report_cli_exits_nonzero_when_no_data(
    db: str, tmp_path: Path
) -> None:
    out_dir = tmp_path / "reports"
    result = runner.invoke(
        app, ["fragility-report", "--output-dir", str(out_dir)]
    )
    assert result.exit_code != 0
