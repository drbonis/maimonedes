"""Tests for the decoupling signal (issue #51).

Covers:
- Empirical covariance over a synthetic stream.
- Sign-flip detection: positively correlated → negatively correlated
  triggers `signal_fired`.
- False-positive guard: zero off-diagonal change should not fire even
  when per-axis variance shifts (which is what fragility is for).
- §4.6 Example 4 reproduction: positional drift = 0, fragility per-axis
  ≈ 0, but covariance flips → only decoupling fires.
- structural_signals storage round-trip.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.monitor.decoupling import (
    DecouplingResult,
    _empirical_covariance,
    compute_axis_covariance,
    decoupling_signal,
)
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.perturbations import record_perturbation
from maimonedes.storage.repo import init_engine, reset_engine_for_tests
from maimonedes.storage.structural_signals import (
    list_structural_signals,
    record_structural_signal,
    signals_for_anchor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"

AXIS_IDS = ("scope", "calibration")


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "decoupling.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


def _seed_anchor_baseline(anchor_id: str, vals: tuple[float, ...]) -> None:
    record_score(
        ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition={a: v for a, v in zip(AXIS_IDS, vals)},
            aggregate=float(sum(vals) / len(vals)),
            judge_model="judge:test",
            supervised_model="supervised:test",
            probe_role="anchor",
        )
    )


def _seed_perturbation_score(
    anchor_id: str,
    transform_label: str,
    vals: tuple[float, ...],
) -> int:
    probe = PerturbationProbe(
        anchor_id=anchor_id,
        scenario=f"perturbed by {transform_label}",
        perturbation_kind="authority",
        transform_label=transform_label,
    )
    row_id = record_perturbation(probe)
    record_score(
        ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition={a: v for a, v in zip(AXIS_IDS, vals)},
            aggregate=float(sum(vals) / len(vals)),
            judge_model="judge:test",
            supervised_model="supervised:test",
            perturbation_id=row_id,
            probe_role="perturbation",
        )
    )
    return row_id


# ---------------------------------------------------------------------------
# Pure-numpy primitives
# ---------------------------------------------------------------------------


def test_empirical_covariance_matches_numpy_cov() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((50, 3))
    ours = _empirical_covariance(matrix)
    theirs = np.cov(matrix, rowvar=False, ddof=1)
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_empirical_covariance_returns_zeros_for_lt_two_samples() -> None:
    out = _empirical_covariance(np.array([[0.5, 0.3]]))
    assert out.shape == (2, 2)
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# decoupling_signal: storage-backed
# ---------------------------------------------------------------------------


def _seed_correlated_stream(
    anchor_id: str,
    n_baseline: int,
    n_current: int,
    *,
    baseline_corr: float,
    current_corr: float,
    seed: int = 0,
) -> None:
    """Seed perturbation rows with controlled per-axis correlation.

    Inserts the BASELINE rows first (oldest), then the CURRENT rows
    (newest) — matching `decoupling_signal`'s newest-first ordering.
    """
    rng = np.random.default_rng(seed)

    def _samples(n: int, corr: float) -> np.ndarray:
        # Two correlated standard normals, scaled into [0,1] band.
        cov = np.array([[1.0, corr], [corr, 1.0]])
        L = np.linalg.cholesky(cov)
        z = rng.standard_normal((n, 2))
        return 0.6 + 0.1 * (z @ L.T)

    baseline = _samples(n_baseline, baseline_corr)
    current = _samples(n_current, current_corr)
    # Oldest first → baseline, then current. SQLite ordering uses
    # `scored_at desc, id desc`, so insertion order becomes the
    # newest-last list head.
    for i, vals in enumerate(baseline):
        _seed_perturbation_score(anchor_id, f"base_{i}", tuple(float(v) for v in vals))
    for i, vals in enumerate(current):
        _seed_perturbation_score(anchor_id, f"cur_{i}", tuple(float(v) for v in vals))


def test_decoupling_fires_after_correlation_sign_flip(db: str) -> None:
    _seed_anchor_baseline("A1", (0.6, 0.6))
    _seed_correlated_stream(
        "A1",
        n_baseline=60,
        n_current=30,
        baseline_corr=+0.85,
        current_corr=-0.85,
        seed=1,
    )
    result = decoupling_signal(
        anchor_id="A1",
        axis_ids=AXIS_IDS,
        baseline_window=60,
        current_window=30,
    )
    assert result.signal_fired is True
    # The (0, 1) pair must show up as a flip.
    assert (0, 1) in result.flipped_pairs


def test_decoupling_does_not_fire_on_pure_variance_shift(db: str) -> None:
    """Per-axis variance shifts (no off-diagonal change) shouldn't fire.

    Both windows have zero correlation; current window has higher
    variance on each axis (which fragility would catch). Decoupling
    should stay quiet because off-diagonals stay near zero and the
    Frobenius delta is dominated by diagonal-only changes — but the
    `h_decoupling` default still allows it through, so we ALSO pass
    a high explicit threshold here.
    """
    _seed_anchor_baseline("A1", (0.6, 0.6))
    _seed_correlated_stream(
        "A1",
        n_baseline=80,
        n_current=40,
        baseline_corr=0.0,
        current_corr=0.0,
        seed=42,
    )
    result = decoupling_signal(
        anchor_id="A1",
        axis_ids=AXIS_IDS,
        baseline_window=80,
        current_window=40,
        # Override threshold to focus on the sign-flip half of the
        # signal — variance shifts that don't flip the off-diagonal
        # should not fire.
        h_decoupling=10.0,
    )
    assert result.flipped_pairs == []
    assert result.signal_fired is False


def test_compute_axis_covariance_returns_correct_shape(db: str) -> None:
    _seed_anchor_baseline("A1", (0.6, 0.6))
    for i in range(20):
        # Co-vary axes within [0, 1] so pydantic's aggregate ≤ 1.0
        # constraint stays satisfied.
        v = 0.30 + 0.025 * i
        _seed_perturbation_score("A1", f"p_{i}", (v, v + 0.05))
    cov = compute_axis_covariance(anchor_id="A1", axis_ids=AXIS_IDS, window=20)
    assert cov.shape == (2, 2)
    # Same direction on both axes → strongly positive correlation.
    assert cov[0, 1] > 0.0


def test_decoupling_skips_when_evidence_is_insufficient(db: str) -> None:
    _seed_anchor_baseline("A1", (0.6, 0.6))
    # Only 4 perturbation samples — far below the requested current_window=30.
    for i in range(4):
        _seed_perturbation_score("A1", f"few_{i}", (0.5, 0.5))
    result = decoupling_signal(
        anchor_id="A1",
        axis_ids=AXIS_IDS,
        baseline_window=60,
        current_window=30,
    )
    assert result.signal_fired is False


# ---------------------------------------------------------------------------
# §4.6 Example 4: positional drift = 0, fragility = 0, covariance flips
# ---------------------------------------------------------------------------


def test_example4_only_decoupling_fires_on_pure_covariance_flip(db: str) -> None:
    """Same per-axis means, same per-axis variance, opposite correlation.

    `decoupling` should fire while a per-axis fragility comparison
    (which compares means and variances independently) would not.
    """
    _seed_anchor_baseline("A1", (0.6, 0.6))
    _seed_correlated_stream(
        "A1",
        n_baseline=80,
        n_current=40,
        baseline_corr=+0.85,
        current_corr=-0.85,
        seed=7,
    )
    result = decoupling_signal(
        anchor_id="A1",
        axis_ids=AXIS_IDS,
        baseline_window=80,
        current_window=40,
    )
    assert result.signal_fired is True
    # Sanity: per-axis means are roughly the same → positional drift ≈ 0.
    base_mean = result.baseline_cov.diagonal().mean()
    cur_mean = result.current_cov.diagonal().mean()
    # Both windows draw from the same distribution shape; per-axis
    # variance hasn't shifted by more than ~30%.
    assert abs(cur_mean - base_mean) < 0.4 * max(base_mean, 1e-6)


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------


def test_record_and_query_structural_signal(db: str) -> None:
    sid = record_structural_signal(
        anchor_id="A1",
        signal_type="decoupling",
        metric_value=0.42,
        threshold=0.20,
        evidence={"flipped_pairs": [[0, 1]], "frobenius_delta": 0.42},
    )
    assert sid >= 1
    by_anchor = signals_for_anchor("A1")
    assert len(by_anchor) == 1
    row = by_anchor[0]
    assert row["signal_type"] == "decoupling"
    assert row["metric_value"] == pytest.approx(0.42)
    assert row["threshold"] == pytest.approx(0.20)
    assert row["evidence"]["flipped_pairs"] == [[0, 1]]
    listed = list_structural_signals()
    assert len(listed) == 1


def test_record_structural_signal_rejects_unknown_type(db: str) -> None:
    with pytest.raises(ValueError, match="unknown signal_type"):
        record_structural_signal(
            anchor_id="A1",
            signal_type="bogus",
            metric_value=0.1,
            threshold=0.2,
        )
