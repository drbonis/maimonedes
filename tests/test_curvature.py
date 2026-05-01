"""Tests for the curvature signal (issue #51)."""
from __future__ import annotations

import numpy as np
import pytest

from maimonedes.monitor.curvature import (
    CurvatureResult,
    compute_curvature,
    curvature_drift,
    curvature_per_anchor,
)
from maimonedes.monitor.metric import (
    MetricMLP,
    RiemannianMetric,
    fit_metric_from_pairs,
)


def _metric_with_constant_g(g_const: np.ndarray) -> RiemannianMetric:
    """Fit a tiny MLP to a constant `g_const` everywhere.

    With repeated training pairs across a small grid the MLP collapses
    onto the constant; close enough for curvature comparisons.
    `c_arr` width matches `g_const`'s dimensionality so the API
    contract on `k` is satisfied.
    """
    k = g_const.shape[0]
    rng = np.random.default_rng(0)
    c_arr = rng.uniform(0.0, 1.0, size=(16, k))
    g_arr = np.tile(g_const[None, :, :], (16, 1, 1))
    return fit_metric_from_pairs(
        c_arr,
        g_arr,
        k=k,
        hidden=8,
        depth=1,
        epochs=600,
        lr=0.05,
        seed=0,
        policy_id="curv",
    )


def test_compute_curvature_eigenvalue_ratio() -> None:
    """A diagonal `g = diag(λ_max, λ_min)` returns `λ_max / λ_min`."""
    g = np.diag([4.0, 1.0])
    metric = _metric_with_constant_g(g)
    kappa = compute_curvature(metric, np.array([0.5, 0.5]))
    # Allow some tolerance; the MLP fit isn't exact.
    assert kappa == pytest.approx(4.0, abs=0.5)


def test_compute_curvature_isotropic_metric_is_one() -> None:
    """`g = c · I` has condition number 1 regardless of scale."""
    metric = _metric_with_constant_g(np.eye(2) * 2.5)
    kappa = compute_curvature(metric, np.array([0.5, 0.5]))
    assert kappa == pytest.approx(1.0, abs=0.2)


def test_curvature_drift_fires_on_anisotropic_increase() -> None:
    """Isotropic baseline + anisotropic current → fires on relative increase."""
    metric_baseline = _metric_with_constant_g(np.eye(2) * 1.0)  # κ = 1
    metric_current = _metric_with_constant_g(np.diag([10.0, 1.0]))  # κ = 10

    res = curvature_drift(
        anchor_id="A1",
        metric_baseline=metric_baseline,
        metric_current=metric_current,
        c=np.array([0.5, 0.5]),
    )
    assert res.signal_fired is True
    assert res.relative_increase > 1.0


def test_curvature_drift_does_not_fire_when_metrics_match() -> None:
    """Two metrics fit to the same data should not register a drift."""
    g = np.diag([2.0, 1.0])
    metric_a = _metric_with_constant_g(g)
    # Re-fit with a different seed so weights differ but g is similar.
    c_arr = np.array([[0.5, 0.5]] * 8)
    g_arr = np.tile(g[None, :, :], (8, 1, 1))
    metric_b = fit_metric_from_pairs(
        c_arr, g_arr, k=2, hidden=8, depth=1, epochs=600, lr=0.05, seed=1, policy_id="b"
    )
    res = curvature_drift(
        anchor_id="A1",
        metric_baseline=metric_a,
        metric_current=metric_b,
        c=np.array([0.5, 0.5]),
    )
    # Both have κ ≈ 2.0; relative increase should be small.
    assert res.signal_fired is False
    assert abs(res.relative_increase) < 0.3


def test_curvature_drift_threshold_override() -> None:
    metric_baseline = _metric_with_constant_g(np.diag([2.0, 1.0]))  # κ ≈ 2
    metric_current = _metric_with_constant_g(np.diag([3.0, 1.0]))  # κ ≈ 3 → +50%

    # Default h=0.5 puts +50% right at the boundary; tightening to
    # 0.2 forces a fire, loosening to 1.0 suppresses it.
    fire_loose = curvature_drift(
        anchor_id="A1",
        metric_baseline=metric_baseline,
        metric_current=metric_current,
        c=np.array([0.5, 0.5]),
        h_curvature=0.2,
    )
    suppress = curvature_drift(
        anchor_id="A1",
        metric_baseline=metric_baseline,
        metric_current=metric_current,
        c=np.array([0.5, 0.5]),
        h_curvature=2.0,
    )
    assert fire_loose.signal_fired is True
    assert suppress.signal_fired is False


def test_curvature_drift_raises_on_dimensionality_mismatch() -> None:
    metric_2d = _metric_with_constant_g(np.eye(2))
    metric_3d = _metric_with_constant_g(np.eye(3))
    with pytest.raises(ValueError, match="dimensionality mismatch"):
        curvature_drift(
            anchor_id="A1",
            metric_baseline=metric_2d,
            metric_current=metric_3d,
            c=np.array([0.5, 0.5]),
        )


def test_curvature_per_anchor_routes_positions() -> None:
    metric_baseline = _metric_with_constant_g(np.eye(2))
    metric_current = _metric_with_constant_g(np.diag([5.0, 1.0]))
    positions = {
        "A1": np.array([0.5, 0.5]),
        "A2": np.array([0.7, 0.3]),
    }
    results = curvature_per_anchor(
        positions,
        metric_baseline=metric_baseline,
        metric_current=metric_current,
    )
    assert set(results) == {"A1", "A2"}
    for r in results.values():
        assert isinstance(r, CurvatureResult)
        assert r.signal_fired is True
