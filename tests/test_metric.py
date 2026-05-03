"""Tests for the Phase 5 Riemannian metric learner (issue #50).

Covers:
- Synthetic Jacobian field reconstruction (analytic g(c)).
- §4.6 Example 1 (interior vs near-boundary risk amplification).
- §4.6 Example 3 (drift trajectory with growing Riemannian/Euclidean ratio).
- Positive-definiteness invariant for random points.
- Persistence round-trip (.npz save/load).
- CLI smoke (`fit-metric`, `metric-distance`).
- Localizer integration (Riemannian distance via `metric=` kwarg).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.monitor.metric import (
    MetricMLP,
    RiemannianMetric,
    _build_lower_triangular,
    _build_lower_triangular_grad,
    _frobenius_loss_and_grad,
    compute_ratio_surface,
    euclidean_distance,
    fit_metric,
    fit_metric_from_pairs,
    metric_at,
    riemannian_distance,
    worst_fragility_axis_pair,
)
from maimonedes.monitor.localizer import boundary_distance
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.perturbations import record_perturbation
from maimonedes.storage.metric_fits import (
    latest_metric_fit_for_policy,
    list_metric_fits,
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
    db_path = tmp_path / "metric.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


# ---------------------------------------------------------------------------
# Pure-numpy primitives
# ---------------------------------------------------------------------------


def test_lower_triangular_construction_is_positive_definite() -> None:
    rng = np.random.default_rng(0)
    k = 3
    for _ in range(50):
        params = rng.standard_normal(k * (k + 1) // 2)
        L = _build_lower_triangular(params, k)
        # Strictly lower-triangular layout (diagonal is exp() so > 0).
        for i in range(k):
            for j in range(i + 1, k):
                assert L[i, j] == 0.0
            assert L[i, i] > 0.0
        g = L @ L.T
        eigvals = np.linalg.eigvalsh(g)
        assert eigvals.min() > 0.0


def test_backprop_matches_finite_difference() -> None:
    """Gradient check: analytic ≈ finite-difference for hidden + output layers."""
    mlp = MetricMLP.init(k=3, hidden=8, depth=2, seed=42)
    c = np.array([0.5, 0.7, 0.3])
    g_target = np.array(
        [[2.0, 0.3, 0.0], [0.3, 1.0, 0.1], [0.0, 0.1, 0.5]]
    )

    params, acts = mlp.forward(c)
    L = _build_lower_triangular(params, 3)
    _, dL = _frobenius_loss_and_grad(L, g_target)
    grad_params = _build_lower_triangular_grad(dL, params, 3)
    grads = mlp.backward(acts, grad_params)

    eps = 1e-6

    def loss_fn() -> float:
        p, _ = mlp.forward(c)
        L_local = _build_lower_triangular(p, 3)
        loss, _ = _frobenius_loss_and_grad(L_local, g_target)
        return loss

    for layer_idx in range(len(mlp.weights)):
        W, b = mlp.weights[layer_idx]
        # Spot-check a few weight entries per layer.
        for (i, j) in [(0, 0), (W.shape[0] - 1, W.shape[1] - 1)]:
            original = W[i, j]
            W[i, j] = original + eps
            l_plus = loss_fn()
            W[i, j] = original - eps
            l_minus = loss_fn()
            W[i, j] = original
            fd = (l_plus - l_minus) / (2 * eps)
            analytic = grads[layer_idx][0][i, j]
            if abs(fd) + abs(analytic) > 1e-9:
                assert abs(fd - analytic) < max(1e-3 * abs(fd), 1e-5), (
                    f"layer {layer_idx} W[{i},{j}]: fd={fd} analytic={analytic}"
                )


# ---------------------------------------------------------------------------
# Acceptance criterion: synthetic Jacobian field reconstruction
# ---------------------------------------------------------------------------


def test_fit_reconstructs_diagonal_metric_field() -> None:
    """Anisotropic diagonal field: eigenvalue grows with `1 - c0`.

    The MLP should learn g(c) ≈ diag(1 + 5(1 - c0), 1) within a small
    tolerance at unseen points.
    """
    rng = np.random.default_rng(0)
    n_samples = 120
    k = 2
    c_array = rng.uniform(0.0, 1.0, size=(n_samples, k))
    g_array = np.zeros((n_samples, k, k))
    for i in range(n_samples):
        eig = 1.0 + 5.0 * (1.0 - c_array[i, 0])
        g_array[i] = np.diag([eig, 1.0])

    metric = fit_metric_from_pairs(
        c_array,
        g_array,
        k=k,
        hidden=24,
        depth=2,
        epochs=400,
        lr=0.02,
        seed=0,
        policy_id="synth",
    )

    for c_test, eig_true in [
        (np.array([0.1, 0.5]), 1.0 + 5.0 * 0.9),
        (np.array([0.5, 0.5]), 1.0 + 5.0 * 0.5),
        (np.array([0.9, 0.5]), 1.0 + 5.0 * 0.1),
    ]:
        g_pred = metric_at(metric, c_test)
        assert g_pred[0, 0] == pytest.approx(eig_true, abs=0.4)
        assert g_pred[1, 1] == pytest.approx(1.0, abs=0.3)


def test_metric_at_is_positive_definite_on_random_points() -> None:
    """1000 random `c` samples → all eigenvalues of `metric_at(c)` > 0."""
    rng = np.random.default_rng(3)
    c_array = rng.uniform(-1.0, 2.0, size=(40, 2))
    g_array = np.tile(np.eye(2)[None, :, :], (40, 1, 1))
    metric = fit_metric_from_pairs(
        c_array,
        g_array,
        k=2,
        hidden=12,
        depth=2,
        epochs=80,
        lr=0.02,
        seed=1,
        policy_id="pd",
    )
    test_points = rng.uniform(-1.5, 2.5, size=(1000, 2))
    for c in test_points:
        g = metric_at(metric, c)
        eigvals = np.linalg.eigvalsh(g)
        assert eigvals.min() > 0.0


# ---------------------------------------------------------------------------
# §4.6 Example 1: interior vs near-boundary
# ---------------------------------------------------------------------------


def _make_radial_metric(k: int = 2) -> RiemannianMetric:
    """Synthesise the §4.6 boundary geometry directly.

    Avoids re-fitting an MLP for every test by constructing a
    `RiemannianMetric` whose forward pass evaluates an analytic
    radial field. The resulting metric is identity in the safe
    interior (sum > 1.50) and grows quadratically as the system
    approaches the boundary line `c0 + c1 = 1.10`.
    """
    rng = np.random.default_rng(7)
    boundary_offset = 1.10
    n = 240
    c_arr = rng.uniform(0.3, 1.0, size=(n, k))
    g_arr = np.zeros((n, k, k))
    for i in range(n):
        d_b = c_arr[i].sum() - boundary_offset
        # Scale: 1 in the safe interior (d_b >= 0.4); rises sharply
        # as d_b shrinks toward the boundary.
        if d_b >= 0.4:
            scale = 1.0
        else:
            scale = max(1.0, (0.4 / max(d_b, 0.05)) ** 2)
        g_arr[i] = np.eye(k) * scale
    return fit_metric_from_pairs(
        c_arr,
        g_arr,
        k=k,
        hidden=24,
        depth=2,
        epochs=800,
        lr=0.02,
        seed=0,
        policy_id="example1",
    )


def test_example1_interior_riemannian_is_smaller_than_boundary() -> None:
    """System A (deep interior) → Riemannian < Euclidean.

    System B (near boundary) → Riemannian ≫ Euclidean.
    """
    metric = _make_radial_metric()
    # System A: starts at (0.91, 0.88), moves to (0.82, 0.79). Sum stays > 1.6.
    a0, a1 = np.array([0.91, 0.88]), np.array([0.82, 0.79])
    # System B: starts at (0.64, 0.61), moves to (0.55, 0.52). Sum drops near boundary.
    b0, b1 = np.array([0.64, 0.61]), np.array([0.55, 0.52])

    eucl_a = euclidean_distance(a0, a1)
    eucl_b = euclidean_distance(b0, b1)
    riem_a = riemannian_distance(metric, a0, a1)
    riem_b = riemannian_distance(metric, b0, b1)

    # Same Euclidean displacement (≈ 0.127), wildly different risk.
    assert eucl_a == pytest.approx(eucl_b, abs=1e-3)
    # System A is interior: Riemannian on the order of Euclidean.
    assert riem_a < 1.5 * eucl_a
    # System B is near-boundary: Riemannian dwarfs Euclidean.
    assert riem_b > 3.0 * eucl_b


# ---------------------------------------------------------------------------
# §4.6 Example 3: temporal drift trajectory amplification
# ---------------------------------------------------------------------------


def test_example3_riemannian_amplifies_in_dangerous_region() -> None:
    """t₀→t₁ Riemannian/Euclidean ≈ 1.x; t₀→t₂ ratio is materially larger."""
    metric = _make_radial_metric()
    t0 = np.array([0.89, 0.85])
    t1 = np.array([0.81, 0.79])
    t2 = np.array([0.71, 0.68])

    eucl_01 = euclidean_distance(t0, t1)
    eucl_02 = euclidean_distance(t0, t2)
    riem_01 = riemannian_distance(metric, t0, t1)
    riem_02 = riemannian_distance(metric, t0, t2)

    ratio_01 = riem_01 / eucl_01
    ratio_02 = riem_02 / eucl_02
    # t0→t1 stays in the gentle region; ratio close to 1.
    assert ratio_01 < 2.5
    # t0→t2 enters the steep region near c1+c2=1.10; ratio amplifies.
    assert ratio_02 > ratio_01 * 1.5
    # The §4.6 narrative says t0→t2 Riemannian is much larger absolutely.
    assert riem_02 > 2.0 * riem_01


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    c_arr = rng.uniform(0.0, 1.0, size=(20, 2))
    g_arr = np.tile(np.eye(2)[None, :, :], (20, 1, 1))
    metric = fit_metric_from_pairs(
        c_arr,
        g_arr,
        k=2,
        hidden=8,
        depth=2,
        epochs=20,
        seed=0,
        policy_id="rt",
        sub_condition_ids=("a", "b"),
    )
    out = tmp_path / "metric.npz"
    metric.save(out)
    assert out.exists()

    loaded = RiemannianMetric.load(out)
    assert loaded.policy_id == "rt"
    assert loaded.k == 2
    assert loaded.sub_condition_ids == ("a", "b")
    test_c = np.array([0.3, 0.5])
    g0 = metric_at(metric, test_c)
    g1 = metric_at(loaded, test_c)
    np.testing.assert_allclose(g0, g1, atol=1e-12)


# ---------------------------------------------------------------------------
# Localizer integration: opt-in metric kwarg changes ranking
# ---------------------------------------------------------------------------


def test_boundary_distance_with_metric_returns_riemannian() -> None:
    """When `metric` is supplied, `boundary_distance` reports the Riemannian length.

    Using a flat (≈ identity) metric yields a value close to the
    rubric-weighted Euclidean distance up to scale; using the radial
    metric near the boundary inflates the value.
    """
    flat_metric = fit_metric_from_pairs(
        np.array([[0.5, 0.5], [0.6, 0.7]]),
        np.tile(np.eye(2)[None, :, :], (2, 1, 1)),
        k=2,
        hidden=4,
        depth=1,
        epochs=10,
        seed=0,
        policy_id="flat",
        sub_condition_ids=("a", "b"),
    )
    radial = _make_radial_metric()

    score = ComplianceScore(
        anchor_id="A1",
        policy_id="x",
        per_sub_condition={"a": 0.30, "b": 0.30},
        aggregate=0.30,
        judge_model="judge:test",
        supervised_model="supervised:test",
        probe_role="anchor",
    )
    thresholds = {"a": 0.5, "b": 0.5}
    weights = {"a": 0.5, "b": 0.5}

    eucl = boundary_distance(score, thresholds=thresholds, weights=weights)
    flat_riem = boundary_distance(
        score, thresholds=thresholds, weights=weights, metric=flat_metric
    )
    # Flat metric ≈ identity → Riemannian length close to Euclidean
    # (allow generous tolerance because the MLP is tiny / barely fit).
    assert flat_riem == pytest.approx(eucl, abs=0.5)

    # Radial metric near boundary produces a longer geodesic.
    radial_riem = boundary_distance(
        score, thresholds=thresholds, weights=weights, metric=radial
    )
    assert radial_riem > eucl


# ---------------------------------------------------------------------------
# CLI smoke: fit-metric + metric-distance round-trip
# ---------------------------------------------------------------------------


SCOPE_SUB_IDS = (
    "flags_physician_review",
    "expresses_uncertainty",
    "avoids_unqualified_prescriptive_language",
    "recommendation_within_scope",
    "recommendation_appropriate_specificity",
)


def _seed_anchor_for_metric(anchor_id: str, base: float) -> None:
    """Seed one anchor + four perturbations against the 5-axis scope rubric.

    Pads the per-axis vector to all five rubric sub-conditions so
    `_collect_anchor_pairs` accepts the row.
    """
    def _score(values: list[float], probe_role: str, perturbation_id: int | None = None) -> ComplianceScore:
        return ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition={s: v for s, v in zip(SCOPE_SUB_IDS, values)},
            aggregate=float(sum(values) / len(values)),
            judge_model="judge:test",
            supervised_model="supervised:test",
            perturbation_id=perturbation_id,
            probe_role=probe_role,
        )

    record_score(_score([base] * 5, probe_role="anchor"))
    perturbation_specs = [
        ("authority:gp", "authority", [-0.15, 0.0, -0.05, 0.0, 0.0]),
        ("boundary:should", "boundary", [0.0, -0.20, 0.0, -0.05, 0.0]),
        ("demographic:elder", "demographic", [-0.05, -0.05, -0.05, 0.0, -0.10]),
        ("paraphrase:1", "paraphrase", [-0.10, -0.05, 0.0, -0.05, -0.05]),
    ]
    for label, kind, deltas in perturbation_specs:
        probe = PerturbationProbe(
            anchor_id=anchor_id,
            scenario=f"perturbed by {label}",
            perturbation_kind=kind,  # type: ignore[arg-type]
            transform_label=label,
        )
        row_id = record_perturbation(probe)
        values = [max(0.0, base + d) for d in deltas]
        record_score(_score(values, probe_role="perturbation", perturbation_id=row_id))


def test_cli_fit_and_distance_roundtrip(db: str, tmp_path: Path) -> None:
    """`maimonedes fit-metric` persists; `maimonedes metric-distance` reads."""
    # Two anchors with usable Jacobians.
    _seed_anchor_for_metric("A1", base=0.85)
    _seed_anchor_for_metric("A2", base=0.60)

    out_dir = tmp_path / "models"
    result = runner.invoke(
        app,
        [
            "fit-metric",
            "--epochs",
            "30",
            "--hidden",
            "8",
            "--output-dir",
            str(out_dir),
        ],
    )
    if result.exit_code != 0:
        print("OUTPUT:", result.output)
        if result.exception is not None:
            import traceback
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)
    assert result.exit_code == 0, result.output
    assert "metric_fit_id=" in result.output
    fits = list_metric_fits()
    assert len(fits) == 1
    npz_path = Path(str(fits[0]["path"]))
    assert npz_path.exists()

    # metric-distance: pick two 5-axis compliance points and check both
    # numbers print. The 5-axis layout matches the seeded anchors above.
    c0 = "0.85,0.80,0.85,0.80,0.85"
    c1 = "0.55,0.50,0.55,0.50,0.55"
    dist_result = runner.invoke(app, ["metric-distance", c0, c1])
    assert dist_result.exit_code == 0, dist_result.output
    assert "euclidean_distance" in dist_result.output
    assert "riemannian_distance" in dist_result.output
    assert "ratio (riem/eucl)" in dist_result.output

    # latest_metric_fit_for_policy resolves the same row.
    latest = latest_metric_fit_for_policy("scope_of_practice")
    assert latest is not None
    assert latest["id"] == fits[0]["id"]


def test_fit_metric_errors_when_no_anchors_exist(db: str, tmp_path: Path) -> None:
    out_dir = tmp_path / "models"
    result = runner.invoke(
        app,
        ["fit-metric", "--epochs", "5", "--output-dir", str(out_dir)],
    )
    # No anchors seeded → fit raises ValueError → CLI exits 2.
    assert result.exit_code == 2, result.output
    assert "no anchor Jacobians" in result.output


# ---------------------------------------------------------------------------
# 3D ratio-surface helpers (dashboard `06_metric.py` 3D view)
# ---------------------------------------------------------------------------


def test_compute_ratio_surface_fixed_reference_shape_and_finite() -> None:
    metric = _make_radial_metric()
    xs, ys, z = compute_ratio_surface(
        metric,
        axis_indices=(0, 1),
        pinned=(1.0, 1.0),
        mode="fixed_reference",
        reference=(1.0, 1.0),
        resolution=8,
    )
    assert len(xs) == 8 and len(ys) == 8
    assert len(z) == 8 and all(len(row) == 8 for row in z)
    z_arr = np.asarray(z, dtype=float)
    # All non-NaN cells must be positive (Riemannian distance / Euclidean).
    assert np.all((np.isnan(z_arr)) | (z_arr > 0))
    # The reference cell at (x=1, y=1) is NaN (Euclidean=0 there).
    assert np.isnan(z_arr[-1, -1])
    # At least some cells are not at the reference, so they're finite.
    assert np.isfinite(z_arr).any()


def test_compute_ratio_surface_local_stretch_is_positive() -> None:
    """Local stretch √λ_max(g(c)) is well-defined everywhere (no reference)."""
    metric = _make_radial_metric()
    xs, ys, z = compute_ratio_surface(
        metric,
        axis_indices=(0, 1),
        pinned=(0.5, 0.5),
        mode="local_stretch",
        resolution=6,
    )
    z_arr = np.asarray(z, dtype=float)
    assert z_arr.shape == (6, 6)
    assert np.all(np.isfinite(z_arr))
    assert np.all(z_arr > 0)


def test_compute_ratio_surface_fixed_reference_amplifies_near_boundary() -> None:
    """The §4.6 narrative: cells near the violation boundary should have a
    Riemannian/Euclidean ratio > 1 (the radial metric has scale > 1 there)."""
    metric = _make_radial_metric()
    _, _, z = compute_ratio_surface(
        metric,
        axis_indices=(0, 1),
        pinned=(1.0, 1.0),
        mode="fixed_reference",
        reference=(1.0, 1.0),
        resolution=10,
    )
    z_arr = np.asarray(z, dtype=float)
    # Cell at (~0.0, ~0.0) — corner, deepest into the violation region —
    # is far from the reference and crosses the high-scale region.
    corner = z_arr[0, 0]
    assert np.isfinite(corner)
    assert corner > 1.0, (
        f"corner ratio {corner:.3f} should be > 1 for the radial metric"
    )


def test_compute_ratio_surface_rejects_invalid_args() -> None:
    metric = _make_radial_metric()
    with pytest.raises(ValueError, match="axis_indices"):
        compute_ratio_surface(
            metric,
            axis_indices=(0, 0),
            pinned=(0.0, 0.0),
            resolution=4,
        )
    with pytest.raises(ValueError, match="resolution"):
        compute_ratio_surface(
            metric,
            axis_indices=(0, 1),
            pinned=(0.0, 0.0),
            resolution=1,
        )
    with pytest.raises(ValueError, match="unknown mode"):
        compute_ratio_surface(
            metric,
            axis_indices=(0, 1),
            pinned=(0.0, 0.0),
            mode="bogus",
            resolution=4,
        )
    with pytest.raises(ValueError, match="pinned has length"):
        compute_ratio_surface(
            metric,
            axis_indices=(0, 1),
            pinned=(0.0,),  # k=2 but only 1 pinned
            resolution=4,
        )


def test_worst_fragility_axis_pair_falls_back_when_no_data(db: str) -> None:
    """No perturbation data in the DB → fallback to the supplied default."""
    pair = worst_fragility_axis_pair(
        sub_condition_ids=("a", "b", "c"), fallback=(0, 2)
    )
    assert pair == (0, 2)


def test_worst_fragility_axis_pair_picks_most_negative_axes(db: str) -> None:
    """Seed perturbation data where two axes have strongly negative drops.
    The helper should rank those two as the worst-fragility pair."""
    sub_ids = ("a", "b", "c", "d")
    # Anchor baseline (probe_role=anchor) at score=0.9 across all axes.
    record_score(
        ComplianceScore(
            anchor_id="ANCH",
            policy_id="p",
            per_sub_condition={s: 0.9 for s in sub_ids},
            aggregate=0.9,
            judge_model="j",
            supervised_model="s",
        )
    )
    # Three perturbation_probes with scores that drop axes 'b' and 'd'
    # the most. axes 'a' and 'c' are roughly stable.
    drops = [
        {"a": 0.85, "b": 0.10, "c": 0.85, "d": 0.05},
        {"a": 0.85, "b": 0.05, "c": 0.85, "d": 0.10},
        {"a": 0.85, "b": 0.10, "c": 0.85, "d": 0.10},
    ]
    for i, per_sub in enumerate(drops):
        probe = PerturbationProbe(
            anchor_id="ANCH",
            scenario=f"perturb-{i}",
            perturbation_kind="authority",
            transform_label=f"authority:{i}",
            generator_metadata={},
        )
        pid = record_perturbation(probe)
        record_score(
            ComplianceScore(
                anchor_id="ANCH",
                policy_id="p",
                per_sub_condition=per_sub,
                aggregate=float(np.mean(list(per_sub.values()))),
                judge_model="j",
                supervised_model="s",
                perturbation_id=pid,
                probe_role="perturbation",
            )
        )
    pair = worst_fragility_axis_pair(sub_condition_ids=sub_ids, fallback=(0, 1))
    # 'b' and 'd' are at indices 1 and 3.
    assert set(pair) == {1, 3}
