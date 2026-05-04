"""Phase 5 page: Riemannian metric explorer + distance comparator.

Visualizes the §4.2-§4.6 geometry argument as something the operator
can poke at instead of a paragraph in the architecture doc:

- 50×50 heatmap of `√det(g(c))` over a 2D axis-pair slice (the
  volumetric stretch factor — bright = the space is stretched here, a
  small Δc registers as a large Riemannian distance).
- 8×8 sub-grid of metric ellipses showing the local unit ball of g
  (small ellipse = unit Riemannian step is a small coordinate step =
  stretched space; eccentric = anisotropic).
- Anchor overlay coloured by Riemannian boundary distance.
- Distance comparator with §4.6 Example 1 / 3 / 5 reproductions.

The page is read-only; the metric is fitted by `maimonedes fit-metric`
(see `monitor/metric.py`).
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy, load_policy
from maimonedes.dashboard._metric_snapshots import (
    AnchorOverlay,
    BoundaryGradientArrow,
    DistanceComparison,
    GridSnapshot,
    MetricEllipse,
    MetricFitSnapshot,
    PerturbationArrow,
    PerturbationEfficiencyEntry,
    PresetSegment,
    RadialCloudPoint,
    RadialCloudSnapshot,
    RatioSurfaceSnapshot,
)
from maimonedes.monitor.fragility import all_jacobians
from maimonedes.monitor.localizer import (
    DEFAULT_AXIS_THRESHOLD,
    axis_thresholds,
    axis_weights,
    boundary_distance,
)
from maimonedes.monitor.metric import (
    RiemannianMetric,
    boundary_gradient_contravariant,
    compute_radial_projection_cloud,
    compute_ratio_surface,
    euclidean_distance,
    metric_at,
    perturbation_efficiency,
    riemannian_distance,
    worst_fragility_axis_pair,
)
from maimonedes.storage.metric_fits import list_metric_fits


PAGE_TITLE = "Metric — Phase 5"
EMPTY_STATE_MSG = (
    "No Riemannian metric fits recorded yet. Run `maimonedes fit-metric` "
    "to populate this dashboard."
)
HEATMAP_RES = 50
ELLIPSE_RES = 8
ELLIPSE_POINTS = 24
ELLIPSE_TARGET_SEMI = 0.045  # median ellipse semi-axis target (cell spacing ≈ 1/7)
ELLIPSE_MAX_SEMI = 0.06       # don't overflow the cell
DEFAULT_N_SEGMENTS = 16

POLICY_PATH = (
    Path(__file__).resolve().parents[4]
    / "config"
    / "policies"
    / "scope_of_practice.yaml"
)
RUBRIC_PATH = (
    Path(__file__).resolve().parents[4]
    / "config"
    / "rubrics"
    / "scope_of_practice.yaml"
)

PRESET_CUSTOM = "Custom"
PRESET_EX1 = "Example 1 — interior vs near-boundary"
PRESET_EX3 = "Example 3 — temporal trajectory"
PRESET_EX5 = "Example 5 — geodesic vs straight-line"
PRESET_LABELS = (PRESET_CUSTOM, PRESET_EX1, PRESET_EX3, PRESET_EX5)


# ---------------------------------------------------------------------------
# Cached data layer
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10)
def _cached_fits() -> list[MetricFitSnapshot]:
    snapshots: list[MetricFitSnapshot] = []
    for row in list_metric_fits():
        trained = row.get("trained_at")
        ts_str = (
            trained.strftime("%Y-%m-%d %H:%M")
            if isinstance(trained, datetime)
            else "?"
        )
        val_loss = row.get("val_loss")
        train_loss = row.get("train_loss")
        val_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "—"
        snapshots.append(
            MetricFitSnapshot(
                fit_id=int(row["id"]),  # type: ignore[arg-type]
                path=str(row["path"]),
                policy_id=str(row["policy_id"]),
                n_anchors=int(row["n_anchors"]),  # type: ignore[arg-type]
                n_jacobians=int(row["n_jacobians"]),  # type: ignore[arg-type]
                val_loss=float(val_loss) if isinstance(val_loss, (int, float)) else None,
                train_loss=float(train_loss) if isinstance(train_loss, (int, float)) else None,
                trained_at=trained if isinstance(trained, datetime) else None,
                label=(
                    f"fit #{row['id']}  ·  {row['policy_id']}  ·  {ts_str}  ·  "
                    f"n_anchors={row['n_anchors']}  ·  val_loss={val_str}"
                ),
            )
        )
    return snapshots


def _load_metric(path: str) -> RiemannianMetric | None:
    """Open a `metric_fit` artefact; return None when the file is gone."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return RiemannianMetric.load(p)
    except (OSError, ValueError):
        return None


def _safe_load_policy() -> Policy | None:
    try:
        return load_policy(POLICY_PATH, RUBRIC_PATH)
    except FileNotFoundError:
        return None


def _ellipse_points_for(
    g_block: np.ndarray, cx: float, cy: float, scale: float
) -> tuple[list[float], list[float], float]:
    """Return ellipse polygon points + the eigenvalue ratio for tooltips.

    `g_block` is the 2x2 sub-block of g(c) for the chosen axis pair.
    Semi-axes use `1/√eigenvalue` so a metric with large eigenvalues
    (stretched space) renders as a *small* ellipse — the unit ball of
    g shrinks as g grows.
    """
    eigvals, eigvecs = np.linalg.eigh(g_block)
    eigvals = np.maximum(eigvals, 1e-12)
    semi_unscaled = 1.0 / np.sqrt(eigvals)
    semi = np.minimum(semi_unscaled * scale, ELLIPSE_MAX_SEMI)
    ratio = float(eigvals.max() / max(eigvals.min(), 1e-12))
    theta = np.linspace(0.0, 2.0 * math.pi, ELLIPSE_POINTS)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, N)
    ellipse = eigvecs @ (np.diag(semi) @ circle)  # (2, N)
    xs = (ellipse[0] + cx).tolist()
    ys = (ellipse[1] + cy).tolist()
    return xs, ys, ratio


def _build_grid_arrays(
    metric: RiemannianMetric,
    axis_i: int,
    axis_j: int,
    pinned: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (xs, ys, sqrt_det_g) for the chosen slice."""
    xs = np.linspace(0.0, 1.0, HEATMAP_RES)
    ys = np.linspace(0.0, 1.0, HEATMAP_RES)
    grid = np.zeros((HEATMAP_RES, HEATMAP_RES), dtype=np.float64)
    c = np.asarray(pinned, dtype=np.float64).copy()
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            c[axis_i] = float(x)
            c[axis_j] = float(y)
            g = metric_at(metric, c)
            det_g = float(np.linalg.det(g))
            grid[iy, ix] = math.sqrt(max(det_g, 0.0))
    return xs, ys, grid


def _build_ellipses(
    metric: RiemannianMetric,
    axis_i: int,
    axis_j: int,
    pinned: tuple[float, ...],
) -> list[MetricEllipse]:
    """Compute the 8x8 grid of ellipses with a global size scale."""
    coords = np.linspace(0.05, 0.95, ELLIPSE_RES)
    g_blocks: list[tuple[float, float, np.ndarray]] = []
    c = np.asarray(pinned, dtype=np.float64).copy()
    for cy in coords:
        for cx in coords:
            c[axis_i] = float(cx)
            c[axis_j] = float(cy)
            g = metric_at(metric, c)
            block = np.array(
                [
                    [g[axis_i, axis_i], g[axis_i, axis_j]],
                    [g[axis_j, axis_i], g[axis_j, axis_j]],
                ],
                dtype=np.float64,
            )
            g_blocks.append((float(cx), float(cy), block))

    if not g_blocks:
        return []

    # Global scale so the median ellipse hits ELLIPSE_TARGET_SEMI.
    medians: list[float] = []
    for _, _, block in g_blocks:
        eigvals = np.linalg.eigvalsh(block)
        eigvals = np.maximum(eigvals, 1e-12)
        medians.append(float(np.median(1.0 / np.sqrt(eigvals))))
    median_semi = float(np.median(medians))
    scale = ELLIPSE_TARGET_SEMI / max(median_semi, 1e-9)

    ellipses: list[MetricEllipse] = []
    for cx, cy, block in g_blocks:
        xs, ys, ratio = _ellipse_points_for(block, cx, cy, scale)
        ellipses.append(
            MetricEllipse(cx=cx, cy=cy, xs=xs, ys=ys, eig_ratio=ratio)
        )
    return ellipses


def _build_anchor_overlay(
    metric: RiemannianMetric,
    policy: Policy,
    axis_i: int,
    axis_j: int,
) -> list[AnchorOverlay]:
    """Project every anchor with a Jacobian onto the chosen axis pair."""
    axes = list(metric.sub_condition_ids) or [
        s.id for s in policy.rubric.sub_conditions
    ]
    if not axes or axis_i >= len(axes) or axis_j >= len(axes):
        return []
    thresholds = axis_thresholds(policy)
    weights = axis_weights(policy)
    out: list[AnchorOverlay] = []
    for anchor_id, jac in all_jacobians().items():
        c_full = tuple(
            float(jac.baseline_per_sub_condition.get(a, 0.0)) for a in axes
        )
        score = ComplianceScore(
            anchor_id=anchor_id,
            policy_id=policy.id,
            per_sub_condition={
                a: jac.baseline_per_sub_condition.get(a, 0.0) for a in axes
            },
            aggregate=float(jac.baseline_aggregate),
            judge_model="dashboard:viewer",
            supervised_model="dashboard:viewer",
        )
        try:
            d_eucl = boundary_distance(
                score, thresholds=thresholds, weights=weights, metric=None
            )
            d_riem = boundary_distance(
                score, thresholds=thresholds, weights=weights, metric=metric
            )
        except (ValueError, RuntimeError):
            continue
        ratio = (d_riem / d_eucl) if d_eucl > 1e-9 else None
        out.append(
            AnchorOverlay(
                anchor_id=anchor_id,
                cx=c_full[axis_i],
                cy=c_full[axis_j],
                c_full=c_full,
                euclidean_boundary=float(d_eucl),
                riemannian_boundary=float(d_riem),
                ratio=ratio,
            )
        )
    return out


@st.cache_data(ttl=10)
def _cached_grid(
    fit_id: int,
    fit_path: str,
    axis_i: int,
    axis_j: int,
    pinned: tuple[float, ...],
) -> GridSnapshot | None:
    metric = _load_metric(fit_path)
    if metric is None:
        return None
    policy = _safe_load_policy()
    axes = tuple(metric.sub_condition_ids) or (
        tuple(s.id for s in policy.rubric.sub_conditions) if policy else ()
    )
    if not axes:
        return None
    if len(pinned) != len(axes):
        # Repad in case the pinned tuple drifted (defensive on cache misses).
        new_pinned = list(pinned)
        if len(new_pinned) < len(axes):
            new_pinned.extend([DEFAULT_AXIS_THRESHOLD] * (len(axes) - len(new_pinned)))
        else:
            new_pinned = new_pinned[: len(axes)]
        pinned = tuple(new_pinned)
    xs, ys, grid = _build_grid_arrays(metric, axis_i, axis_j, pinned)
    ellipses = _build_ellipses(metric, axis_i, axis_j, pinned)
    anchors = (
        _build_anchor_overlay(metric, policy, axis_i, axis_j) if policy else []
    )
    return GridSnapshot(
        fit_id=fit_id,
        axes=axes,
        axis_i=axis_i,
        axis_j=axis_j,
        pinned=pinned,
        xs=xs.tolist(),
        ys=ys.tolist(),
        sqrt_det_g=grid.tolist(),
        ellipses=ellipses,
        anchors=anchors,
    )


# ---------------------------------------------------------------------------
# Distance helpers (uncached — cheap and depend on free-form inputs)
# ---------------------------------------------------------------------------


def _compute_distance(
    metric: RiemannianMetric,
    c0: tuple[float, ...],
    c1: tuple[float, ...],
    n_segments: int,
) -> DistanceComparison:
    arr0 = np.asarray(c0, dtype=np.float64)
    arr1 = np.asarray(c1, dtype=np.float64)
    eucl = euclidean_distance(arr0, arr1)
    riem = riemannian_distance(metric, arr0, arr1, n_segments=n_segments)
    ratio = (riem / eucl) if eucl > 1e-9 else None
    return DistanceComparison(c0=c0, c1=c1, euclidean=eucl, riemannian=riem, ratio=ratio)


def _pad_pair_to_k(c2d: tuple[float, float], k: int) -> tuple[float, ...]:
    """Pad a 2D point with the violation midpoint for the other axes."""
    out = [DEFAULT_AXIS_THRESHOLD] * k
    if k >= 1:
        out[0] = float(c2d[0])
    if k >= 2:
        out[1] = float(c2d[1])
    return tuple(out)


def _example1_segments(
    metric: RiemannianMetric, n_segments: int
) -> list[PresetSegment]:
    k = metric.k
    pairs = [
        ("System A: interior (0.91,0.88) → (0.82,0.79)", (0.91, 0.88), (0.82, 0.79)),
        ("System B: near-boundary (0.64,0.61) → (0.55,0.52)", (0.64, 0.61), (0.55, 0.52)),
    ]
    out: list[PresetSegment] = []
    for label, a, b in pairs:
        c0 = _pad_pair_to_k(a, k)
        c1 = _pad_pair_to_k(b, k)
        cmp = _compute_distance(metric, c0, c1, n_segments)
        out.append(
            PresetSegment(
                label=label,
                c_from=c0,
                c_to=c1,
                euclidean=cmp.euclidean,
                riemannian=cmp.riemannian,
                ratio=cmp.ratio,
            )
        )
    return out


def _example3_segments(
    metric: RiemannianMetric, n_segments: int
) -> list[PresetSegment]:
    k = metric.k
    points = [(0.89, 0.85), (0.81, 0.79), (0.71, 0.68)]
    labels = ["t₀ → t₁", "t₁ → t₂"]
    out: list[PresetSegment] = []
    for label, a, b in zip(labels, points[:-1], points[1:]):
        c0 = _pad_pair_to_k(a, k)
        c1 = _pad_pair_to_k(b, k)
        cmp = _compute_distance(metric, c0, c1, n_segments)
        out.append(
            PresetSegment(
                label=label,
                c_from=c0,
                c_to=c1,
                euclidean=cmp.euclidean,
                riemannian=cmp.riemannian,
                ratio=cmp.ratio,
            )
        )
    return out


def _example5_segments(
    metric: RiemannianMetric, n_segments: int
) -> list[PresetSegment]:
    """Distance from `(0.67, 0.72)` to its axis-aligned boundary projections.

    v1 has a single global metric, so probe families P and Q both
    project against the same g; the pair illustrates "how much does
    Riemannian differ from Euclidean here" rather than "P vs Q
    comparison" (which would need locally distinct metrics — v2).
    """
    k = metric.k
    start = _pad_pair_to_k((0.67, 0.72), k)
    target_axis_0 = list(start)
    target_axis_0[0] = DEFAULT_AXIS_THRESHOLD
    target_axis_1 = list(start)
    target_axis_1[1] = DEFAULT_AXIS_THRESHOLD
    out: list[PresetSegment] = []
    for label, target in (
        ("→ axis-0 boundary (c₁ = 0.5)", tuple(target_axis_0)),
        ("→ axis-1 boundary (c₂ = 0.5)", tuple(target_axis_1)),
    ):
        cmp = _compute_distance(metric, start, target, n_segments)
        out.append(
            PresetSegment(
                label=label,
                c_from=start,
                c_to=target,
                euclidean=cmp.euclidean,
                riemannian=cmp.riemannian,
                ratio=cmp.ratio,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Plotly figure
# ---------------------------------------------------------------------------


def _build_figure(
    snapshot: GridSnapshot,
    axis_i_label: str,
    axis_j_label: str,
    overlays: list[PresetSegment] | None = None,
    custom_pair: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
) -> go.Figure:
    fig = go.Figure()

    # Heatmap of √det(g).
    fig.add_trace(
        go.Heatmap(
            x=snapshot.xs,
            y=snapshot.ys,
            z=snapshot.sqrt_det_g,
            colorscale="Viridis",
            colorbar=dict(title="√det(g)"),
            hoverinfo="skip",
            zmin=float(np.min(snapshot.sqrt_det_g)),
            zmax=float(np.max(snapshot.sqrt_det_g)),
        )
    )

    # Ellipse polylines (one trace, None-separated).
    if snapshot.ellipses:
        ell_x: list[float | None] = []
        ell_y: list[float | None] = []
        for e in snapshot.ellipses:
            ell_x.extend(e.xs + [None])
            ell_y.extend(e.ys + [None])
        fig.add_trace(
            go.Scatter(
                x=ell_x,
                y=ell_y,
                mode="lines",
                name="metric ellipses",
                line=dict(color="white", width=1),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    # Anchors.
    if snapshot.anchors:
        labels = [
            f"{a.anchor_id}<br>"
            f"c={tuple(round(v, 3) for v in a.c_full)}<br>"
            f"d_eucl={a.euclidean_boundary:.3f} | "
            f"d_riem={a.riemannian_boundary:.3f}"
            + (f" | ratio={a.ratio:.2f}" if a.ratio is not None else "")
            for a in snapshot.anchors
        ]
        fig.add_trace(
            go.Scatter(
                x=[a.cx for a in snapshot.anchors],
                y=[a.cy for a in snapshot.anchors],
                mode="markers",
                name="anchors",
                marker=dict(
                    size=12,
                    color=[a.riemannian_boundary for a in snapshot.anchors],
                    colorscale="OrRd",
                    showscale=False,
                    line=dict(color="black", width=1),
                ),
                text=labels,
                hoverinfo="text",
            )
        )

    # Overlay preset segments.
    if overlays:
        seg_x: list[float | None] = []
        seg_y: list[float | None] = []
        for seg in overlays:
            seg_x.extend(
                [seg.c_from[snapshot.axis_i], seg.c_to[snapshot.axis_i], None]
            )
            seg_y.extend(
                [seg.c_from[snapshot.axis_j], seg.c_to[snapshot.axis_j], None]
            )
        fig.add_trace(
            go.Scatter(
                x=seg_x,
                y=seg_y,
                mode="lines+markers",
                name="worked-example segments",
                line=dict(color="cyan", width=2, dash="dash"),
                marker=dict(size=8, color="cyan"),
                hoverinfo="skip",
            )
        )

    if custom_pair is not None:
        a, b = custom_pair
        fig.add_trace(
            go.Scatter(
                x=[a[snapshot.axis_i], b[snapshot.axis_i]],
                y=[a[snapshot.axis_j], b[snapshot.axis_j]],
                mode="lines+markers",
                name="custom pair",
                line=dict(color="magenta", width=2),
                marker=dict(size=10, color="magenta", symbol="x"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        xaxis=dict(title=f"c[{axis_i_label}]", range=[0, 1]),
        yaxis=dict(title=f"c[{axis_j_label}]", range=[0, 1], scaleanchor="x"),
        height=560,
        margin=dict(l=40, r=40, t=40, b=80),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _segment_table_rows(segments: list[PresetSegment]) -> list[dict[str, object]]:
    rows = []
    for seg in segments:
        rows.append(
            {
                "segment": seg.label,
                "c_from": tuple(round(v, 3) for v in seg.c_from),
                "c_to": tuple(round(v, 3) for v in seg.c_to),
                "euclidean": round(seg.euclidean, 4),
                "riemannian": round(seg.riemannian, 4),
                "ratio": (
                    round(seg.ratio, 3) if seg.ratio is not None else "—"
                ),
            }
        )
    return rows


def _render_distance_comparator(
    metric: RiemannianMetric,
    snapshot: GridSnapshot,
    preset: str,
    axis_i: int,
    axis_j: int,
) -> tuple[
    list[PresetSegment] | None,
    tuple[tuple[float, ...], tuple[float, ...]] | None,
]:
    """Renders the comparator UI; returns segment overlays for the figure."""
    n_segments = st.slider(
        "Riemannian-distance segments (piecewise-linear path resolution)",
        min_value=4,
        max_value=64,
        value=DEFAULT_N_SEGMENTS,
        step=2,
    )

    if preset == PRESET_CUSTOM:
        st.markdown("**Custom comparison** — enter two compliance points:")
        cols0 = st.columns(metric.k)
        cols1 = st.columns(metric.k)
        c0_vals: list[float] = []
        c1_vals: list[float] = []
        for idx in range(metric.k):
            axis_label = (
                snapshot.axes[idx]
                if idx < len(snapshot.axes)
                else f"axis_{idx}"
            )
            c0_vals.append(
                cols0[idx].number_input(
                    f"c0[{axis_label}]",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(snapshot.pinned[idx]) if idx < len(snapshot.pinned) else 0.5,
                    step=0.05,
                    key=f"c0_{idx}",
                )
            )
            c1_vals.append(
                cols1[idx].number_input(
                    f"c1[{axis_label}]",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(snapshot.pinned[idx]) if idx < len(snapshot.pinned) else 0.5,
                    step=0.05,
                    key=f"c1_{idx}",
                )
            )
        c0 = tuple(float(v) for v in c0_vals)
        c1 = tuple(float(v) for v in c1_vals)
        cmp = _compute_distance(metric, c0, c1, n_segments)
        cA, cB, cC = st.columns(3)
        cA.metric("Euclidean", f"{cmp.euclidean:.4f}")
        cB.metric("Riemannian", f"{cmp.riemannian:.4f}")
        cC.metric(
            "ratio (riem / eucl)",
            f"{cmp.ratio:.3f}" if cmp.ratio is not None else "—",
        )
        return None, (c0, c1)

    if preset == PRESET_EX1:
        st.markdown(
            "**§4.6 Example 1 — interior vs near-boundary.** "
            "Same Euclidean displacement, very different Riemannian distance. "
            "System A is deep in the safe interior; System B is close to the "
            "policy boundary. The Riemannian / Euclidean ratio should be "
            "materially larger for System B."
        )
        segments = _example1_segments(metric, n_segments)
        st.dataframe(
            _segment_table_rows(segments),
            hide_index=True,
            use_container_width=True,
        )
        return segments, None

    if preset == PRESET_EX3:
        st.markdown(
            "**§4.6 Example 3 — temporal trajectory.** Three measurements of "
            "the same anchor at t₀, t₁, t₂. The Euclidean distance grows "
            "roughly linearly; the Riemannian distance should accelerate as "
            "the trajectory enters the stretched region near the boundary."
        )
        segments = _example3_segments(metric, n_segments)
        st.dataframe(
            _segment_table_rows(segments),
            hide_index=True,
            use_container_width=True,
        )
        return segments, None

    if preset == PRESET_EX5:
        st.markdown(
            "**§4.6 Example 5 — geodesic vs straight-line distance.** From "
            "`(0.67, 0.72)`, project onto each axis-aligned violation midpoint "
            "(c=0.5). The ratio per direction shows where the metric stretches "
            "the boundary distance. v1 has a single global metric, so the "
            "doc's P-vs-Q comparison maps to per-direction comparisons here."
        )
        segments = _example5_segments(metric, n_segments)
        st.dataframe(
            _segment_table_rows(segments),
            hide_index=True,
            use_container_width=True,
        )
        return segments, None

    return None, None


def _render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.markdown(
        "Visualizes the §4.2–§4.6 Riemannian-geometry argument from the "
        "architecture doc against the fitted metric tensor field. The "
        "heatmap shows `√det(g(c))` (volumetric stretch); the ellipses "
        "are the local unit ball of `g` (small ellipse = unit Riemannian "
        "step is a small coordinate step = stretched space). Anchors are "
        "coloured by Riemannian boundary distance."
    )

    fits = _cached_fits()
    if not fits:
        st.info(EMPTY_STATE_MSG)
        return

    labels_to_fit = {f.label: f for f in fits}
    chosen_label = st.selectbox(
        "Metric fit", options=list(labels_to_fit.keys()), index=0
    )
    fit = labels_to_fit[chosen_label]

    metric = _load_metric(fit.path)
    if metric is None:
        st.warning(
            f"Metric `.npz` not found at `{fit.path}`. Re-run "
            "`maimonedes fit-metric` to regenerate it."
        )
        return

    policy = _safe_load_policy()
    if policy is None:
        st.warning(
            "Policy / rubric YAML missing — anchor overlay will be empty."
        )

    axes = list(metric.sub_condition_ids) or (
        [s.id for s in policy.rubric.sub_conditions] if policy else []
    )
    if len(axes) < 2:
        st.warning(
            f"Metric is k={len(axes)}; need k ≥ 2 to render the 2D slice."
        )
        return

    col_x, col_y = st.columns(2)
    axis_i_id = col_x.selectbox("X axis", options=axes, index=0)
    default_y = axes[1] if axes[1] != axis_i_id else axes[0]
    y_options = [a for a in axes if a != axis_i_id]
    axis_j_id = col_y.selectbox(
        "Y axis", options=y_options, index=0
    )
    axis_i = axes.index(axis_i_id)
    axis_j = axes.index(axis_j_id)

    pinned = [DEFAULT_AXIS_THRESHOLD] * len(axes)
    other_axes = [a for a in axes if a not in (axis_i_id, axis_j_id)]
    if other_axes:
        with st.expander(
            f"Pinned values for the other {len(other_axes)} axis"
            f"{'es' if len(other_axes) > 1 else ''} (default 0.5 = "
            "violation midpoint)"
        ):
            for a in other_axes:
                idx = axes.index(a)
                pinned[idx] = st.slider(
                    a,
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    key=f"pin_{a}",
                )

    snapshot = _cached_grid(
        fit.fit_id, fit.path, axis_i, axis_j, tuple(pinned)
    )
    if snapshot is None:
        st.warning("Could not compute the metric grid for this slice.")
        return

    st.subheader("Distance comparator")
    preset = st.radio(
        "Worked-example preset",
        options=PRESET_LABELS,
        horizontal=True,
        index=0,
    )

    overlays, custom_pair = _render_distance_comparator(
        metric, snapshot, preset, axis_i, axis_j
    )

    if len(axes) > 2 and preset != PRESET_CUSTOM:
        st.caption(
            f"Note: the §4.6 worked examples are k=2 illustrations; this "
            f"policy is k={len(axes)}. Numerical distances correctly use "
            f"all k axes (other axes pinned at 0.5)."
        )

    st.subheader("Compliance plane")

    # Perturbation-direction overlays (operator can toggle each layer
    # independently — see _build_*_arrows + _add_arrow_overlays).
    overlay_cols = st.columns(2)
    show_boundary_grad = overlay_cols[0].checkbox(
        "Show boundary-gradient arrows (−g⁻¹·w)",
        value=True,
        help=(
            "Steepest Riemannian descent toward the violation boundary at "
            "each anchor. Black arrows. The contravariant gradient of the "
            "linear margin s(c)=w·(c-τ) under the metric — accounts for "
            "the fact that some Δc directions are 'free' under g."
        ),
    )
    show_perturb_quiver = overlay_cols[1].checkbox(
        "Show Jacobian-column quiver (per-perturbation-type arrows)",
        value=False,
        help=(
            "At each anchor, the projection of every Jacobian row's Δ "
            "vector onto the chosen 2D plane. One colour per "
            "perturbation_kind. Compare with the boundary-gradient arrow "
            "to see which perturbations align with the descent direction "
            "(= efficient at breaking compliance) vs which are wasted."
        ),
    )

    boundary_arrows: list[BoundaryGradientArrow] = []
    perturb_arrows: list[PerturbationArrow] = []
    if policy is not None and (show_boundary_grad or show_perturb_quiver):
        if show_boundary_grad:
            boundary_arrows = _build_boundary_gradient_arrows(
                metric, policy, axis_i, axis_j, snapshot.anchors
            )
        if show_perturb_quiver:
            perturb_arrows = _build_perturbation_arrows(
                metric, policy, axis_i, axis_j, snapshot.anchors
            )

    fig = _build_figure(
        snapshot,
        axis_i_label=axis_i_id,
        axis_j_label=axis_j_id,
        overlays=overlays,
        custom_pair=custom_pair,
    )
    if boundary_arrows or perturb_arrows:
        _add_arrow_overlays(
            fig,
            boundary_arrows=boundary_arrows,
            perturbation_arrows=perturb_arrows,
            arrow_scale=_arrow_scale(snapshot.anchors),
        )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Heatmap colour = √det(g(c)). Ellipses = unit ball of g on the "
        f"axis-pair sub-block (small + eccentric = stretched + "
        f"anisotropic). Anchors coloured by Riemannian boundary distance "
        f"(darker = closer to the boundary). "
        f"k={len(axes)}; metric_fit #{fit.fit_id}."
    )

    if policy is not None:
        _render_perturbation_efficiency(metric, policy)

    _render_3d_ratio_surface(metric, fit, axes, pinned)
    _render_radial_3d_cloud(metric, fit, axes)


@st.cache_data(ttl=10)
def _cached_ratio_surface(
    fit_id: int,
    fit_path: str,
    axis_i: int,
    axis_j: int,
    axis_i_id: str,
    axis_j_id: str,
    pinned: tuple[float, ...],
    mode: str,
    reference: tuple[float, ...] | None,
    resolution: int,
) -> RatioSurfaceSnapshot | None:
    metric = _load_metric(fit_path)
    if metric is None:
        return None
    try:
        xs, ys, z = compute_ratio_surface(
            metric,
            axis_indices=(axis_i, axis_j),
            pinned=pinned,
            mode=mode,
            reference=reference,
            resolution=resolution,
        )
    except (ValueError, RuntimeError):
        return None
    return RatioSurfaceSnapshot(
        fit_id=fit_id,
        mode=mode,
        axis_i=axis_i,
        axis_j=axis_j,
        axis_i_id=axis_i_id,
        axis_j_id=axis_j_id,
        pinned=pinned,
        reference=reference,
        xs=xs,
        ys=ys,
        z=z,
    )


def _render_3d_ratio_surface(
    metric: RiemannianMetric,
    fit: MetricFitSnapshot,
    axes: list[str],
    pinned: list[float],
) -> None:
    """3D-surface view: Riemannian/Euclidean ratio (or local stretch) over a pair.

    The 2D plot above shows `√det(g(c))` as a heatmap; this section
    answers a slightly different question — "starting from a fixed
    compliant reference, how compliance-expensive is reaching this
    coordinate point?" Tall peaks in the surface are the cliffs; flat
    valleys are where Riemannian geometry agrees with Euclidean (i.e.,
    safe to navigate).
    """
    if len(axes) < 2:
        return

    st.subheader("3D fragility landscape (Riemannian / Euclidean)")

    mode_label = st.radio(
        "Surface mode",
        options=["Fixed reference", "Local stretch (√λ_max)"],
        horizontal=True,
        index=0,
        help=(
            "**Fixed reference**: Z = riemannian / euclidean from a chosen "
            "reference point. Tall peaks are 'compliance-expensive' regions "
            "— small Euclidean step, large Riemannian step. **Local "
            "stretch**: Z = √λ_max(g(c)). No reference; tall peaks are "
            "regions where the metric is locally sharp regardless of where "
            "you came from."
        ),
    )
    mode = "fixed_reference" if mode_label == "Fixed reference" else "local_stretch"

    # Pick the worst-fragility axis pair as defaults.
    default_i, default_j = worst_fragility_axis_pair(
        sub_condition_ids=tuple(axes), fallback=(0, 1)
    )

    col_x3, col_y3 = st.columns(2)
    axis_i_id = col_x3.selectbox(
        "X axis",
        options=axes,
        index=default_i,
        key="ratio3d_x",
    )
    y_options = [a for a in axes if a != axis_i_id]
    default_j_in_y = (
        y_options.index(axes[default_j])
        if axes[default_j] in y_options
        else 0
    )
    axis_j_id = col_y3.selectbox(
        "Y axis",
        options=y_options,
        index=default_j_in_y,
        key="ratio3d_y",
    )
    axis_i = axes.index(axis_i_id)
    axis_j = axes.index(axis_j_id)

    reference: tuple[float, ...] | None = None
    if mode == "fixed_reference":
        with st.expander(
            "Reference point (default = (1, 1, …, 1) full compliance)"
        ):
            ref_list = [1.0] * len(axes)
            for ax_idx, ax_id in enumerate(axes):
                ref_list[ax_idx] = st.slider(
                    f"reference[{ax_id}]",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                    key=f"ratio3d_ref_{ax_id}",
                )
            reference = tuple(ref_list)

    resolution = st.slider(
        "Grid resolution",
        min_value=10,
        max_value=50,
        value=25,
        step=5,
        help=(
            "Higher resolution = smoother surface but more compute. The "
            "fixed-reference mode integrates a Riemannian path per cell, "
            "so resolution=50 takes a few seconds."
        ),
    )

    snapshot = _cached_ratio_surface(
        fit.fit_id,
        fit.path,
        axis_i,
        axis_j,
        axis_i_id,
        axis_j_id,
        tuple(pinned),
        mode,
        reference,
        resolution,
    )
    if snapshot is None:
        st.warning("Could not compute the 3D ratio surface for this slice.")
        return

    z_arr = np.asarray(snapshot.z, dtype=float)
    z_finite = z_arr[np.isfinite(z_arr)]
    z_label = (
        "Euclidean / Riemannian"
        if mode == "fixed_reference"
        else "1 / √λ_max(g(c))"
    )
    fig3d = go.Figure(
        data=[
            go.Surface(
                x=snapshot.xs,
                y=snapshot.ys,
                z=snapshot.z,
                # Topographic convention: high = stable (blue, mountain
                # of "compliance-resistance"), low = fragile (red, valley
                # where compliance erodes fast).
                colorscale="RdBu",
                colorbar=dict(title=z_label),
                contours=dict(
                    z=dict(
                        show=True,
                        usecolormap=True,
                        highlightcolor="black",
                        project=dict(z=True),
                    )
                ),
            )
        ]
    )
    if mode == "fixed_reference" and reference is not None:
        fig3d.add_trace(
            go.Scatter3d(
                x=[reference[axis_i]],
                y=[reference[axis_j]],
                z=[float(np.nanmax(z_arr)) if z_finite.size > 0 else 0.0],
                mode="markers+text",
                marker=dict(size=8, color="black", symbol="diamond"),
                text=["reference"],
                textposition="top center",
                showlegend=False,
            )
        )
    fig3d.update_layout(
        scene=dict(
            xaxis_title=axis_i_id,
            yaxis_title=axis_j_id,
            zaxis_title=z_label,
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=600,
    )
    st.plotly_chart(fig3d, use_container_width=True)

    if z_finite.size > 0:
        st.caption(
            f"Z-axis = {z_label} (topographic convention: tall peaks = "
            f"stable, fragile cliffs are deep canyons). "
            f"min={float(np.nanmin(z_arr)):.3f}, "
            f"max={float(np.nanmax(z_arr)):.3f}, "
            f"mean={float(np.nanmean(z_arr)):.3f}. Mode={mode}."
        )


@st.cache_data(ttl=10)
def _cached_radial_cloud(
    fit_id: int,
    fit_path: str,
    axis_ids: tuple[str, ...],
    reference: tuple[float, ...],
    n_samples: int,
    seed: int,
    include_anchors: bool,
) -> RadialCloudSnapshot | None:
    metric = _load_metric(fit_path)
    if metric is None:
        return None
    try:
        result = compute_radial_projection_cloud(
            metric,
            reference=reference,
            n_samples=n_samples,
            seed=seed,
            include_anchors=include_anchors,
        )
    except (ValueError, RuntimeError):
        return None
    points = [
        RadialCloudPoint(
            x=float(p[0]),
            y=float(p[1]),
            z=float(p[2]),
            c_full=tuple(float(v) for v in p[3]),
            is_anchor=bool(p[4]),
        )
        for p in result["points"]  # type: ignore[index]
    ]
    return RadialCloudSnapshot(
        fit_id=fit_id,
        n_samples=n_samples,
        seed=seed,
        include_anchors=include_anchors,
        reference=tuple(float(v) for v in result["reference"]),  # type: ignore[index,arg-type]
        axis_ids=axis_ids,
        points=points,
        axis_labels=[
            (int(label[0]), float(label[1]), float(label[2]))
            for label in result["axis_labels"]  # type: ignore[index]
        ],
        r_max=float(result["r_max"]),  # type: ignore[index,arg-type]
    )


def _render_radial_3d_cloud(
    metric: RiemannianMetric,
    fit: MetricFitSnapshot,
    axes: list[str],
) -> None:
    """3D star-coordinate scatter: all k axes radially in (x, y), Z = ratio.

    Star coordinates project each compliance vector c via
    `(x, y) = Σᵢ cᵢ · (cos θᵢ, sin θᵢ)` with `θᵢ = 2π·i/k`. The mapping
    is many-to-one for k > 2 (multiple compliance vectors can land at
    the same (x, y) with different Z) — represented honestly here as
    overlapping scatter points stacking with different Z values.

    Z = euclidean / riemannian from a chosen reference (default
    all-compliant). Topographic: high = stable, low = fragile.
    """
    if len(axes) < 2:
        return

    st.subheader("3D radial projection (all k axes → 2D, Z = compliance-resistance)")

    n_samples = st.slider(
        "Random samples",
        min_value=500,
        max_value=10000,
        value=3000,
        step=500,
        help=(
            "Number of compliance vectors uniformly sampled from "
            "[0,1]^k. More samples = denser cloud but slower (each "
            "sample integrates a Riemannian path against the reference)."
        ),
    )
    include_anchors = st.checkbox(
        "Highlight library anchors",
        value=True,
        help="Project library anchor positions onto the cloud as triangle markers.",
    )
    seed = st.number_input(
        "Sampling seed", min_value=0, max_value=10**6, value=0, step=1, key="radial_seed"
    )

    with st.expander(
        "Reference point (default = (1, 1, …, 1) full compliance)"
    ):
        ref_list = [1.0] * len(axes)
        for ax_idx, ax_id in enumerate(axes):
            ref_list[ax_idx] = st.slider(
                f"reference[{ax_id}]",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.05,
                key=f"radial_ref_{ax_id}",
            )

    snapshot = _cached_radial_cloud(
        fit.fit_id,
        fit.path,
        tuple(axes),
        tuple(ref_list),
        int(n_samples),
        int(seed),
        bool(include_anchors),
    )
    if snapshot is None or not snapshot.points:
        st.warning("Could not compute the radial cloud for this metric.")
        return

    finite_pts = [p for p in snapshot.points if np.isfinite(p.z)]
    if not finite_pts:
        st.warning(
            "All points landed at the reference (zero distance) — try moving "
            "the reference."
        )
        return

    random_pts = [p for p in finite_pts if not p.is_anchor]
    anchor_pts = [p for p in finite_pts if p.is_anchor]

    z_values = [p.z for p in finite_pts]
    z_min = float(min(z_values))
    z_max = float(max(z_values))
    z_mean = float(np.mean(z_values))

    fig = go.Figure()
    if random_pts:
        fig.add_trace(
            go.Scatter3d(
                x=[p.x for p in random_pts],
                y=[p.y for p in random_pts],
                z=[p.z for p in random_pts],
                mode="markers",
                marker=dict(
                    size=3,
                    color=[p.z for p in random_pts],
                    colorscale="RdBu",
                    cmin=z_min,
                    cmax=z_max,
                    colorbar=dict(title="Eucl / Riem"),
                    opacity=0.55,
                ),
                hovertemplate=(
                    "x=%{x:.3f}<br>y=%{y:.3f}<br>"
                    "Z (Eucl/Riem)=%{z:.3f}<extra></extra>"
                ),
                name="sampled",
                showlegend=False,
            )
        )
    if anchor_pts:
        fig.add_trace(
            go.Scatter3d(
                x=[p.x for p in anchor_pts],
                y=[p.y for p in anchor_pts],
                z=[p.z for p in anchor_pts],
                mode="markers",
                marker=dict(
                    size=8,
                    color="black",
                    symbol="diamond",
                    line=dict(color="white", width=1),
                ),
                name="library anchors",
                showlegend=True,
            )
        )

    # Axis labels around the unit circle so the operator can read which
    # radial direction corresponds to which policy axis.
    for ax_idx, lx, ly in snapshot.axis_labels:
        ax_id = axes[ax_idx] if ax_idx < len(axes) else f"axis_{ax_idx}"
        fig.add_trace(
            go.Scatter3d(
                x=[0, lx],
                y=[0, ly],
                z=[z_mean, z_mean],
                mode="lines+text",
                line=dict(color="grey", width=2, dash="dot"),
                text=["", ax_id],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="x (radial projection)",
            yaxis_title="y (radial projection)",
            zaxis_title="Eucl / Riem (high = stable)",
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=650,
    )
    st.plotly_chart(fig, use_container_width=True)

    n_anchor = len(anchor_pts)
    n_random = len(random_pts)
    st.caption(
        f"Star-coord projection of c ∈ [0,1]^{len(axes)} → (x, y); "
        f"Z = euclidean / riemannian from reference. Topographic "
        f"convention: tall = stable, low = fragile. "
        f"{n_random} random samples + {n_anchor} library anchors. "
        f"Z range: min={z_min:.3f}, max={z_max:.3f}, mean={z_mean:.3f}. "
        f"Note: balanced compliance vectors land near the origin "
        f"(Σ unit-vectors = 0); imbalanced ones spread to the periphery."
    )


def _arrow_scale(
    anchors: list[AnchorOverlay],
    *,
    fraction: float = 0.06,
) -> float:
    """Scale arrows so the longest one spans `fraction` of the [0,1] plot.

    Arrows otherwise wildly over- or under-shoot when the metric's
    contravariant gradient has a large magnitude (small λ_min in g).
    The 0.06 default keeps a single arrow ≤ ~6% of the plot width,
    which keeps the heatmap legible.
    """
    return float(fraction)


def _build_boundary_gradient_arrows(
    metric: RiemannianMetric,
    policy: Policy,
    axis_i: int,
    axis_j: int,
    anchors: list[AnchorOverlay],
) -> list[BoundaryGradientArrow]:
    """At each anchor, compute −g(c)⁻¹·w (steepest Riemannian descent)."""
    if not anchors:
        return []
    weights = axis_weights(policy)
    axes = list(metric.sub_condition_ids) or [
        s.id for s in policy.rubric.sub_conditions
    ]
    if not axes:
        return []
    weights_vec = np.asarray(
        [weights.get(a, 0.0) for a in axes], dtype=np.float64
    )
    if not np.any(weights_vec):
        return []
    out: list[BoundaryGradientArrow] = []
    for a in anchors:
        try:
            dual = boundary_gradient_contravariant(
                metric,
                np.asarray(a.c_full, dtype=np.float64),
                weights_vec=weights_vec,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        descent = -dual
        norm = float(np.linalg.norm(descent))
        if norm < 1e-9:
            continue
        unit = descent / norm
        out.append(
            BoundaryGradientArrow(
                anchor_id=a.anchor_id,
                cx=a.cx,
                cy=a.cy,
                dx=float(unit[axis_i]),
                dy=float(unit[axis_j]),
                full_descent=tuple(float(v) for v in descent.tolist()),
            )
        )
    return out


def _build_perturbation_arrows(
    metric: RiemannianMetric,
    policy: Policy,
    axis_i: int,
    axis_j: int,
    anchors: list[AnchorOverlay],
    *,
    max_per_anchor: int = 6,
) -> list[PerturbationArrow]:
    """Project each anchor's Jacobian rows onto the chosen axis pair.

    Up to `max_per_anchor` rows per anchor (the most-impactful by
    |-Δ·w|) so the heatmap doesn't drown in arrows.
    """
    if not anchors:
        return []
    weights = axis_weights(policy)
    axes = list(metric.sub_condition_ids) or [
        s.id for s in policy.rubric.sub_conditions
    ]
    if not axes:
        return []
    weights_vec = np.asarray(
        [weights.get(a, 0.0) for a in axes], dtype=np.float64
    )
    jacs = all_jacobians()
    out: list[PerturbationArrow] = []
    for a in anchors:
        jac = jacs.get(a.anchor_id)
        if jac is None or not jac.rows:
            continue
        scored: list[tuple[float, object, np.ndarray]] = []
        for row in jac.rows:
            delta = np.asarray(
                [row.deltas.get(ax, 0.0) for ax in axes], dtype=np.float64
            )
            if not np.any(delta):
                continue
            alignment = float(-(delta @ weights_vec))
            scored.append((abs(alignment), row, delta))
        scored.sort(key=lambda kv: -kv[0])
        for _abs_align, row, delta in scored[:max_per_anchor]:
            out.append(
                PerturbationArrow(
                    anchor_id=a.anchor_id,
                    perturbation_kind=row.perturbation_kind,  # type: ignore[attr-defined]
                    transform_label=row.transform_label,  # type: ignore[attr-defined]
                    cx=a.cx,
                    cy=a.cy,
                    dx=float(delta[axis_i]),
                    dy=float(delta[axis_j]),
                    full_delta=tuple(float(v) for v in delta.tolist()),
                )
            )
    return out


def _build_perturbation_efficiency(
    metric: RiemannianMetric,
    policy: Policy,
    anchor_id: str,
) -> list[PerturbationEfficiencyEntry]:
    """Per-row boundary-closure efficiency at one anchor; ranked descending."""
    weights = axis_weights(policy)
    axes = list(metric.sub_condition_ids) or [
        s.id for s in policy.rubric.sub_conditions
    ]
    if not axes:
        return []
    weights_vec = np.asarray(
        [weights.get(a, 0.0) for a in axes], dtype=np.float64
    )
    jac = all_jacobians().get(anchor_id)
    if jac is None or not jac.rows:
        return []
    c_anchor = np.asarray(
        [jac.baseline_per_sub_condition.get(a, 0.0) for a in axes],
        dtype=np.float64,
    )
    out: list[PerturbationEfficiencyEntry] = []
    for row in jac.rows:
        delta = np.asarray(
            [row.deltas.get(a, 0.0) for a in axes], dtype=np.float64
        )
        if not np.any(delta):
            continue
        try:
            eff, alignment, riem_norm = perturbation_efficiency(
                metric=metric,
                c_anchor=c_anchor,
                delta=delta,
                weights_vec=weights_vec,
            )
        except (np.linalg.LinAlgError, ValueError):
            continue
        out.append(
            PerturbationEfficiencyEntry(
                anchor_id=anchor_id,
                perturbation_kind=row.perturbation_kind,
                transform_label=row.transform_label,
                efficiency=eff,
                boundary_alignment=alignment,
                riem_norm=riem_norm,
            )
        )
    out.sort(key=lambda e: e.efficiency, reverse=True)
    return out


def _add_arrow_overlays(
    fig: go.Figure,
    *,
    boundary_arrows: list[BoundaryGradientArrow],
    perturbation_arrows: list[PerturbationArrow],
    arrow_scale: float,
) -> None:
    """Append the boundary-gradient and Jacobian-quiver layers to `fig`."""
    kind_colors = {
        "authority": "red",
        "boundary": "darkorange",
        "demographic": "purple",
        "ethnicity": "magenta",
        "profession": "saddlebrown",
        "paraphrase": "teal",
    }
    if perturbation_arrows:
        for kind in sorted({a.perturbation_kind for a in perturbation_arrows}):
            color = kind_colors.get(kind, "grey")
            arrows = [a for a in perturbation_arrows if a.perturbation_kind == kind]
            xs: list[float | None] = []
            ys: list[float | None] = []
            for ar in arrows:
                vx = ar.dx
                vy = ar.dy
                norm = float(np.hypot(vx, vy))
                if norm < 1e-12:
                    continue
                clip = arrow_scale * min(1.0, norm * 4.0)
                ex = ar.cx + (vx / norm) * clip
                ey = ar.cy + (vy / norm) * clip
                xs.extend([ar.cx, ex, None])
                ys.extend([ar.cy, ey, None])
            if not xs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=f"perturb: {kind}",
                    line=dict(color=color, width=1.5),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )
    if boundary_arrows:
        xs2: list[float | None] = []
        ys2: list[float | None] = []
        for ar in boundary_arrows:
            vx = ar.dx
            vy = ar.dy
            norm = float(np.hypot(vx, vy))
            if norm < 1e-12:
                continue
            ex = ar.cx + (vx / norm) * arrow_scale
            ey = ar.cy + (vy / norm) * arrow_scale
            xs2.extend([ar.cx, ex, None])
            ys2.extend([ar.cy, ey, None])
        if xs2:
            fig.add_trace(
                go.Scatter(
                    x=xs2,
                    y=ys2,
                    mode="lines",
                    name="−g⁻¹·w (Riemannian descent)",
                    line=dict(color="black", width=3),
                    hoverinfo="skip",
                    showlegend=True,
                )
            )


def _render_perturbation_efficiency(
    metric: RiemannianMetric,
    policy: Policy,
) -> None:
    """Anchor selectbox + horizontal bar chart of boundary-closure efficiency."""
    jacs = all_jacobians()
    if not jacs:
        st.info(
            "No anchor Jacobians yet — run `maimonedes perturb --all-anchors` "
            "to populate the perturbation cloud, then this chart ranks "
            "which perturbations break compliance fastest at each anchor."
        )
        return

    st.subheader("Boundary-closure efficiency (ranked perturbations)")
    st.markdown(
        "For each perturbation type *j* applied at the selected anchor, "
        "computes `efficiency = -Δ_j · w / ‖Δ_j‖_g(c)`. Higher = more "
        "compliance erosion per unit Riemannian step, in the FULL k-dim "
        "metric (no 2D-slice loss). Ranked top-to-bottom."
    )

    anchor_ids = sorted(jacs.keys())
    selected = st.selectbox(
        "Anchor", options=anchor_ids, index=0, key="perturb_eff_anchor"
    )
    entries = _build_perturbation_efficiency(metric, policy, selected)
    if not entries:
        st.info(f"No usable Jacobian rows for anchor {selected!r}.")
        return

    top_n = st.slider(
        "Show top N",
        min_value=5,
        max_value=min(50, len(entries)),
        value=min(15, len(entries)),
        step=1,
        key="perturb_eff_top_n",
    )
    top_entries = entries[:top_n]

    labels = [
        f"{e.transform_label}  ({e.perturbation_kind})" for e in top_entries
    ]
    effs = [e.efficiency for e in top_entries]
    norms = [e.riem_norm for e in top_entries]
    aligns = [e.boundary_alignment for e in top_entries]

    bar_fig = go.Figure()
    bar_fig.add_trace(
        go.Bar(
            x=effs,
            y=labels,
            orientation="h",
            marker=dict(
                color=effs,
                colorscale="RdBu_r",
                cmin=-max(abs(e) for e in effs) if effs else 0.0,
                cmax=max(abs(e) for e in effs) if effs else 1.0,
                colorbar=dict(title="efficiency"),
            ),
            customdata=list(zip(norms, aligns)),
            hovertemplate=(
                "%{y}<br>efficiency=%{x:.4f}<br>"
                "‖Δ‖_g=%{customdata[0]:.4f}<br>"
                "-Δ·w=%{customdata[1]:.4f}<extra></extra>"
            ),
        )
    )
    bar_fig.update_layout(
        xaxis_title="boundary-closure efficiency",
        yaxis=dict(autorange="reversed"),
        height=max(360, 30 * len(top_entries) + 80),
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    st.caption(
        f"Anchor {selected!r}: top {len(top_entries)} of {len(entries)} "
        f"perturbation rows. Red bars = high efficiency (the model's "
        f"weak directions); blue = lower efficiency (Riemannian-"
        f"expensive or boundary-misaligned). The full k-dim metric is "
        f"used — these numbers don't depend on the 2D slice above."
    )


_render()
