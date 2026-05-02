"""Phase 5 page: decoupling alarm — per-anchor covariance shift.

Visualizes the §4.6 Example 4 / §6.4 *structural reorganization* claim:
the off-diagonal covariance between policy axes flipped sign, even
when no individual axis crossed a violation line. Two intuition pumps:

1. k×k covariance heatmaps (baseline + current) side by side — sign
   flips show as red↔blue cell flips with a yellow border.
2. 2D axis-pair scatter clouds — perturbation-stage scores from each
   window plotted on the same axes; a tight diagonal cluster in
   baseline that rotates orthogonally in the current window is the
   §4.6 Example 4 case made literal. PCA principal axes overlaid
   per window so the rotation angle is rendered as a number.

Reads compliance scores directly via `monitor/decoupling`. No
`metric_fit` dependency — this signal lives entirely in the score
covariance, not in the learned metric.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from maimonedes.core.policy import Policy, load_policy
from maimonedes.dashboard._decoupling_snapshots import (
    AnchorDecouplingRow,
    CovarianceMatrix,
    DecouplingSnapshot,
    ScatterCloud,
    StructuralAlertRow,
)
from maimonedes.monitor.decoupling import (
    SIGN_FLIP_EPS,
    anchors_with_perturbation_scores,
    decoupling_signal,
)
from maimonedes.storage.structural_signals import list_structural_signals


PAGE_TITLE = "Decoupling — Phase 5"
EMPTY_STATE_MSG = (
    "No anchor has any perturbation-stage scores yet. Run "
    "`maimonedes perturb --all-anchors` (and ensure scoring fires) to "
    "populate this dashboard."
)
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
DEFAULT_BASELINE_WINDOW = 50
DEFAULT_CURRENT_WINDOW = 20


# ---------------------------------------------------------------------------
# Cached data layer
# ---------------------------------------------------------------------------


def _safe_load_policy() -> Policy | None:
    try:
        return load_policy(POLICY_PATH, RUBRIC_PATH)
    except FileNotFoundError:
        return None


def _axis_ids_from_policy(policy: Policy | None) -> tuple[str, ...]:
    if policy is None:
        return tuple()
    return tuple(s.id for s in policy.rubric.sub_conditions)


def _matrix_to_dataclass(
    axis_ids: tuple[str, ...], cov: np.ndarray
) -> CovarianceMatrix:
    return CovarianceMatrix(axis_ids=axis_ids, matrix=cov.tolist())


@st.cache_data(ttl=10)
def _cached_anchor_list() -> list[str]:
    return list(anchors_with_perturbation_scores())


@st.cache_data(ttl=10)
def _cached_snapshot(
    anchor_id: str,
    baseline_window: int,
    current_window: int,
    axis_ids: tuple[str, ...],
    all_anchor_ids: tuple[str, ...],
) -> DecouplingSnapshot | None:
    if not axis_ids:
        return None

    rows: list[AnchorDecouplingRow] = []
    for aid in all_anchor_ids:
        try:
            res = decoupling_signal(
                anchor_id=aid,
                axis_ids=axis_ids,
                baseline_window=baseline_window,
                current_window=current_window,
            )
        except ValueError:
            continue
        flipped_labels = tuple(
            (axis_ids[i], axis_ids[j]) for (i, j) in res.flipped_pairs
        )
        rows.append(
            AnchorDecouplingRow(
                anchor_id=aid,
                evidence_count=int(res.evidence_count),
                frobenius_delta=float(res.frobenius_delta),
                n_flipped_pairs=len(res.flipped_pairs),
                flipped_pair_labels=flipped_labels,
                h_decoupling=float(res.h_decoupling),
                fired=bool(res.signal_fired),
            )
        )

    rows.sort(key=lambda r: -r.frobenius_delta)

    snap = DecouplingSnapshot(
        anchor_id=anchor_id,
        baseline_window=baseline_window,
        current_window=current_window,
        axis_ids=axis_ids,
        rows=rows,
    )

    # Compute the chosen anchor's covariance pair separately so the
    # heatmaps + flipped pairs are immediately available.
    try:
        chosen = decoupling_signal(
            anchor_id=anchor_id,
            axis_ids=axis_ids,
            baseline_window=baseline_window,
            current_window=current_window,
        )
    except ValueError:
        return snap
    snap.selected_baseline_cov = _matrix_to_dataclass(axis_ids, chosen.baseline_cov)
    snap.selected_current_cov = _matrix_to_dataclass(axis_ids, chosen.current_cov)
    snap.selected_flipped_pairs = tuple(chosen.flipped_pairs)
    return snap


@st.cache_data(ttl=10)
def _cached_signals_feed(limit: int = 20) -> list[StructuralAlertRow]:
    feed = list_structural_signals(signal_type="decoupling", limit=limit)
    out: list[StructuralAlertRow] = []
    for row in feed:
        evidence = row.get("evidence")
        if isinstance(evidence, dict):
            try:
                ev_str = json.dumps(evidence, sort_keys=True, default=str)
            except (TypeError, ValueError):
                ev_str = repr(evidence)
        else:
            ev_str = str(evidence) if evidence is not None else ""
        out.append(
            StructuralAlertRow(
                fired_at=row.get("fired_at") if isinstance(row.get("fired_at"), datetime) else None,
                anchor_id=str(row.get("anchor_id")),
                metric_value=float(row.get("metric_value", 0.0)),
                threshold=float(row.get("threshold", 0.0)),
                evidence=ev_str,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Scatter rotation helpers (uncached — cheap; depends on slider+axis pair)
# ---------------------------------------------------------------------------


def _scatter_arrays(
    anchor_id: str,
    axis_ids: tuple[str, ...],
    baseline_window: int,
    current_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (baseline_matrix, current_matrix) of perturbation scores.

    Reuses `monitor.decoupling._recent_perturbation_scores` so the
    page's window-slicing semantics match `decoupling_signal` exactly.
    """
    from maimonedes.monitor.decoupling import _recent_perturbation_scores

    total = baseline_window + current_window
    scores = _recent_perturbation_scores(anchor_id, limit=total)
    current = scores[:current_window]
    baseline = scores[current_window:]

    def _to_matrix(ss):
        if not ss:
            return np.zeros((0, len(axis_ids)), dtype=np.float64)
        return np.asarray(
            [[s.per_sub_condition.get(a, 0.0) for a in axis_ids] for s in ss],
            dtype=np.float64,
        )

    return _to_matrix(baseline), _to_matrix(current)


def _principal_axis(matrix: np.ndarray) -> np.ndarray | None:
    """Unit vector along the first PC of `matrix` (rows = samples)."""
    if matrix.shape[0] < 2:
        return None
    pca = PCA(n_components=1)
    pca.fit(matrix)
    v = pca.components_[0]
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return None
    return v / norm


def _pca_line_from(
    matrix: np.ndarray, axis_i: int, axis_j: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return (start, end) endpoints of the PC1 line projected onto axes."""
    if matrix.shape[0] < 2:
        return None
    sub = matrix[:, [axis_i, axis_j]]
    v = _principal_axis(sub)
    if v is None:
        return None
    centroid = sub.mean(axis=0)
    span = max(float(sub.max() - sub.min()), 0.1)
    start = tuple(centroid - 0.5 * span * v)
    end = tuple(centroid + 0.5 * span * v)
    return (
        (float(start[0]), float(start[1])),
        (float(end[0]), float(end[1])),
    )


def _rotation_angle_deg(
    baseline: np.ndarray, current: np.ndarray, axis_i: int, axis_j: int
) -> float | None:
    """Angle between the PC1 directions of the two clouds (axis-pair slice)."""
    if baseline.shape[0] < 2 or current.shape[0] < 2:
        return None
    sub_b = baseline[:, [axis_i, axis_j]]
    sub_c = current[:, [axis_i, axis_j]]
    v_b = _principal_axis(sub_b)
    v_c = _principal_axis(sub_c)
    if v_b is None or v_c is None:
        return None
    dot = float(np.dot(v_b, v_c))
    cross = float(abs(v_b[0] * v_c[1] - v_b[1] * v_c[0]))
    angle = math.degrees(math.atan2(cross, dot))
    # Clamp to [0, 90] — orientation only, not direction.
    if angle > 90.0:
        angle = 180.0 - angle
    return angle


def _build_scatter_cloud(
    anchor_id: str,
    axis_ids: tuple[str, ...],
    axis_i: int,
    axis_j: int,
    baseline_window: int,
    current_window: int,
) -> ScatterCloud | None:
    if axis_i >= len(axis_ids) or axis_j >= len(axis_ids):
        return None
    baseline, current = _scatter_arrays(
        anchor_id, axis_ids, baseline_window, current_window
    )
    bx = baseline[:, axis_i].tolist() if baseline.shape[0] else []
    by = baseline[:, axis_j].tolist() if baseline.shape[0] else []
    cx = current[:, axis_i].tolist() if current.shape[0] else []
    cy = current[:, axis_j].tolist() if current.shape[0] else []
    return ScatterCloud(
        anchor_id=anchor_id,
        axis_i=axis_i,
        axis_j=axis_j,
        axis_i_id=axis_ids[axis_i],
        axis_j_id=axis_ids[axis_j],
        baseline_x=bx,
        baseline_y=by,
        current_x=cx,
        current_y=cy,
        baseline_pca_line=_pca_line_from(baseline, axis_i, axis_j),
        current_pca_line=_pca_line_from(current, axis_i, axis_j),
        rotation_degrees=_rotation_angle_deg(baseline, current, axis_i, axis_j),
    )


# ---------------------------------------------------------------------------
# Tables + figures
# ---------------------------------------------------------------------------


def _table_dataframe(snapshot: DecouplingSnapshot) -> pd.DataFrame:
    rows = []
    for r in snapshot.rows:
        rows.append(
            {
                "anchor": r.anchor_id,
                "evidence_count": r.evidence_count,
                "frobenius_delta": round(r.frobenius_delta, 4),
                "n_flipped": r.n_flipped_pairs,
                "flipped_pairs": "; ".join(
                    f"({a}, {b})" for a, b in r.flipped_pair_labels
                ) or "—",
                "h_decoupling": round(r.h_decoupling, 4),
                "fired": "✓" if r.fired else "✗",
            }
        )
    return pd.DataFrame(rows)


def _color_frobenius(val: object, threshold: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    if not isinstance(val, (int, float)):
        return ""
    f = float(val)
    if f > threshold:
        return "background-color: rgba(220, 80, 80, 0.55)"
    if f > 0.5 * threshold:
        return "background-color: rgba(220, 170, 80, 0.45)"
    return ""


def _styled_dataframe(df: pd.DataFrame, h_threshold_proxy: float):
    if "frobenius_delta" not in df.columns:
        return df
    return df.style.format(
        {
            "frobenius_delta": "{:.4f}",
            "h_decoupling": "{:.4f}",
        }
    ).map(
        lambda v: _color_frobenius(v, h_threshold_proxy),
        subset=["frobenius_delta"],
    )


def _heatmap_pair_figure(
    baseline: CovarianceMatrix,
    current: CovarianceMatrix,
    flipped_pairs: tuple[tuple[int, int], ...],
) -> go.Figure:
    from plotly.subplots import make_subplots

    z_max = max(
        float(np.max(np.abs(np.asarray(baseline.matrix)))) if baseline.matrix else 0.0,
        float(np.max(np.abs(np.asarray(current.matrix)))) if current.matrix else 0.0,
    )
    if z_max < 1e-9:
        z_max = 1e-3  # avoid degenerate colorscale

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("baseline window", "current window"),
        horizontal_spacing=0.12,
    )
    for col_idx, mat in ((1, baseline), (2, current)):
        fig.add_trace(
            go.Heatmap(
                x=list(mat.axis_ids),
                y=list(mat.axis_ids),
                z=mat.matrix,
                zmin=-z_max,
                zmax=z_max,
                colorscale="RdBu",
                zmid=0,
                showscale=col_idx == 2,
                colorbar=dict(title="cov") if col_idx == 2 else None,
            ),
            row=1,
            col=col_idx,
        )

    # Highlight flipped (i, j) cells with a yellow rectangle on both panels.
    for (i, j) in flipped_pairs:
        for col_idx in (1, 2):
            xref = "x" if col_idx == 1 else "x2"
            yref = "y" if col_idx == 1 else "y2"
            for (xi, yi) in ((i, j), (j, i)):
                fig.add_shape(
                    type="rect",
                    x0=xi - 0.5,
                    x1=xi + 0.5,
                    y0=yi - 0.5,
                    y1=yi + 0.5,
                    line=dict(color="#f5d142", width=3),
                    xref=xref,
                    yref=yref,
                )
    fig.update_layout(
        height=420,
        margin=dict(l=40, r=40, t=60, b=80),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _scatter_figure(cloud: ScatterCloud) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cloud.baseline_x,
            y=cloud.baseline_y,
            mode="markers",
            name="baseline window",
            marker=dict(size=10, color="#4a90e2", symbol="circle", opacity=0.7),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cloud.current_x,
            y=cloud.current_y,
            mode="markers",
            name="current window",
            marker=dict(size=10, color="#d0021b", symbol="triangle-up", opacity=0.7),
            hoverinfo="skip",
        )
    )
    if cloud.baseline_pca_line is not None:
        (sx, sy), (ex, ey) = cloud.baseline_pca_line
        fig.add_trace(
            go.Scatter(
                x=[sx, ex],
                y=[sy, ey],
                mode="lines",
                name="baseline PC1",
                line=dict(color="#4a90e2", dash="dash", width=2),
                hoverinfo="skip",
            )
        )
    if cloud.current_pca_line is not None:
        (sx, sy), (ex, ey) = cloud.current_pca_line
        fig.add_trace(
            go.Scatter(
                x=[sx, ex],
                y=[sy, ey],
                mode="lines",
                name="current PC1",
                line=dict(color="#d0021b", dash="dash", width=2),
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        xaxis=dict(title=f"c[{cloud.axis_i_id}]"),
        yaxis=dict(title=f"c[{cloud.axis_j_id}]", scaleanchor="x"),
        height=460,
        margin=dict(l=40, r=40, t=20, b=80),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _signals_dataframe(rows: list[StructuralAlertRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fired_at": (
                    r.fired_at.strftime("%Y-%m-%d %H:%M:%S")
                    if r.fired_at is not None
                    else "—"
                ),
                "anchor_id": r.anchor_id,
                "metric_value": round(r.metric_value, 4),
                "threshold": round(r.threshold, 4),
                "evidence": r.evidence,
            }
            for r in rows
        ]
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def _render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.markdown(
        "Decoupling is the §6.4 *structural reorganization* alarm: the "
        "off-diagonal covariance between policy axes flipped sign, even "
        "when no individual axis crossed a violation line. §4.6 Example 4 "
        "is the canonical case — scope and calibration were correlated "
        "under perturbation, suddenly they aren't."
    )

    anchor_ids = _cached_anchor_list()
    if not anchor_ids:
        st.info(EMPTY_STATE_MSG)
        return

    policy = _safe_load_policy()
    axis_ids = _axis_ids_from_policy(policy)
    if not axis_ids:
        st.warning(
            "Policy / rubric YAML missing — cannot determine axis order."
        )
        return

    # Default to the first anchor that has a fired signal at the live thresholds.
    default_anchor = anchor_ids[0]
    snap_for_default = _cached_snapshot(
        default_anchor,
        DEFAULT_BASELINE_WINDOW,
        DEFAULT_CURRENT_WINDOW,
        axis_ids,
        tuple(anchor_ids),
    )
    if snap_for_default is not None:
        for r in snap_for_default.rows:
            if r.fired:
                default_anchor = r.anchor_id
                break

    anchor_id = st.selectbox(
        "Anchor",
        options=anchor_ids,
        index=anchor_ids.index(default_anchor) if default_anchor in anchor_ids else 0,
    )

    col_b, col_c = st.columns(2)
    baseline_window = col_b.slider(
        "baseline_window",
        min_value=5,
        max_value=200,
        value=DEFAULT_BASELINE_WINDOW,
        step=5,
        help=(
            "Most-recent N perturbation-stage scores (after the current "
            "window) used for the baseline covariance estimate."
        ),
    )
    current_window = col_c.slider(
        "current_window",
        min_value=5,
        max_value=100,
        value=DEFAULT_CURRENT_WINDOW,
        step=5,
        help=(
            "Most-recent M perturbation-stage scores used for the current "
            "covariance estimate. Sliders only change the live view; "
            "persisted thresholds are unchanged."
        ),
    )

    snapshot = _cached_snapshot(
        anchor_id,
        baseline_window,
        current_window,
        axis_ids,
        tuple(anchor_ids),
    )
    if snapshot is None or not snapshot.rows:
        st.warning(
            "No anchors had enough perturbation-stage scores to compute "
            "covariance (need ≥ 2 in each window). Run more perturbations "
            "or shrink the windows."
        )
        return

    selected_row = next(
        (r for r in snapshot.rows if r.anchor_id == anchor_id), None
    )
    h_proxy = (
        float(selected_row.h_decoupling)
        if selected_row is not None and selected_row.h_decoupling > 0
        else 0.05
    )

    st.subheader("Per-anchor decoupling signal")
    df = _table_dataframe(snapshot)
    st.dataframe(_styled_dataframe(df, h_proxy), use_container_width=True)

    insufficient_for_cov = (
        selected_row is None or selected_row.evidence_count < 2
    )
    below_window = (
        selected_row is not None
        and 2 <= selected_row.evidence_count < current_window
    )
    if insufficient_for_cov:
        st.warning(
            f"Anchor `{anchor_id}` has fewer than 2 perturbation-stage "
            "score(s) in the current window — covariance is undefined; "
            "heatmaps and scatter rotation skipped."
        )
    else:
        if below_window:
            st.warning(
                f"Anchor `{anchor_id}` has only "
                f"{selected_row.evidence_count} perturbation-stage scores "
                f"in the current window of {current_window} — covariance "
                f"is noisy. Run more perturbations or shrink the slider."
            )
        st.subheader("Covariance heatmaps (baseline vs current)")
        if (
            snapshot.selected_baseline_cov is not None
            and snapshot.selected_current_cov is not None
        ):
            fig_pair = _heatmap_pair_figure(
                snapshot.selected_baseline_cov,
                snapshot.selected_current_cov,
                snapshot.selected_flipped_pairs,
            )
            st.plotly_chart(fig_pair, use_container_width=True)
            st.caption(
                "Divergent colormap centred at zero (positive correlation = "
                "blue, negative = red). Yellow borders mark cells in "
                f"`flipped_pairs` (sign |Δ| > {SIGN_FLIP_EPS:g})."
            )

        st.subheader("Axis-pair scatter rotation")
        flipped_default: tuple[int, int] | None = (
            snapshot.selected_flipped_pairs[0]
            if snapshot.selected_flipped_pairs
            else None
        )
        col_x, col_y = st.columns(2)
        default_x_idx = flipped_default[0] if flipped_default else 0
        default_y_idx = flipped_default[1] if flipped_default else 1
        if default_y_idx >= len(axis_ids):
            default_y_idx = 1 if len(axis_ids) > 1 else 0
        axis_i_id = col_x.selectbox(
            "X axis", options=list(axis_ids), index=default_x_idx
        )
        y_options = [a for a in axis_ids if a != axis_i_id]
        if not y_options:
            st.warning("Need at least 2 axes for the scatter rotation view.")
        else:
            default_y = (
                axis_ids[default_y_idx]
                if axis_ids[default_y_idx] in y_options
                else y_options[0]
            )
            axis_j_id = col_y.selectbox(
                "Y axis", options=y_options, index=y_options.index(default_y)
            )
            axis_i = list(axis_ids).index(axis_i_id)
            axis_j = list(axis_ids).index(axis_j_id)

            cloud = _build_scatter_cloud(
                anchor_id, axis_ids, axis_i, axis_j, baseline_window, current_window
            )
            if cloud is None:
                st.write("(no scatter data for this axis pair)")
            else:
                st.plotly_chart(_scatter_figure(cloud), use_container_width=True)
                rot = (
                    f"{cloud.rotation_degrees:.1f}°"
                    if cloud.rotation_degrees is not None
                    else "—"
                )
                st.caption(
                    f"PC1 rotation between windows: **{rot}**. A rotation "
                    "near 90° is the §4.6 Example 4 sign flip; near 0° = "
                    "the cluster shape is unchanged. Decoupling shows up "
                    "as a rotation of the perturbation cluster's principal "
                    "axis."
                )

    st.subheader("Recorded decoupling alarms")
    feed = _cached_signals_feed()
    if not feed:
        st.write("No decoupling alarms have been recorded yet.")
    else:
        st.dataframe(_signals_dataframe(feed), use_container_width=True)


_render()
