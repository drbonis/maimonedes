"""Phase 5 page: curvature alarm — κ comparison + eigen spectra + signals feed.

Visualizes the §6.4 *geometric alarm* claim against the persisted
metric_fits. Pick two fits (typically baseline-vs-after-feedback or
t₀-vs-t₁), see which anchors got steeper at their position, and read
the persisted `structural_signals` feed for previously-recorded alerts.

v1 curvature scalar is `κ(g) = λ_max / λ_min` per `monitor/curvature.py`
— "how anisotropic is the metric here?". The eigenvalue-spectrum bars
make the change visible per direction; the ellipse-pair view shows the
change for one operator-chosen axis pair.
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

from maimonedes.dashboard._curvature_snapshots import (
    AnchorCurvatureRow,
    CurvatureFitOption,
    CurvatureSnapshot,
    EigenSpectrum,
    EllipsePairData,
    StructuralAlertRow,
)
from maimonedes.monitor.curvature import (
    DEFAULT_H_CURVATURE,
    compute_curvature,
)
from maimonedes.monitor.fragility import all_jacobians
from maimonedes.monitor.metric import (
    RiemannianMetric,
    metric_at,
    position_vector,
)
from maimonedes.storage.metric_fits import list_metric_fits
from maimonedes.storage.structural_signals import list_structural_signals


PAGE_TITLE = "Curvature — Phase 5"
EMPTY_STATE_MSG = (
    "Need at least two `metric_fits` to compare curvature. Run "
    "`maimonedes fit-metric` twice — once before and once after the "
    "drift / feedback cycle."
)
ELLIPSE_POINTS = 24
ELLIPSE_TARGET_SEMI = 0.20  # anchor-centric ellipses, no per-cell collision


# ---------------------------------------------------------------------------
# Cached data layer
# ---------------------------------------------------------------------------


@st.cache_data(ttl=10)
def _cached_fit_options() -> list[CurvatureFitOption]:
    out: list[CurvatureFitOption] = []
    for row in list_metric_fits():
        trained = row.get("trained_at")
        ts_str = (
            trained.strftime("%Y-%m-%d %H:%M")
            if isinstance(trained, datetime)
            else "?"
        )
        val_loss = row.get("val_loss")
        val_str = (
            f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "—"
        )
        out.append(
            CurvatureFitOption(
                fit_id=int(row["id"]),  # type: ignore[arg-type]
                path=str(row["path"]),
                policy_id=str(row["policy_id"]),
                n_anchors=int(row["n_anchors"]),  # type: ignore[arg-type]
                val_loss=float(val_loss) if isinstance(val_loss, (int, float)) else None,
                trained_at=trained if isinstance(trained, datetime) else None,
                label=(
                    f"fit #{row['id']}  ·  {row['policy_id']}  ·  {ts_str}  ·  "
                    f"n_anchors={row['n_anchors']}  ·  val_loss={val_str}"
                ),
            )
        )
    return out


def _load_metric(path: str) -> RiemannianMetric | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return RiemannianMetric.load(p)
    except (OSError, ValueError):
        return None


@st.cache_data(ttl=10)
def _cached_snapshot(
    baseline_fit_id: int,
    baseline_path: str,
    current_fit_id: int,
    current_path: str,
    h_curvature: float,
) -> CurvatureSnapshot | None:
    metric_baseline = _load_metric(baseline_path)
    metric_current = _load_metric(current_path)
    if metric_baseline is None or metric_current is None:
        return None
    if metric_baseline.k != metric_current.k:
        return None

    axes = (
        metric_baseline.sub_condition_ids
        or metric_current.sub_condition_ids
        or tuple()
    )
    if not axes:
        # Fall back to indexed axis names so downstream rendering still works.
        axes = tuple(f"axis_{i}" for i in range(metric_baseline.k))

    rows: list[AnchorCurvatureRow] = []
    spectra: dict[str, EigenSpectrum] = {}
    for anchor_id, jac in all_jacobians().items():
        if set(axes).difference(jac.baseline_per_sub_condition.keys()):
            # Skip anchors whose Jacobian doesn't carry every metric axis.
            continue
        c = position_vector(jac.baseline_per_sub_condition, axes)
        try:
            kappa_b = compute_curvature(metric_baseline, c)
            kappa_c = compute_curvature(metric_current, c)
        except ValueError as exc:
            rows.append(
                AnchorCurvatureRow(
                    anchor_id=anchor_id,
                    c_anchor=tuple(float(v) for v in c),
                    kappa_baseline=float("nan"),
                    kappa_current=float("nan"),
                    relative_increase=float("nan"),
                    fired=False,
                    error=str(exc),
                )
            )
            continue
        rel = (kappa_c - kappa_b) / kappa_b if kappa_b > 0 else float("nan")
        rows.append(
            AnchorCurvatureRow(
                anchor_id=anchor_id,
                c_anchor=tuple(float(v) for v in c),
                kappa_baseline=float(kappa_b),
                kappa_current=float(kappa_c),
                relative_increase=float(rel),
                fired=bool(rel > h_curvature),
            )
        )
        g_b = metric_at(metric_baseline, c)
        g_c = metric_at(metric_current, c)
        spectra[anchor_id] = EigenSpectrum(
            anchor_id=anchor_id,
            axis_ids=tuple(axes),
            eigvals_baseline=sorted(np.linalg.eigvalsh(g_b).tolist()),
            eigvals_current=sorted(np.linalg.eigvalsh(g_c).tolist()),
        )

    rows.sort(
        key=lambda r: (
            -r.relative_increase if not math.isnan(r.relative_increase) else 0.0
        )
    )
    return CurvatureSnapshot(
        baseline_fit_id=baseline_fit_id,
        current_fit_id=current_fit_id,
        axis_ids=tuple(axes),
        rows=rows,
        spectra=spectra,
    )


@st.cache_data(ttl=10)
def _cached_signals_feed(limit: int = 20) -> list[StructuralAlertRow]:
    feed = list_structural_signals(signal_type="curvature", limit=limit)
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
# Ellipse helpers (mirror metric page; anchor-centric scale)
# ---------------------------------------------------------------------------


def _ellipse_xy_for_block(
    g_block: np.ndarray, cx: float, cy: float, scale: float
) -> tuple[list[float], list[float]]:
    eigvals, eigvecs = np.linalg.eigh(g_block)
    eigvals = np.maximum(eigvals, 1e-12)
    semi = (1.0 / np.sqrt(eigvals)) * scale
    theta = np.linspace(0.0, 2.0 * math.pi, ELLIPSE_POINTS)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    ell = eigvecs @ (np.diag(semi) @ circle)
    return (ell[0] + cx).tolist(), (ell[1] + cy).tolist()


def _build_ellipse_pair(
    metric_baseline: RiemannianMetric,
    metric_current: RiemannianMetric,
    c_anchor: np.ndarray,
    axis_i: int,
    axis_j: int,
) -> EllipsePairData:
    g_b = metric_at(metric_baseline, c_anchor)
    g_c = metric_at(metric_current, c_anchor)
    block_b = np.array(
        [[g_b[axis_i, axis_i], g_b[axis_i, axis_j]],
         [g_b[axis_j, axis_i], g_b[axis_j, axis_j]]],
        dtype=np.float64,
    )
    block_c = np.array(
        [[g_c[axis_i, axis_i], g_c[axis_i, axis_j]],
         [g_c[axis_j, axis_i], g_c[axis_j, axis_j]]],
        dtype=np.float64,
    )
    eigvals_b = np.maximum(np.linalg.eigvalsh(block_b), 1e-12)
    eigvals_c = np.maximum(np.linalg.eigvalsh(block_c), 1e-12)
    median_semi = float(
        np.median([1.0 / np.sqrt(eigvals_b.min()), 1.0 / np.sqrt(eigvals_c.min())])
    )
    scale = ELLIPSE_TARGET_SEMI / max(median_semi, 1e-9)
    cx = float(c_anchor[axis_i])
    cy = float(c_anchor[axis_j])
    bx, by = _ellipse_xy_for_block(block_b, cx, cy, scale)
    cx_, cy_ = _ellipse_xy_for_block(block_c, cx, cy, scale)
    return EllipsePairData(
        anchor_id="",  # caller sets
        axis_i=axis_i,
        axis_j=axis_j,
        cx=cx,
        cy=cy,
        baseline_xs=bx,
        baseline_ys=by,
        current_xs=cx_,
        current_ys=cy_,
    )


# ---------------------------------------------------------------------------
# Tables + figures
# ---------------------------------------------------------------------------


def _table_dataframe(snapshot: CurvatureSnapshot) -> pd.DataFrame:
    rows = []
    for r in snapshot.rows:
        rec: dict[str, object] = {"anchor": r.anchor_id}
        for idx, axis in enumerate(snapshot.axis_ids):
            rec[f"c[{axis}]"] = round(r.c_anchor[idx], 3) if idx < len(r.c_anchor) else None
        rec["κ_baseline"] = round(r.kappa_baseline, 3) if not math.isnan(r.kappa_baseline) else None
        rec["κ_current"] = round(r.kappa_current, 3) if not math.isnan(r.kappa_current) else None
        rec["relative_increase"] = (
            round(r.relative_increase, 3)
            if not math.isnan(r.relative_increase)
            else None
        )
        rec["fired"] = "✓" if r.fired else "✗"
        if r.error:
            rec["note"] = r.error
        rows.append(rec)
    return pd.DataFrame(rows)


def _color_relative_increase(val: object, threshold: float) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    if not isinstance(val, (int, float)):
        return ""
    f = float(val)
    if f > threshold:
        return "background-color: rgba(220, 80, 80, 0.55)"
    if f > 0:
        return "background-color: rgba(220, 170, 80, 0.45)"
    return "background-color: rgba(120, 200, 120, 0.35)"


def _styled_dataframe(df: pd.DataFrame, threshold: float):
    if "relative_increase" not in df.columns:
        return df
    return df.style.format(
        {
            "κ_baseline": "{:.3f}",
            "κ_current": "{:.3f}",
            "relative_increase": "{:+.3f}",
        },
        na_rep="—",
    ).map(lambda v: _color_relative_increase(v, threshold), subset=["relative_increase"])


def _spectrum_figure(spectrum: EigenSpectrum) -> go.Figure:
    indices = list(range(1, len(spectrum.eigvals_baseline) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=[f"λ{i}" for i in indices],
            x=spectrum.eigvals_baseline,
            orientation="h",
            name="baseline",
            marker_color="#4a90e2",
        )
    )
    fig.add_trace(
        go.Bar(
            y=[f"λ{i}" for i in indices],
            x=spectrum.eigvals_current,
            orientation="h",
            name="current",
            marker_color="#d0021b",
        )
    )
    fig.update_layout(
        barmode="group",
        xaxis=dict(title="eigenvalue magnitude"),
        yaxis=dict(title="rank (ascending)"),
        height=320,
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _ellipse_pair_figure(
    pair: EllipsePairData, axis_i_label: str, axis_j_label: str
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pair.baseline_xs,
            y=pair.baseline_ys,
            mode="lines",
            name="baseline",
            line=dict(color="#4a90e2", width=2),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pair.current_xs,
            y=pair.current_ys,
            mode="lines",
            name="current",
            line=dict(color="#d0021b", width=2),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pair.cx],
            y=[pair.cy],
            mode="markers",
            name="anchor",
            marker=dict(size=10, color="black", symbol="cross"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        xaxis=dict(title=f"c[{axis_i_label}]"),
        yaxis=dict(title=f"c[{axis_j_label}]", scaleanchor="x"),
        height=420,
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


def _select_default_anchor(snapshot: CurvatureSnapshot) -> str | None:
    if not snapshot.rows:
        return None
    # Already sorted descending by relative_increase. Pick the first non-error
    # row; if all rows errored, fall back to the first.
    for r in snapshot.rows:
        if r.error is None and not math.isnan(r.relative_increase):
            return r.anchor_id
    return snapshot.rows[0].anchor_id


def _render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.markdown(
        "Curvature is the §6.4 *geometric alarm*: the metric at an anchor "
        "location is steeper than it was, even when the anchor's position "
        "has not moved. v1 measures `κ(g) = λ_max / λ_min` (anisotropy at "
        "the anchor); the full Ricci scalar is deferred to v3 per the "
        "issue note in `monitor/curvature.py`."
    )

    fits = _cached_fit_options()
    if len(fits) < 2:
        st.info(EMPTY_STATE_MSG)
        return

    labels_to_fit = {f.label: f for f in fits}
    col_b, col_c = st.columns(2)
    baseline_label = col_b.selectbox(
        "Baseline fit",
        options=list(labels_to_fit.keys()),
        index=min(1, len(fits) - 1),
    )
    current_label = col_c.selectbox(
        "Current fit",
        options=list(labels_to_fit.keys()),
        index=0,
    )
    baseline_fit = labels_to_fit[baseline_label]
    current_fit = labels_to_fit[current_label]

    if baseline_fit.policy_id != current_fit.policy_id:
        st.error(
            "Baseline and current fits must share a `policy_id`. Re-fit "
            "the metric on a single policy across both runs."
        )
        return
    if baseline_fit.fit_id == current_fit.fit_id:
        st.warning(
            "Baseline and current point at the same fit — curvature change "
            "should be zero everywhere. This is a sanity-check view only."
        )

    threshold = st.slider(
        "h_curvature (relative-increase threshold for `fired`)",
        min_value=0.0,
        max_value=2.0,
        value=float(DEFAULT_H_CURVATURE),
        step=0.05,
        help=(
            "Fires when `(κ_current − κ_baseline) / κ_baseline > h_curvature`. "
            "This slider only affects the live view; the persisted threshold "
            "is set when the CLI / orchestrator records `structural_signals`."
        ),
    )

    snapshot = _cached_snapshot(
        baseline_fit.fit_id,
        baseline_fit.path,
        current_fit.fit_id,
        current_fit.path,
        threshold,
    )
    if snapshot is None:
        st.warning(
            "Could not load both metric artefacts. Re-run "
            "`maimonedes fit-metric` to regenerate the `.npz` files."
        )
        return

    if not snapshot.rows:
        st.info(
            "No anchors with a Jacobian aligned to both metrics' axes. Run "
            "`maimonedes perturb --all-anchors` to seed the fragility data, "
            "then re-fit."
        )
    else:
        st.subheader("Per-anchor κ comparison")
        df = _table_dataframe(snapshot)
        # Apply live `fired` recompute against the slider (snapshot was cached
        # at the chosen threshold but the threshold also keys the cache).
        st.dataframe(_styled_dataframe(df, threshold), use_container_width=True)

    if snapshot.rows:
        st.subheader("Per-anchor eigenvalue spectrum")
        anchor_options = [r.anchor_id for r in snapshot.rows]
        default_anchor = _select_default_anchor(snapshot) or anchor_options[0]
        anchor_id = st.selectbox(
            "Anchor",
            options=anchor_options,
            index=anchor_options.index(default_anchor),
        )
        spectrum = snapshot.spectra.get(anchor_id)
        if spectrum is not None:
            log_scale = st.checkbox(
                "Log scale",
                value=False,
                help="Useful when the eigenvalue spread spans multiple orders of magnitude.",
            )
            fig = _spectrum_figure(spectrum)
            if log_scale:
                fig.update_xaxes(type="log")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Bars grouped per λ rank: longer current-vs-baseline ratio = "
                "metric got more anisotropic at this anchor (curvature ↑)."
            )

            # Ellipse pair on operator-chosen axis pair.
            axes = list(snapshot.axis_ids)
            if len(axes) >= 2:
                st.subheader("2D ellipse pair (chosen axis pair)")
                col_x, col_y = st.columns(2)
                axis_i_id = col_x.selectbox(
                    "X axis", options=axes, index=0, key="curv_x"
                )
                y_options = [a for a in axes if a != axis_i_id]
                axis_j_id = col_y.selectbox(
                    "Y axis", options=y_options, index=0, key="curv_y"
                )
                axis_i = axes.index(axis_i_id)
                axis_j = axes.index(axis_j_id)
                metric_baseline = _load_metric(baseline_fit.path)
                metric_current = _load_metric(current_fit.path)
                if metric_baseline is not None and metric_current is not None:
                    target_row = next(
                        (r for r in snapshot.rows if r.anchor_id == anchor_id),
                        None,
                    )
                    if target_row is not None:
                        c_arr = np.asarray(target_row.c_anchor, dtype=np.float64)
                        pair = _build_ellipse_pair(
                            metric_baseline,
                            metric_current,
                            c_arr,
                            axis_i,
                            axis_j,
                        )
                        st.plotly_chart(
                            _ellipse_pair_figure(pair, axis_i_id, axis_j_id),
                            use_container_width=True,
                        )
                        st.caption(
                            "Both ellipses share a global scale; tightening "
                            "from baseline (blue) to current (red) on the "
                            "chosen axis pair is the geometric alarm signal."
                        )

    st.subheader("Recorded curvature alarms")
    feed = _cached_signals_feed()
    if not feed:
        st.write("No curvature alarms have been recorded yet.")
    else:
        st.dataframe(_signals_dataframe(feed), use_container_width=True)


_render()
