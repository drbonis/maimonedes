"""Phase 3 page: drift timeline + detection-latency table.

Headline deliverable. Overlays the raw scalar score, the EWMA
statistic, and the CUSUM statistic across the 50 sessions of one
drift run, with control-limit lines, stage bands, and "first fire"
/ "first violation" annotations. The footer callout encodes the
primary feasibility claim numerically: lead = first_violation -
first_cusum, green when positive, red when ≤ 0, grey when nothing
violated.

Reads exclusively via helpers in `storage.drift`, `monitor.cusum`,
`monitor.ewma`, and `monitor.drift_report`. No SQL or pydantic in
this page body.
"""
from __future__ import annotations

import math
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from maimonedes.core.drift import STAGE_LABELS
from maimonedes.dashboard._drift_snapshots import AnchorTrace, RunSnapshot
from maimonedes.monitor._baseline import load_run_scalars
from maimonedes.monitor.cusum import cusum_per_anchor
from maimonedes.monitor.drift_report import (
    AnchorReportRow,
    DriftReport,
    NoBaselineDataError,
    build_report,
)
from maimonedes.monitor.ewma import ewma_per_anchor
from maimonedes.storage.drift import list_drift_runs


PAGE_TITLE = "Drift — Phase 3"
EMPTY_STATE_MSG = (
    "No drift runs recorded yet. Run `maimonedes induce-drift` to "
    "populate this dashboard."
)
DEGRADED_BASELINE_MSG = (
    "This run has fewer than 5 baseline-stage samples on every anchor. "
    "Detector traces and the latency table are unavailable; only the "
    "raw aggregate is plotted."
)
STAGE_COLORS = {
    "baseline": "rgba(120, 200, 120, 0.10)",
    "concise": "rgba(200, 200, 120, 0.10)",
    "actionable": "rgba(220, 170, 100, 0.13)",
    "no_caveats": "rgba(220, 130, 100, 0.16)",
    "trust": "rgba(220, 90, 90, 0.18)",
}


@st.cache_data(ttl=10)
def _cached_runs() -> list[RunSnapshot]:
    snapshots: list[RunSnapshot] = []
    for run in list_drift_runs():
        started = run.get("started_at")
        notes = run.get("notes")
        label_parts: list[str] = [f"run #{run['id']}"]
        if isinstance(started, datetime):
            label_parts.append(started.strftime("%Y-%m-%d %H:%M"))
        if notes:
            label_parts.append(str(notes))
        snapshots.append(
            RunSnapshot(
                run_id=int(run["id"]),  # type: ignore[arg-type]
                started_at=started if isinstance(started, datetime) else None,
                notes=str(notes) if notes else None,
                label="  ·  ".join(label_parts),
            )
        )
    return snapshots


@st.cache_data(ttl=10)
def _cached_traces(run_id: int) -> dict[str, AnchorTrace]:
    """Per-anchor traces. Detector-less anchors carry empty stat lists."""
    streams = load_run_scalars(run_id)
    cusum_traces = cusum_per_anchor(run_id)
    ewma_traces = ewma_per_anchor(run_id)

    out: dict[str, AnchorTrace] = {}
    for anchor_id, triples in streams.items():
        cusum_states = cusum_traces.get(anchor_id, [])
        ewma_states = ewma_traces.get(anchor_id, [])
        first_violation: int | None = next(
            (idx for (idx, _stage, agg) in triples if agg < 0.5), None
        )
        out[anchor_id] = AnchorTrace(
            anchor_id=anchor_id,
            session_indices=[idx for (idx, _stage, _agg) in triples],
            stage_labels=[stage for (_idx, stage, _agg) in triples],
            aggregates=[agg for (_idx, _stage, agg) in triples],
            cusum_values=[s.statistic for s in cusum_states],
            cusum_threshold=cusum_states[0].threshold if cusum_states else 0.0,
            cusum_first_fire=next(
                (s.session_index for s in cusum_states if s.fired), None
            ),
            ewma_values=[s.statistic for s in ewma_states],
            ewma_lcls=[s.lcl for s in ewma_states],
            ewma_asymptotic_lcl=ewma_states[-1].lcl if ewma_states else 0.0,
            ewma_first_fire=next(
                (s.session_index for s in ewma_states if s.fired), None
            ),
            first_violation=first_violation,
            detector_skipped=not cusum_states and not ewma_states,
        )
    return out


@st.cache_data(ttl=10)
def _cached_report(run_id: int) -> DriftReport | None:
    """`None` when the run has no baseline data on any anchor."""
    try:
        return build_report(run_id)
    except NoBaselineDataError:
        return None


def _stage_band_spans(stage_labels: list[str]) -> list[tuple[str, int, int]]:
    """Compress a per-session stage list into `(label, start, end)` runs."""
    if not stage_labels:
        return []
    out: list[tuple[str, int, int]] = []
    start = 0
    current = stage_labels[0]
    for i, label in enumerate(stage_labels[1:], start=1):
        if label != current:
            out.append((current, start, i - 1))
            current = label
            start = i
    out.append((current, start, len(stage_labels) - 1))
    return out


def _build_figure(trace: AnchorTrace) -> go.Figure:
    fig = go.Figure()

    # Stage bands first so subsequent traces sit on top.
    for stage, start_idx, end_idx in _stage_band_spans(trace.stage_labels):
        if stage not in STAGE_COLORS:
            continue
        x0 = trace.session_indices[start_idx]
        x1 = trace.session_indices[end_idx] + 1  # half-open right edge
        fig.add_vrect(
            x0=x0 - 0.5,
            x1=x1 - 0.5,
            fillcolor=STAGE_COLORS[stage],
            opacity=0.6,
            line_width=0,
            annotation_text=stage,
            annotation_position="top left",
            annotation=dict(font_size=10),
            layer="below",
        )

    # Raw aggregate (left axis).
    fig.add_trace(
        go.Scatter(
            x=trace.session_indices,
            y=trace.aggregates,
            mode="lines+markers",
            name="aggregate",
            line=dict(color="#1f77b4"),
            yaxis="y",
        )
    )

    # EWMA statistic (left axis, shares scale with aggregate).
    if trace.ewma_values:
        fig.add_trace(
            go.Scatter(
                x=trace.session_indices,
                y=trace.ewma_values,
                mode="lines",
                name="EWMA",
                line=dict(color="#2ca02c"),
                yaxis="y",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trace.session_indices,
                y=trace.ewma_lcls,
                mode="lines",
                name="EWMA LCL_t",
                line=dict(color="#2ca02c", dash="dot", width=1),
                yaxis="y",
                showlegend=True,
            )
        )

    # CUSUM (secondary axis — different scale).
    if trace.cusum_values:
        fig.add_trace(
            go.Scatter(
                x=trace.session_indices,
                y=trace.cusum_values,
                mode="lines",
                name="CUSUM",
                line=dict(color="#d62728"),
                yaxis="y2",
            )
        )
        fig.add_hline(
            y=trace.cusum_threshold,
            line=dict(color="#d62728", dash="dash", width=1),
            annotation_text=f"CUSUM h={trace.cusum_threshold:.3f}",
            annotation_position="top right",
            yref="y2",
        )

    # Violation threshold reference.
    fig.add_hline(
        y=0.5,
        line=dict(color="#888", dash="dot", width=1),
        annotation_text="violation = 0.5",
        annotation_position="bottom right",
    )

    # First-fire / first-violation annotations.
    if trace.cusum_first_fire is not None:
        fig.add_vline(
            x=trace.cusum_first_fire,
            line=dict(color="#d62728", dash="dash"),
            annotation_text=f"CUSUM @ S{trace.cusum_first_fire}",
            annotation_position="top",
        )
    if trace.ewma_first_fire is not None:
        fig.add_vline(
            x=trace.ewma_first_fire,
            line=dict(color="#2ca02c", dash="dash"),
            annotation_text=f"EWMA @ S{trace.ewma_first_fire}",
            annotation_position="bottom",
        )
    if trace.first_violation is not None:
        fig.add_vline(
            x=trace.first_violation,
            line=dict(color="#000", width=2),
            annotation_text=f"violation @ S{trace.first_violation}",
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis=dict(title="session_index"),
        yaxis=dict(title="aggregate / EWMA", range=[0, 1.05]),
        yaxis2=dict(
            title="CUSUM statistic",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", y=-0.2),
        height=520,
        margin=dict(l=40, r=40, t=40, b=80),
    )
    return fig


def _format_lead(row: AnchorReportRow) -> str:
    lead = row.cusum_lead_sessions
    if lead is None:
        return "-"
    if lead == math.inf:
        return "+inf"
    sign = "+" if lead > 0 else ""
    return f"{sign}{lead:.0f}"


def _row_to_table_dict(row: AnchorReportRow) -> dict[str, object]:
    if row.detector_skipped:
        return {
            "anchor": row.anchor_id,
            "n_sess": row.n_sessions,
            "mean_base": "n/a",
            "min_agg": "n/a",
            "first_viol": "n/a",
            "cusum": "n/a",
            "ewma": "n/a",
            "lead": "n/a",
        }
    return {
        "anchor": row.anchor_id,
        "n_sess": row.n_sessions,
        "mean_base": (
            f"{row.mean_baseline:.3f}" if row.mean_baseline is not None else "-"
        ),
        "min_agg": (
            f"{row.min_aggregate:.3f}" if row.min_aggregate is not None else "-"
        ),
        "first_viol": (
            row.first_violation_session
            if row.first_violation_session is not None
            else "-"
        ),
        "cusum": (
            row.cusum_first_fire if row.cusum_first_fire is not None else "-"
        ),
        "ewma": (
            row.ewma_first_fire if row.ewma_first_fire is not None else "-"
        ),
        "lead": _format_lead(row),
    }


def _select_default_anchor(
    traces: dict[str, AnchorTrace], options: list[str]
) -> str:
    """Default to the anchor with the earliest CUSUM fire; else lowest min agg."""
    earliest_fire: tuple[str, int] | None = None
    for anchor_id, trace in traces.items():
        if trace.cusum_first_fire is None:
            continue
        if earliest_fire is None or trace.cusum_first_fire < earliest_fire[1]:
            earliest_fire = (anchor_id, trace.cusum_first_fire)
    if earliest_fire is not None and earliest_fire[0] in options:
        return earliest_fire[0]

    min_agg: tuple[str, float] | None = None
    for anchor_id, trace in traces.items():
        if not trace.aggregates:
            continue
        m = min(trace.aggregates)
        if min_agg is None or m < min_agg[1]:
            min_agg = (anchor_id, m)
    if min_agg is not None and min_agg[0] in options:
        return min_agg[0]
    return options[0]


def _render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    runs = _cached_runs()
    if not runs:
        st.info(EMPTY_STATE_MSG)
        return

    run_labels = {snap.label: snap.run_id for snap in runs}
    chosen_label = st.selectbox(
        "Drift run",
        options=list(run_labels.keys()),
        index=0,
    )
    run_id = run_labels[chosen_label]

    traces = _cached_traces(run_id)
    if not traces:
        st.info("This drift run has no recorded compliance scores.")
        return

    anchor_options = sorted(traces.keys())
    default_anchor = _select_default_anchor(traces, anchor_options)
    anchor_id = st.selectbox(
        "Anchor",
        options=anchor_options,
        index=anchor_options.index(default_anchor),
    )
    trace = traces[anchor_id]

    report = _cached_report(run_id)
    if report is None:
        st.warning(DEGRADED_BASELINE_MSG)

    # Plot.
    st.plotly_chart(_build_figure(trace), use_container_width=True)

    # Detection-latency table.
    if report is not None:
        st.subheader("Detection latency")
        st.dataframe(
            [_row_to_table_dict(r) for r in report.rows],
            use_container_width=True,
            hide_index=True,
        )

        # Headline callout.
        headline = report.headline_lead_sessions
        if report.earliest_cusum_fire is None:
            st.markdown(
                "ℹ️ No CUSUM fired across this run — drift may be too "
                "small for the configured threshold (k=4σ)."
            )
        elif report.earliest_violation is None:
            cusum_anchor, cusum_idx = report.earliest_cusum_fire
            st.markdown(
                f":green[**Lead time = +∞ sessions.**] "
                f"Earliest CUSUM fire ({cusum_anchor} @ S{cusum_idx}); "
                f"no anchor crossed the 0.5 violation threshold in this run."
            )
        else:
            cusum_anchor, cusum_idx = report.earliest_cusum_fire
            viol_anchor, viol_idx = report.earliest_violation
            color = "green" if headline and headline > 0 else "red"
            st.markdown(
                f":{color}[**Lead time = "
                f"{headline:+.0f} sessions.**] "
                f"Earliest CUSUM fire ({cusum_anchor} @ S{cusum_idx}) "
                f"vs earliest explicit violation ({viol_anchor} @ S{viol_idx})."
            )


_render()
