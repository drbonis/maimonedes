"""Phase 4 page: recovery before/after bars + feedback quote blocks.

Headline deliverable. Per anchor: three grouped bars showing the
baseline (parent drift baseline-stage mean), the drift-low (parent
drift's worst aggregate), and the recovery (post-feedback aggregate).
The synthesized feedback text is rendered as a blockquote in an
expander panel co-located with its empirical evidence so the
natural-language recommendation is right next to the numerical
result.

Reads exclusively via helpers in `storage.recovery`, `storage.drift`,
`monitor.recovery_report`, and `feedback.contrastive`. No SQL or
pydantic in this page body.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from maimonedes.dashboard._recovery_snapshots import (
    AnchorBarsSnapshot,
    RecoveryRunSnapshot,
)
from maimonedes.feedback.contrastive import (
    fragility_pair,
    temporal_pair,
)
from maimonedes.monitor.recovery_report import (
    NoRecoveryDataError,
    OrphanRecoveryRunError,
    RecoveryReport,
    build_report,
)
from maimonedes.storage.drift import scores_for_run
from maimonedes.storage.recovery import (
    feedbacks_for_run,
    list_recovery_runs,
    scores_for_recovery_run,
)


PAGE_TITLE = "Recovery — Phase 4"
EMPTY_STATE_MSG = (
    "No recovery runs recorded yet. Run `maimonedes apply-feedback "
    "<drift-run-id>` to populate this dashboard."
)
DEGRADED_RUN_MSG = (
    "This recovery run has no scored anchors yet. The synthesized "
    "feedback texts are still shown below."
)


@st.cache_data(ttl=10)
def _cached_runs() -> list[RecoveryRunSnapshot]:
    snapshots: list[RecoveryRunSnapshot] = []
    for run in list_recovery_runs():
        started = run.get("started_at")
        notes = run.get("notes")
        kind = str(run.get("contrastive_kind", ""))
        label_parts: list[str] = [f"recovery #{run['id']}"]
        label_parts.append(f"parent drift #{run['parent_drift_run_id']}")
        label_parts.append(f"kind={kind}")
        if isinstance(started, datetime):
            label_parts.append(started.strftime("%Y-%m-%d %H:%M"))
        if notes:
            label_parts.append(str(notes))
        snapshots.append(
            RecoveryRunSnapshot(
                recovery_run_id=int(run["id"]),  # type: ignore[arg-type]
                parent_drift_run_id=int(run["parent_drift_run_id"]),  # type: ignore[arg-type]
                started_at=(
                    started if isinstance(started, datetime) else None
                ),
                notes=str(notes) if notes else None,
                contrastive_kind=kind,
                label="  ·  ".join(label_parts),
            )
        )
    return snapshots


@st.cache_data(ttl=10)
def _cached_report(recovery_run_id: int) -> RecoveryReport | None:
    try:
        return build_report(recovery_run_id)
    except (NoRecoveryDataError, OrphanRecoveryRunError):
        return None


@st.cache_data(ttl=10)
def _cached_bars(
    recovery_run_id: int,
    parent_drift_run_id: int,
    contrastive_kind: str,
) -> list[AnchorBarsSnapshot]:
    """Build the per-anchor bar trio + the contrastive pair texts.

    `baseline` = mean of the parent drift's baseline-stage scores for
    that anchor (a robust single-bar summary of "how it scored before
    drift bit"). `drift_low` = the parent drift's worst aggregate.
    `recovery` = the latest post-feedback anchor aggregate.
    """
    drift_streams = scores_for_run(parent_drift_run_id)
    recovery_streams = scores_for_recovery_run(recovery_run_id)
    feedbacks = feedbacks_for_run(recovery_run_id)
    anchor_ids = sorted(set(feedbacks) | set(recovery_streams))

    out: list[AnchorBarsSnapshot] = []
    for anchor_id in anchor_ids:
        drift_scores = drift_streams.get(anchor_id, [])
        baseline_scores = [
            s.aggregate
            for s in drift_scores
            # Heuristic: highest scores are baseline-stage. Reuses the
            # localizer's stage-label-free convention so we don't need
            # to thread DriftSession joins through the dashboard.
            if s.aggregate >= max((x.aggregate for x in drift_scores), default=0) - 0.05
        ]
        baseline = (
            sum(baseline_scores) / len(baseline_scores) if baseline_scores else None
        )
        drift_low = (
            min(s.aggregate for s in drift_scores) if drift_scores else None
        )
        recovery_anchor_scores = [
            s for s in recovery_streams.get(anchor_id, []) if s.probe_role == "anchor"
        ]
        recovery = (
            recovery_anchor_scores[-1].aggregate
            if recovery_anchor_scores
            else None
        )

        feedback_text = (
            feedbacks[anchor_id].feedback_text
            if anchor_id in feedbacks
            else None
        )

        # Pull the contrastive pair texts so the expander panel can
        # show the safe / near-boundary outputs that motivated the
        # feedback.
        if contrastive_kind == "fragility":
            from maimonedes.core.policy import load_policy
            from pathlib import Path

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
            try:
                policy = load_policy(POLICY_PATH, RUBRIC_PATH)
                pair = fragility_pair(anchor_id, policy=policy)
            except Exception:
                pair = None
        else:
            pair = temporal_pair(anchor_id, drift_run_id=parent_drift_run_id)

        safe_text = pair.safe_text if pair is not None else None
        near_boundary_text = (
            pair.near_boundary_text if pair is not None else None
        )

        out.append(
            AnchorBarsSnapshot(
                anchor_id=anchor_id,
                baseline=baseline,
                drift_low=drift_low,
                recovery=recovery,
                feedback_text=feedback_text,
                safe_text=safe_text,
                near_boundary_text=near_boundary_text,
            )
        )
    return out


def _bars_dataframe(bars: list[AnchorBarsSnapshot]) -> pd.DataFrame:
    rows = []
    for b in bars:
        rows.append({
            "anchor": b.anchor_id,
            "baseline": b.baseline,
            "drift_low": b.drift_low,
            "recovery": b.recovery,
        })
    df = pd.DataFrame(rows)
    if "anchor" in df.columns:
        df = df.set_index("anchor")
    return df


def _verdict_color(verdict: str) -> str:
    return {"closed_loop": "green", "partial": "orange", "failed": "red"}.get(
        verdict, "grey"
    )


def _render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    runs = _cached_runs()
    if not runs:
        st.info(EMPTY_STATE_MSG)
        return

    run_labels = {snap.label: snap for snap in runs}
    chosen_label = st.selectbox(
        "Recovery run",
        options=list(run_labels.keys()),
        index=0,
    )
    snap = run_labels[chosen_label]

    report = _cached_report(snap.recovery_run_id)
    if report is None:
        st.warning(DEGRADED_RUN_MSG)

    bars = _cached_bars(
        snap.recovery_run_id,
        snap.parent_drift_run_id,
        snap.contrastive_kind,
    )
    if not bars:
        st.info("This recovery run has no per-anchor data to plot.")
        return

    has_any_recovery = any(b.recovery is not None for b in bars)
    if not has_any_recovery:
        st.warning(DEGRADED_RUN_MSG)
    else:
        st.subheader("Before / after compliance")
        bars_df = _bars_dataframe(bars)
        st.bar_chart(bars_df, use_container_width=True)
        # 0.5 violation reference line — a horizontal line on a bar chart
        # isn't natively supported by st.bar_chart, so we render the
        # threshold as a caption right below the chart for now.
        st.caption(
            "Violation threshold: 0.5 (anchors above this line are recovered)."
        )

    st.subheader("Per-anchor detail")
    for b in bars:
        title = f"{b.anchor_id}"
        if b.recovery is not None and b.drift_low is not None:
            delta = b.recovery - b.drift_low
            title += f"  Δ={delta:+.3f}"
        with st.expander(title):
            cols = st.columns(3)
            cols[0].metric(
                "baseline",
                f"{b.baseline:.3f}" if b.baseline is not None else "—",
            )
            cols[1].metric(
                "drift_low",
                f"{b.drift_low:.3f}" if b.drift_low is not None else "—",
            )
            cols[2].metric(
                "recovery",
                f"{b.recovery:.3f}" if b.recovery is not None else "—",
            )

            if b.feedback_text:
                st.markdown(f"> {b.feedback_text}")
            else:
                st.markdown("(no feedback recorded for this anchor)")

            if b.safe_text or b.near_boundary_text:
                with st.expander("Contrastive pair (collapsed)"):
                    if b.safe_text:
                        st.markdown("**Safe output:**")
                        st.code(b.safe_text)
                    if b.near_boundary_text:
                        st.markdown("**Near-boundary output:**")
                        st.code(b.near_boundary_text)

    if report is not None:
        verdict_color = _verdict_color(report.verdict)
        delta = report.mean_delta_toward_baseline
        delta_str = f"{delta:+.3f}" if delta is not None else "n/a"
        st.markdown(
            f":{verdict_color}[**verdict: {report.verdict}**]  ·  "
            f"mean Δ-toward-baseline = {delta_str}  ·  "
            f"recovered = {report.recovered_count}/{report.total_anchors}"
        )


_render()
