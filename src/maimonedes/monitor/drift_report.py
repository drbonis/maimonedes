"""Phase 3 detection-latency rollup.

Pulls per-anchor scalar streams for one drift run, runs CUSUM and
EWMA, and computes the first-violation index. The result is a list
of `AnchorReportRow` plus aggregate footer numbers — the CLI and
dashboard format the same data structure.

Phase 5 add-ons:
- Issue #50: when a `RiemannianMetric` is supplied via `metric=`,
  each row gains Euclidean / Riemannian displacement columns that
  compare how far the anchor's worst-session score sits from its
  baseline under flat L2 vs the learned geodesic-style integral.
- Issue #51: structural signals (decoupling + curvature) layer on
  top: decoupling fires when the per-axis covariance flips, curvature
  fires when the local metric becomes more anisotropic. Both are
  enabled with the same `metric=` flag (curvature needs two metric
  fits — passed as `metric_baseline=` and `metric_current=`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from maimonedes.monitor._baseline import (
    InsufficientBaseline,
    load_run_scalars,
)
from maimonedes.monitor.cusum import cusum_per_anchor
from maimonedes.monitor.curvature import (
    CurvatureResult,
    DEFAULT_H_CURVATURE,
    curvature_drift,
)
from maimonedes.monitor.decoupling import (
    DecouplingResult,
    decoupling_signal,
)
from maimonedes.monitor.ewma import ewma_per_anchor
from maimonedes.monitor.metric import (
    RiemannianMetric,
    euclidean_distance,
    riemannian_distance,
)
from maimonedes.storage.drift import scores_for_run


@dataclass(frozen=True)
class AnchorReportRow:
    """One anchor's view of detection latency for a drift run.

    `euclidean_displacement` / `riemannian_displacement` are populated
    only when `build_report(..., metric=...)` is called; `decoupling`
    and `curvature` are populated only when their respective inputs
    are supplied. All three default to None so existing callers stay
    unchanged.
    """

    anchor_id: str
    n_sessions: int
    n_baseline: int
    mean_baseline: float | None
    min_aggregate: float | None
    first_violation_session: int | None
    cusum_first_fire: int | None
    ewma_first_fire: int | None
    euclidean_displacement: float | None = None
    riemannian_displacement: float | None = None
    decoupling: DecouplingResult | None = None
    curvature: CurvatureResult | None = None

    @property
    def detector_skipped(self) -> bool:
        """True when the anchor has no detector traces (short baseline)."""
        return self.cusum_first_fire is None and self.ewma_first_fire is None and self.n_baseline < 5

    @property
    def cusum_lead_sessions(self) -> float | None:
        """`first_violation_session - cusum_first_fire`.

        - `+inf` if violation never occurred but CUSUM fired.
        - `None` if CUSUM didn't fire (cannot compute lead time).
        """
        if self.cusum_first_fire is None:
            return None
        if self.first_violation_session is None:
            return math.inf
        return float(self.first_violation_session - self.cusum_first_fire)

    @property
    def decoupling_fired(self) -> bool:
        return self.decoupling is not None and self.decoupling.signal_fired

    @property
    def curvature_fired(self) -> bool:
        return self.curvature is not None and self.curvature.signal_fired


@dataclass(frozen=True)
class DriftReport:
    rows: list[AnchorReportRow]
    earliest_cusum_fire: tuple[str, int] | None  # (anchor_id, session_index)
    earliest_violation: tuple[str, int] | None
    headline_lead_sessions: float | None  # violation - cusum, +inf if no violation
    metric_id: int | None = None  # populated when --metric was supplied

    @property
    def has_data(self) -> bool:
        return any(r.n_sessions > 0 for r in self.rows)

    @property
    def has_any_baseline(self) -> bool:
        return any(r.n_baseline >= 5 for r in self.rows)

    @property
    def has_riemannian_columns(self) -> bool:
        return any(r.riemannian_displacement is not None for r in self.rows)

    @property
    def has_structural_columns(self) -> bool:
        return any(
            r.decoupling is not None or r.curvature is not None
            for r in self.rows
        )


class NoBaselineDataError(Exception):
    """Raised when a drift run has no baseline-stage sessions across any anchor."""


def _compute_displacement_columns(
    run_id: int,
    *,
    metric: RiemannianMetric,
) -> dict[str, tuple[float | None, float | None]]:
    """Per-anchor (Euclidean, Riemannian) displacements for the worst score.

    Reads the full per-axis stream for the run (via `scores_for_run`),
    builds each anchor's mean baseline position c_baseline (over its
    early scores — proxy for the baseline stage), then locates the
    worst (lowest aggregate) score and computes both distances from
    c_baseline to that worst score's per-axis position.
    """
    streams_full = scores_for_run(run_id)
    axes = list(metric.sub_condition_ids) if metric.sub_condition_ids else None
    out: dict[str, tuple[float | None, float | None]] = {}
    for anchor_id, scores in streams_full.items():
        if not scores:
            out[anchor_id] = (None, None)
            continue
        per_axis_keys = axes or sorted(scores[0].per_sub_condition.keys())
        if not per_axis_keys:
            out[anchor_id] = (None, None)
            continue
        # Baseline window: the highest-aggregate sample is closest to
        # the uncontaminated state (matches localizer's baseline pick).
        sorted_by_agg = sorted(scores, key=lambda s: s.aggregate, reverse=True)
        baseline_pool = sorted_by_agg[: max(1, len(sorted_by_agg) // 2)]
        c_baseline = np.mean(
            [
                [s.per_sub_condition.get(a, 0.0) for a in per_axis_keys]
                for s in baseline_pool
            ],
            axis=0,
        )
        worst = min(scores, key=lambda s: s.aggregate)
        c_worst = np.asarray(
            [worst.per_sub_condition.get(a, 0.0) for a in per_axis_keys],
            dtype=np.float64,
        )
        if c_worst.size != metric.k:
            out[anchor_id] = (None, None)
            continue
        d_eucl = euclidean_distance(c_baseline, c_worst)
        d_riem = riemannian_distance(metric, c_baseline, c_worst)
        out[anchor_id] = (d_eucl, d_riem)
    return out


def _structural_signals_for_anchor(
    anchor_id: str,
    *,
    axis_ids: tuple[str, ...] | list[str] | None,
    metric_baseline: RiemannianMetric | None,
    metric_current: RiemannianMetric | None,
    anchor_position: np.ndarray | None,
    decoupling_baseline_window: int,
    decoupling_current_window: int,
    h_curvature: float,
    h_decoupling: float | None,
) -> tuple[DecouplingResult | None, CurvatureResult | None]:
    """Run decoupling + curvature for one anchor, returning (None, None) when inputs missing."""
    decoupling: DecouplingResult | None = None
    curvature: CurvatureResult | None = None
    if axis_ids:
        try:
            decoupling = decoupling_signal(
                anchor_id=anchor_id,
                axis_ids=axis_ids,
                baseline_window=decoupling_baseline_window,
                current_window=decoupling_current_window,
                h_decoupling=h_decoupling,
            )
        except ValueError:
            # Insufficient samples — leave as None.
            decoupling = None
    if (
        metric_baseline is not None
        and metric_current is not None
        and anchor_position is not None
    ):
        try:
            curvature = curvature_drift(
                anchor_id=anchor_id,
                metric_baseline=metric_baseline,
                metric_current=metric_current,
                c=anchor_position,
                h_curvature=h_curvature,
            )
        except ValueError:
            curvature = None
    return decoupling, curvature


def build_report(
    run_id: int,
    *,
    violation_threshold: float = 0.5,
    k: float = 4.0,
    lambda_: float = 0.2,
    L: float = 3.0,
    metric: RiemannianMetric | None = None,
    metric_id: int | None = None,
    metric_baseline: RiemannianMetric | None = None,
    metric_current: RiemannianMetric | None = None,
    decoupling_baseline_window: int = 50,
    decoupling_current_window: int = 20,
    h_curvature: float = DEFAULT_H_CURVATURE,
    h_decoupling: float | None = None,
) -> DriftReport:
    """Assemble the detection-latency report for one drift run.

    Raises `NoBaselineDataError` when no anchor has at least 5
    baseline-stage samples — the caller (CLI) translates this to a
    clear exit-code-2 message rather than producing a half-formed
    report.

    Optional add-ons:
    - `metric=...` (issue #50): per-row Euclidean/Riemannian
      displacement columns relative to the worst-session score.
    - `metric_baseline=` + `metric_current=` (issue #51): per-row
      curvature-drift signal — fires when the local condition number
      jumped between fits.
    - decoupling signal (issue #51): always runs when `metric` is
      supplied (uses its `sub_condition_ids` for the canonical axis
      order); disabled by passing `metric=None`.
    """
    streams = load_run_scalars(run_id)
    if not streams:
        raise NoBaselineDataError(
            f"drift run {run_id} has no compliance scores"
        )

    cusum_traces = cusum_per_anchor(run_id, k=k)
    ewma_traces = ewma_per_anchor(run_id, lambda_=lambda_, L=L)

    displacements: dict[str, tuple[float | None, float | None]] = {}
    if metric is not None:
        displacements = _compute_displacement_columns(run_id, metric=metric)

    structural_axes: tuple[str, ...] | None = None
    if metric is not None and metric.sub_condition_ids:
        structural_axes = tuple(metric.sub_condition_ids)
    elif metric_current is not None and metric_current.sub_condition_ids:
        structural_axes = tuple(metric_current.sub_condition_ids)

    full_streams = scores_for_run(run_id) if structural_axes is not None else {}

    rows: list[AnchorReportRow] = []
    any_baseline = False
    for anchor_id in sorted(streams):
        triples = streams[anchor_id]
        baselines = [agg for (_idx, stage, agg) in triples if stage == "baseline"]
        n_baseline = len(baselines)
        if n_baseline >= 5:
            any_baseline = True

        all_aggs = [agg for (_idx, _stage, agg) in triples]
        first_violation: int | None = None
        for idx, _stage, agg in triples:
            if agg < violation_threshold:
                first_violation = idx
                break

        cusum_states = cusum_traces.get(anchor_id, [])
        cusum_fire = next((s.session_index for s in cusum_states if s.fired), None)
        ewma_states = ewma_traces.get(anchor_id, [])
        ewma_fire = next((s.session_index for s in ewma_states if s.fired), None)

        eucl_disp, riem_disp = displacements.get(anchor_id, (None, None))

        anchor_position: np.ndarray | None = None
        if structural_axes is not None:
            anchor_scores = full_streams.get(anchor_id, [])
            if anchor_scores:
                # Mean across baseline-stage observations — proxied by
                # the highest-aggregate window (matches the localizer).
                sorted_by_agg = sorted(
                    anchor_scores, key=lambda s: s.aggregate, reverse=True
                )
                head = sorted_by_agg[: max(1, len(sorted_by_agg) // 2)]
                anchor_position = np.mean(
                    [
                        [s.per_sub_condition.get(a, 0.0) for a in structural_axes]
                        for s in head
                    ],
                    axis=0,
                )

        decoupling_res, curvature_res = _structural_signals_for_anchor(
            anchor_id,
            axis_ids=structural_axes,
            metric_baseline=metric_baseline,
            metric_current=metric_current,
            anchor_position=anchor_position,
            decoupling_baseline_window=decoupling_baseline_window,
            decoupling_current_window=decoupling_current_window,
            h_curvature=h_curvature,
            h_decoupling=h_decoupling,
        )

        rows.append(
            AnchorReportRow(
                anchor_id=anchor_id,
                n_sessions=len(triples),
                n_baseline=n_baseline,
                mean_baseline=(
                    sum(baselines) / n_baseline if n_baseline > 0 else None
                ),
                min_aggregate=min(all_aggs) if all_aggs else None,
                first_violation_session=first_violation,
                cusum_first_fire=cusum_fire,
                ewma_first_fire=ewma_fire,
                euclidean_displacement=eucl_disp,
                riemannian_displacement=riem_disp,
                decoupling=decoupling_res,
                curvature=curvature_res,
            )
        )

    if not any_baseline:
        raise NoBaselineDataError(
            f"drift run {run_id}: no anchor has 5+ baseline samples"
        )

    earliest_cusum: tuple[str, int] | None = None
    for r in rows:
        if r.cusum_first_fire is None:
            continue
        if earliest_cusum is None or r.cusum_first_fire < earliest_cusum[1]:
            earliest_cusum = (r.anchor_id, r.cusum_first_fire)

    earliest_violation: tuple[str, int] | None = None
    for r in rows:
        if r.first_violation_session is None:
            continue
        if (
            earliest_violation is None
            or r.first_violation_session < earliest_violation[1]
        ):
            earliest_violation = (r.anchor_id, r.first_violation_session)

    headline: float | None
    if earliest_cusum is None:
        headline = None
    elif earliest_violation is None:
        headline = math.inf
    else:
        headline = float(earliest_violation[1] - earliest_cusum[1])

    return DriftReport(
        rows=rows,
        earliest_cusum_fire=earliest_cusum,
        earliest_violation=earliest_violation,
        headline_lead_sessions=headline,
        metric_id=metric_id,
    )


__all__ = [
    "AnchorReportRow",
    "DriftReport",
    "NoBaselineDataError",
    "build_report",
]
