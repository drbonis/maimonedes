"""Phase 3 detection-latency rollup.

Pulls per-anchor scalar streams for one drift run, runs CUSUM and
EWMA, and computes the first-violation index. The result is a list
of `AnchorReportRow` plus aggregate footer numbers — the CLI and
dashboard format the same data structure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from maimonedes.monitor._baseline import (
    InsufficientBaseline,
    load_run_scalars,
)
from maimonedes.monitor.cusum import cusum_per_anchor
from maimonedes.monitor.ewma import ewma_per_anchor


@dataclass(frozen=True)
class AnchorReportRow:
    """One anchor's view of detection latency for a drift run."""

    anchor_id: str
    n_sessions: int
    n_baseline: int
    mean_baseline: float | None
    min_aggregate: float | None
    first_violation_session: int | None
    cusum_first_fire: int | None
    ewma_first_fire: int | None

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


@dataclass(frozen=True)
class DriftReport:
    rows: list[AnchorReportRow]
    earliest_cusum_fire: tuple[str, int] | None  # (anchor_id, session_index)
    earliest_violation: tuple[str, int] | None
    headline_lead_sessions: float | None  # violation - cusum, +inf if no violation

    @property
    def has_data(self) -> bool:
        return any(r.n_sessions > 0 for r in self.rows)

    @property
    def has_any_baseline(self) -> bool:
        return any(r.n_baseline >= 5 for r in self.rows)


class NoBaselineDataError(Exception):
    """Raised when a drift run has no baseline-stage sessions across any anchor."""


def build_report(
    run_id: int,
    *,
    violation_threshold: float = 0.5,
    k: float = 4.0,
    lambda_: float = 0.2,
    L: float = 3.0,
) -> DriftReport:
    """Assemble the detection-latency report for one drift run.

    Raises `NoBaselineDataError` when no anchor has at least 5
    baseline-stage samples — the caller (CLI) translates this to a
    clear exit-code-2 message rather than producing a half-formed
    report.
    """
    streams = load_run_scalars(run_id)
    if not streams:
        raise NoBaselineDataError(
            f"drift run {run_id} has no compliance scores"
        )

    cusum_traces = cusum_per_anchor(run_id, k=k)
    ewma_traces = ewma_per_anchor(run_id, lambda_=lambda_, L=L)

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
    )


__all__ = [
    "AnchorReportRow",
    "DriftReport",
    "NoBaselineDataError",
    "build_report",
]
