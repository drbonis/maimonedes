"""Phase 4 before/after rollup.

Pulls per-anchor pre-feedback worst score (from the parent drift run)
and post-feedback aggregate (from the recovery run), computes delta-
toward-baseline, returns a `RecoveryReport`. The CLI and dashboard
both format the same data structure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.feedback import Feedback
from maimonedes.storage.drift import (
    get_drift_run,
    list_drift_sessions,
    scores_for_run,
)
from maimonedes.storage.recovery import (
    feedbacks_for_run,
    get_recovery_run,
    scores_for_recovery_run,
)


VIOLATION_THRESHOLD = 0.5
RecoveryVerdict = Literal["closed_loop", "partial", "failed"]


@dataclass(frozen=True)
class RecoveryAnchorRow:
    """One anchor's before/after view of a recovery run."""

    anchor_id: str
    pre_worst_aggregate: float | None
    pre_worst_session: int | None
    post_aggregate: float | None
    feedback: Feedback | None

    @property
    def delta_toward_baseline(self) -> float | None:
        if self.pre_worst_aggregate is None or self.post_aggregate is None:
            return None
        return self.post_aggregate - self.pre_worst_aggregate

    @property
    def recovered(self) -> bool:
        return (
            self.post_aggregate is not None
            and self.post_aggregate >= VIOLATION_THRESHOLD
        )


@dataclass(frozen=True)
class RecoveryReport:
    rows: list[RecoveryAnchorRow]
    parent_drift_run_id: int
    contrastive_kind: str
    mean_delta_toward_baseline: float | None
    recovered_count: int
    verdict: RecoveryVerdict

    @property
    def total_anchors(self) -> int:
        return len(self.rows)


class NoRecoveryDataError(Exception):
    """Raised when a recovery run has no scored anchors."""


class OrphanRecoveryRunError(Exception):
    """Raised when a recovery run's parent drift run has been deleted."""


def _pre_worst(
    scores: list[ComplianceScore],
    session_index_by_id: dict[int, int],
) -> tuple[float | None, int | None]:
    """Return (worst_aggregate, session_index) — None when stream is empty."""
    if not scores:
        return None, None
    worst = min(scores, key=lambda s: s.aggregate)
    session_index: int | None = None
    if worst.drift_session_id is not None:
        session_index = session_index_by_id.get(worst.drift_session_id)
    return worst.aggregate, session_index


def build_report(recovery_run_id: int) -> RecoveryReport:
    """Assemble the before/after report for one recovery run.

    Raises `OrphanRecoveryRunError` when the parent drift run cannot
    be found, and `NoRecoveryDataError` when the recovery run has
    zero compliance scores.
    """
    run = get_recovery_run(recovery_run_id)
    if run is None:
        raise NoRecoveryDataError(
            f"recovery_run {recovery_run_id} not found"
        )
    parent_drift_run_id = int(run["parent_drift_run_id"])  # type: ignore[arg-type]

    if get_drift_run(parent_drift_run_id) is None:
        raise OrphanRecoveryRunError(
            f"recovery_run {recovery_run_id}: parent drift_run "
            f"{parent_drift_run_id} not found"
        )
    drift_streams = scores_for_run(parent_drift_run_id)
    drift_sessions = list_drift_sessions(parent_drift_run_id)
    session_index_by_id = {s.id: s.session_index for s in drift_sessions}

    recovery_streams = scores_for_recovery_run(recovery_run_id)
    feedbacks = feedbacks_for_run(recovery_run_id)

    # Build a row per anchor that has either a feedback or a recovery score.
    anchor_ids = sorted(set(feedbacks) | set(recovery_streams))
    if not anchor_ids:
        raise NoRecoveryDataError(
            f"recovery_run {recovery_run_id} has no scored anchors"
        )

    rows: list[RecoveryAnchorRow] = []
    deltas: list[float] = []
    recovered_count = 0
    for anchor_id in anchor_ids:
        drift_scores = drift_streams.get(anchor_id, [])
        pre_aggregate, pre_session = _pre_worst(drift_scores, session_index_by_id)

        recovery_anchor_scores = [
            s for s in recovery_streams.get(anchor_id, []) if s.probe_role == "anchor"
        ]
        post_aggregate = (
            recovery_anchor_scores[-1].aggregate
            if recovery_anchor_scores
            else None
        )
        row = RecoveryAnchorRow(
            anchor_id=anchor_id,
            pre_worst_aggregate=pre_aggregate,
            pre_worst_session=pre_session,
            post_aggregate=post_aggregate,
            feedback=feedbacks.get(anchor_id),
        )
        rows.append(row)
        delta = row.delta_toward_baseline
        if delta is not None:
            deltas.append(delta)
        if row.recovered:
            recovered_count += 1

    mean_delta = sum(deltas) / len(deltas) if deltas else None

    if mean_delta is None:
        verdict: RecoveryVerdict = "failed"
    elif mean_delta > 0 and recovered_count > 0:
        verdict = "closed_loop"
    elif mean_delta > 0:
        verdict = "partial"
    else:
        verdict = "failed"

    return RecoveryReport(
        rows=rows,
        parent_drift_run_id=parent_drift_run_id,
        contrastive_kind=str(run["contrastive_kind"]),
        mean_delta_toward_baseline=mean_delta,
        recovered_count=recovered_count,
        verdict=verdict,
    )


__all__ = [
    "NoRecoveryDataError",
    "OrphanRecoveryRunError",
    "RecoveryAnchorRow",
    "RecoveryReport",
    "RecoveryVerdict",
    "VIOLATION_THRESHOLD",
    "build_report",
]
