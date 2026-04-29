"""Phase 4 localizer.

Ranks anchors by a rubric-weighted Euclidean distance to the per-axis
policy boundary so the closed-loop orchestrator targets the anchors
that have actually moved into violation territory. Per the locked
decision in the planning thread, axis thresholds are uniform 0.5
(midpoint of normalised score) and the rubric weights enter via a
weighted L2 distance metric. Only axes BELOW their threshold
contribute — overcompliance does not increase the distance.

Distance:
    d(s) = sqrt( Σ w_i · max(0, τ_i - s_i)^2 )
    τ_i = 0.5
    w_i = rubric.sub_conditions[i].weight
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy
from maimonedes.storage.drift import scores_for_run


DEFAULT_AXIS_THRESHOLD = 0.5


@dataclass(frozen=True)
class LocalizationResult:
    """One anchor's distance-ranked view of a run."""

    anchor_id: str
    distance: float
    worst_score: ComplianceScore
    baseline_score: ComplianceScore | None


def axis_thresholds(policy: Policy) -> dict[str, float]:
    """Per-axis violation midpoint. v1: uniform 0.5 across every axis.

    The function exists so the per-axis-threshold variant (deferred)
    can plug in without changing call sites.
    """
    return {s.id: DEFAULT_AXIS_THRESHOLD for s in policy.rubric.sub_conditions}


def axis_weights(policy: Policy) -> dict[str, float]:
    """Passthrough of `rubric.sub_conditions[i].weight`, keyed by id."""
    return {s.id: s.weight for s in policy.rubric.sub_conditions}


def boundary_distance(
    score: ComplianceScore,
    *,
    thresholds: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Rubric-weighted L2 distance from a score vector to the boundary.

    Axes at or above their threshold contribute zero. Axes below their
    threshold contribute `w_i · (τ_i - s_i)^2`. Distance is the sqrt
    of the sum.
    """
    accum = 0.0
    for sub_id, threshold in thresholds.items():
        s_i = score.per_sub_condition.get(sub_id, 0.0)
        margin = threshold - s_i
        if margin <= 0:
            continue
        w_i = weights.get(sub_id, 0.0)
        accum += w_i * margin * margin
    return math.sqrt(accum)


def worst_session_score(scores: Sequence[ComplianceScore]) -> ComplianceScore:
    """Return the score with the lowest aggregate (the localizer's "now")."""
    if not scores:
        raise ValueError("worst_session_score: scores must be non-empty")
    return min(scores, key=lambda s: s.aggregate)


def _baseline_anchor_score(scores: Sequence[ComplianceScore]) -> ComplianceScore | None:
    """Pick the anchor's most recent baseline-stage observation.

    `scores_for_run` does not carry stage labels on the row itself,
    so we approximate "baseline" as "the highest aggregate observed
    in this anchor's stream" — under the contamination schedule
    baseline-stage scores ARE the highest the anchor reaches. This
    matches the visual intent of the Phase 4 "before/after" plot:
    `baseline` is the level the anchor was scoring before drift bit.
    """
    if not scores:
        return None
    return max(scores, key=lambda s: s.aggregate)


def localize(
    run_id: int,
    *,
    policy: Policy,
    top_k: int | None = None,
) -> list[LocalizationResult]:
    """Rank anchors by descending boundary distance for one drift run.

    Anchors with no scores at all are omitted from the result rather
    than listed with distance 0 (which would suggest a safe anchor
    rather than an absent one).
    """
    streams = scores_for_run(run_id)
    if not streams:
        return []

    thresholds = axis_thresholds(policy)
    weights = axis_weights(policy)

    results: list[LocalizationResult] = []
    for anchor_id, scores in streams.items():
        if not scores:
            continue
        worst = worst_session_score(scores)
        baseline = _baseline_anchor_score(scores)
        distance = boundary_distance(
            worst, thresholds=thresholds, weights=weights
        )
        results.append(
            LocalizationResult(
                anchor_id=anchor_id,
                distance=distance,
                worst_score=worst,
                baseline_score=baseline,
            )
        )

    results.sort(key=lambda r: r.distance, reverse=True)
    if top_k is not None:
        results = results[:top_k]
    return results


__all__ = [
    "DEFAULT_AXIS_THRESHOLD",
    "LocalizationResult",
    "axis_thresholds",
    "axis_weights",
    "boundary_distance",
    "localize",
    "worst_session_score",
]
