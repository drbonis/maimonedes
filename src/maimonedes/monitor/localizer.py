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

Riemannian variant (issue #50): when a `RiemannianMetric` is supplied
via `metric=...`, the boundary distance switches to the Riemannian
length from the score's current position to its axis-wise projection
on the boundary box (still using the per-axis midpoint thresholds).
Local geometry comes from the learned metric tensor; v2 can swap in
a true geodesic ODE for off-diagonal-coupled boundaries.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy
from maimonedes.monitor.metric import RiemannianMetric, riemannian_distance
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
    metric: RiemannianMetric | None = None,
) -> float:
    """Rubric-weighted distance from a score vector to the boundary.

    Default (`metric=None`): rubric-weighted Euclidean L2 — axes at or
    above their threshold contribute zero, axes below contribute
    `w_i · (τ_i - s_i)^2`. Distance is the sqrt of the sum.

    Riemannian (`metric` set): integrate the learned metric along a
    straight-line path in compliance space from the score's current
    position to its axis-wise projection on the boundary box. Weights
    are folded into the path's per-axis displacement so an axis's
    relative importance still drives the magnitude.
    """
    if metric is None:
        accum = 0.0
        for sub_id, threshold in thresholds.items():
            s_i = score.per_sub_condition.get(sub_id, 0.0)
            margin = threshold - s_i
            if margin <= 0:
                continue
            w_i = weights.get(sub_id, 0.0)
            accum += w_i * margin * margin
        return math.sqrt(accum)

    # Riemannian: project onto the boundary box (only axes that crossed
    # the threshold contribute), apply weights to the displacement, and
    # measure path length under the learned metric.
    axes = sorted(thresholds.keys())
    if metric.sub_condition_ids:
        # Keep the metric's training axis order so g(c) is evaluated on
        # vectors aligned with the original Cholesky parameters.
        axes = [a for a in metric.sub_condition_ids if a in thresholds]
        if not axes:
            return 0.0
    c_now = np.asarray(
        [score.per_sub_condition.get(a, 0.0) for a in axes],
        dtype=np.float64,
    )
    c_target = c_now.copy()
    has_violation = False
    for i, sub_id in enumerate(axes):
        threshold = thresholds[sub_id]
        margin = threshold - c_now[i]
        if margin > 0:
            has_violation = True
            w_i = weights.get(sub_id, 1.0)
            # Weight scales the *amount* we move toward the boundary;
            # higher-weight axes pull harder on the path.
            c_target[i] = c_now[i] + math.sqrt(max(w_i, 0.0)) * margin
    if not has_violation:
        return 0.0
    return riemannian_distance(metric, c_now, c_target)


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
    metric: RiemannianMetric | None = None,
) -> list[LocalizationResult]:
    """Rank anchors by descending boundary distance for one drift run.

    Anchors with no scores at all are omitted from the result rather
    than listed with distance 0 (which would suggest a safe anchor
    rather than an absent one). When `metric` is supplied, the per-
    anchor distance is geodesic-style (Riemannian) instead of flat;
    ranking can change because the same Euclidean displacement can
    correspond to wildly different Riemannian distances depending on
    where the score sits in compliance space.
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
            worst, thresholds=thresholds, weights=weights, metric=metric
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
