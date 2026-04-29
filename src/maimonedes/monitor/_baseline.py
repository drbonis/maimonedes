"""Shared helpers for the Phase 3 detectors.

Both CUSUM (#23) and EWMA (#24) tune their thresholds from the
baseline window of a drift run — the sessions tagged
`stage_label = "baseline"`. The detectors are otherwise independent;
only the loader and the (μ₀, σ) computation are shared.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from maimonedes.core.compliance import ComplianceScore
from maimonedes.storage.drift import (
    list_drift_sessions,
    scores_for_run,
)


MIN_BASELINE_N = 5
FLAT_SIGMA_FALLBACK = 0.01


class InsufficientBaseline(Exception):
    """Raised when an anchor has fewer than `MIN_BASELINE_N` baseline scores."""


def compute_baseline_stats(
    values: Sequence[float],
    *,
    min_n: int = MIN_BASELINE_N,
    label: str = "<unknown>",
) -> tuple[float, float]:
    """Return `(mean, sigma)` of the baseline window.

    Raises `InsufficientBaseline` if fewer than `min_n` samples. Falls
    back to `FLAT_SIGMA_FALLBACK` and emits a warning when σ is
    numerically zero (a degenerate case the detectors must still
    define behaviour for).
    """
    if len(values) < min_n:
        raise InsufficientBaseline(
            f"{label}: need at least {min_n} baseline samples, got {len(values)}"
        )
    arr = np.asarray(values, dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if sigma == 0.0:
        warnings.warn(
            f"{label}: baseline sigma is zero; falling back to "
            f"{FLAT_SIGMA_FALLBACK} so detector statistics remain defined",
            stacklevel=2,
        )
        sigma = FLAT_SIGMA_FALLBACK
    return mu, sigma


def load_run_scalars(
    run_id: int,
) -> dict[str, list[tuple[int, str, float]]]:
    """Per-anchor `(session_index, stage_label, aggregate)` triples.

    Result is keyed by `anchor_id`; each list is ordered by
    `session_index`. Anchors with no scores are absent from the
    mapping. Used by both detectors to pull the streams they monitor.
    """
    scores: dict[str, list[ComplianceScore]] = scores_for_run(run_id)
    if not scores:
        return {}

    sessions = list_drift_sessions(run_id)
    stage_by_id: dict[int, tuple[int, str]] = {
        s.id: (s.session_index, s.stage_label) for s in sessions
    }

    out: dict[str, list[tuple[int, str, float]]] = {}
    for anchor_id, anchor_scores in scores.items():
        triples: list[tuple[int, str, float]] = []
        for score in anchor_scores:
            if score.drift_session_id is None:
                continue
            sess = stage_by_id.get(score.drift_session_id)
            if sess is None:
                continue
            session_index, stage_label = sess
            triples.append((session_index, stage_label, score.aggregate))
        triples.sort(key=lambda t: t[0])
        if triples:
            out[anchor_id] = triples
    return out


def split_baseline(
    triples: Sequence[tuple[int, str, float]],
) -> tuple[list[float], list[float]]:
    """Return `(baseline_values, full_values)` for one anchor's stream.

    `baseline_values` is the subset where `stage_label == "baseline"`,
    preserving session order. `full_values` is every aggregate in the
    stream — the caller runs the detector across this full sequence.
    """
    baseline = [agg for (_idx, stage, agg) in triples if stage == "baseline"]
    full = [agg for (_idx, _stage, agg) in triples]
    return baseline, full


__all__ = [
    "FLAT_SIGMA_FALLBACK",
    "InsufficientBaseline",
    "MIN_BASELINE_N",
    "compute_baseline_stats",
    "load_run_scalars",
    "split_baseline",
]
