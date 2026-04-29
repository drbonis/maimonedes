"""Phase 3 lower-side CUSUM detector.

Page test, lower-side only. We monitor the scalar `aggregate`
compliance score per anchor across the drift run; the contamination
schedule pushes scores downward, so the upper-side branch would only
add false alarms. Threshold `h = k·σ` with `k = 4` by default; `K`
(slack) defaults to `0.5σ`. Both σ and μ₀ are estimated from the
baseline window (`stage_label == "baseline"`) of the run itself —
that is the only stage where the supervised system is uncontaminated.

Recurrence:
    S_lo[t] = max(0, S_lo[t-1] + (μ₀ - K) - x[t])

Fires when `S_lo[t] >= h`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from maimonedes.monitor._baseline import (
    InsufficientBaseline,  # re-exported for callers
    compute_baseline_stats,
    load_run_scalars,
    split_baseline,
)


@dataclass(frozen=True)
class CusumState:
    """One step of the CUSUM trace; safe to plot or persist."""

    session_index: int
    statistic: float
    fired: bool
    threshold: float


@dataclass
class LowerCusum:
    """Lower-side Page CUSUM with baseline-tuned (K, h)."""

    target: float = 0.0
    sigma: float = 0.0
    K: float = 0.0
    h: float = 0.0
    k_threshold: float = 4.0
    _statistic: float = field(default=0.0, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, baseline: Sequence[float], *, label: str = "<unknown>") -> None:
        """Estimate μ₀ and σ from the baseline window; set K and h."""
        mu, sigma = compute_baseline_stats(baseline, label=label)
        self.target = mu
        self.sigma = sigma
        self.K = 0.5 * sigma
        self.h = self.k_threshold * sigma
        self._statistic = 0.0
        self._fitted = True

    def step(self, x: float, session_index: int) -> CusumState:
        """Advance the running statistic by one observation."""
        if not self._fitted:
            raise RuntimeError("LowerCusum: call fit() before step()")
        self._statistic = max(
            0.0, self._statistic + (self.target - self.K) - x
        )
        return CusumState(
            session_index=session_index,
            statistic=self._statistic,
            fired=self._statistic >= self.h,
            threshold=self.h,
        )

    def run(
        self,
        stream: Sequence[float],
        *,
        session_indices: Sequence[int] | None = None,
    ) -> list[CusumState]:
        """Reset the running statistic and replay the full stream."""
        if not self._fitted:
            raise RuntimeError("LowerCusum: call fit() before run()")
        self._statistic = 0.0
        if session_indices is None:
            session_indices = list(range(len(stream)))
        if len(session_indices) != len(stream):
            raise ValueError("session_indices and stream length must match")
        return [self.step(x, idx) for idx, x in zip(session_indices, stream)]

    def first_fire_index(
        self,
        stream: Sequence[float],
        *,
        session_indices: Sequence[int] | None = None,
    ) -> int | None:
        """Return the session_index of the first firing step, or None."""
        for state in self.run(stream, session_indices=session_indices):
            if state.fired:
                return state.session_index
        return None


def cusum_per_anchor(
    run_id: int, *, k: float = 4.0
) -> dict[str, list[CusumState]]:
    """CUSUM trace per anchor for a drift run.

    Anchors with `< MIN_BASELINE_N` baseline samples are skipped:
    `InsufficientBaseline` raised by `fit()` is caught here and the
    anchor is omitted from the result so callers can render the rest
    of the run without special-casing every detector site.
    """
    streams = load_run_scalars(run_id)
    out: dict[str, list[CusumState]] = {}
    for anchor_id, triples in streams.items():
        baseline, full = split_baseline(triples)
        detector = LowerCusum(k_threshold=k)
        try:
            detector.fit(baseline, label=f"cusum:{anchor_id}")
        except InsufficientBaseline:
            continue
        indices = [idx for (idx, _stage, _agg) in triples]
        out[anchor_id] = detector.run(full, session_indices=indices)
    return out


__all__ = [
    "CusumState",
    "InsufficientBaseline",
    "LowerCusum",
    "cusum_per_anchor",
]
