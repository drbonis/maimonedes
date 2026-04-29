"""Phase 3 lower-side EWMA detector.

Comparison detector for CUSUM. Same baseline-tuned σ, same lower-side
posture (drift in the contamination experiment is downward by
construction), but with a time-varying lower control limit so the
detector does not fire spuriously at very small `t`.

Recurrence:
    Z[0]   = μ₀
    Z[t]   = λ · x[t] + (1 - λ) · Z[t-1]

Time-varying lower control limit:
    LCL[t] = μ₀ - L · σ · sqrt(λ / (2-λ) · (1 - (1-λ)^{2t}))

Fires when `Z[t] < LCL[t]`. Asymptotically `LCL → μ₀ - L·σ·sqrt(λ/(2-λ))`,
which is what the dashboard's flat reference line shows.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from maimonedes.monitor._baseline import (
    InsufficientBaseline,  # re-exported for callers
    compute_baseline_stats,
    load_run_scalars,
    split_baseline,
)


@dataclass(frozen=True)
class EwmaState:
    """One step of the EWMA trace; safe to plot or persist."""

    session_index: int
    statistic: float
    lcl: float
    fired: bool


@dataclass
class LowerEwma:
    """Lower-side EWMA with baseline-tuned (μ₀, σ) and time-varying LCL."""

    target: float = 0.0
    sigma: float = 0.0
    lambda_: float = 0.2
    L: float = 3.0
    _statistic: float = field(default=0.0, init=False, repr=False)
    _t: int = field(default=0, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, baseline: Sequence[float], *, label: str = "<unknown>") -> None:
        mu, sigma = compute_baseline_stats(baseline, label=label)
        self.target = mu
        self.sigma = sigma
        self._statistic = mu
        self._t = 0
        self._fitted = True

    @property
    def asymptotic_lcl(self) -> float:
        """`μ₀ - L·σ·sqrt(λ/(2-λ))` — the flat reference line on plots."""
        var_factor = self.lambda_ / (2.0 - self.lambda_)
        return self.target - self.L * self.sigma * math.sqrt(var_factor)

    def _lcl_at(self, t: int) -> float:
        # t = 1, 2, ... ; the (1 - (1-λ)^{2t}) factor approaches 1 as t grows.
        if t <= 0:
            raise ValueError("LCL is defined for t >= 1")
        var_factor = self.lambda_ / (2.0 - self.lambda_)
        decay = (1.0 - self.lambda_) ** (2 * t)
        return self.target - self.L * self.sigma * math.sqrt(var_factor * (1.0 - decay))

    def step(self, x: float, session_index: int) -> EwmaState:
        if not self._fitted:
            raise RuntimeError("LowerEwma: call fit() before step()")
        self._t += 1
        self._statistic = self.lambda_ * x + (1.0 - self.lambda_) * self._statistic
        lcl = self._lcl_at(self._t)
        return EwmaState(
            session_index=session_index,
            statistic=self._statistic,
            lcl=lcl,
            fired=self._statistic < lcl,
        )

    def run(
        self,
        stream: Sequence[float],
        *,
        session_indices: Sequence[int] | None = None,
    ) -> list[EwmaState]:
        if not self._fitted:
            raise RuntimeError("LowerEwma: call fit() before run()")
        # Reset to the post-fit initial state.
        self._statistic = self.target
        self._t = 0
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
        for state in self.run(stream, session_indices=session_indices):
            if state.fired:
                return state.session_index
        return None


def ewma_per_anchor(
    run_id: int,
    *,
    lambda_: float = 0.2,
    L: float = 3.0,
) -> dict[str, list[EwmaState]]:
    """EWMA trace per anchor for a drift run.

    Anchors with `< MIN_BASELINE_N` baseline samples are skipped; the
    `InsufficientBaseline` raised by `fit()` is caught here and the
    anchor is omitted from the result.
    """
    streams = load_run_scalars(run_id)
    out: dict[str, list[EwmaState]] = {}
    for anchor_id, triples in streams.items():
        baseline, full = split_baseline(triples)
        detector = LowerEwma(lambda_=lambda_, L=L)
        try:
            detector.fit(baseline, label=f"ewma:{anchor_id}")
        except InsufficientBaseline:
            continue
        indices = [idx for (idx, _stage, _agg) in triples]
        out[anchor_id] = detector.run(full, session_indices=indices)
    return out


__all__ = [
    "EwmaState",
    "InsufficientBaseline",
    "LowerEwma",
    "ewma_per_anchor",
]
