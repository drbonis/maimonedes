"""Phase 5 curvature signal (issue #51).

Detects geometric reorganisation: the supervised system's compliance
landscape becomes more sharply curved at an anchor's location even
though the anchor's positional drift is zero. v1 expresses
"curvature" as the condition number of the local metric tensor —
`κ(g) = λ_max / λ_min` — which captures the §6.4 narrative ("the
metric at an anchor location shows increasing curvature — the
compliance landscape is becoming steeper") without needing to compute
a full Ricci scalar (deferred to v3 per the issue's implementation
notes).

Signal fires when `(κ_current - κ_baseline) / κ_baseline > h_curvature`.
Default `h_curvature = 0.5` — a 50% relative increase — picked to
match the same "structural alarm" sensitivity that CUSUM operates at.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from maimonedes.monitor.metric import RiemannianMetric, metric_at


DEFAULT_H_CURVATURE = 0.5


@dataclass(frozen=True)
class CurvatureResult:
    """One anchor's curvature-drift verdict.

    Stored as the condition-number scalar at both fits plus the
    relative-increase ratio for convenient threshold tuning.
    """

    anchor_id: str
    c: tuple[float, ...]
    baseline_curvature: float
    current_curvature: float
    relative_increase: float
    signal_fired: bool
    h_curvature: float


def compute_curvature(metric: RiemannianMetric, c: np.ndarray) -> float:
    """Scalar curvature at point `c` — condition number `λ_max / λ_min`.

    Raises `ValueError` if the metric is degenerate (smallest eigen-
    value below 1e-12). The Cholesky parameterisation guarantees PD
    so this only triggers in pathological numerical cases.
    """
    g = metric_at(metric, c)
    eigvals = np.linalg.eigvalsh(g)
    lam_min = float(eigvals.min())
    lam_max = float(eigvals.max())
    if lam_min <= 1e-12:
        raise ValueError(
            f"compute_curvature: metric is near-degenerate at c={c.tolist()}: "
            f"min eigenvalue {lam_min:.3e}"
        )
    return lam_max / lam_min


def curvature_drift(
    *,
    anchor_id: str,
    metric_baseline: RiemannianMetric,
    metric_current: RiemannianMetric,
    c: np.ndarray,
    h_curvature: float = DEFAULT_H_CURVATURE,
) -> CurvatureResult:
    """Compare condition numbers at the same `c` between two metric fits.

    Both metrics must agree on `k` (the issue's v1 only supports per-
    policy metrics; cross-policy curvature is v3). Fires when the
    relative increase exceeds `h_curvature`.
    """
    if metric_baseline.k != metric_current.k:
        raise ValueError(
            f"curvature_drift: metric dimensionality mismatch — "
            f"baseline k={metric_baseline.k}, current k={metric_current.k}"
        )
    c_arr = np.asarray(c, dtype=np.float64).reshape(-1)
    if c_arr.size != metric_baseline.k:
        raise ValueError(
            f"curvature_drift: c shape {c_arr.shape}; expected ({metric_baseline.k},)"
        )

    base_kappa = compute_curvature(metric_baseline, c_arr)
    cur_kappa = compute_curvature(metric_current, c_arr)
    rel_increase = (cur_kappa - base_kappa) / base_kappa
    fired = rel_increase > h_curvature

    return CurvatureResult(
        anchor_id=anchor_id,
        c=tuple(float(v) for v in c_arr),
        baseline_curvature=base_kappa,
        current_curvature=cur_kappa,
        relative_increase=rel_increase,
        signal_fired=fired,
        h_curvature=h_curvature,
    )


def curvature_per_anchor(
    anchor_positions: dict[str, np.ndarray],
    *,
    metric_baseline: RiemannianMetric,
    metric_current: RiemannianMetric,
    h_curvature: float = DEFAULT_H_CURVATURE,
) -> dict[str, CurvatureResult]:
    """Run `curvature_drift` for every (anchor_id, c) pair."""
    out: dict[str, CurvatureResult] = {}
    for aid, c in anchor_positions.items():
        out[aid] = curvature_drift(
            anchor_id=aid,
            metric_baseline=metric_baseline,
            metric_current=metric_current,
            c=c,
            h_curvature=h_curvature,
        )
    return out


__all__ = [
    "CurvatureResult",
    "DEFAULT_H_CURVATURE",
    "compute_curvature",
    "curvature_drift",
    "curvature_per_anchor",
]
