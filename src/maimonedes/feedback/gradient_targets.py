"""Phase 5 gradient-guided probe synthesis (issue #52, §6.5).

The K-NN exemplar synthesizer (#42) is *target-driven*: the GP proposes
an embedding-space target and the LLM generates text near it. This
module is *gradient-driven*: each GP seed is stepped along the
descent direction of the Stage-2 score for a chosen axis, in
embedding space, until either

1. predicted compliance crosses a violation threshold (we've reached
   the policy boundary in the model's view), or
2. GP posterior σ exceeds a cap (we've left the model's confidence
   region — further gradient extrapolation is operationally
   meaningless).

The final embedding of each trajectory is then fed back into
`KnnExemplarSynthesizer.synthesize(...)` (with re-embed + τ check +
LLM-as-validator gate) so generated probes are still subject to the
existing quality bar — gradient stepping does not let bad text slip
through.

Gradient computation uses central-differences numerical
differentiation. Stage-2 heads are sklearn (Ridge or MLPRegressor
inside Pipeline(StandardScaler, head)); they don't expose backprop,
so analytic gradients would have to special-case Ridge vs MLP. Two
batched forward passes per gradient step keeps the code uniform across
head families and is fast enough at the v1 scale (768-dim Bio_ClinicalBERT
embeddings, ≤10 steps per seed, ≤10 seeds per call).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from maimonedes.core.policy import Policy
from maimonedes.models.stage2 import Stage2Model
from maimonedes.monitor.fragility import aggregated_fragility
from maimonedes.monitor.gp_layer import ComplianceGP, propose_targets


log = logging.getLogger(__name__)

DEFAULT_VIOLATION_THRESHOLD = 0.5
DEFAULT_UNCERTAINTY_CAP_FACTOR = 2.0
DEFAULT_MAX_STEPS = 10
DEFAULT_FD_EPS = 1e-3


@dataclass(frozen=True)
class GradientTarget:
    """One gradient-stepped embedding-space target.

    `embedding` is the FINAL position after `n_steps_taken` steps along
    the descent direction. `expected_score` and `uncertainty` are the
    GP's posterior at that final point. `axis_id` records which
    Stage-2 axis the gradient descended (the policy's worst-fragility
    axis by default).
    """

    embedding: list[float]
    expected_score: float
    uncertainty: float
    axis_id: str
    n_steps_taken: int
    stop_reason: str  # "violation" | "uncertainty_cap" | "max_steps" | "zero_gradient"


def _resolve_axis_id(axis_id: str | None, *, policy: Policy) -> str:
    """When `axis_id is None`, default to the policy's worst-fragility axis.

    "Worst" = the column whose `mean_delta` across all (perturbation_kind, *)
    rows is most negative (largest compliance loss under perturbation).
    Falls back to the policy rubric's first sub-condition when no
    fragility data is available.
    """
    if axis_id is not None:
        return axis_id
    sub_ids = {s.id for s in policy.rubric.sub_conditions}
    try:
        table = aggregated_fragility(include_synthesized=False)
    except Exception as exc:
        log.warning(
            "gradient_targets.fragility_unavailable",
            extra={"error": str(exc)},
        )
        table = None
    if table is not None and table.cells:
        # Filter to per-axis cells (skip aggregate column).
        per_axis = [c for c in table.cells if c.column in sub_ids]
        if per_axis:
            worst = min(per_axis, key=lambda c: c.mean_delta)
            return worst.column
    # No fragility signal — fall back to first sub-condition.
    return policy.rubric.sub_conditions[0].id


def compute_score_gradient(
    stage2: Stage2Model,
    embedding: np.ndarray | list[float],
    axis_id: str,
    *,
    eps: float = DEFAULT_FD_EPS,
) -> np.ndarray:
    """Unit-norm descent direction `-∇_e score[axis_id] / ‖∇_e score[axis_id]‖`.

    Stage-2 heads are sklearn (Ridge or MLPRegressor in
    `Pipeline(StandardScaler, head)`), so we use central-differences
    finite-differencing instead of an analytic per-family path. Two
    batched forward passes (each on a `(d, d)` matrix) compute every
    coordinate's partial in one shot, so wall-time scales with one
    matrix multiply per call rather than `d` separate predicts.

    Returns a zero vector when the gradient is degenerate (e.g. heads
    that flat-line in this region of embedding space).
    """
    if axis_id not in stage2.heads:
        raise ValueError(
            f"axis_id={axis_id!r} not in stage2.heads (keys: "
            f"{sorted(stage2.heads.keys())})"
        )
    head = stage2.heads[axis_id]
    e = np.asarray(embedding, dtype=float).reshape(-1)
    d = e.shape[0]
    eye = np.eye(d, dtype=float)
    plus = np.tile(e, (d, 1)) + eps * eye
    minus = np.tile(e, (d, 1)) - eps * eye
    s_plus = np.asarray(head.predict(plus), dtype=float).reshape(-1)
    s_minus = np.asarray(head.predict(minus), dtype=float).reshape(-1)
    grad = (s_plus - s_minus) / (2.0 * eps)
    descent = -grad
    norm = float(np.linalg.norm(descent))
    if norm == 0.0:
        return np.zeros_like(descent)
    return descent / norm


def _kernel_length_scale(gp: ComplianceGP) -> float:
    """Best-effort scalar length-scale for step-size defaulting.

    Stationary RBF: `length_scale` is a scalar; for the Gibbs kernel
    (#49) we use `exp(a)` (the `ℓ(0)` value) as a sane scalar
    representative. Falls back to 1.0 when nothing scalar-shaped is
    discoverable.
    """
    kernel = gp.gp.kernel_
    # GibbsKernel from #49.
    if hasattr(kernel, "_length_scale") and hasattr(kernel, "a"):
        try:
            return float(np.exp(getattr(kernel, "a")))
        except Exception:
            pass
    length_scale = getattr(kernel, "length_scale", None)
    if length_scale is None and hasattr(kernel, "k2"):
        length_scale = getattr(kernel.k2, "length_scale", None)
    if length_scale is None:
        return 1.0
    if hasattr(length_scale, "__len__"):
        return float(np.mean(length_scale))
    return float(length_scale)


def _uncertainty_cap(gp: ComplianceGP, *, factor: float) -> float:
    """`factor × σ_train_max` — the boundary of the GP's confidence region."""
    if gp.training_embeddings.size == 0:
        return float("inf")
    _, std_train = gp.predict(gp.training_embeddings)
    sigma_max = float(np.max(std_train)) if std_train.size > 0 else 0.0
    return factor * sigma_max if sigma_max > 0.0 else float("inf")


def propose_gradient_targets(
    *,
    gp: ComplianceGP,
    stage2: Stage2Model,
    policy: Policy,
    n_targets: int = 10,
    step: float | None = None,
    axis_id: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    violation_threshold: float = DEFAULT_VIOLATION_THRESHOLD,
    uncertainty_cap_factor: float = DEFAULT_UNCERTAINTY_CAP_FACTOR,
    candidate_pool_size: int = 1000,
    seed: int = 0,
) -> list[GradientTarget]:
    """For each GP-proposed seed, step along the Stage-2 descent direction.

    Each step recomputes the gradient at the new position (the head is
    nonlinear under MLP), so the trajectory follows the field rather
    than a single straight line. The step size defaults to a tenth of
    the GP kernel's length scale — large enough to make progress, small
    enough that the local-linear gradient direction stays meaningful.

    Stops when:
    - GP posterior σ exceeds `uncertainty_cap_factor × σ_train_max` —
      the seed has left the GP's confidence region.
    - Stage-2 predicted compliance for `axis_id` < `violation_threshold`
      — the trajectory crossed the policy boundary in the model's view.
    - `max_steps` taken — the gradient hasn't converged into either
      criterion within budget; we return the best-effort point.
    - The gradient is identically zero (degenerate head) — no descent
      direction available, return the seed embedding unchanged.
    """
    if n_targets < 1:
        raise ValueError("n_targets must be >= 1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    resolved_axis = _resolve_axis_id(axis_id, policy=policy)
    if resolved_axis not in stage2.heads:
        raise ValueError(
            f"resolved axis_id={resolved_axis!r} not in stage2.heads "
            f"(available: {sorted(stage2.heads.keys())})"
        )

    if step is None:
        ell = _kernel_length_scale(gp)
        step = 0.1 * ell
    if step <= 0.0:
        raise ValueError(f"step must be positive; got {step}")

    seeds = propose_targets(
        gp,
        n_targets=n_targets,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
    )
    if not seeds:
        return []

    cap = _uncertainty_cap(gp, factor=uncertainty_cap_factor)
    head = stage2.heads[resolved_axis]

    out: list[GradientTarget] = []
    for seed_t in seeds:
        e = np.asarray(seed_t.embedding, dtype=float)
        stop_reason = "max_steps"
        n_steps_taken = 0
        for step_idx in range(max_steps):
            direction = compute_score_gradient(stage2, e, resolved_axis)
            if not np.any(direction):
                stop_reason = "zero_gradient"
                break
            e_next = e + step * direction
            n_steps_taken = step_idx + 1
            # Uncertainty cap is checked before the boundary crossing
            # so a step into wildly-uncertain territory doesn't get
            # silently labelled "violation".
            _, std_next = gp.predict(e_next.reshape(1, -1))
            if float(std_next[0]) > cap:
                stop_reason = "uncertainty_cap"
                e = e_next
                break
            s2 = float(np.clip(head.predict(e_next.reshape(1, -1))[0], 0.0, 1.0))
            e = e_next
            if s2 < violation_threshold:
                stop_reason = "violation"
                break

        mean, std = gp.predict(e.reshape(1, -1))
        out.append(
            GradientTarget(
                embedding=[float(v) for v in e],
                expected_score=float(mean[0]),
                uncertainty=float(std[0]),
                axis_id=resolved_axis,
                n_steps_taken=n_steps_taken,
                stop_reason=stop_reason,
            )
        )
    return out


__all__ = [
    "DEFAULT_FD_EPS",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_UNCERTAINTY_CAP_FACTOR",
    "DEFAULT_VIOLATION_THRESHOLD",
    "GradientTarget",
    "compute_score_gradient",
    "propose_gradient_targets",
]
