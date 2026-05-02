"""Phase 5 Riemannian metric learner: MLP on Phase-2 Jacobians.

Implements a small feed-forward network that maps a compliance-space
coordinate c ∈ ℝ^k to the lower-triangular Cholesky factor L(c) of
the local metric tensor g(c) = L(c) L(c)^T (positive-definite by
construction). Trained on per-anchor empirical Jacobians from
`monitor/fragility.py`, where each anchor contributes the target
`g_target = J^T J / ‖J‖²` (Fisher-information style construction).

Implementation choice: PyTorch is not a project dependency. The MLP
uses pure-numpy forward + backward passes plus a tiny Adam optimizer.
Persistence is `numpy.savez_compressed` to `.npz` (the issue spec
mentions `.pt` but our model is numpy-native — same role, different
wire format).

`riemannian_distance(metric, c0, c1, *, n_segments)` integrates the
local metric along a straight-line path in coordinate space:
    d ≈ Σ_i √((Δc_i)^T g(c_mid_i) (Δc_i))
This captures the §4.6 amplification factors without solving the
geodesic ODE; v2 can swap in a true geodesic.

Opt-in integration with monitors: every drift / boundary detector
takes an optional `metric` kwarg; default `None` keeps the current
Euclidean behaviour.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from maimonedes.core.policy import Policy
from maimonedes.monitor.fragility import AGGREGATE_COLUMN, Jacobian, all_jacobians


# ---------------------------------------------------------------------------
# Cholesky parameterisation
# ---------------------------------------------------------------------------

def _tri_param_count(k: int) -> int:
    """Number of free entries in a k×k lower-triangular matrix (incl. diag)."""
    return k * (k + 1) // 2


def _build_lower_triangular(params: np.ndarray, k: int) -> np.ndarray:
    """Pack `k(k+1)/2` raw params into a lower-triangular L with positive diag.

    Diagonal entries `L_ii` are exponentiated so g = L L^T is strictly
    positive-definite for any finite raw param vector. Off-diagonal
    entries are passed through unchanged. Layout (k=3 example):

        params = [d0, d1, d2, o10, o20, o21]
        L      = [[exp(d0), 0,        0      ],
                  [o10,     exp(d1),  0      ],
                  [o20,     o21,      exp(d2)]]
    """
    if params.shape[-1] != _tri_param_count(k):
        raise ValueError(
            f"params shape {params.shape} incompatible with k={k}"
        )
    L = np.zeros((k, k), dtype=np.float64)
    # Diagonals first (k entries), then off-diagonals row-major below the diag.
    for i in range(k):
        L[i, i] = np.exp(params[i])
    idx = k
    for i in range(1, k):
        for j in range(i):
            L[i, j] = params[idx]
            idx += 1
    return L


def _build_lower_triangular_grad(
    dL: np.ndarray, params: np.ndarray, k: int
) -> np.ndarray:
    """Backprop ∂loss/∂params from ∂loss/∂L (k×k, lower-tri only).

    Mirror of `_build_lower_triangular`: diagonals propagate via the
    chain rule for `L_ii = exp(raw_i)`, off-diagonals are identity.
    """
    out = np.zeros(_tri_param_count(k), dtype=np.float64)
    for i in range(k):
        # L_ii = exp(params[i]) → dL_ii/dparam_i = L_ii itself.
        out[i] = dL[i, i] * np.exp(params[i])
    idx = k
    for i in range(1, k):
        for j in range(i):
            out[idx] = dL[i, j]
            idx += 1
    return out


# ---------------------------------------------------------------------------
# Numpy MLP
# ---------------------------------------------------------------------------

@dataclass
class MetricMLP:
    """Numpy MLP `c → L_params ∈ ℝ^{k(k+1)/2}` with `depth` ReLU layers.

    `weights` is a list of (W, b) tuples, one per layer including the
    output. `forward` returns the raw param vector; `metric_at` uses
    `_build_lower_triangular` to construct `L` and then `g = L L^T`.
    """

    k: int
    hidden: int
    depth: int
    weights: list[tuple[np.ndarray, np.ndarray]]

    @classmethod
    def init(
        cls,
        k: int,
        *,
        hidden: int = 32,
        depth: int = 2,
        seed: int = 0,
    ) -> "MetricMLP":
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        rng = np.random.default_rng(seed)
        weights: list[tuple[np.ndarray, np.ndarray]] = []
        prev_dim = k
        out_dim = _tri_param_count(k)
        for layer in range(depth):
            # He initialisation for ReLU.
            scale = np.sqrt(2.0 / max(prev_dim, 1))
            W = rng.standard_normal((hidden, prev_dim)) * scale
            b = np.zeros(hidden)
            weights.append((W, b))
            prev_dim = hidden
        # Output layer maps to the Cholesky parameters; small init keeps
        # the initial metric near identity (L_ii = exp(0) = 1).
        scale_out = np.sqrt(1.0 / max(prev_dim, 1))
        W_out = rng.standard_normal((out_dim, prev_dim)) * scale_out * 0.1
        b_out = np.zeros(out_dim)
        weights.append((W_out, b_out))
        return cls(k=k, hidden=hidden, depth=depth, weights=weights)

    def forward(self, c: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Compute L_params for one `c` vector; cache activations for backward."""
        a = np.asarray(c, dtype=np.float64).reshape(-1)
        if a.size != self.k:
            raise ValueError(f"c shape {a.shape}; expected ({self.k},)")
        activations: list[np.ndarray] = [a]
        for layer_idx, (W, b) in enumerate(self.weights):
            z = W @ a + b
            if layer_idx < self.depth:
                a = np.maximum(0.0, z)  # ReLU on hidden layers
            else:
                a = z  # linear output (Cholesky raw params)
            activations.append(a)
        return activations[-1], activations

    def predict_L(self, c: np.ndarray) -> np.ndarray:
        params, _ = self.forward(c)
        return _build_lower_triangular(params, self.k)

    def predict_g(self, c: np.ndarray) -> np.ndarray:
        L = self.predict_L(c)
        return L @ L.T

    # ---- backprop helpers --------------------------------------------------

    def backward(
        self,
        activations: list[np.ndarray],
        grad_params: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Backprop ∂loss/∂params through the MLP; returns matching grad shapes.

        `grad_params` is `∂loss/∂(output L_params)`, shape `(k(k+1)/2,)`.
        `activations[i]` is the input to layer `i` (= output of layer
        `i-1`); `activations[i+1]` is the output of layer `i`. Hidden
        layers used ReLU, the output layer is linear.
        """
        grads: list[tuple[np.ndarray, np.ndarray]] = [None] * len(self.weights)  # type: ignore[list-item]
        delta = grad_params  # ∂loss/∂(output of current layer)
        for layer_idx in reversed(range(len(self.weights))):
            W, _ = self.weights[layer_idx]
            is_hidden = layer_idx < self.depth
            if is_hidden:
                # delta currently is ∂L/∂a[layer_idx]; multiply by ReLU'
                # to get ∂L/∂z[layer_idx]. ReLU(z) > 0 ⇔ z > 0.
                a_out = activations[layer_idx + 1]
                delta = delta * (a_out > 0).astype(np.float64)
            a_in = activations[layer_idx]
            grad_W = np.outer(delta, a_in)
            grad_b = delta.copy()
            grads[layer_idx] = (grad_W, grad_b)
            if layer_idx > 0:
                delta = W.T @ delta  # ∂L/∂a[layer_idx-1]
        return grads


# ---------------------------------------------------------------------------
# Adam optimizer (stateful tiny helper)
# ---------------------------------------------------------------------------

@dataclass
class _AdamState:
    m: list[tuple[np.ndarray, np.ndarray]]
    v: list[tuple[np.ndarray, np.ndarray]]
    t: int = 0

    @classmethod
    def for_mlp(cls, mlp: MetricMLP) -> "_AdamState":
        m = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in mlp.weights]
        v = [(np.zeros_like(W), np.zeros_like(b)) for (W, b) in mlp.weights]
        return cls(m=m, v=v)


def _adam_step(
    mlp: MetricMLP,
    grads: list[tuple[np.ndarray, np.ndarray]],
    state: _AdamState,
    *,
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    state.t += 1
    bc1 = 1.0 - beta1 ** state.t
    bc2 = 1.0 - beta2 ** state.t
    new_weights: list[tuple[np.ndarray, np.ndarray]] = []
    for layer_idx, ((W, b), (gW, gb)) in enumerate(zip(mlp.weights, grads)):
        mW, mb = state.m[layer_idx]
        vW, vb = state.v[layer_idx]
        mW_new = beta1 * mW + (1.0 - beta1) * gW
        mb_new = beta1 * mb + (1.0 - beta1) * gb
        vW_new = beta2 * vW + (1.0 - beta2) * (gW * gW)
        vb_new = beta2 * vb + (1.0 - beta2) * (gb * gb)
        state.m[layer_idx] = (mW_new, mb_new)
        state.v[layer_idx] = (vW_new, vb_new)
        W_new = W - lr * (mW_new / bc1) / (np.sqrt(vW_new / bc2) + eps)
        b_new = b - lr * (mb_new / bc1) / (np.sqrt(vb_new / bc2) + eps)
        new_weights.append((W_new, b_new))
    mlp.weights = new_weights


# ---------------------------------------------------------------------------
# RiemannianMetric dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiemannianMetric:
    """Trained metric: holds the MLP + provenance metadata.

    The `model` field is the numpy `MetricMLP` (not a `torch.nn.Module`
    despite the issue spec's wording — see module docstring). `path`
    is set after `save(...)` and on `load(...)`.
    """

    model: MetricMLP
    policy_id: str
    trained_at: datetime
    n_anchors: int
    n_jacobians: int
    eval_metrics: dict[str, float]
    path: Path | None = None
    sub_condition_ids: tuple[str, ...] = field(default_factory=tuple)

    @property
    def k(self) -> int:
        return self.model.k

    # ------------- persistence -------------

    def save(self, path: Path) -> "RiemannianMetric":
        """Write to `.npz`. Returns self with `path` populated."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pack weights into a flat dict of named arrays.
        data: dict[str, Any] = {
            "k": np.asarray(self.model.k),
            "hidden": np.asarray(self.model.hidden),
            "depth": np.asarray(self.model.depth),
            "policy_id": np.asarray(self.policy_id),
            "trained_at": np.asarray(self.trained_at.isoformat()),
            "n_anchors": np.asarray(self.n_anchors),
            "n_jacobians": np.asarray(self.n_jacobians),
            "eval_metrics_json": np.asarray(
                json.dumps(self.eval_metrics, sort_keys=True)
            ),
            "sub_condition_ids_json": np.asarray(
                json.dumps(list(self.sub_condition_ids))
            ),
        }
        for layer_idx, (W, b) in enumerate(self.model.weights):
            data[f"W{layer_idx}"] = W
            data[f"b{layer_idx}"] = b
        np.savez_compressed(path, **data)
        self.path = path
        return self

    @classmethod
    def load(cls, path: Path) -> "RiemannianMetric":
        path = Path(path)
        with np.load(path, allow_pickle=False) as f:
            k = int(f["k"])
            hidden = int(f["hidden"])
            depth = int(f["depth"])
            policy_id = str(f["policy_id"])
            trained_at = datetime.fromisoformat(str(f["trained_at"]))
            n_anchors = int(f["n_anchors"])
            n_jacobians = int(f["n_jacobians"])
            eval_metrics = json.loads(str(f["eval_metrics_json"]))
            try:
                sub_ids_raw = json.loads(str(f["sub_condition_ids_json"]))
                sub_condition_ids = tuple(sub_ids_raw)
            except Exception:
                sub_condition_ids = tuple()
            n_layers = depth + 1  # hidden layers + output
            weights: list[tuple[np.ndarray, np.ndarray]] = []
            for layer_idx in range(n_layers):
                weights.append((f[f"W{layer_idx}"].copy(), f[f"b{layer_idx}"].copy()))
        mlp = MetricMLP(k=k, hidden=hidden, depth=depth, weights=weights)
        return cls(
            model=mlp,
            policy_id=policy_id,
            trained_at=trained_at,
            n_anchors=n_anchors,
            n_jacobians=n_jacobians,
            eval_metrics=eval_metrics,
            path=path,
            sub_condition_ids=sub_condition_ids,
        )


# ---------------------------------------------------------------------------
# Loss + training loop
# ---------------------------------------------------------------------------

def _frobenius_loss_and_grad(
    L: np.ndarray, g_target: np.ndarray
) -> tuple[float, np.ndarray]:
    """Loss = ‖L L^T - g_target‖_F²; returns scalar loss + ∂loss/∂L (k×k).

    For symmetric `g_target`, `diff = L L^T - g_target` is symmetric and
    ∂‖diff‖_F²/∂L = 4 · diff @ L  (since g = L L^T, see issue's
    Cholesky-output backprop note).
    """
    g_pred = L @ L.T
    diff = g_pred - g_target
    loss = float(np.sum(diff * diff))
    grad_L = 4.0 * (diff @ L)
    return loss, grad_L


def _l2_penalty(mlp: MetricMLP) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    """L2 norm of all weights + matching gradient (biases excluded by convention)."""
    penalty = 0.0
    grads: list[tuple[np.ndarray, np.ndarray]] = []
    for (W, b) in mlp.weights:
        penalty += float(np.sum(W * W))
        grads.append((2.0 * W, np.zeros_like(b)))
    return penalty, grads


def fit_metric_from_pairs(
    c_array: np.ndarray,
    g_array: np.ndarray,
    *,
    k: int,
    hidden: int = 32,
    depth: int = 2,
    epochs: int = 200,
    l2: float = 1e-3,
    lr: float = 1e-2,
    seed: int = 0,
    val_fraction: float = 0.2,
    policy_id: str = "<unknown>",
    n_anchors: int = 0,
    n_jacobians: int = 0,
    sub_condition_ids: tuple[str, ...] = (),
) -> RiemannianMetric:
    """Fit `MetricMLP` to (c_i, g_i) pairs.

    Splits inputs into train / val by `val_fraction` (deterministic
    via `seed`). Reports `train_loss`, `val_loss` in the returned
    `RiemannianMetric.eval_metrics`. With < 5 samples the val split
    collapses to the full train set (val_loss == train_loss).
    """
    c_array = np.asarray(c_array, dtype=np.float64)
    g_array = np.asarray(g_array, dtype=np.float64)
    if c_array.shape[1] != k:
        raise ValueError(f"c_array width {c_array.shape[1]}; expected k={k}")
    if g_array.shape[1:] != (k, k):
        raise ValueError(f"g_array shape {g_array.shape}; expected (n, {k}, {k})")
    n_samples = c_array.shape[0]
    if n_samples == 0:
        raise ValueError("fit_metric_from_pairs: zero training pairs")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    if n_samples >= 5:
        n_val = max(1, int(round(n_samples * val_fraction)))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
    else:
        val_idx = perm
        train_idx = perm

    mlp = MetricMLP.init(k=k, hidden=hidden, depth=depth, seed=seed)
    state = _AdamState.for_mlp(mlp)

    def _epoch_loss(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0
        total = 0.0
        for i in indices:
            params, _ = mlp.forward(c_array[i])
            L = _build_lower_triangular(params, k)
            loss, _ = _frobenius_loss_and_grad(L, g_array[i])
            total += loss
        return total / len(indices)

    train_idx_local = list(train_idx)
    final_train_loss = float("inf")
    final_val_loss = float("inf")
    for epoch in range(epochs):
        rng.shuffle(train_idx_local)
        epoch_loss = 0.0
        for i in train_idx_local:
            params, activations = mlp.forward(c_array[i])
            L = _build_lower_triangular(params, k)
            loss, dL = _frobenius_loss_and_grad(L, g_array[i])
            epoch_loss += loss
            grad_params = _build_lower_triangular_grad(dL, params, k)
            grads = mlp.backward(activations, grad_params)
            if l2 > 0.0:
                # L2 regularisation gradient (no per-sample averaging).
                _, l2_grads = _l2_penalty(mlp)
                grads = [
                    (gW + l2 * lW, gb + l2 * lb)
                    for (gW, gb), (lW, lb) in zip(grads, l2_grads)
                ]
            _adam_step(mlp, grads, state, lr=lr)
        final_train_loss = epoch_loss / max(len(train_idx_local), 1)
        final_val_loss = _epoch_loss(val_idx)

    return RiemannianMetric(
        model=mlp,
        policy_id=policy_id,
        trained_at=datetime.now(timezone.utc),
        n_anchors=n_anchors,
        n_jacobians=n_jacobians,
        eval_metrics={
            "train_loss": float(final_train_loss),
            "val_loss": float(final_val_loss),
        },
        sub_condition_ids=sub_condition_ids,
    )


# ---------------------------------------------------------------------------
# Anchor pair extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _AnchorPair:
    anchor_id: str
    c: np.ndarray
    g_target: np.ndarray
    n_perturbations: int


def _columns_for_axes(jac: Jacobian) -> list[str]:
    return [c for c in jac.columns if c != AGGREGATE_COLUMN]


def _anchor_pair_from_jacobian(jac: Jacobian) -> _AnchorPair | None:
    """Build (c_anchor, g_target) for one anchor's Jacobian, or None.

    `c_anchor` is the per-axis baseline; `g_target = J^T J / ‖J‖²` over
    the rows in the Jacobian, where each row's per-axis Δ becomes one
    row of J. Returns None when the anchor has zero perturbation rows.
    """
    axes = _columns_for_axes(jac)
    if not axes:
        return None
    c = np.asarray(
        [jac.baseline_per_sub_condition.get(a, 0.0) for a in axes],
        dtype=np.float64,
    )
    if not jac.rows:
        return None
    J = np.zeros((len(jac.rows), len(axes)), dtype=np.float64)
    for i, row in enumerate(jac.rows):
        for j, a in enumerate(axes):
            J[i, j] = row.deltas.get(a, 0.0)
    norm_sq = float(np.sum(J * J))
    if norm_sq <= 1e-12:
        # Anchor is fully insensitive to its perturbation cloud → metric
        # is undefined here. Skip with a small ridge to keep training
        # stable on sister anchors.
        return None
    g_target = (J.T @ J) / norm_sq
    return _AnchorPair(
        anchor_id=jac.anchor_id,
        c=c,
        g_target=g_target,
        n_perturbations=len(jac.rows),
    )


def _collect_anchor_pairs_window(
    policy: Policy,
    *,
    start_session_id: int | None,
    end_session_id: int | None,
) -> list[_AnchorPair]:
    """Window-scoped anchor pairs (issue #55).

    Bypasses `monitor.fragility.all_jacobians()` (which has no
    drift-session filter) and pulls compliance scores directly from
    the DB, restricted to `drift_session_id` in
    `[start_session_id, end_session_id]`. Anchor scores
    (`probe_role="anchor"`) form the per-axis baseline; perturbation
    scores (`probe_role="perturbation"`) build the Jacobian's J matrix.

    Anchors with no anchor- or perturbation-scores in the window are
    skipped; the caller decides whether the resulting empty result is
    a usable training set.
    """
    from sqlalchemy import select

    from maimonedes.storage.compliance import ComplianceScoreRow, row_to_score
    from maimonedes.storage.repo import get_session

    expected_axes = [s.id for s in policy.rubric.sub_conditions]
    with get_session() as session:
        stmt = select(ComplianceScoreRow).where(
            ComplianceScoreRow.policy_id == policy.id
        )
        if start_session_id is not None:
            stmt = stmt.where(
                ComplianceScoreRow.drift_session_id >= start_session_id
            )
        if end_session_id is not None:
            stmt = stmt.where(
                ComplianceScoreRow.drift_session_id <= end_session_id
            )
        rows = session.execute(stmt).scalars().all()
        scores = [row_to_score(r) for r in rows]

    by_anchor: dict[str, list] = {}
    for s in scores:
        by_anchor.setdefault(s.anchor_id, []).append(s)

    out: list[_AnchorPair] = []
    for anchor_id, anchor_scores in by_anchor.items():
        anchor_only = [s for s in anchor_scores if s.probe_role == "anchor"]
        perturbed = [s for s in anchor_scores if s.probe_role == "perturbation"]
        if not anchor_only or not perturbed:
            continue
        # Per-axis baseline = mean of anchor-stage scores in the window.
        c_per_axis = {
            a: float(np.mean([s.per_sub_condition.get(a, 0.0) for s in anchor_only]))
            for a in expected_axes
        }
        c = np.asarray([c_per_axis[a] for a in expected_axes], dtype=np.float64)
        J = np.zeros((len(perturbed), len(expected_axes)), dtype=np.float64)
        for i, s in enumerate(perturbed):
            for j, a in enumerate(expected_axes):
                J[i, j] = s.per_sub_condition.get(a, 0.0) - c_per_axis[a]
        norm_sq = float(np.sum(J * J))
        if norm_sq <= 1e-12:
            continue
        g_target = (J.T @ J) / norm_sq
        out.append(
            _AnchorPair(
                anchor_id=anchor_id,
                c=c,
                g_target=g_target,
                n_perturbations=len(perturbed),
            )
        )
    return out


def _collect_anchor_pairs(policy: Policy) -> list[_AnchorPair]:
    """Build training pairs for every anchor with a usable Jacobian."""
    expected_axes = [s.id for s in policy.rubric.sub_conditions]
    out: list[_AnchorPair] = []
    for anchor_id, jac in all_jacobians().items():
        # Filter to anchors whose Jacobian columns line up with the
        # current rubric — drift between policy versions would
        # otherwise sneak in stale axes.
        axes = _columns_for_axes(jac)
        if set(axes) != set(expected_axes):
            continue
        # Re-order to the policy's canonical axis order so every
        # anchor's c-vector lives in the same coordinate system.
        c_dict = jac.baseline_per_sub_condition
        c = np.asarray(
            [c_dict.get(a, 0.0) for a in expected_axes],
            dtype=np.float64,
        )
        J = np.zeros((len(jac.rows), len(expected_axes)), dtype=np.float64)
        for i, row in enumerate(jac.rows):
            for j, a in enumerate(expected_axes):
                J[i, j] = row.deltas.get(a, 0.0)
        norm_sq = float(np.sum(J * J))
        if norm_sq <= 1e-12 or not jac.rows:
            continue
        g_target = (J.T @ J) / norm_sq
        out.append(
            _AnchorPair(
                anchor_id=anchor_id,
                c=c,
                g_target=g_target,
                n_perturbations=len(jac.rows),
            )
        )
    return out


def fit_metric(
    *,
    policy: Policy,
    hidden: int = 32,
    depth: int = 2,
    l2: float = 1e-3,
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    start_session_id: int | None = None,
    end_session_id: int | None = None,
) -> RiemannianMetric:
    """Pull anchor Jacobians from the DB and fit a `RiemannianMetric`.

    With `start_session_id` / `end_session_id` set, the training data is
    scoped to compliance scores whose `drift_session_id` falls in that
    range (issue #55) — used by `drift-report` to construct the two
    metric fits curvature compares. Default behaviour (both kwargs
    `None`) is unchanged: pulls every anchor Jacobian via
    `monitor.fragility.all_jacobians()`.

    Raises `ValueError` when no anchors have usable Jacobians (e.g.,
    `maimonedes perturb` has not been run yet, or the session window
    is empty).
    """
    if start_session_id is not None or end_session_id is not None:
        pairs = _collect_anchor_pairs_window(
            policy,
            start_session_id=start_session_id,
            end_session_id=end_session_id,
        )
        if not pairs:
            raise ValueError(
                f"fit_metric: no anchor scores in drift_session_id range "
                f"[{start_session_id}, {end_session_id}]"
            )
    else:
        pairs = _collect_anchor_pairs(policy)
        if not pairs:
            raise ValueError(
                "fit_metric: no anchor Jacobians found; run `maimonedes perturb` first"
            )
    expected_axes = tuple(s.id for s in policy.rubric.sub_conditions)
    k = len(expected_axes)
    c_array = np.stack([p.c for p in pairs], axis=0)
    g_array = np.stack([p.g_target for p in pairs], axis=0)
    n_jacobians = sum(p.n_perturbations for p in pairs)
    return fit_metric_from_pairs(
        c_array,
        g_array,
        k=k,
        hidden=hidden,
        depth=depth,
        epochs=epochs,
        l2=l2,
        lr=lr,
        seed=seed,
        policy_id=policy.id,
        n_anchors=len(pairs),
        n_jacobians=n_jacobians,
        sub_condition_ids=expected_axes,
    )


# ---------------------------------------------------------------------------
# Public read-side API
# ---------------------------------------------------------------------------

def metric_at(metric: RiemannianMetric, c: np.ndarray) -> np.ndarray:
    """Local metric tensor `g(c) ∈ ℝ^{k×k}` (positive-definite)."""
    return metric.model.predict_g(np.asarray(c, dtype=np.float64))


def riemannian_distance(
    metric: RiemannianMetric,
    c0: np.ndarray,
    c1: np.ndarray,
    *,
    n_segments: int = 16,
) -> float:
    """Piecewise-linear Riemannian distance from c0 to c1.

    For each of `n_segments` equal sub-intervals of the straight-line
    coordinate path, integrates `√((Δc)^T g(c_mid) (Δc))` and sums.
    The path stays straight in coordinate space — geodesic-ODE solving
    is deferred to v2 (issue's "implementation notes").
    """
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    c0_arr = np.asarray(c0, dtype=np.float64)
    c1_arr = np.asarray(c1, dtype=np.float64)
    if c0_arr.shape != c1_arr.shape:
        raise ValueError(f"c0 shape {c0_arr.shape} != c1 shape {c1_arr.shape}")
    if c0_arr.ndim != 1 or c0_arr.size != metric.k:
        raise ValueError(f"c0/c1 must be ({metric.k},) vectors")
    delta_total = c1_arr - c0_arr
    seg_delta = delta_total / n_segments
    total = 0.0
    for i in range(n_segments):
        t_mid = (i + 0.5) / n_segments
        c_mid = c0_arr + t_mid * delta_total
        g_mid = metric_at(metric, c_mid)
        quad = float(seg_delta @ g_mid @ seg_delta)
        # Numerical floor — quad should be >= 0 for PD g, but tiny
        # negative values can appear from rounding.
        total += np.sqrt(max(quad, 0.0))
    return total


def euclidean_distance(c0: np.ndarray, c1: np.ndarray) -> float:
    """Plain L2 distance — convenience for side-by-side reporting."""
    return float(np.linalg.norm(np.asarray(c1) - np.asarray(c0)))


def position_vector(score_per_sub: dict[str, float], axes: Iterable[str]) -> np.ndarray:
    """Project a `per_sub_condition` dict onto a fixed axis order."""
    return np.asarray([score_per_sub.get(a, 0.0) for a in axes], dtype=np.float64)


__all__ = [
    "MetricMLP",
    "RiemannianMetric",
    "euclidean_distance",
    "fit_metric",
    "fit_metric_from_pairs",
    "metric_at",
    "position_vector",
    "riemannian_distance",
]
