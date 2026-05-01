"""Phase 5 Gaussian-process layer.

Fits a GaussianProcessRegressor over `(text_embedding, aggregate_score)`
pairs and proposes embedding-space targets for high-uncertainty
regions weighted by proximity to the violation boundary. The probe
synthesizer (#42) takes those embeddings as input and generates new
clinical scenarios via K-NN exemplar prompting.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
import scipy.optimize
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, RBF
from sklearn.preprocessing import StandardScaler

from maimonedes.core.policy import Policy
from maimonedes.llm.embed_client import EmbedClient
from maimonedes.storage.llm_calls import pair_supervised_with_scores


log = logging.getLogger(__name__)


@dataclass
class TrainingPoint:
    text: str
    aggregate: float


@dataclass
class ComplianceGP:
    """A fitted Gaussian process over compliance-aggregate scores.

    `training_embeddings` and the inputs to `predict` are in the
    ORIGINAL Bio_ClinicalBERT space. Scaling for the GP is applied
    internally via `scaler` so callers don't have to thread the
    transformation through. `scaler is None` is supported for
    backward-compatibility with v1 (un-scaled) fits, though new
    fits always populate it.
    """

    policy_id: str
    trained_at: datetime
    n_samples: int
    embedding_model: str
    feature_dim: int
    gp: GaussianProcessRegressor
    training_embeddings: np.ndarray  # shape (n, feature_dim) in raw space
    training_aggregates: np.ndarray  # shape (n,)
    log_marginal_likelihood: float
    kernel_repr: str
    library_anchor_embeddings: dict[str, list[float]] = field(default_factory=dict)
    scaler: StandardScaler | None = None
    pca: PCA | None = None  # optional dim-reduction; None = use scaled space
    # Optional text payload, used by the dashboard for hover tooltips.
    # Default-empty for backward compat with pre-existing pickled artefacts.
    training_texts: list[str] = field(default_factory=list)
    library_anchor_texts: dict[str, str] = field(default_factory=dict)

    def _to_gp_space(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply scaler → optional PCA so the GP sees its native space."""
        x = embeddings
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if self.pca is not None:
            x = self.pca.transform(x)
        return x

    def predict(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(mean, std)` for each row of `embeddings` (raw space)."""
        mean, std = self.gp.predict(self._to_gp_space(embeddings), return_std=True)
        return np.asarray(mean), np.asarray(std)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ComplianceGP":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise ValueError(f"{path}: pickle did not contain a ComplianceGP")
        return obj


@dataclass(frozen=True)
class GPTarget:
    """One proposed embedding-space target for the probe synthesizer."""

    embedding: list[float]
    expected_score: float  # GP posterior mean at this point
    uncertainty: float  # GP posterior std at this point
    score: float  # uncertainty × max(0, 1 − 2·|0.5 − expected_score|)


def _boundary_seeking_score(
    uncertainty: np.ndarray, mean: np.ndarray
) -> np.ndarray:
    """Per-candidate score: peaks at mean=0.5 (boundary), falls to 0 at extremes.

    `uncertainty × max(0, 1 − 2·|0.5 − mean|)` weights uncertain candidates
    near the violation boundary (where being wrong about safety matters) above
    equally-uncertain candidates deep in compliant or violation territory.
    """
    boundary_weight = np.maximum(0.0, 1.0 - 2.0 * np.abs(0.5 - mean))
    return uncertainty * boundary_weight


def _fetch_training_points(
    policy_id: str, *, supervised_backend_prefix: str = "ollama-supervised"
) -> list[TrainingPoint]:
    """Pull (supervised_text, aggregate_score) pairs via chronological pairing.

    Phase 1+ orchestrators don't wire `compliance_scores.llm_call_id`,
    so we fall back to timestamp-ordered pairing. See
    `storage.llm_calls.pair_supervised_with_scores`.
    """
    pairs = pair_supervised_with_scores(
        policy_id, supervised_backend_prefix=supervised_backend_prefix
    )
    return [
        TrainingPoint(text=p.supervised_text, aggregate=float(p.score.aggregate))
        for p in pairs
    ]


def _default_kernel() -> Kernel:
    # length_scale upper bound bumped to 100 — earlier fits in raw 768-dim
    # space pinned at the old 10 because the curse of dimensionality
    # makes 768-dim Euclidean distances large; PCA preprocessing makes
    # the old (0.1, 10) range workable, but we leave the higher ceiling
    # so the optimizer has headroom either way.
    return ConstantKernel(1.0, (1e-3, 1e6)) * RBF(
        length_scale=1.0, length_scale_bounds=(0.1, 100.0)
    )


# ---- Non-stationary Gibbs kernel (issue #49) -------------------------------
#
# A Gibbs kernel with position-dependent length-scale ℓ(x):
#
#   k(x, y) = σ² · (2 ℓ(x) ℓ(y) / (ℓ(x)² + ℓ(y)²))^(d/2)
#                · exp(-‖x - y‖² / (ℓ(x)² + ℓ(y)²))
#
#   ℓ(x) = exp(a + b · x[0] + c · x[1])
#
# Three degrees of freedom in ℓ — a constant (a) and log-linear slopes
# along the first two coordinates (b, c). x[0] / x[1] are interpreted
# as the leading PCA components in the GP's input space, matching the
# §5.5.5 prediction that compliance geometry is most anisotropic along
# the principal-variance axes.
#
# Hyperparameters (σ, a, b, c) are NOT exposed via sklearn's `Hyperparameter`
# protocol — sklearn's L-BFGS-B path requires positive bounds and analytic
# kernel-gradient code. Both are awkward here (a, b, c are real-valued; the
# Gibbs gradient is messy). Instead we run scipy's L-BFGS-B over the GP log-
# marginal-likelihood externally and pass the optimized kernel to
# GaussianProcessRegressor with `optimizer=None`.


class GibbsKernel(Kernel):
    """Non-stationary Gibbs kernel; see module-level explanation."""

    def __init__(
        self,
        sigma: float = 1.0,
        a: float = 0.0,
        b: float = 0.0,
        c: float = 0.0,
        sigma_bounds: tuple[float, float] = (1e-3, 1e3),
        a_bounds: tuple[float, float] = (-3.0, 3.0),
        b_bounds: tuple[float, float] = (-1.0, 1.0),
        c_bounds: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.sigma = sigma
        self.a = a
        self.b = b
        self.c = c
        self.sigma_bounds = sigma_bounds
        self.a_bounds = a_bounds
        self.b_bounds = b_bounds
        self.c_bounds = c_bounds

    def is_stationary(self) -> bool:
        return False

    def _length_scale(self, X: np.ndarray) -> np.ndarray:
        """ℓ(x) = exp(a + b · x[0] + c · x[1]). Falls back gracefully for d<2."""
        d = X.shape[1]
        log_ell = np.full(X.shape[0], self.a, dtype=float)
        if d >= 1:
            log_ell = log_ell + self.b * X[:, 0]
        if d >= 2:
            log_ell = log_ell + self.c * X[:, 1]
        return np.exp(log_ell)

    def __call__(
        self,
        X: np.ndarray,
        Y: np.ndarray | None = None,
        eval_gradient: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if eval_gradient:
            # Hyperparameters are externally fitted — sklearn shouldn't
            # request the gradient path. If it ever does, fail loud.
            raise NotImplementedError(
                "GibbsKernel hyperparameters are externally optimized; "
                "instantiate a GaussianProcessRegressor with optimizer=None "
                "so sklearn doesn't request kernel gradients."
            )
        X_arr = np.asarray(X, dtype=float)
        Y_arr = X_arr if Y is None else np.asarray(Y, dtype=float)
        d = X_arr.shape[1]

        ell_X = self._length_scale(X_arr)  # (n,)
        ell_Y = ell_X if Y is None else self._length_scale(Y_arr)  # (m,)

        ell_X_sq = ell_X[:, None] ** 2  # (n, 1)
        ell_Y_sq = ell_Y[None, :] ** 2  # (1, m)
        s = ell_X_sq + ell_Y_sq  # (n, m)
        prefactor = (2.0 * ell_X[:, None] * ell_Y[None, :] / s) ** (d / 2.0)

        # ‖x - y‖² without materializing the (n, m, d) tensor.
        Xsq = (X_arr ** 2).sum(axis=1)
        Ysq = (Y_arr ** 2).sum(axis=1)
        sqdist = Xsq[:, None] + Ysq[None, :] - 2.0 * (X_arr @ Y_arr.T)
        np.maximum(sqdist, 0.0, out=sqdist)  # numerical floor

        return (self.sigma ** 2) * prefactor * np.exp(-sqdist / s)

    def diag(self, X: np.ndarray) -> np.ndarray:
        # k(x, x) = σ² · (2 ℓ²/2 ℓ²)^(d/2) · exp(0) = σ²
        return np.full(np.asarray(X).shape[0], self.sigma ** 2, dtype=float)

    def __repr__(self) -> str:
        return (
            f"GibbsKernel(sigma={self.sigma:.4g}, a={self.a:.4g}, "
            f"b={self.b:.4g}, c={self.c:.4g})"
        )


def non_stationary_kernel(
    *,
    sigma: float = 1.0,
    a: float = 0.0,
    b: float = 0.0,
    c: float = 0.0,
) -> Kernel:
    """Return a Gibbs kernel with the supplied initial hyperparameters.

    Hyperparameters are externally optimized by `_fit_gibbs_hyperparameters`
    when this kernel is used inside `fit_compliance_gp(..., kernel="non_stationary")`.
    """
    return GibbsKernel(sigma=sigma, a=a, b=b, c=c)


def _gibbs_log_marginal_likelihood(
    theta: np.ndarray,
    X: np.ndarray,
    y_normalized: np.ndarray,
    alpha: float,
) -> float:
    """Negative log-marginal-likelihood for L-BFGS-B (minimization)."""
    sigma, a, b, c = theta
    kernel = GibbsKernel(sigma=float(sigma), a=float(a), b=float(b), c=float(c))
    K = kernel(X) + alpha * np.eye(X.shape[0])
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return 1e10  # PD failure → reject this hyperparameter set
    alpha_vec = scipy.linalg.cho_solve((L, True), y_normalized)
    n = X.shape[0]
    lml = (
        -0.5 * float(y_normalized @ alpha_vec)
        - float(np.log(np.diag(L)).sum())
        - 0.5 * n * float(np.log(2.0 * np.pi))
    )
    return -lml


def _fit_gibbs_hyperparameters(
    X: np.ndarray,
    y_normalized: np.ndarray,
    *,
    alpha: float,
    n_restarts: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    """Maximize LML over (σ, a, b, c) with L-BFGS-B + multi-start.

    Returns (sigma, a, b, c, log_marginal_likelihood).
    """
    rng = np.random.default_rng(seed)
    bounds = [(0.01, 100.0), (-3.0, 3.0), (-1.0, 1.0), (-1.0, 1.0)]

    # First restart is the deterministic starting point (stationary-RBF-ish).
    initial = [np.array([1.0, 0.0, 0.0, 0.0], dtype=float)]
    for _ in range(n_restarts):
        initial.append(
            np.array(
                [
                    rng.uniform(0.5, 5.0),
                    rng.uniform(-1.0, 1.0),
                    rng.uniform(-0.5, 0.5),
                    rng.uniform(-0.5, 0.5),
                ],
                dtype=float,
            )
        )

    best_theta: np.ndarray | None = None
    best_neg_lml = np.inf
    for theta_0 in initial:
        try:
            result = scipy.optimize.minimize(
                _gibbs_log_marginal_likelihood,
                x0=theta_0,
                args=(X, y_normalized, alpha),
                method="L-BFGS-B",
                bounds=bounds,
            )
        except Exception as exc:
            log.warning(
                "gp_layer.gibbs_optimizer_failed",
                extra={"theta_0": theta_0.tolist(), "error": str(exc)},
            )
            continue
        if not np.isfinite(result.fun):
            continue
        if result.fun < best_neg_lml:
            best_neg_lml = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)

    if best_theta is None or not np.isfinite(best_neg_lml):
        raise RuntimeError(
            "all GibbsKernel hyperparameter restarts failed to converge"
        )

    sigma, a, b, c = best_theta.tolist()
    return float(sigma), float(a), float(b), float(c), -best_neg_lml


# sklearn's default `alpha=1e-10` assumes near-perfect observations.
# Real compliance_scores have substantial duplicate-text rows
# (anchors scored repeatedly across run-once / drift / recovery)
# producing different aggregates due to judge noise. With alpha=1e-2
# the GP attributes that variance to observation noise instead of
# trying to fit it through the kernel — kernel matrix stays well-
# conditioned and the optimizer converges without hitting bounds.
DEFAULT_ALPHA = 1e-2

# Default PCA reduction. Bio_ClinicalBERT outputs 768-dim; with ~1700
# points the curse of dimensionality forces every pairwise distance
# into a narrow band and the RBF kernel can't find local structure.
# 50 components captures most of the variance while keeping average
# pairwise distance in a range the kernel can model.
DEFAULT_PCA_COMPONENTS = 50


def _fit_gp_arrays(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kernel: Kernel | str | None,
    alpha: float = DEFAULT_ALPHA,
    n_restarts_optimizer: int = 5,
) -> GaussianProcessRegressor:
    """Fit a GP given pre-computed input arrays and a kernel selector.

    Split out from `fit_compliance_gp` so tests can drive the kernel
    branching without seeding the database / embed pipeline.
    """
    if isinstance(kernel, str):
        if kernel == "stationary":
            kernel_obj: Kernel = _default_kernel()
            use_external = False
        elif kernel == "non_stationary":
            # Externally optimize on the same y normalization sklearn
            # would apply with normalize_y=True, so the resulting
            # log_marginal_likelihood_value_ is comparable to the
            # stationary path's.
            y_arr = np.asarray(y, dtype=float)
            y_mean = float(np.mean(y_arr))
            y_std = float(np.std(y_arr)) or 1.0
            y_norm = (y_arr - y_mean) / y_std
            sigma_opt, a_opt, b_opt, c_opt, _ = _fit_gibbs_hyperparameters(
                X,
                y_norm,
                alpha=alpha,
                n_restarts=n_restarts_optimizer,
                seed=0,
            )
            kernel_obj = GibbsKernel(
                sigma=sigma_opt, a=a_opt, b=b_opt, c=c_opt
            )
            use_external = True
        else:
            raise ValueError(
                f"unknown kernel selector {kernel!r}; "
                "use 'stationary', 'non_stationary', or a Kernel instance"
            )
    elif kernel is None:
        kernel_obj = _default_kernel()
        use_external = False
    else:
        kernel_obj = kernel
        use_external = False

    gp = GaussianProcessRegressor(
        kernel=kernel_obj,
        normalize_y=True,
        n_restarts_optimizer=0 if use_external else n_restarts_optimizer,
        optimizer=None if use_external else "fmin_l_bfgs_b",
        alpha=alpha,
        random_state=0,
    )
    gp.fit(X, y)
    return gp


def fit_compliance_gp(
    *,
    embed_client: EmbedClient,
    policy: Policy,
    embedding_model: str | None = None,
    kernel: Kernel | str | None = None,
    min_samples: int = 20,
    library_anchor_texts: dict[str, str] | None = None,
    n_restarts_optimizer: int = 5,
    alpha: float = DEFAULT_ALPHA,
    pca_components: int | None = DEFAULT_PCA_COMPONENTS,
) -> ComplianceGP:
    """Pull (text, aggregate) pairs, embed, fit, return ComplianceGP.

    `library_anchor_texts={anchor_id: scenario_text}` is embedded at fit
    time and cached on the artefact so the GP dashboard can do nearest-
    library-anchor lookups without re-embedding the curated probes.

    `kernel` accepts a sklearn `Kernel` instance, the string
    `"stationary"` (default) for the v1 ConstantKernel*RBF, or
    `"non_stationary"` for the Gibbs kernel from issue #49 — fitted
    via external scipy L-BFGS-B since its hyperparameters are
    real-valued and can't be expressed as positive sklearn
    hyperparameters.
    """
    points = _fetch_training_points(policy.id)
    if len(points) < min_samples:
        raise ValueError(
            f"only {len(points)} samples available; need at least "
            f"{min_samples} to fit a GP for policy {policy.id!r}"
        )

    embedding_model = embedding_model or "bio_clinicalbert"
    embeddings: list[list[float]] = []
    aggregates: list[float] = []
    for p in points:
        resp = embed_client.embed(p.text, model=embedding_model)
        embeddings.append(resp.embedding)
        aggregates.append(p.aggregate)

    X = np.asarray(embeddings, dtype=float)
    y = np.asarray(aggregates, dtype=float)
    feature_dim = X.shape[1]

    # Scale features so the kernel's length_scale settles in a finite
    # range. Without this, raw 768-dim Bio_ClinicalBERT magnitudes
    # drive the kernel optimization to its boundaries.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA dim-reduction. With raw 768-dim and ~1700 samples
    # the GP can't find local structure; PCA to ~50 dimensions keeps
    # the principal variance and lets the kernel discriminate
    # neighborhoods.
    pca: PCA | None = None
    if pca_components is not None and pca_components < X_scaled.shape[1]:
        n_components = min(pca_components, X_scaled.shape[0] - 1)
        if n_components >= 1:
            pca = PCA(n_components=n_components, random_state=0)
            X_gp = pca.fit_transform(X_scaled)
        else:
            X_gp = X_scaled
    else:
        X_gp = X_scaled

    gp = _fit_gp_arrays(
        X_gp,
        y,
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
    )

    library_anchors: dict[str, list[float]] = {}
    library_text_payload: dict[str, str] = {}
    if library_anchor_texts:
        for anchor_id, text in library_anchor_texts.items():
            try:
                resp = embed_client.embed(text, model=embedding_model)
                library_anchors[anchor_id] = resp.embedding
                library_text_payload[anchor_id] = text
            except Exception as exc:
                log.warning(
                    "gp_layer.library_anchor_embed_failed",
                    extra={"anchor_id": anchor_id, "error": str(exc)},
                )

    return ComplianceGP(
        policy_id=policy.id,
        trained_at=datetime.now(timezone.utc),
        n_samples=len(points),
        embedding_model=embedding_model,
        feature_dim=feature_dim,
        gp=gp,
        training_embeddings=X,  # store RAW so propose_targets perturbs in raw space
        training_aggregates=y,
        log_marginal_likelihood=float(gp.log_marginal_likelihood_value_),
        kernel_repr=str(gp.kernel_),
        library_anchor_embeddings=library_anchors,
        scaler=scaler,
        pca=pca,
        training_texts=[p.text for p in points],
        library_anchor_texts=library_text_payload,
    )


def _stratified_seed_indices(
    aggregates: np.ndarray,
    *,
    n_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample `n_candidates` training-row indices stratified by score quartile.

    Compliance distributions are typically skewed toward compliance —
    uniform sampling oversamples high-score regions and starves the
    boundary. Quartile stratification forces equal candidate share in
    each quartile, so perturbations of low-score training points get
    represented in the candidate pool.

    Falls back to uniform sampling when training data has fewer than
    four distinct score levels (the bucketization would degenerate).
    """
    n_train = aggregates.shape[0]
    if n_train < 4:
        return rng.integers(0, n_train, size=n_candidates)
    qs = np.quantile(aggregates, [0.25, 0.5, 0.75])
    buckets = [
        np.where(aggregates <= qs[0])[0],
        np.where((aggregates > qs[0]) & (aggregates <= qs[1]))[0],
        np.where((aggregates > qs[1]) & (aggregates <= qs[2]))[0],
        np.where(aggregates > qs[2])[0],
    ]
    buckets = [b for b in buckets if len(b) > 0]
    if len(buckets) <= 1:
        # All scores collapse into one quartile — fall back to uniform.
        return rng.integers(0, n_train, size=n_candidates)
    per_bucket = n_candidates // len(buckets)
    remainder = n_candidates - per_bucket * len(buckets)
    chunks: list[np.ndarray] = []
    for i, bucket in enumerate(buckets):
        n_for_bucket = per_bucket + (1 if i < remainder else 0)
        chunks.append(rng.choice(bucket, size=n_for_bucket, replace=True))
    return np.concatenate(chunks)


def propose_targets(
    gp: ComplianceGP,
    *,
    n_targets: int = 10,
    candidate_pool_size: int = 1000,
    seed: int = 0,
    perturbation_sigma: float | None = None,
    stratify_by_score: bool = True,
) -> list[GPTarget]:
    """Propose top-K embedding-space targets ranked by uncertainty × boundary weight.

    Candidates are perturbed in the GP's NATIVE input space (scaled,
    optionally PCA-projected) so the noise actually reaches the GP's
    posterior. Seeds are sampled by score-quartile stratification by
    default (equal candidate share from each quartile of training
    aggregates) so a compliance-skewed distribution doesn't starve
    the boundary. Top-K are inverse-transformed back to raw 768-dim
    Bio_ClinicalBERT space before being returned, so the synthesizer
    (#42) and dashboard receive embeddings in the same coordinate
    system as the library anchors.

    `perturbation_sigma` overrides the kernel's length-scale-derived
    perturbation magnitude when set — useful for sweeps.
    `stratify_by_score=False` reverts to uniform-random seeding.
    """
    rng = np.random.default_rng(seed)
    if gp.training_embeddings.size == 0:
        return []

    # Project training embeddings into the GP's input space.
    seeds_gp_space = gp._to_gp_space(gp.training_embeddings)
    n_train, gp_dim = seeds_gp_space.shape

    if perturbation_sigma is None:
        try:
            length_scale = getattr(gp.gp.kernel_, "length_scale", 1.0)
            if isinstance(length_scale, np.ndarray):
                sigma = float(np.mean(length_scale))
            else:
                sigma = float(length_scale)
        except Exception:
            sigma = 1.0
    else:
        sigma = float(perturbation_sigma)

    n_candidates = max(1, candidate_pool_size)
    if stratify_by_score:
        seed_indices = _stratified_seed_indices(
            gp.training_aggregates, n_candidates=n_candidates, rng=rng
        )
    else:
        seed_indices = rng.integers(0, n_train, size=n_candidates)
    seeds = seeds_gp_space[seed_indices]
    noise = rng.normal(0.0, sigma, size=seeds.shape)
    candidates_gp = seeds + noise

    # Predict directly on GP-space candidates (skipping `gp.predict`
    # which would re-transform them).
    mean, std = gp.gp.predict(candidates_gp, return_std=True)
    score = _boundary_seeking_score(std, mean)

    order = np.argsort(-score, kind="stable")
    top_idx = order[:n_targets]

    # Map top candidates back to raw 768-dim space: PCA inverse → scaler inverse.
    top_gp = candidates_gp[top_idx]
    if gp.pca is not None:
        top_scaled = gp.pca.inverse_transform(top_gp)
    else:
        top_scaled = top_gp
    if gp.scaler is not None:
        top_raw = gp.scaler.inverse_transform(top_scaled)
    else:
        top_raw = top_scaled

    return [
        GPTarget(
            embedding=[float(v) for v in top_raw[k]],
            expected_score=float(mean[i]),
            uncertainty=float(std[i]),
            score=float(score[i]),
        )
        for k, i in enumerate(top_idx)
    ]


@dataclass(frozen=True)
class QuartileDiagnostic:
    """Per-PCA-1 quartile summary for the non-stationary kernel diagnostic."""

    quartile: int  # 1 .. 4
    n: int
    pc1_mean: float
    ell_mean: float  # mean ℓ(x) across this quartile's training points
    sigma_pred_mean: float  # mean GP posterior std across this quartile
    score_var: float  # empirical variance of training aggregates in this quartile


def quartile_diagnostics(gp: ComplianceGP) -> list[QuartileDiagnostic]:
    """Per-PCA-1-quartile summary of length-scale and posterior uncertainty.

    Splits training points into PC1 quartiles and reports mean ℓ(x), mean
    posterior std, and the empirical score variance per quartile. The §5.5.5
    prediction is that the highest-score-variance quartile (prescriptive
    region) gets the smallest ℓ and the lowest-variance quartile (referral
    region) gets the largest ℓ — only meaningful when the kernel is
    `GibbsKernel`. For a stationary kernel `ell_mean` is constant.
    """
    if gp.training_embeddings.size == 0:
        return []

    X_gp = gp._to_gp_space(gp.training_embeddings)
    pc1 = X_gp[:, 0] if X_gp.shape[1] >= 1 else np.zeros(X_gp.shape[0])
    aggregates = gp.training_aggregates

    if isinstance(gp.gp.kernel_, GibbsKernel):
        ell_per_point = gp.gp.kernel_._length_scale(X_gp)
    else:
        # Best-effort fallback for non-Gibbs kernels.
        length_scale = getattr(gp.gp.kernel_, "length_scale", 1.0)
        if isinstance(length_scale, np.ndarray):
            length_scale = float(np.mean(length_scale))
        ell_per_point = np.full(X_gp.shape[0], float(length_scale))

    _, sigma_pred = gp.gp.predict(X_gp, return_std=True)

    # Rank-based bucketing — guarantees 4 quartiles even when PC1 has
    # heavy ties (e.g. low-dim deterministic test embeddings).
    n = pc1.shape[0]
    order = np.argsort(pc1, kind="stable")
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    quartile_idx = np.minimum(rank * 4 // max(n, 1), 3)

    out: list[QuartileDiagnostic] = []
    for q in range(4):
        mask = quartile_idx == q
        n_q = int(mask.sum())
        if n_q == 0:
            continue
        out.append(
            QuartileDiagnostic(
                quartile=q + 1,
                n=n_q,
                pc1_mean=float(np.mean(pc1[mask])),
                ell_mean=float(np.mean(ell_per_point[mask])),
                sigma_pred_mean=float(np.mean(sigma_pred[mask])),
                score_var=float(np.var(aggregates[mask])),
            )
        )
    return out


__all__ = [
    "ComplianceGP",
    "GPTarget",
    "GibbsKernel",
    "QuartileDiagnostic",
    "TrainingPoint",
    "fit_compliance_gp",
    "non_stationary_kernel",
    "propose_targets",
    "quartile_diagnostics",
]
