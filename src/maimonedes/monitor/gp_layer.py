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


def fit_compliance_gp(
    *,
    embed_client: EmbedClient,
    policy: Policy,
    embedding_model: str | None = None,
    kernel: Kernel | None = None,
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

    gp = GaussianProcessRegressor(
        kernel=kernel or _default_kernel(),
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        alpha=alpha,
        random_state=0,
    )
    gp.fit(X_gp, y)

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


__all__ = [
    "ComplianceGP",
    "GPTarget",
    "TrainingPoint",
    "fit_compliance_gp",
    "propose_targets",
]
