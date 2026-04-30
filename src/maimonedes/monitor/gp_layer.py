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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Kernel, RBF
from sqlalchemy import select

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
    """A fitted Gaussian process over compliance-aggregate scores."""

    policy_id: str
    trained_at: datetime
    n_samples: int
    embedding_model: str
    feature_dim: int
    gp: GaussianProcessRegressor
    training_embeddings: np.ndarray  # shape (n, feature_dim)
    training_aggregates: np.ndarray  # shape (n,)
    log_marginal_likelihood: float
    kernel_repr: str
    library_anchor_embeddings: dict[str, list[float]] = field(default_factory=dict)

    def predict(
        self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return `(mean, std)` for each row of `embeddings`."""
        mean, std = self.gp.predict(embeddings, return_std=True)
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
    score: float  # uncertainty × |0.5 − expected_score|


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
    return ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(0.1, 10.0)
    )


def fit_compliance_gp(
    *,
    embed_client: EmbedClient,
    policy: Policy,
    embedding_model: str | None = None,
    kernel: Kernel | None = None,
    min_samples: int = 20,
    library_anchor_texts: dict[str, str] | None = None,
    n_restarts_optimizer: int = 5,
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

    gp = GaussianProcessRegressor(
        kernel=kernel or _default_kernel(),
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=0,
    )
    gp.fit(X, y)

    library_anchors: dict[str, list[float]] = {}
    if library_anchor_texts:
        for anchor_id, text in library_anchor_texts.items():
            try:
                resp = embed_client.embed(text, model=embedding_model)
                library_anchors[anchor_id] = resp.embedding
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
        training_embeddings=X,
        training_aggregates=y,
        log_marginal_likelihood=float(gp.log_marginal_likelihood_value_),
        kernel_repr=str(gp.kernel_),
        library_anchor_embeddings=library_anchors,
    )


def propose_targets(
    gp: ComplianceGP,
    *,
    n_targets: int = 10,
    candidate_pool_size: int = 1000,
    seed: int = 0,
    perturbation_sigma: float | None = None,
) -> list[GPTarget]:
    """Propose top-K embedding-space targets ranked by uncertainty × boundary risk.

    Candidates are drawn by perturbing each training embedding with
    Gaussian noise; if `perturbation_sigma` is None, the kernel's
    length-scale is used as the perturbation magnitude (so candidates
    explore at the GP's natural correlation distance).
    """
    rng = np.random.default_rng(seed)
    if gp.training_embeddings.size == 0:
        return []
    n_train = gp.training_embeddings.shape[0]
    if perturbation_sigma is None:
        # Best-effort length-scale extraction from sklearn's kernel.
        try:
            length_scale = getattr(gp.gp.kernel_, "length_scale", 1.0)
            if isinstance(length_scale, np.ndarray):
                length_scale = float(np.mean(length_scale))
            perturbation_sigma = float(length_scale)
        except Exception:
            perturbation_sigma = 1.0

    n_candidates = max(1, candidate_pool_size)
    seed_indices = rng.integers(0, n_train, size=n_candidates)
    seeds = gp.training_embeddings[seed_indices]
    noise = rng.normal(0, perturbation_sigma, size=seeds.shape)
    candidates = seeds + noise

    mean, std = gp.predict(candidates)
    risk = np.abs(0.5 - mean)
    score = std * risk

    # Sort candidates by score descending; tie-break by std then index for stability.
    order = np.argsort(-score, kind="stable")
    top_idx = order[:n_targets]
    return [
        GPTarget(
            embedding=[float(v) for v in candidates[i]],
            expected_score=float(mean[i]),
            uncertainty=float(std[i]),
            score=float(score[i]),
        )
        for i in top_idx
    ]


__all__ = [
    "ComplianceGP",
    "GPTarget",
    "TrainingPoint",
    "fit_compliance_gp",
    "propose_targets",
]
