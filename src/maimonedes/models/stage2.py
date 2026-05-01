"""Phase 5 Stage-2 classifier — offline training + on-disk model artefact.

Per the locked decision in the planning thread: per-axis Ridge
regression on Bio_ClinicalBERT embeddings. Aggregation matches the
rubric's weighted sum exactly so Stage-1 (LLM judge) and Stage-2
(linear heads) are directly comparable.
"""
from __future__ import annotations

import logging
import pickle
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy
from maimonedes.llm.embed_client import EmbedClient
from maimonedes.storage.llm_calls import pair_supervised_with_scores

HeadKind = Literal["ridge", "mlp"]


log = logging.getLogger(__name__)


@dataclass
class TrainingExample:
    """One (text, score_vector, anchor_id) triple after the SQL pull."""

    anchor_id: str
    text: str
    score: ComplianceScore


@dataclass
class AxisMetrics:
    """Per-axis agreement metric vs Stage-1 on the held-out split."""

    sub_id: str
    mae: float
    spearman_rho: float


@dataclass
class Stage2Model:
    """Trained per-axis Stage-2 classifier — picklable artefact.

    `heads` carries the per-axis predictor, which is either a bare
    `sklearn.linear_model.Ridge` (legacy v1 artefacts saved before
    #45) or a `Pipeline(StandardScaler, head)` (post-#45 trainer).
    Both expose `.predict(X) -> ndarray`, so callers stay agnostic.
    """

    policy_id: str
    trained_at: datetime
    n_samples: int
    n_train: int
    n_eval: int
    embedding_model: str
    feature_dim: int
    heads: dict[str, Pipeline | Ridge]
    agreement_metrics: list[AxisMetrics]
    agreement_status: str  # "green" | "amber" | "red"
    head_kind: HeadKind = "ridge"

    def predict_per_axis(self, embedding: Sequence[float]) -> dict[str, float]:
        x = np.asarray(embedding, dtype=float).reshape(1, -1)
        out: dict[str, float] = {}
        for sub_id, head in self.heads.items():
            raw = float(head.predict(x)[0])
            out[sub_id] = float(np.clip(raw, 0.0, 1.0))
        return out

    def predict_aggregate(
        self,
        embedding: Sequence[float],
        *,
        policy: Policy,
    ) -> tuple[float, dict[str, float]]:
        """Return `(aggregate, per_sub_condition)`. Aggregation matches Judge."""
        per_sub = self.predict_per_axis(embedding)
        total = sum(s.weight * per_sub[s.id] for s in policy.rubric.sub_conditions)
        return min(1.0, max(0.0, total)), per_sub

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Stage2Model":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise ValueError(f"{path}: pickle did not contain a Stage2Model")
        return obj


# ---- training pipeline -----------------------------------------------------


def _fetch_training_examples(
    policy_id: str, *, supervised_backend_prefix: str = "ollama-supervised"
) -> list[TrainingExample]:
    """Pull (supervised_text, score) pairs via the chronological pairing helper.

    See `storage.llm_calls.pair_supervised_with_scores` for the
    rationale (Phase 1+ orchestrators do not wire
    `compliance_scores.llm_call_id`).
    """
    pairs = pair_supervised_with_scores(
        policy_id, supervised_backend_prefix=supervised_backend_prefix
    )
    return [
        TrainingExample(anchor_id=p.anchor_id, text=p.supervised_text, score=p.score)
        for p in pairs
    ]


def _stratified_split(
    examples: list[TrainingExample],
    *,
    eval_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """80/20 split, stratified by anchor_id where possible.

    Anchors with one example go to the training set (eval needs ≥1 ground
    truth per anchor for spearman to make sense; we'd lose the per-axis
    structure on a single sample anyway).
    """
    rng = np.random.default_rng(seed)
    by_anchor: dict[str, list[TrainingExample]] = {}
    for ex in examples:
        by_anchor.setdefault(ex.anchor_id, []).append(ex)
    train: list[TrainingExample] = []
    evalset: list[TrainingExample] = []
    for anchor_id, group in by_anchor.items():
        if len(group) < 2:
            train.extend(group)
            continue
        rng.shuffle(group)
        n_eval = max(1, int(round(len(group) * eval_fraction)))
        evalset.extend(group[:n_eval])
        train.extend(group[n_eval:])
    return train, evalset


def _spearman_rho(y_true: list[float], y_pred: list[float]) -> float:
    """Spearman ρ; defined to be 0 when one of the arrays has zero variance."""
    if len(y_true) < 2:
        return 0.0
    if len(set(y_true)) < 2 or len(set(y_pred)) < 2:
        return 0.0
    rho, _p = stats.spearmanr(y_true, y_pred)
    if np.isnan(rho):
        return 0.0
    return float(rho)


GREEN_MAE_MAX = 0.16
GREEN_RHO_MIN = 0.6
AMBER_MAE_MAX = 0.25
AMBER_RHO_MIN = 0.4


def _grade_agreement(metrics: list[AxisMetrics]) -> str:
    if not metrics:
        return "red"
    statuses: list[str] = []
    for m in metrics:
        if m.spearman_rho >= GREEN_RHO_MIN and m.mae <= GREEN_MAE_MAX:
            statuses.append("green")
        elif m.spearman_rho >= AMBER_RHO_MIN or m.mae <= AMBER_MAE_MAX:
            statuses.append("amber")
        else:
            statuses.append("red")
    if all(s == "green" for s in statuses):
        return "green"
    if any(s == "red" for s in statuses):
        return "red"
    return "amber"


def _build_head(
    head_kind: HeadKind, *, ridge_alpha: float, seed: int
) -> Pipeline:
    """Per-axis pipeline: StandardScaler → Ridge or MLPRegressor.

    Wrapping every head in a `Pipeline` keeps `Stage2Model.predict_per_axis`
    agnostic to the head family. The scaler is fit on the training
    embeddings; at predict time it normalises new embeddings before
    they hit the regressor.
    """
    if head_kind == "ridge":
        regressor = Ridge(alpha=ridge_alpha, random_state=seed)
    elif head_kind == "mlp":
        regressor = MLPRegressor(
            hidden_layer_sizes=(256,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=500,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown head_kind {head_kind!r}; expected 'ridge' or 'mlp'")
    return Pipeline(steps=[("scaler", StandardScaler()), ("head", regressor)])


def train_stage2(
    *,
    policy: Policy,
    embed_client: EmbedClient,
    embedding_model: str | None = None,
    min_samples: int = 50,
    eval_fraction: float = 0.2,
    ridge_alpha: float = 1.0,
    seed: int = 0,
    head_kind: HeadKind = "ridge",
) -> Stage2Model:
    """Pull training data, embed, fit per-axis heads, return artefact.

    Each axis gets its own `Pipeline(StandardScaler, head)`. The head
    family is chosen by `head_kind`:
    - `"ridge"` (default) — `sklearn.linear_model.Ridge(alpha=ridge_alpha)`.
      Same behaviour as pre-#45 trainer.
    - `"mlp"` — `sklearn.neural_network.MLPRegressor` with a single
      256-unit ReLU hidden layer. For axes whose Ridge ρ is weak but
      MAE is small (linear underfit), the MLP often recovers ranking
      power. Cost: ~1–2 minutes total train time across axes.

    Raises `ValueError` if fewer than `min_samples` (text, score) pairs are
    available — agreement metrics on tiny datasets are meaningless.
    """
    examples = _fetch_training_examples(policy.id)
    if len(examples) < min_samples:
        raise ValueError(
            f"only {len(examples)} samples available; need at least "
            f"{min_samples} to train a Stage-2 model for policy {policy.id!r}"
        )

    train, evalset = _stratified_split(examples, eval_fraction=eval_fraction, seed=seed)
    log.info(
        "stage2.split",
        extra={"n_total": len(examples), "n_train": len(train), "n_eval": len(evalset)},
    )

    embedding_model = embedding_model or "bio_clinicalbert"

    train_embeddings: list[list[float]] = []
    for ex in train:
        resp = embed_client.embed(ex.text, model=embedding_model)
        train_embeddings.append(resp.embedding)
    eval_embeddings: list[list[float]] = []
    for ex in evalset:
        resp = embed_client.embed(ex.text, model=embedding_model)
        eval_embeddings.append(resp.embedding)

    feature_dim = len(train_embeddings[0]) if train_embeddings else 0
    if feature_dim == 0:
        raise ValueError("training embeddings are empty; cannot fit a model")

    X_train = np.asarray(train_embeddings, dtype=float)
    X_eval = np.asarray(eval_embeddings, dtype=float) if eval_embeddings else None

    sub_conditions = policy.rubric.sub_conditions
    heads: dict[str, Pipeline | Ridge] = {}
    metrics: list[AxisMetrics] = []
    for sub in sub_conditions:
        y_train = np.asarray(
            [ex.score.per_sub_condition.get(sub.id, 0.0) for ex in train],
            dtype=float,
        )
        pipeline = _build_head(head_kind, ridge_alpha=ridge_alpha, seed=seed)
        pipeline.fit(X_train, y_train)
        heads[sub.id] = pipeline

        if X_eval is not None and X_eval.shape[0] > 0:
            y_eval = [ex.score.per_sub_condition.get(sub.id, 0.0) for ex in evalset]
            preds = np.clip(pipeline.predict(X_eval), 0.0, 1.0).tolist()
            mae = float(np.mean(np.abs(np.asarray(y_eval) - np.asarray(preds))))
            rho = _spearman_rho(list(y_eval), preds)
        else:
            mae = float("nan")
            rho = float("nan")
        metrics.append(AxisMetrics(sub_id=sub.id, mae=mae, spearman_rho=rho))

    status = _grade_agreement(metrics) if X_eval is not None and X_eval.shape[0] > 0 else "amber"

    return Stage2Model(
        policy_id=policy.id,
        trained_at=datetime.now(timezone.utc),
        n_samples=len(examples),
        n_train=len(train),
        n_eval=len(evalset),
        embedding_model=embedding_model,
        feature_dim=feature_dim,
        heads=heads,
        agreement_metrics=metrics,
        agreement_status=status,
        head_kind=head_kind,
    )


__all__ = [
    "AxisMetrics",
    "HeadKind",
    "Stage2Model",
    "TrainingExample",
    "train_stage2",
]
