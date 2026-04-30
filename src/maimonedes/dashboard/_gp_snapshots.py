"""Picklable snapshot types for the GP page's `st.cache_data` callers.

Same gotcha as the drift / recovery pages: classes defined inside a
numeric-prefixed page module aren't picklable through `st.cache_data`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class GPFitSnapshot:
    """Picklable summary used by the gp-fit dropdown."""

    fit_id: int
    path: str
    policy_id: str
    n_samples: int
    log_marginal_likelihood: float
    kernel_name: str
    embedding_model: str
    trained_at: datetime | None
    label: str


@dataclass
class GPScatterPoint:
    """One point in the 2D embedding-projection scatter plot."""

    kind: str  # "training" | "target" | "anchor"
    x: float
    y: float
    aggregate: float | None  # only for "training"
    score: float | None  # GP target priority — only for "target"
    label: str  # tooltip
    uncertainty: float | None = None  # GP target — only for "target"


@dataclass
class GPSnapshot:
    """Full per-fit data the dashboard renders."""

    fit_id: int
    points: list[GPScatterPoint] = field(default_factory=list)
    target_rows: list[dict] = field(default_factory=list)
    projection_method: str = "PCA"  # "UMAP" or "PCA" — what produced `points`


__all__ = ["GPFitSnapshot", "GPScatterPoint", "GPSnapshot"]
