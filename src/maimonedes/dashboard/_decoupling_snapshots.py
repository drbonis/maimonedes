"""Picklable snapshot types for the decoupling page's `st.cache_data` callers.

Same caching gotcha as `_drift_snapshots`: page filenames under
`dashboard/pages/` start with digits, so classes defined inside the
page module aren't picklable through `st.cache_data`. Define them
here instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AnchorDecouplingRow:
    """Per-anchor row of the decoupling comparison table."""

    anchor_id: str
    evidence_count: int
    frobenius_delta: float
    n_flipped_pairs: int
    flipped_pair_labels: tuple[tuple[str, str], ...]
    h_decoupling: float
    fired: bool


@dataclass
class CovarianceMatrix:
    """Symmetric k×k covariance matrix flattened for caching."""

    axis_ids: tuple[str, ...]
    matrix: list[list[float]]


@dataclass
class ScatterCloud:
    """Two-window scatter projected onto an axis pair (for the rotation view)."""

    anchor_id: str
    axis_i: int
    axis_j: int
    axis_i_id: str
    axis_j_id: str
    baseline_x: list[float]
    baseline_y: list[float]
    current_x: list[float]
    current_y: list[float]
    baseline_pca_line: tuple[tuple[float, float], tuple[float, float]] | None
    current_pca_line: tuple[tuple[float, float], tuple[float, float]] | None
    rotation_degrees: float | None


@dataclass
class StructuralAlertRow:
    """One row of the structural-signals decoupling feed."""

    fired_at: datetime | None
    anchor_id: str
    metric_value: float
    threshold: float
    evidence: str  # compact JSON-ish summary


@dataclass
class DecouplingSnapshot:
    """All page data for one (anchor, baseline_window, current_window) selection."""

    anchor_id: str
    baseline_window: int
    current_window: int
    axis_ids: tuple[str, ...]
    rows: list[AnchorDecouplingRow] = field(default_factory=list)
    selected_baseline_cov: CovarianceMatrix | None = None
    selected_current_cov: CovarianceMatrix | None = None
    selected_flipped_pairs: tuple[tuple[int, int], ...] = field(
        default_factory=tuple
    )


__all__ = [
    "AnchorDecouplingRow",
    "CovarianceMatrix",
    "DecouplingSnapshot",
    "ScatterCloud",
    "StructuralAlertRow",
]
