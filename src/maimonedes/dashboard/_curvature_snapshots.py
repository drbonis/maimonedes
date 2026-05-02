"""Picklable snapshot types for the curvature page's `st.cache_data` callers.

Same caching gotcha as `_drift_snapshots`: page filenames under
`dashboard/pages/` start with digits, so classes defined inside the
page module aren't picklable through `st.cache_data`. Define them
here instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CurvatureFitOption:
    """One row of the metric_fit dropdown."""

    fit_id: int
    path: str
    policy_id: str
    n_anchors: int
    val_loss: float | None
    trained_at: datetime | None
    label: str


@dataclass
class AnchorCurvatureRow:
    """One row of the per-anchor κ comparison table."""

    anchor_id: str
    c_anchor: tuple[float, ...]
    kappa_baseline: float
    kappa_current: float
    relative_increase: float
    fired: bool
    error: str | None = None


@dataclass
class EigenSpectrum:
    """Per-anchor eigenvalue spectrum, baseline + current."""

    anchor_id: str
    axis_ids: tuple[str, ...]
    eigvals_baseline: list[float]
    eigvals_current: list[float]


@dataclass
class EllipsePairData:
    """Two ellipses for the same anchor on the chosen axis pair."""

    anchor_id: str
    axis_i: int
    axis_j: int
    cx: float
    cy: float
    baseline_xs: list[float]
    baseline_ys: list[float]
    current_xs: list[float]
    current_ys: list[float]


@dataclass
class StructuralAlertRow:
    """One row of the structural-signals feed table."""

    fired_at: datetime | None
    anchor_id: str
    metric_value: float
    threshold: float
    evidence: str  # compact JSON-ish summary


@dataclass
class CurvatureSnapshot:
    """Full per-pair view-data the page renders."""

    baseline_fit_id: int
    current_fit_id: int
    axis_ids: tuple[str, ...]
    rows: list[AnchorCurvatureRow] = field(default_factory=list)
    spectra: dict[str, EigenSpectrum] = field(default_factory=dict)


__all__ = [
    "AnchorCurvatureRow",
    "CurvatureFitOption",
    "CurvatureSnapshot",
    "EigenSpectrum",
    "EllipsePairData",
    "StructuralAlertRow",
]
