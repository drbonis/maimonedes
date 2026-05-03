"""Picklable snapshot types for the metric page's `st.cache_data` callers.

Same caching gotcha as the drift / recovery / GP pages: page filenames
under `dashboard/pages/` start with digits, so classes defined inside
the page module aren't picklable through `st.cache_data`. Define them
here instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MetricFitSnapshot:
    """Picklable summary used by the metric-fit dropdown."""

    fit_id: int
    path: str
    policy_id: str
    n_anchors: int
    n_jacobians: int
    val_loss: float | None
    train_loss: float | None
    trained_at: datetime | None
    label: str


@dataclass
class MetricEllipse:
    """One sub-grid ellipse in coordinate (c_i, c_j) space.

    `xs` / `ys` are the ellipse polygon points (already transformed +
    translated to the ellipse's center). `eig_ratio` is the metric's
    anisotropy at this cell — used for tooltips, not the geometry.
    """

    cx: float
    cy: float
    xs: list[float]
    ys: list[float]
    eig_ratio: float


@dataclass
class AnchorOverlay:
    """One anchor projected onto the chosen axis pair."""

    anchor_id: str
    cx: float
    cy: float
    c_full: tuple[float, ...]
    euclidean_boundary: float
    riemannian_boundary: float
    ratio: float | None  # riemannian / euclidean; None when euclidean ≈ 0


@dataclass
class GridSnapshot:
    """Heatmap + ellipses + anchors for one (fit, axis_pair, pins) selection."""

    fit_id: int
    axes: tuple[str, ...]
    axis_i: int
    axis_j: int
    pinned: tuple[float, ...]
    xs: list[float]
    ys: list[float]
    sqrt_det_g: list[list[float]]
    ellipses: list[MetricEllipse] = field(default_factory=list)
    anchors: list[AnchorOverlay] = field(default_factory=list)


@dataclass
class PresetSegment:
    """One (from, to) segment for the §4.6 worked-example presets."""

    label: str
    c_from: tuple[float, ...]
    c_to: tuple[float, ...]
    euclidean: float
    riemannian: float
    ratio: float | None


@dataclass
class DistanceComparison:
    """Result of the comparator panel for the current (c0, c1) pair."""

    c0: tuple[float, ...]
    c1: tuple[float, ...]
    euclidean: float
    riemannian: float
    ratio: float | None


@dataclass
class RatioSurfaceSnapshot:
    """Z-grid for the 3D ratio-surface plot.

    `mode` is `"fixed_reference"` (Z = riemannian/euclidean from `reference`)
    or `"local_stretch"` (Z = √λ_max(g(c))). `reference` is None in the
    local-stretch mode.
    """

    fit_id: int
    mode: str
    axis_i: int
    axis_j: int
    axis_i_id: str
    axis_j_id: str
    pinned: tuple[float, ...]
    reference: tuple[float, ...] | None
    xs: list[float]
    ys: list[float]
    z: list[list[float]]


__all__ = [
    "AnchorOverlay",
    "DistanceComparison",
    "GridSnapshot",
    "MetricEllipse",
    "MetricFitSnapshot",
    "PresetSegment",
    "RatioSurfaceSnapshot",
]
