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
    """Z-grid for the 3D ratio-surface plot (topographic convention).

    Z is `euclidean/riemannian` (fixed_reference) or `1/√λ_max(g(c))`
    (local_stretch). High = stable plateau; low = fragile cliff.
    `reference` is None in local-stretch mode.
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


@dataclass
class RadialCloudPoint:
    """One sampled compliance vector projected to 2D star-coordinates."""

    x: float
    y: float
    z: float  # NaN at the reference point
    c_full: tuple[float, ...]
    is_anchor: bool


@dataclass
class PerturbationArrow:
    """One Jacobian-row delta projected onto the chosen 2D axis pair.

    `(cx, cy)` is the arrow tail (anchor's projected position);
    `(dx, dy)` is the arrow vector (Δ projected onto the same plane).
    `full_delta` carries the unprojected k-vector for hover-text and
    for the bar-chart computation that uses the full-dim metric norm.
    """

    anchor_id: str
    perturbation_kind: str
    transform_label: str
    cx: float
    cy: float
    dx: float
    dy: float
    full_delta: tuple[float, ...]


@dataclass
class BoundaryGradientArrow:
    """Riemannian (contravariant) gradient of the margin at one anchor.

    The arrow points toward fastest *increase* of the margin (away from
    the boundary). Negate to get steepest descent. Direction in 2D is
    the projection of −g(c)⁻¹·w onto the chosen axis pair.
    """

    anchor_id: str
    cx: float
    cy: float
    dx: float
    dy: float
    full_descent: tuple[float, ...]


@dataclass
class PerturbationEfficiencyEntry:
    """One ranked row of the per-anchor boundary-closure efficiency chart.

    `efficiency = -Δ·w / ‖Δ‖_g` (higher = more compliance erosion per
    unit Riemannian step). `riem_norm` is the denominator,
    `boundary_alignment` is the numerator before normalisation.
    """

    anchor_id: str
    perturbation_kind: str
    transform_label: str
    efficiency: float
    boundary_alignment: float
    riem_norm: float


@dataclass
class RadialCloudSnapshot:
    """Star-coordinate scatter cloud + radial axis labels.

    Each point projects c ∈ [0,1]^k via star coords (each axis at angle
    2π·i/k) into (x, y); Z = euclidean/riemannian from `reference`
    (high = stable, low = fragile). The mapping is many-to-one for
    k > 2; collisions stack visually in the scatter.
    """

    fit_id: int
    n_samples: int
    seed: int
    include_anchors: bool
    reference: tuple[float, ...]
    axis_ids: tuple[str, ...]
    points: list[RadialCloudPoint]
    axis_labels: list[tuple[int, float, float]]  # (axis_index, label_x, label_y)
    r_max: float


__all__ = [
    "AnchorOverlay",
    "BoundaryGradientArrow",
    "DistanceComparison",
    "GridSnapshot",
    "MetricEllipse",
    "MetricFitSnapshot",
    "PerturbationArrow",
    "PerturbationEfficiencyEntry",
    "PresetSegment",
    "RadialCloudPoint",
    "RadialCloudSnapshot",
    "RatioSurfaceSnapshot",
]
