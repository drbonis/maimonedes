"""Phase 2 analytical core: empirical Jacobian and aggregated fragility.

Reads from the DB (perturbation_probes + compliance_scores), writes
nothing. The Jacobian for one anchor is the §4.4 table of
Δscore-per-perturbation-direction; the aggregated fragility table
averages those Δs across all anchors so the doc's prediction
("authority + prescriptive framing dominates") is testable.

Aggregation: each (perturbation_kind, sub-condition) cell is the mean
Δ across (anchor × perturbation-of-that-kind). Anchors without a
recorded baseline are skipped with a structured warning, never
synthesised — that would tilt the aggregate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from sqlalchemy import func, select

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationKind
from maimonedes.storage.compliance import (
    ComplianceScoreRow,
    latest_anchor_baseline,
    row_to_score,
)
from maimonedes.storage.perturbations import PerturbationProbeRow
from maimonedes.storage.repo import get_session


log = logging.getLogger(__name__)

AGGREGATE_COLUMN = "aggregate"


@dataclass(frozen=True)
class JacobianRow:
    transform_label: str
    perturbation_kind: PerturbationKind
    deltas: dict[str, float]  # column id ("aggregate" or sub_id) -> Δ


@dataclass(frozen=True)
class Jacobian:
    """Per-anchor §4.4 table.

    `columns` is `["aggregate", *sub_condition_ids]` (sorted within
    each group for stability). `rows` ordered by total |Δ| descending
    so the most-fragile perturbations float to the top.
    """

    anchor_id: str
    baseline_aggregate: float
    baseline_per_sub_condition: dict[str, float]
    columns: list[str]
    rows: list[JacobianRow]

    def cell(self, transform_label: str, column: str) -> float | None:
        for row in self.rows:
            if row.transform_label == transform_label:
                return row.deltas.get(column)
        return None


@dataclass(frozen=True)
class FragilityCell:
    perturbation_kind: PerturbationKind
    column: str  # "aggregate" or a sub_condition_id
    mean_delta: float
    count: int


@dataclass(frozen=True)
class FragilityTable:
    """Aggregated fragility across all anchors.

    Rows = perturbation kinds; columns = aggregate + sub-conditions.
    Empty cells (no observations for that kind × column) are omitted
    rather than zero-filled, so consumers see explicit absences.
    """

    perturbation_kinds: list[str]
    columns: list[str]
    cells: list[FragilityCell] = field(default_factory=list)

    def cell(self, kind: str, column: str) -> FragilityCell | None:
        for c in self.cells:
            if c.perturbation_kind == kind and c.column == column:
                return c
        return None

    def as_grid(self) -> dict[tuple[str, str], FragilityCell]:
        return {(c.perturbation_kind, c.column): c for c in self.cells}


def _columns_for(baseline: ComplianceScore) -> list[str]:
    return [AGGREGATE_COLUMN, *sorted(baseline.per_sub_condition.keys())]


def _delta_row(
    transform_label: str,
    perturbation_kind: PerturbationKind,
    baseline: ComplianceScore,
    score: ComplianceScore,
) -> JacobianRow:
    deltas: dict[str, float] = {
        AGGREGATE_COLUMN: score.aggregate - baseline.aggregate
    }
    for sub_id, base_val in baseline.per_sub_condition.items():
        pert_val = score.per_sub_condition.get(sub_id, 0.0)
        deltas[sub_id] = pert_val - base_val
    return JacobianRow(
        transform_label=transform_label,
        perturbation_kind=perturbation_kind,
        deltas=deltas,
    )


@dataclass(frozen=True)
class _ProbeScorePair:
    """Detachment-safe snapshot of a (perturbation_probe, score) pair."""

    transform_label: str
    perturbation_kind: str
    score: ComplianceScore


def _scored_perturbations(anchor_id: str) -> list[_ProbeScorePair]:
    """Most-recent perturbation score per (anchor, transform_label).

    Re-running `maimonedes perturb` creates new probe rows with the
    same `transform_label` values; the Jacobian wants the LATEST
    direction, not the cartesian product. Without this dedup, pandas
    Styler refuses to render the per-anchor heatmap because the
    DataFrame index has duplicates.

    Materialises the relevant fields inside the session so the
    returned objects don't trigger detached-attribute lookups in the
    caller.
    """
    with get_session() as session:
        # Latest probe row id per transform_label for this anchor.
        latest_ids = session.execute(
            select(func.max(PerturbationProbeRow.id))
            .where(PerturbationProbeRow.anchor_id == anchor_id)
            .group_by(PerturbationProbeRow.transform_label)
        ).scalars().all()
        if not latest_ids:
            return []

        probe_rows = session.execute(
            select(PerturbationProbeRow).where(
                PerturbationProbeRow.id.in_(latest_ids)
            )
        ).scalars().all()

        out: list[_ProbeScorePair] = []
        for probe in probe_rows:
            score_row = session.execute(
                select(ComplianceScoreRow)
                .where(
                    ComplianceScoreRow.perturbation_id == probe.id,
                    ComplianceScoreRow.probe_role == "perturbation",
                )
                .order_by(
                    ComplianceScoreRow.scored_at.desc(),
                    ComplianceScoreRow.id.desc(),
                )
                .limit(1)
            ).scalar_one_or_none()
            if score_row is None:
                continue
            out.append(
                _ProbeScorePair(
                    transform_label=probe.transform_label,
                    perturbation_kind=probe.perturbation_kind,
                    score=row_to_score(score_row),
                )
            )
        return out


def jacobian_for_anchor(anchor_id: str) -> Jacobian | None:
    """Return the per-anchor §4.4 Jacobian, or None if no baseline exists."""
    baseline = latest_anchor_baseline(anchor_id)
    if baseline is None:
        log.warning(
            "fragility.missing_baseline",
            extra={"anchor_id": anchor_id},
        )
        return None

    pairs = _scored_perturbations(anchor_id)
    rows = [
        _delta_row(
            pair.transform_label,
            pair.perturbation_kind,  # type: ignore[arg-type]
            baseline,
            pair.score,
        )
        for pair in pairs
    ]
    rows.sort(
        key=lambda r: sum(abs(v) for v in r.deltas.values()),
        reverse=True,
    )
    return Jacobian(
        anchor_id=anchor_id,
        baseline_aggregate=baseline.aggregate,
        baseline_per_sub_condition=dict(baseline.per_sub_condition),
        columns=_columns_for(baseline),
        rows=rows,
    )


def _all_anchor_ids_with_baseline() -> list[str]:
    with get_session() as session:
        return list(
            session.execute(
                select(ComplianceScoreRow.anchor_id)
                .where(ComplianceScoreRow.probe_role == "anchor")
                .distinct()
            ).scalars().all()
        )


def aggregated_fragility() -> FragilityTable:
    """Mean Δ across all anchors, indexed by (perturbation_kind, column).

    Each (kind, column) cell is the mean over (anchor × perturbation-
    of-that-kind). Anchors without a baseline are skipped (logged at
    WARNING) so the aggregate stays honest.
    """
    accum: dict[tuple[str, str], list[float]] = {}
    columns_seen: set[str] = set()
    kinds_seen: set[str] = set()

    for anchor_id in _all_anchor_ids_with_baseline():
        jac = jacobian_for_anchor(anchor_id)
        if jac is None:
            continue
        columns_seen.update(jac.columns)
        for row in jac.rows:
            kinds_seen.add(row.perturbation_kind)
            for column, delta in row.deltas.items():
                accum.setdefault(
                    (row.perturbation_kind, column), []
                ).append(delta)

    cells = [
        FragilityCell(
            perturbation_kind=kind,  # type: ignore[arg-type]
            column=column,
            mean_delta=float(np.mean(deltas)),
            count=len(deltas),
        )
        for (kind, column), deltas in sorted(accum.items())
    ]
    # Stable, sorted order: aggregate first, then sub-conditions alphabetically.
    columns_list = (
        ([AGGREGATE_COLUMN] if AGGREGATE_COLUMN in columns_seen else [])
        + sorted(columns_seen - {AGGREGATE_COLUMN})
    )
    return FragilityTable(
        perturbation_kinds=sorted(kinds_seen),
        columns=columns_list,
        cells=cells,
    )


def all_jacobians() -> dict[str, Jacobian]:
    """Compute the per-anchor Jacobian for every anchor with a baseline."""
    out: dict[str, Jacobian] = {}
    for anchor_id in _all_anchor_ids_with_baseline():
        jac = jacobian_for_anchor(anchor_id)
        if jac is not None:
            out[anchor_id] = jac
    return out


__all__ = [
    "AGGREGATE_COLUMN",
    "FragilityCell",
    "FragilityTable",
    "Jacobian",
    "JacobianRow",
    "aggregated_fragility",
    "all_jacobians",
    "jacobian_for_anchor",
]
