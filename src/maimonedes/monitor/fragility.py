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

Replicate-aware aggregation: when `maimonedes perturb` is run with
`--replicates N`, each `(anchor, transform_label)` produces N probe
rows sharing a single `run_id` stamped in their `generator_metadata`.
The Jacobian groups by `transform_label` within the most-recent
`run_id`, computes per-axis mean ± std across the N replicates, and
reports both. With N=1 the std is zero by construction.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from sqlalchemy import select

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
    deltas: dict[str, float]  # mean Δ per column (single Δ when n=1)
    std_deltas: dict[str, float] = field(default_factory=dict)  # std Δ across replicates; all-zero when n=1
    n_replicates: int = 1


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
    count: int  # number of contributing anchors (each anchor's mean Δ counts once)
    std_delta: float = 0.0  # between-anchor std (zero when only one anchor contributes)


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


def _delta_row_from_replicates(
    transform_label: str,
    perturbation_kind: PerturbationKind,
    baseline: ComplianceScore,
    scores: list[ComplianceScore],
) -> JacobianRow:
    """Aggregate one or more replicate scores into a single JacobianRow.

    For each column (aggregate + every sub-condition id), computes
    `mean(replicate_score - baseline_score)` and `std(...)` across the
    replicate samples. With one replicate the std is zero.
    """
    if not scores:
        raise ValueError("scores list must be non-empty")

    by_column: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        by_column[AGGREGATE_COLUMN].append(score.aggregate - baseline.aggregate)
        for sub_id, base_val in baseline.per_sub_condition.items():
            pert_val = score.per_sub_condition.get(sub_id, 0.0)
            by_column[sub_id].append(pert_val - base_val)

    deltas = {col: float(np.mean(vals)) for col, vals in by_column.items()}
    std_deltas: dict[str, float] = {}
    for col, vals in by_column.items():
        if len(vals) > 1:
            std_deltas[col] = float(np.std(vals, ddof=1))
        else:
            std_deltas[col] = 0.0

    return JacobianRow(
        transform_label=transform_label,
        perturbation_kind=perturbation_kind,
        deltas=deltas,
        std_deltas=std_deltas,
        n_replicates=len(scores),
    )


@dataclass(frozen=True)
class _ProbeScorePair:
    """Detachment-safe snapshot of a (perturbation_probe, score) pair."""

    transform_label: str
    perturbation_kind: str
    score: ComplianceScore
    run_id: str | None  # from generator_metadata; None for legacy probes


def _parse_metadata(probe: PerturbationProbeRow) -> dict[str, Any]:
    try:
        parsed = json.loads(probe.generator_metadata_json or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _scored_perturbations(anchor_id: str) -> list[_ProbeScorePair]:
    """Replicates of every `transform_label` from the most recent run.

    Behaviour:

    - For probes with `run_id` in `generator_metadata` (post-replicates
      schema): for each `transform_label`, find the highest-id probe
      and use its `run_id` as the canonical "latest run". Return ALL
      probes from that `(transform_label, run_id)` tuple — those are
      the replicates the Jacobian aggregates across.
    - For legacy probes without `run_id`: fall back to the previous
      single-row dedup — the highest-id probe per `transform_label`
      wins. This preserves the prior `re-running perturb means latest
      observation supersedes` semantics.
    """
    with get_session() as session:
        all_probes = (
            session.execute(
                select(PerturbationProbeRow)
                .where(PerturbationProbeRow.anchor_id == anchor_id)
                .order_by(PerturbationProbeRow.id.desc())
            )
            .scalars()
            .all()
        )
        if not all_probes:
            return []

        # First pass: for each transform_label, identify the run_id (or
        # absence thereof) of the latest probe. That's the canonical
        # "current run" for that label.
        latest_run_per_label: dict[str, str | None] = {}
        latest_id_per_label: dict[str, int] = {}
        for probe in all_probes:  # already ordered id desc
            label = probe.transform_label
            if label in latest_run_per_label:
                continue
            metadata = _parse_metadata(probe)
            latest_run_per_label[label] = metadata.get("run_id")
            latest_id_per_label[label] = probe.id

        # Second pass: keep probes that match the canonical run for
        # their label. Legacy (run_id=None): only the highest-id probe
        # qualifies (single observation, no replicates). New (run_id
        # set): all probes sharing that run_id are replicate samples.
        out: list[_ProbeScorePair] = []
        for probe in all_probes:
            label = probe.transform_label
            metadata = _parse_metadata(probe)
            probe_run = metadata.get("run_id")
            expected_run = latest_run_per_label[label]

            if expected_run is None:
                if probe.id != latest_id_per_label[label]:
                    continue
            else:
                if probe_run != expected_run:
                    continue

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
                    run_id=probe_run,
                )
            )
        return out


def jacobian_for_anchor(anchor_id: str) -> Jacobian | None:
    """Return the per-anchor §4.4 Jacobian, or None if no baseline exists.

    With replicate-aware aggregation, each `JacobianRow` carries
    mean Δ + std Δ + n_replicates per column.
    """
    baseline = latest_anchor_baseline(anchor_id)
    if baseline is None:
        log.warning(
            "fragility.missing_baseline",
            extra={"anchor_id": anchor_id},
        )
        return None

    pairs = _scored_perturbations(anchor_id)

    # Group replicates by transform_label.
    grouped: dict[str, list[_ProbeScorePair]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.transform_label].append(pair)

    rows = [
        _delta_row_from_replicates(
            label,
            label_pairs[0].perturbation_kind,  # type: ignore[arg-type]
            baseline,
            [p.score for p in label_pairs],
        )
        for label, label_pairs in grouped.items()
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
    of-that-kind), where each anchor's contribution is its own
    *replicate-mean* Δ for that column. The reported `std_delta` is
    the between-anchor std of those means; the within-anchor replicate
    std is preserved at the per-anchor Jacobian level.

    Anchors without a baseline are skipped (logged at WARNING) so the
    aggregate stays honest.
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

    cells = []
    for (kind, column), deltas in sorted(accum.items()):
        cells.append(
            FragilityCell(
                perturbation_kind=kind,  # type: ignore[arg-type]
                column=column,
                mean_delta=float(np.mean(deltas)),
                std_delta=float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                count=len(deltas),
            )
        )
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
