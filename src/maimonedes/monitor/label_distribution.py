"""Per-axis rubric-label distribution diagnostic.

For each sub-condition in a policy's rubric, summarise the distribution
of normalised values seen across `compliance_scores.per_sub_condition_json`.
The diagnostic distinguishes "head can't fit" (entropy ~ 1.0) from
"labels are near-degenerate" (entropy << 1.0). See issue #46.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import select, text

from maimonedes.core.policy import Policy, SubCondition
from maimonedes.storage.compliance import ComplianceScoreRow
from maimonedes.storage.repo import get_session

NEAR_UNIFORM_THRESHOLD = 0.85


@dataclass(frozen=True)
class Bucket:
    """One row of a per-axis histogram."""

    value: float
    label_id: str | None  # only set for labels-scale axes
    count: int
    share: float  # count / n_rows


@dataclass(frozen=True)
class AxisDistribution:
    """Per-sub-condition distribution summary."""

    sub_id: str
    scale: str
    n_rows: int
    buckets: list[Bucket]
    dominant_share: float  # largest bucket share, or 0.0 if no rows
    entropy_bits: float  # H(P) in bits
    entropy_normalised: float  # H(P) / log2(n_levels)

    @property
    def near_uniform(self) -> bool:
        return self.dominant_share > NEAR_UNIFORM_THRESHOLD


@dataclass(frozen=True)
class LabelDistributionReport:
    policy_id: str
    n_rows_total: int
    axes: list[AxisDistribution]  # sorted by dominant_share desc

    @property
    def flagged_axes(self) -> list[AxisDistribution]:
        return [a for a in self.axes if a.near_uniform]


def _expected_levels(sub: SubCondition) -> int:
    """How many discrete levels the rubric defines for this sub-condition."""
    if sub.scale == "boolean":
        return 2
    if sub.scale == "0-3":
        return 4
    if sub.scale == "labels":
        assert sub.labels is not None
        return len(sub.labels)
    raise ValueError(f"unknown scale {sub.scale!r}")


def _value_to_label_id(sub: SubCondition, value: float) -> str | None:
    """Map a normalised value back to its label id, or None if no match.

    Labels-scale only. Tolerant to small float drift (1e-6).
    """
    if sub.scale != "labels" or sub.labels is None:
        return None
    for label in sub.labels:
        if abs(label.value - value) < 1e-6:
            return label.id
    return None


def _quantise(value: float) -> float:
    """Round to 3 decimals so tiny float drift doesn't fragment buckets."""
    return round(float(value), 3)


def _entropy(counts: list[int]) -> float:
    """Shannon entropy in bits over a count vector."""
    total = sum(counts)
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def _build_axis(
    sub: SubCondition, values: list[float]
) -> AxisDistribution:
    n_rows = len(values)
    counter: Counter[float] = Counter(_quantise(v) for v in values)
    if n_rows == 0:
        return AxisDistribution(
            sub_id=sub.id,
            scale=sub.scale,
            n_rows=0,
            buckets=[],
            dominant_share=0.0,
            entropy_bits=0.0,
            entropy_normalised=0.0,
        )

    items = sorted(counter.items(), key=lambda kv: kv[0])
    buckets = [
        Bucket(
            value=v,
            label_id=_value_to_label_id(sub, v),
            count=c,
            share=c / n_rows,
        )
        for v, c in items
    ]
    dominant = max(b.share for b in buckets)
    h_bits = _entropy([b.count for b in buckets])
    n_levels = _expected_levels(sub)
    h_norm = h_bits / math.log2(n_levels) if n_levels > 1 else 0.0
    return AxisDistribution(
        sub_id=sub.id,
        scale=sub.scale,
        n_rows=n_rows,
        buckets=buckets,
        dominant_share=dominant,
        entropy_bits=h_bits,
        entropy_normalised=h_norm,
    )


def compute_distribution(
    policy: Policy,
    *,
    filter_sql: str | None = None,
) -> LabelDistributionReport:
    """Compute per-axis label distribution from `compliance_scores`.

    Parameters
    ----------
    policy
        Loaded `Policy` — defines which sub-conditions to summarise.
    filter_sql
        Optional SQL fragment appended after `WHERE policy_id = :pid AND `.
        Example: ``judge_model LIKE 'judge%'``. Untrusted user input must
        not be passed here; this is an operator-level diagnostic with no
        web-facing surface.
    """
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    per_axis: dict[str, list[float]] = {sid: [] for sid in sub_ids}

    with get_session() as session:
        stmt = select(ComplianceScoreRow.per_sub_condition_json).where(
            ComplianceScoreRow.policy_id == policy.id
        )
        if filter_sql:
            stmt = stmt.where(text(filter_sql))
        rows = session.execute(stmt).scalars().all()

    n_rows_total = 0
    for raw in rows:
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        n_rows_total += 1
        for sid in sub_ids:
            v = payload.get(sid)
            if isinstance(v, (int, float)):
                per_axis[sid].append(float(v))

    axes = [
        _build_axis(s, per_axis[s.id])
        for s in policy.rubric.sub_conditions
    ]
    axes.sort(key=lambda a: a.dominant_share, reverse=True)
    return LabelDistributionReport(
        policy_id=policy.id,
        n_rows_total=n_rows_total,
        axes=axes,
    )


def format_report(report: LabelDistributionReport) -> list[str]:
    """Plain-text formatting matching drift-report / recovery-report style."""
    lines: list[str] = []
    lines.append(
        f"policy={report.policy_id} n_rows={report.n_rows_total} "
        f"axes={len(report.axes)} threshold={NEAR_UNIFORM_THRESHOLD:.2f}"
    )
    if report.n_rows_total == 0:
        lines.append("(no compliance_scores rows for this policy)")
        return lines

    for axis in report.axes:
        lines.append("")
        flag = "  ⚠ near-uniform — rubric refinement recommended" if axis.near_uniform else ""
        lines.append(f"axis: {axis.sub_id}  scale={axis.scale}{flag}")
        if not axis.buckets:
            lines.append("  (no rows)")
            continue
        for b in axis.buckets:
            label_str = f" ({b.label_id})" if b.label_id else ""
            lines.append(
                f"  {b.value:>5.3f}{label_str:<28s} "
                f"{b.count:>6d}  {b.share * 100:>5.1f}%"
            )
        lines.append(
            f"  -- n={axis.n_rows}  unique={len(axis.buckets)}  "
            f"dominant_share={axis.dominant_share:.3f}  "
            f"entropy={axis.entropy_bits:.3f} bits "
            f"(norm={axis.entropy_normalised:.3f})"
        )
    return lines


__all__ = [
    "AxisDistribution",
    "Bucket",
    "LabelDistributionReport",
    "NEAR_UNIFORM_THRESHOLD",
    "compute_distribution",
    "format_report",
]
