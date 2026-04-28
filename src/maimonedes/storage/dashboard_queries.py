"""Query helpers used by the Streamlit dashboard.

The dashboard module only calls into this layer; no SQL or
pydantic-massaging lives in `dashboard/app.py`. That keeps the
`AppTest` smoke test honest — if the helpers return the right
shapes, the page is just glue.
"""
from __future__ import annotations

from collections.abc import Mapping

from maimonedes.core.compliance import ComplianceScore
from maimonedes.storage.compliance import latest_score_per_anchor, recent_scores


def latest_table_rows() -> list[dict[str, object]]:
    """Return one row per anchor, sorted by anchor_id, ready for a table.

    Each row holds the fields the page renders directly so the dashboard
    doesn't need to peek into pydantic internals.
    """
    latest = latest_score_per_anchor()
    rows: list[dict[str, object]] = []
    for anchor_id in sorted(latest.keys()):
        score = latest[anchor_id]
        rows.append(
            {
                "anchor_id": anchor_id,
                "aggregate": float(score.aggregate),
                "policy_id": score.policy_id,
                "judge_model": score.judge_model,
                "supervised_model": score.supervised_model,
                "scored_at": score.scored_at,
            }
        )
    return rows


def history_for_anchor(anchor_id: str, *, limit: int = 20) -> list[ComplianceScore]:
    """Most-recent scores for one anchor, OLDEST first (chart-friendly)."""
    rows = recent_scores(anchor_id, limit=limit)
    return list(reversed(rows))


def per_sub_condition(scores: Mapping[str, ComplianceScore]) -> dict[str, dict[str, float]]:
    """Flatten {anchor_id: ComplianceScore} into {anchor_id: {sub_id: value}}."""
    return {aid: dict(score.per_sub_condition) for aid, score in scores.items()}


__all__ = [
    "history_for_anchor",
    "latest_table_rows",
    "per_sub_condition",
]
