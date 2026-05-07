"""Contrastive pair extraction for Phase 4.

Two constructors, one shape: `ContrastivePair`. The synthesizer (#31)
takes the same shape from either constructor.

Temporal (`temporal_pair`):
    safe              = latest baseline-stage score for the anchor in the
                        given drift run
    near_boundary     = score at first CUSUM fire if available, else the
                        worst-aggregate session

Fragility (`fragility_pair`):
    safe              = most-recent anchor-baseline score
    near_boundary     = the perturbation row whose aggregate Δ is largest
                        (lowest aggregate) for this anchor

Both pairs report `dropoff_axes`: the rubric sub-condition ids ranked
by descending |safe - near_boundary| per axis. The synthesizer uses
this list as the "dominant gradient direction" hint.
"""
from __future__ import annotations

import logging
from datetime import timezone

from sqlalchemy import select

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.feedback import ContrastivePair
from maimonedes.core.policy import Policy
from maimonedes.monitor.cusum import cusum_per_anchor
from maimonedes.storage.compliance import (
    ComplianceScoreRow,
    latest_anchor_baseline,
    row_to_score,
)
from maimonedes.storage.drift import (
    list_drift_sessions,
    scores_for_run,
)
from maimonedes.storage.llm_calls import (
    SUPERVISED_TO_SCORE_MAX_SECONDS,
    LLMCall,
)
from maimonedes.storage.perturbations import PerturbationProbeRow
from maimonedes.storage.repo import get_session


log = logging.getLogger(__name__)

MISSING_TEXT = "<text not recorded>"
SUPERVISED_BACKEND_PREFIX = "ollama-supervised"


def _llm_response_text(score: ComplianceScore) -> str:
    """Recover the supervised system's response text for a score.

    Two paths, in order:
    1. **FK path.** If `score.llm_call_id` is set and the row exists,
       return its `response_content` — the precise call that produced
       the scored output.
    2. **Chronological fallback.** Legacy rows (created before #44 wired
       the FK) lack `llm_call_id`. Pair them with the most-recent
       supervised LLMCall whose timestamp is at or before
       `score.scored_at`, within `SUPERVISED_TO_SCORE_MAX_SECONDS`.
       This mirrors the heuristic in `pair_supervised_with_scores` so
       the dashboard surfaces real text without requiring an explicit
       `backfill-llm-call-ids` run.

    Returns `MISSING_TEXT` only when neither path resolves a row (e.g.
    a test fixture with no supervised calls at all, or a score whose
    nearest call is outside the staleness window).
    """
    if score.llm_call_id is not None:
        with get_session() as session:
            row = session.get(LLMCall, score.llm_call_id)
            if row is not None:
                return row.response_content

    scored_at = score.scored_at
    if scored_at.tzinfo is None:
        scored_at = scored_at.replace(tzinfo=timezone.utc)
    with get_session() as session:
        row = session.execute(
            select(LLMCall)
            .where(
                LLMCall.backend_name.like(f"{SUPERVISED_BACKEND_PREFIX}%"),
                LLMCall.timestamp <= scored_at,
            )
            .order_by(LLMCall.timestamp.desc(), LLMCall.id.desc())
            .limit(1)
        ).scalar_one_or_none()
        if row is None:
            return MISSING_TEXT
        ts = row.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if (scored_at - ts).total_seconds() > SUPERVISED_TO_SCORE_MAX_SECONDS:
            return MISSING_TEXT
        return row.response_content


def _dropoff_axes(safe: ComplianceScore, near: ComplianceScore) -> list[str]:
    """Sub-condition ids ranked by descending |safe - near| per axis.

    Ties broken by sub-condition id ascending so the order is stable
    for tests + replay.
    """
    keys = set(safe.per_sub_condition) | set(near.per_sub_condition)
    deltas: list[tuple[str, float]] = []
    for key in keys:
        delta = abs(
            safe.per_sub_condition.get(key, 0.0)
            - near.per_sub_condition.get(key, 0.0)
        )
        deltas.append((key, delta))
    deltas.sort(key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _d in deltas]


def temporal_pair(
    anchor_id: str, *, drift_run_id: int
) -> ContrastivePair | None:
    """Build a temporal contrastive pair for one anchor in one drift run.

    Returns None when either side is missing — no baseline-stage scores
    OR no scores at all OR the safe and near-boundary collapsed to the
    same row (run too short / nothing drifted).
    """
    streams = scores_for_run(drift_run_id)
    scores = streams.get(anchor_id)
    if not scores:
        return None

    sessions_by_id = {s.id: s for s in list_drift_sessions(drift_run_id)}

    baseline_scores = [
        s
        for s in scores
        if s.drift_session_id is not None
        and s.drift_session_id in sessions_by_id
        and sessions_by_id[s.drift_session_id].stage_label == "baseline"
    ]
    if not baseline_scores:
        return None
    safe = max(
        baseline_scores,
        key=lambda s: sessions_by_id[s.drift_session_id].session_index,
    )

    cusum_states = cusum_per_anchor(drift_run_id).get(anchor_id, [])
    cusum_fire_idx = next(
        (s.session_index for s in cusum_states if s.fired), None
    )

    near: ComplianceScore | None = None
    if cusum_fire_idx is not None:
        for s in scores:
            if (
                s.drift_session_id is not None
                and s.drift_session_id in sessions_by_id
                and sessions_by_id[s.drift_session_id].session_index
                == cusum_fire_idx
            ):
                near = s
                break

    if near is None:
        near = min(scores, key=lambda s: s.aggregate)

    if near is safe or (
        near.aggregate == safe.aggregate
        and near.per_sub_condition == safe.per_sub_condition
    ):
        return None

    return ContrastivePair(
        anchor_id=anchor_id,
        kind="temporal",
        safe_text=_llm_response_text(safe),
        safe_score=safe,
        near_boundary_text=_llm_response_text(near),
        near_boundary_score=near,
        dropoff_axes=_dropoff_axes(safe, near),
    )


def fragility_pair(
    anchor_id: str,
    *,
    policy: Policy,
    run_context: int | None = None,
) -> ContrastivePair | None:
    """Build a fragility contrastive pair for one anchor.

    `policy` is accepted for forward-compatibility (per-axis-threshold
    or per-rubric-baseline variants may need it later); v1 only uses
    it to verify the anchor's baseline policy matches the supplied
    policy. `run_context` is also forward-compatibility (filtering
    perturbations by a specific run id).
    """
    _ = run_context  # reserved
    safe = latest_anchor_baseline(anchor_id)
    if safe is None:
        return None
    if safe.policy_id != policy.id:
        log.warning(
            "fragility_pair.policy_mismatch",
            extra={
                "anchor_id": anchor_id,
                "anchor_policy": safe.policy_id,
                "supplied_policy": policy.id,
            },
        )
        return None

    with get_session() as session:
        score_row = session.execute(
            select(ComplianceScoreRow)
            .join(
                PerturbationProbeRow,
                ComplianceScoreRow.perturbation_id == PerturbationProbeRow.id,
            )
            .where(
                PerturbationProbeRow.anchor_id == anchor_id,
                ComplianceScoreRow.probe_role == "perturbation",
            )
            .order_by(
                ComplianceScoreRow.aggregate.asc(),
                ComplianceScoreRow.id.desc(),
            )
            .limit(1)
        ).scalar_one_or_none()
        if score_row is None:
            return None
        near = row_to_score(score_row)

    return ContrastivePair(
        anchor_id=anchor_id,
        kind="fragility",
        safe_text=_llm_response_text(safe),
        safe_score=safe,
        near_boundary_text=_llm_response_text(near),
        near_boundary_score=near,
        dropoff_axes=_dropoff_axes(safe, near),
    )


__all__ = ["MISSING_TEXT", "fragility_pair", "temporal_pair"]
