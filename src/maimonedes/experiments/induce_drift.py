"""Phase 3 orchestrator: execute one synthetic drift run.

`run_drift(schedule, ...)` iterates the schedule's session sequence,
prepends the active stage suffix to the supervised system's prompt,
and runs every anchor through supervised + judge. Each (session,
anchor) pair persists one `compliance_scores` row tagged with the
session's `drift_session_id`.

Per-row fault tolerance: a single judge or supervised hiccup is
logged and counted, never propagated. The detectors operate on
whatever data we manage to collect, and partial sessions are far
more useful than aborted ones.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.drift import DriftSchedule, StageLabel
from maimonedes.core.policy import Policy
from maimonedes.core.probe import AnchorProbe, get_anchor_by_id
from maimonedes.llm.client import LLMClient, Message
from maimonedes.llm.recording_client import RecordingClient
from maimonedes.scorer.judge import Judge
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
    finalize_drift_run,
    finalize_drift_session,
)


log = logging.getLogger(__name__)

SUPERVISED_BACKEND_NAME = "ollama-supervised"
JUDGE_BACKEND_NAME = "ollama-judge"


@dataclass
class DriftAnchorOutcome:
    """Per-(session, anchor) result. `score is None` means scoring failed."""

    session_index: int
    stage_label: StageLabel
    anchor_id: str
    score: ComplianceScore | None
    error: str | None


@dataclass
class DriftSessionProgress:
    """Streamed once per (session, anchor) so callers can show live progress."""

    session_index: int
    stage_label: StageLabel
    anchor_id: str
    outcome: DriftAnchorOutcome


@dataclass
class DriftRunSummary:
    """Final per-run rollup printed by the CLI."""

    drift_run_id: int
    total_scored: int
    failure_count: int
    mean_baseline_aggregate: float | None


ProgressCallback = Callable[[DriftSessionProgress], None]


def run_drift(
    schedule: DriftSchedule,
    *,
    policy: Policy,
    anchors: Iterable[AnchorProbe],
    supervised_client: LLMClient,
    judge_client: LLMClient,
    supervised_model: str,
    judge_model: str,
    schedule_path: str,
    replay: bool = False,
    run_notes: str | None = None,
    k_threshold: float = 4.0,
    supervised_temperature: float = 0.0,
    on_progress: ProgressCallback | None = None,
) -> tuple[int, DriftRunSummary]:
    """Execute one full drift run; return `(drift_run_id, summary)`.

    The summary is built from the persisted scores, so a partial run
    (interrupted, or with many anchor failures) still returns a
    useful rollup.
    """
    anchor_list = [a for a in anchors if a.policy_id == policy.id]
    if not anchor_list:
        raise ValueError(
            f"no anchors found that match policy {policy.id!r}"
        )

    supervised_rc = RecordingClient(
        supervised_client, backend_name=SUPERVISED_BACKEND_NAME, replay=replay
    )
    judge_rc = RecordingClient(
        judge_client, backend_name=JUDGE_BACKEND_NAME, replay=replay
    )
    judge = Judge(
        judge_rc, model=judge_model, supervised_model=supervised_model
    )

    drift_run_id = create_drift_run(
        policy_id=policy.id,
        supervised_model=supervised_model,
        judge_model=judge_model,
        schedule_path=schedule_path,
        notes=run_notes,
        k_threshold=k_threshold,
    )

    total_scored = 0
    failure_count = 0
    baseline_aggregates: list[float] = []
    try:
        for session_index, stage_label, suffix_text in schedule.iter_sessions():
            drift_session_id = create_drift_session(
                drift_run_id=drift_run_id,
                session_index=session_index,
                stage_label=stage_label,
                suffix_text=suffix_text,
            )
            for anchor in anchor_list:
                outcome = _run_one(
                    anchor=anchor,
                    policy=policy,
                    supervised_rc=supervised_rc,
                    judge=judge,
                    supervised_model=supervised_model,
                    suffix_text=suffix_text,
                    drift_session_id=drift_session_id,
                    session_index=session_index,
                    stage_label=stage_label,
                    supervised_temperature=supervised_temperature,
                )
                if outcome.score is not None:
                    total_scored += 1
                    if stage_label == "baseline":
                        baseline_aggregates.append(outcome.score.aggregate)
                else:
                    failure_count += 1
                if on_progress is not None:
                    on_progress(
                        DriftSessionProgress(
                            session_index=session_index,
                            stage_label=stage_label,
                            anchor_id=anchor.id,
                            outcome=outcome,
                        )
                    )
            finalize_drift_session(drift_session_id)
    finally:
        finalize_drift_run(drift_run_id)

    mean_baseline = (
        sum(baseline_aggregates) / len(baseline_aggregates)
        if baseline_aggregates
        else None
    )
    summary = DriftRunSummary(
        drift_run_id=drift_run_id,
        total_scored=total_scored,
        failure_count=failure_count,
        mean_baseline_aggregate=mean_baseline,
    )
    return drift_run_id, summary


def _run_one(
    *,
    anchor: AnchorProbe,
    policy: Policy,
    supervised_rc: RecordingClient,
    judge: Judge,
    supervised_model: str,
    suffix_text: str,
    drift_session_id: int,
    session_index: int,
    stage_label: StageLabel,
    supervised_temperature: float,
) -> DriftAnchorOutcome:
    """Run one anchor under the active suffix and persist its score."""
    messages: list[Message] = []
    if suffix_text:
        messages.append(Message(role="system", content=suffix_text))
    messages.append(Message(role="user", content=anchor.scenario))

    try:
        supervised_resp = supervised_rc.chat_completion(
            messages,
            model=supervised_model,
            temperature=supervised_temperature,
        )
        score = judge.score(
            policy,
            anchor,
            supervised_resp.content,
            llm_call_id=supervised_resp.llm_call_id,
        )
        score = score.model_copy(update={"drift_session_id": drift_session_id})
        record_score(score)
    except Exception as exc:  # network, judge JSON, persistence — all soft failures
        log.warning(
            "induce_drift.score_failed",
            extra={
                "session_index": session_index,
                "stage_label": stage_label,
                "anchor_id": anchor.id,
                "error": str(exc),
            },
        )
        return DriftAnchorOutcome(
            session_index=session_index,
            stage_label=stage_label,
            anchor_id=anchor.id,
            score=None,
            error=str(exc),
        )

    return DriftAnchorOutcome(
        session_index=session_index,
        stage_label=stage_label,
        anchor_id=anchor.id,
        score=score,
        error=None,
    )


__all__ = [
    "DriftAnchorOutcome",
    "DriftRunSummary",
    "DriftSessionProgress",
    "JUDGE_BACKEND_NAME",
    "ProgressCallback",
    "SUPERVISED_BACKEND_NAME",
    "run_drift",
]
