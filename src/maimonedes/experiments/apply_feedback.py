"""Phase 4 orchestrator: close the loop on a drift run.

`apply_feedback(parent_drift_run_id, ...)` localizes the worst-affected
anchors of a parent drift run, builds a contrastive pair per anchor,
synthesizes per-anchor feedback, and re-runs each anchor (and, in the
fragility scenario, every existing perturbation for that anchor) with
the synthesized feedback prepended as a system prompt. Recovery scores
are persisted with `recovery_run_id` set so the Phase 4 dashboard +
recovery-report can read them via `storage.recovery.scores_for_recovery_run`.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from sqlalchemy import select

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.feedback import ContrastiveKind, Feedback
from maimonedes.core.policy import Policy
from maimonedes.core.probe import AnchorProbe
from maimonedes.feedback.contrastive import (
    fragility_pair,
    temporal_pair,
)
from maimonedes.feedback.synthesizer import FeedbackSynthesizer
from maimonedes.llm.client import LLMClient, Message
from maimonedes.llm.recording_client import RecordingClient
from maimonedes.monitor.localizer import LocalizationResult, localize
from maimonedes.scorer.judge import Judge
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import get_drift_run
from maimonedes.storage.perturbations import PerturbationProbeRow
from maimonedes.storage.recovery import (
    create_recovery_run,
    finalize_recovery_run,
    record_feedback,
)
from maimonedes.storage.repo import get_session


log = logging.getLogger(__name__)

SUPERVISED_BACKEND_NAME = "ollama-supervised"
JUDGE_BACKEND_NAME = "ollama-judge"
FEEDBACK_BACKEND_NAME = "ollama-feedback"


@dataclass
class AnchorRecoveryOutcome:
    """Per-anchor result of one apply-feedback step."""

    anchor_id: str
    pre_aggregate: float | None
    post_aggregate: float | None
    feedback_text: str | None
    perturbation_count: int
    failure_count: int
    error: str | None


@dataclass
class RecoveryProgress:
    """Streamed once per (anchor, step) so callers can show live progress."""

    anchor_id: str
    step: str  # "localize" | "synthesize" | "evaluate" | "perturb"
    detail: str


@dataclass
class RecoveryRunSummary:
    recovery_run_id: int
    anchor_count: int
    failure_count: int
    mean_delta_toward_baseline: float | None


ProgressCallback = Callable[[RecoveryProgress], None]


def apply_feedback(
    parent_drift_run_id: int,
    *,
    policy: Policy,
    anchors: Iterable[AnchorProbe],
    supervised_client: LLMClient,
    judge_client: LLMClient,
    supervised_model: str,
    judge_model: str,
    contrastive_kind: ContrastiveKind = "temporal",
    top_k: int = 3,
    selected_anchor_ids: Sequence[str] | None = None,
    replay: bool = False,
    run_notes: str | None = None,
    supervised_temperature: float = 0.0,
    on_progress: ProgressCallback | None = None,
) -> tuple[int, RecoveryRunSummary]:
    """Run the Phase 4 closed-loop step on a parent drift run."""
    parent = get_drift_run(parent_drift_run_id)
    if parent is None:
        raise ValueError(f"drift_run {parent_drift_run_id} not found")

    localizations = localize(
        parent_drift_run_id,
        policy=policy,
        top_k=top_k if selected_anchor_ids is None else None,
    )
    if selected_anchor_ids is not None:
        wanted = set(selected_anchor_ids)
        localizations = [r for r in localizations if r.anchor_id in wanted]
    if not localizations:
        raise ValueError(
            f"no affected anchors found in drift_run {parent_drift_run_id}"
        )

    supervised_rc = RecordingClient(
        supervised_client, backend_name=SUPERVISED_BACKEND_NAME, replay=replay
    )
    judge_rc = RecordingClient(
        judge_client, backend_name=JUDGE_BACKEND_NAME, replay=replay
    )
    feedback_rc = RecordingClient(
        judge_client, backend_name=FEEDBACK_BACKEND_NAME, replay=replay
    )
    judge = Judge(judge_rc, model=judge_model, supervised_model=supervised_model)
    synthesizer = FeedbackSynthesizer(feedback_rc, model=judge_model)

    anchors_by_id = {a.id: a for a in anchors}

    recovery_run_id = create_recovery_run(
        parent_drift_run_id=parent_drift_run_id,
        supervised_model=supervised_model,
        judge_model=judge_model,
        contrastive_kind=contrastive_kind,
        notes=run_notes,
    )

    failure_count = 0
    deltas: list[float] = []
    anchor_count = 0

    try:
        for loc in localizations:
            anchor_count += 1
            outcome = _run_one_anchor(
                loc=loc,
                anchors_by_id=anchors_by_id,
                policy=policy,
                supervised_rc=supervised_rc,
                judge=judge,
                synthesizer=synthesizer,
                supervised_model=supervised_model,
                supervised_temperature=supervised_temperature,
                contrastive_kind=contrastive_kind,
                parent_drift_run_id=parent_drift_run_id,
                recovery_run_id=recovery_run_id,
                on_progress=on_progress,
            )
            failure_count += outcome.failure_count
            if outcome.pre_aggregate is not None and outcome.post_aggregate is not None:
                deltas.append(outcome.post_aggregate - outcome.pre_aggregate)
    finally:
        finalize_recovery_run(recovery_run_id)

    mean_delta = sum(deltas) / len(deltas) if deltas else None
    summary = RecoveryRunSummary(
        recovery_run_id=recovery_run_id,
        anchor_count=anchor_count,
        failure_count=failure_count,
        mean_delta_toward_baseline=mean_delta,
    )
    return recovery_run_id, summary


def _run_one_anchor(
    *,
    loc: LocalizationResult,
    anchors_by_id: dict[str, AnchorProbe],
    policy: Policy,
    supervised_rc: RecordingClient,
    judge: Judge,
    synthesizer: FeedbackSynthesizer,
    supervised_model: str,
    supervised_temperature: float,
    contrastive_kind: ContrastiveKind,
    parent_drift_run_id: int,
    recovery_run_id: int,
    on_progress: ProgressCallback | None,
) -> AnchorRecoveryOutcome:
    anchor_id = loc.anchor_id
    anchor = anchors_by_id.get(anchor_id)
    if anchor is None:
        log.warning(
            "apply_feedback.unknown_anchor",
            extra={"anchor_id": anchor_id, "kind": contrastive_kind},
        )
        return AnchorRecoveryOutcome(
            anchor_id=anchor_id,
            pre_aggregate=loc.worst_score.aggregate,
            post_aggregate=None,
            feedback_text=None,
            perturbation_count=0,
            failure_count=1,
            error="unknown_anchor",
        )

    if on_progress is not None:
        on_progress(
            RecoveryProgress(
                anchor_id=anchor_id,
                step="localize",
                detail=f"distance={loc.distance:.3f}",
            )
        )

    if contrastive_kind == "temporal":
        pair = temporal_pair(anchor_id, drift_run_id=parent_drift_run_id)
    else:
        pair = fragility_pair(anchor_id, policy=policy)

    if pair is None:
        log.warning(
            "apply_feedback.no_pair",
            extra={"anchor_id": anchor_id, "kind": contrastive_kind},
        )
        return AnchorRecoveryOutcome(
            anchor_id=anchor_id,
            pre_aggregate=loc.worst_score.aggregate,
            post_aggregate=None,
            feedback_text=None,
            perturbation_count=0,
            failure_count=1,
            error="no_contrastive_pair",
        )

    try:
        synthesized = synthesizer.synthesize(policy, pair)
    except Exception as exc:
        log.warning(
            "apply_feedback.synthesize_failed",
            extra={"anchor_id": anchor_id, "error": str(exc)},
        )
        return AnchorRecoveryOutcome(
            anchor_id=anchor_id,
            pre_aggregate=loc.worst_score.aggregate,
            post_aggregate=None,
            feedback_text=None,
            perturbation_count=0,
            failure_count=1,
            error=f"synthesize: {exc}",
        )

    feedback = Feedback(
        recovery_run_id=recovery_run_id,
        parent_drift_run_id=parent_drift_run_id,
        anchor_id=anchor_id,
        contrastive_kind=contrastive_kind,
        feedback_text=synthesized.text,
        llm_call_id=synthesized.llm_call_id,
    )
    record_feedback(feedback)

    if on_progress is not None:
        on_progress(
            RecoveryProgress(
                anchor_id=anchor_id,
                step="synthesize",
                detail=synthesized.text[:80] + ("..." if len(synthesized.text) > 80 else ""),
            )
        )

    pre = loc.worst_score.aggregate
    post: float | None = None
    failure_count = 0

    try:
        score = _evaluate_with_feedback(
            scenario=anchor.scenario,
            anchor=anchor,
            policy=policy,
            feedback_text=synthesized.text,
            supervised_rc=supervised_rc,
            judge=judge,
            supervised_model=supervised_model,
            supervised_temperature=supervised_temperature,
            recovery_run_id=recovery_run_id,
            probe_role="anchor",
            perturbation_id=None,
        )
        post = score.aggregate
        if on_progress is not None:
            on_progress(
                RecoveryProgress(
                    anchor_id=anchor_id,
                    step="evaluate",
                    detail=f"pre={pre:.3f} post={post:.3f}",
                )
            )
    except Exception as exc:
        log.warning(
            "apply_feedback.evaluate_failed",
            extra={"anchor_id": anchor_id, "error": str(exc)},
        )
        failure_count += 1

    perturbation_count = 0
    if contrastive_kind == "fragility":
        perturbation_count, perturbation_failures = _rerun_perturbations(
            anchor_id=anchor_id,
            anchor=anchor,
            policy=policy,
            feedback_text=synthesized.text,
            supervised_rc=supervised_rc,
            judge=judge,
            supervised_model=supervised_model,
            supervised_temperature=supervised_temperature,
            recovery_run_id=recovery_run_id,
            on_progress=on_progress,
        )
        failure_count += perturbation_failures

    return AnchorRecoveryOutcome(
        anchor_id=anchor_id,
        pre_aggregate=pre,
        post_aggregate=post,
        feedback_text=synthesized.text,
        perturbation_count=perturbation_count,
        failure_count=failure_count,
        error=None,
    )


def _evaluate_with_feedback(
    *,
    scenario: str,
    anchor: AnchorProbe,
    policy: Policy,
    feedback_text: str,
    supervised_rc: RecordingClient,
    judge: Judge,
    supervised_model: str,
    supervised_temperature: float,
    recovery_run_id: int,
    probe_role: str,
    perturbation_id: int | None,
) -> ComplianceScore:
    messages = [
        Message(role="system", content=feedback_text),
        Message(role="user", content=scenario),
    ]
    supervised_resp = supervised_rc.chat_completion(
        messages,
        model=supervised_model,
        temperature=supervised_temperature,
    )
    score = judge.score(policy, anchor, supervised_resp.content)
    update: dict[str, object] = {"recovery_run_id": recovery_run_id}
    if probe_role == "perturbation":
        update["probe_role"] = "perturbation"
        update["perturbation_id"] = perturbation_id
    score = score.model_copy(update=update)
    record_score(score)
    return score


def _rerun_perturbations(
    *,
    anchor_id: str,
    anchor: AnchorProbe,
    policy: Policy,
    feedback_text: str,
    supervised_rc: RecordingClient,
    judge: Judge,
    supervised_model: str,
    supervised_temperature: float,
    recovery_run_id: int,
    on_progress: ProgressCallback | None,
) -> tuple[int, int]:
    """Re-run every existing perturbation for this anchor under feedback.

    Reuses the existing `perturbation_probes` rows so the Phase 2
    fragility pipeline can compare before/after on the same probes by
    `transform_label`. Returns (count, failure_count).
    """
    with get_session() as session:
        probe_rows = (
            session.execute(
                select(PerturbationProbeRow).where(
                    PerturbationProbeRow.anchor_id == anchor_id
                )
            )
            .scalars()
            .all()
        )
        # Detach so we can use the rows after the session closes.
        for r in probe_rows:
            session.expunge(r)

    count = 0
    failures = 0
    for probe in probe_rows:
        try:
            _evaluate_with_feedback(
                scenario=probe.scenario,
                anchor=anchor,
                policy=policy,
                feedback_text=feedback_text,
                supervised_rc=supervised_rc,
                judge=judge,
                supervised_model=supervised_model,
                supervised_temperature=supervised_temperature,
                recovery_run_id=recovery_run_id,
                probe_role="perturbation",
                perturbation_id=probe.id,
            )
            count += 1
            if on_progress is not None:
                on_progress(
                    RecoveryProgress(
                        anchor_id=anchor_id,
                        step="perturb",
                        detail=f"{probe.transform_label} (count={count})",
                    )
                )
        except Exception as exc:
            failures += 1
            log.warning(
                "apply_feedback.perturbation_eval_failed",
                extra={
                    "anchor_id": anchor_id,
                    "transform_label": probe.transform_label,
                    "error": str(exc),
                },
            )
    return count, failures


__all__ = [
    "AnchorRecoveryOutcome",
    "FEEDBACK_BACKEND_NAME",
    "JUDGE_BACKEND_NAME",
    "ProgressCallback",
    "RecoveryProgress",
    "RecoveryRunSummary",
    "SUPERVISED_BACKEND_NAME",
    "apply_feedback",
]
