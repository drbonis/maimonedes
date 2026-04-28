"""Phase 2 orchestrator: run a perturbation cloud for one anchor.

`run_perturbations(anchor_id, ...)` is the end-to-end pipeline:
ask each generator for its perturbation list, persist the probe,
send the perturbed scenario through the supervised system, score
the supervised output with the judge, persist the resulting
`ComplianceScore` with `probe_role="perturbation"` and
`perturbation_id` set.

Per-row fault tolerance is the contract: a single judge or
supervised hiccup is logged and counted but does not poison the
remaining N-1 rows. The Jacobian still wants whatever data we
managed to collect.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import (
    PerturbationGenerator,
    PerturbationProbe,
)
from maimonedes.core.policy import Policy
from maimonedes.core.probe import AnchorProbe, get_anchor_by_id
from maimonedes.llm.client import LLMClient, Message
from maimonedes.llm.recording_client import RecordingClient
from maimonedes.scorer.judge import Judge
from maimonedes.storage.compliance import (
    latest_anchor_baseline,
    record_score,
)
from maimonedes.storage.perturbations import record_perturbation


log = logging.getLogger(__name__)

SUPERVISED_BACKEND_NAME = "ollama-supervised"
JUDGE_BACKEND_NAME = "ollama-judge"


@dataclass
class PerturbationOutcome:
    """Per-perturbation result. `score is None` means scoring failed."""

    probe: PerturbationProbe
    score: ComplianceScore | None
    delta_aggregate: float | None
    error: str | None


@dataclass
class PerturbationProgress:
    """Streamed at each completed probe so callers can show live progress.

    `index` is 1-based; `total` is the count returned by every generator
    *combined*, computed up-front before any scoring begins so the
    `[index/total]` ratio doesn't shift mid-run.
    """

    anchor_id: str
    index: int
    total: int
    outcome: PerturbationOutcome


ProgressCallback = Callable[[PerturbationProgress], None]
GenerationCallback = Callable[[str, int], None]  # generator_name, n_probes


def run_perturbations(
    anchor_id: str,
    *,
    policy: Policy,
    anchors: Iterable[AnchorProbe],
    supervised_client: LLMClient,
    judge_client: LLMClient,
    generators: list[PerturbationGenerator],
    supervised_model: str,
    judge_model: str,
    replay: bool = False,
    supervised_temperature: float = 0.0,
    on_outcome: ProgressCallback | None = None,
    on_generation: GenerationCallback | None = None,
) -> list[PerturbationOutcome]:
    """Generate, run, score, and persist perturbations for one anchor.

    `on_outcome` fires once per probe, immediately after that probe's
    score is persisted (or its failure is recorded). `on_generation`
    fires once per generator with the count of probes that generator
    produced — useful for printing "generating..." progress before the
    scoring loop starts.
    """
    anchor = get_anchor_by_id(list(anchors), anchor_id)
    if anchor.policy_id != policy.id:
        raise ValueError(
            f"anchor {anchor.id!r} declares policy {anchor.policy_id!r} "
            f"but loaded policy is {policy.id!r}"
        )

    baseline = latest_anchor_baseline(anchor.id)
    baseline_aggregate = baseline.aggregate if baseline is not None else None

    supervised_rc = RecordingClient(
        supervised_client, backend_name=SUPERVISED_BACKEND_NAME, replay=replay
    )
    judge_rc = RecordingClient(
        judge_client, backend_name=JUDGE_BACKEND_NAME, replay=replay
    )
    judge = Judge(
        judge_rc, model=judge_model, supervised_model=supervised_model
    )

    # Generate every probe up-front so `total` is known before the
    # scoring loop streams progress events. Paraphrase's LLM call
    # happens here too — emitting `on_generation` keeps the UI honest
    # about why the first second or two is silent.
    all_probes: list[PerturbationProbe] = []
    for gen in generators:
        try:
            probes = gen.generate(anchor)
        except Exception as exc:  # generator-level failure → skip this generator
            log.warning(
                "perturbation_session.generator_failed",
                extra={
                    "anchor_id": anchor.id,
                    "generator": type(gen).__name__,
                    "error": str(exc),
                },
            )
            if on_generation is not None:
                on_generation(type(gen).__name__, 0)
            continue
        if on_generation is not None:
            on_generation(type(gen).__name__, len(probes))
        all_probes.extend(probes)

    total = len(all_probes)
    outcomes: list[PerturbationOutcome] = []
    for idx, probe in enumerate(all_probes, start=1):
        outcome = _run_single(
            anchor=anchor,
            probe=probe,
            policy=policy,
            supervised_rc=supervised_rc,
            judge=judge,
            supervised_model=supervised_model,
            supervised_temperature=supervised_temperature,
            baseline_aggregate=baseline_aggregate,
        )
        outcomes.append(outcome)
        if on_outcome is not None:
            on_outcome(
                PerturbationProgress(
                    anchor_id=anchor.id,
                    index=idx,
                    total=total,
                    outcome=outcome,
                )
            )

    return outcomes


def _run_single(
    *,
    anchor: AnchorProbe,
    probe: PerturbationProbe,
    policy: Policy,
    supervised_rc: RecordingClient,
    judge: Judge,
    supervised_model: str,
    supervised_temperature: float,
    baseline_aggregate: float | None,
) -> PerturbationOutcome:
    try:
        row_id = record_perturbation(probe)
    except Exception as exc:
        log.warning(
            "perturbation_session.persist_probe_failed",
            extra={"transform_label": probe.transform_label, "error": str(exc)},
        )
        return PerturbationOutcome(
            probe=probe, score=None, delta_aggregate=None, error=str(exc)
        )

    try:
        supervised_resp = supervised_rc.chat_completion(
            [Message(role="user", content=probe.scenario)],
            model=supervised_model,
            temperature=supervised_temperature,
        )
        score = judge.score(policy, anchor, supervised_resp.content)
        score = score.model_copy(
            update={
                "perturbation_id": row_id,
                "probe_role": "perturbation",
            }
        )
        record_score(score)
    except Exception as exc:
        log.warning(
            "perturbation_session.score_failed",
            extra={
                "transform_label": probe.transform_label,
                "error": str(exc),
            },
        )
        return PerturbationOutcome(
            probe=probe, score=None, delta_aggregate=None, error=str(exc)
        )

    delta = (
        score.aggregate - baseline_aggregate
        if baseline_aggregate is not None
        else None
    )
    return PerturbationOutcome(
        probe=probe, score=score, delta_aggregate=delta, error=None
    )


__all__ = [
    "GenerationCallback",
    "JUDGE_BACKEND_NAME",
    "PerturbationOutcome",
    "PerturbationProgress",
    "ProgressCallback",
    "SUPERVISED_BACKEND_NAME",
    "run_perturbations",
]
