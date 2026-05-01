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
import secrets
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
# Paraphrase generator's LLM calls are wrapped at construction time
# (in `cli._build_generators`); this name is used for the
# `RecordingClient(backend_name=...)` so the audit trail in
# `llm_calls` separates paraphrase-generation requests from
# supervised + judge calls.
PARAPHRASE_BACKEND_NAME = "ollama-paraphrase"


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

    `index` is 1-based within a single replicate; `total` is the
    per-replicate probe count. `replicate_index` is 0-based and
    `replicates_total` reflects the orchestrator's `replicates`
    parameter (1 when not in replicate mode).
    """

    anchor_id: str
    index: int
    total: int
    outcome: PerturbationOutcome
    replicate_index: int = 0
    replicates_total: int = 1


ProgressCallback = Callable[[PerturbationProgress], None]
GenerationCallback = Callable[[str, int], None]  # generator_name, n_probes


def _new_run_id() -> str:
    """Short opaque run identifier stamped into every probe's metadata.

    All probes from a single `run_perturbations` invocation share this
    id; the Jacobian aggregator uses it to pick replicates from the
    most recent run when computing per-cell mean ± std.
    """
    return secrets.token_hex(8)


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
    replicates: int = 1,
) -> list[PerturbationOutcome]:
    """Generate, run, score, and persist perturbations for one anchor.

    `on_outcome` fires once per probe, immediately after that probe's
    score is persisted (or its failure is recorded). `on_generation`
    fires once per generator with the count of probes that generator
    produced — useful for printing "generating..." progress before the
    scoring loop starts.

    When `replicates > 1`, the entire generate-and-score pipeline runs
    N times for the anchor. Each replicate gets a fresh call into
    every generator (so paraphrase produces different rewrites each
    replicate at its own temperature=0.7) and fresh supervised + judge
    calls. All probes from a single `run_perturbations` invocation
    share a single `run_id` stamped into their `generator_metadata`,
    plus a per-replicate `replicate_index`. The Jacobian aggregator
    uses these to group replicates of the same `(anchor, transform_label)`
    condition and report mean ± std.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")

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

    run_id = _new_run_id()
    outcomes: list[PerturbationOutcome] = []

    for replicate_idx in range(replicates):
        # Generate every probe for this replicate up-front so `total`
        # is known before the scoring loop streams progress events.
        # `on_generation` fires only on the first replicate so the UI
        # doesn't repeat the "generating ..." headers N times.
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
                        "replicate_index": replicate_idx,
                    },
                )
                if on_generation is not None and replicate_idx == 0:
                    on_generation(type(gen).__name__, 0)
                continue
            if on_generation is not None and replicate_idx == 0:
                on_generation(type(gen).__name__, len(probes))
            all_probes.extend(probes)

        total = len(all_probes)
        for idx, probe in enumerate(all_probes, start=1):
            stamped = probe.model_copy(
                update={
                    "generator_metadata": {
                        **probe.generator_metadata,
                        "run_id": run_id,
                        "replicate_index": replicate_idx,
                    }
                }
            )
            outcome = _run_single(
                anchor=anchor,
                probe=stamped,
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
                        replicate_index=replicate_idx,
                        replicates_total=replicates,
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
        score = judge.score(
            policy,
            anchor,
            supervised_resp.content,
            llm_call_id=supervised_resp.llm_call_id,
        )
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
    "PARAPHRASE_BACKEND_NAME",
    "PerturbationOutcome",
    "PerturbationProgress",
    "ProgressCallback",
    "SUPERVISED_BACKEND_NAME",
    "run_perturbations",
]
