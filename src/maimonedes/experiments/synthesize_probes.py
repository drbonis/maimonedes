"""Phase 5 orchestrator: GP target → KNN synthesizer → optional scorer.

Glues together the GP layer (#39), the K-NN exemplar synthesizer (#42),
and an optional scoring step. Per the locked decisions in the
planning thread:
- Default scorer is the LLM judge; `--scorer classifier` switches to
  a Stage-2 model.
- Validator model is configurable so cheap LLMs can do the quality
  gate at scale.
- Generated scenarios are still EVALUATED by sending them to the
  supervised system first; the chosen scorer rates the supervised
  output (Stage-1 LLM judge or Stage-2 classifier on the embedding).
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy
from maimonedes.core.probe import AnchorProbe
from maimonedes.feedback.probe_synthesis import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TAU,
    KnnExemplarSynthesizer,
    SynthesisResult,
    to_synthesized_probe,
)
from maimonedes.llm.client import LLMClient, Message
from maimonedes.llm.embed_client import EmbedClient
from maimonedes.llm.recording_client import RecordingClient
from maimonedes.llm.recording_embed_client import RecordingEmbedClient
from maimonedes.models.stage2 import Stage2Model
from maimonedes.monitor.gp_layer import ComplianceGP, GPTarget, propose_targets
from maimonedes.scorer.judge import Judge
from maimonedes.scorer.stage2 import Stage2Scorer
from maimonedes.storage.compliance import record_score
from maimonedes.storage.synthesized_probes import record_synthesized_probe


log = logging.getLogger(__name__)

GENERATOR_BACKEND_NAME = "ollama-synth-generator"
VALIDATOR_BACKEND_NAME = "ollama-synth-validator"
SUPERVISED_BACKEND_NAME = "ollama-supervised"
JUDGE_BACKEND_NAME = "ollama-judge"
EMBED_BACKEND_NAME = "clinicalbert"

ScorerKind = Literal["judge", "classifier"]


@dataclass
class TargetOutcome:
    """One target → one synthesized probe → optionally one score."""

    target: GPTarget
    synthesis: SynthesisResult
    synthesized_probe_id: int | None
    score: ComplianceScore | None
    error: str | None


@dataclass
class SynthesisRunSummary:
    n_targets: int
    n_approved: int
    n_rejected_tau: int
    n_rejected_validator: int
    n_scored: int
    mean_aggregate: float | None
    generation_method: str = "knn_exemplar"


@dataclass
class SynthesisProgress:
    """Streamed once per target so callers can show live progress."""

    target_index: int
    target: GPTarget
    outcome: TargetOutcome


ProgressCallback = Callable[[SynthesisProgress], None]


def synthesize_probes(
    *,
    gp_fit: ComplianceGP,
    n_targets: int,
    embed_client: EmbedClient,
    generator_client: LLMClient,
    validator_client: LLMClient,
    supervised_client: LLMClient,
    judge_client: LLMClient,
    embedding_model: str,
    generator_model: str,
    validator_model: str,
    supervised_model: str,
    judge_model: str,
    policy: Policy,
    anchors: Iterable[AnchorProbe],
    scorer: ScorerKind = "judge",
    stage2_model: Stage2Model | None = None,
    gp_fit_id: int | None = None,
    tau: float = DEFAULT_TAU,
    max_retries: int = DEFAULT_MAX_RETRIES,
    replay: bool = False,
    on_progress: ProgressCallback | None = None,
    candidate_pool_size: int = 1000,
    seed: int = 0,
) -> SynthesisRunSummary:
    """Generate `n_targets` synthesized probes; optionally score each."""
    if scorer == "classifier" and stage2_model is None:
        raise ValueError(
            "scorer='classifier' requires a Stage2Model passed via stage2_model"
        )

    anchor_list = [a for a in anchors if a.policy_id == policy.id]
    if not anchor_list:
        raise ValueError(
            f"no anchors match policy {policy.id!r}; cannot KNN-prompt"
        )

    embed_rc = RecordingEmbedClient(
        embed_client, backend_name=EMBED_BACKEND_NAME, replay=replay
    )
    generator_rc = RecordingClient(
        generator_client,
        backend_name=GENERATOR_BACKEND_NAME,
        replay=replay,
    )
    validator_rc = RecordingClient(
        validator_client,
        backend_name=VALIDATOR_BACKEND_NAME,
        replay=replay,
    )
    supervised_rc = RecordingClient(
        supervised_client,
        backend_name=SUPERVISED_BACKEND_NAME,
        replay=replay,
    )
    judge_rc = RecordingClient(
        judge_client, backend_name=JUDGE_BACKEND_NAME, replay=replay
    )
    judge = Judge(judge_rc, model=judge_model, supervised_model=supervised_model)

    synthesizer = KnnExemplarSynthesizer(
        generator_client=generator_rc,
        validator_client=validator_rc,
        embed_client=embed_rc,
        generator_model=generator_model,
        validator_model=validator_model,
        library_anchors=anchor_list,
        policy=policy,
        embedding_model=embedding_model,
        tau=tau,
        max_retries=max_retries,
    )

    targets = propose_targets(
        gp_fit,
        n_targets=n_targets,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
    )

    n_approved = 0
    n_rejected_tau = 0
    n_rejected_validator = 0
    n_scored = 0
    aggregates: list[float] = []

    for i, target in enumerate(targets, start=1):
        result = synthesizer.synthesize(target.embedding)
        probe = to_synthesized_probe(
            result, policy_id=policy.id, gp_fit_id=gp_fit_id
        )
        try:
            probe_id = record_synthesized_probe(probe)
        except Exception as exc:
            log.warning(
                "synthesize_probes.persist_failed",
                extra={"target_index": i, "error": str(exc)},
            )
            outcome = TargetOutcome(
                target=target,
                synthesis=result,
                synthesized_probe_id=None,
                score=None,
                error=str(exc),
            )
            if on_progress is not None:
                on_progress(SynthesisProgress(target_index=i, target=target, outcome=outcome))
            continue

        score: ComplianceScore | None = None
        if result.status == "approved":
            n_approved += 1
            try:
                score = _score_synthesized(
                    scenario=result.scenario,
                    probe_id=probe_id,
                    policy=policy,
                    supervised_rc=supervised_rc,
                    supervised_model=supervised_model,
                    judge=judge,
                    scorer=scorer,
                    stage2_model=stage2_model,
                    embed_client=embed_rc,
                    embedding_model=embedding_model,
                )
                if score is not None:
                    n_scored += 1
                    aggregates.append(score.aggregate)
            except Exception as exc:
                log.warning(
                    "synthesize_probes.score_failed",
                    extra={"target_index": i, "error": str(exc)},
                )
                outcome = TargetOutcome(
                    target=target,
                    synthesis=result,
                    synthesized_probe_id=probe_id,
                    score=None,
                    error=str(exc),
                )
                if on_progress is not None:
                    on_progress(SynthesisProgress(target_index=i, target=target, outcome=outcome))
                continue
        elif result.status == "rejected_validator":
            n_rejected_validator += 1
        else:
            n_rejected_tau += 1

        outcome = TargetOutcome(
            target=target,
            synthesis=result,
            synthesized_probe_id=probe_id,
            score=score,
            error=None,
        )
        if on_progress is not None:
            on_progress(SynthesisProgress(target_index=i, target=target, outcome=outcome))

    mean_aggregate = (
        sum(aggregates) / len(aggregates) if aggregates else None
    )
    return SynthesisRunSummary(
        n_targets=len(targets),
        n_approved=n_approved,
        n_rejected_tau=n_rejected_tau,
        n_rejected_validator=n_rejected_validator,
        n_scored=n_scored,
        mean_aggregate=mean_aggregate,
    )


def _score_synthesized(
    *,
    scenario: str,
    probe_id: int,
    policy: Policy,
    supervised_rc: RecordingClient,
    supervised_model: str,
    judge: Judge,
    scorer: ScorerKind,
    stage2_model: Stage2Model | None,
    embed_client: EmbedClient,
    embedding_model: str,
) -> ComplianceScore:
    """Run supervised on the synthesized scenario, score with the chosen scorer."""
    supervised_response = supervised_rc.chat_completion(
        [Message(role="user", content=scenario)],
        model=supervised_model,
        temperature=0.0,
    )

    # Build a synthetic anchor with the synthesized scenario as its
    # `scenario`. The anchor_id is namespaced so dashboards / queries
    # can trivially distinguish synthesized-anchor scores.
    anchor = AnchorProbe(
        id=f"synth-{probe_id}",
        scenario=scenario,
        policy_id=policy.id,
        expected_baseline_compliance=0.5,
    )

    if scorer == "judge":
        score = judge.score(policy, anchor, supervised_response.content)
    else:
        assert stage2_model is not None  # validated by caller
        s2 = Stage2Scorer(stage2_model, embed_client, policy)
        score = s2.score(
            anchor,
            supervised_response.content,
            supervised_model=supervised_model,
        )
    score = score.model_copy(
        update={
            "synthesized_probe_id": probe_id,
            "probe_role": "anchor",
        }
    )
    record_score(score)
    return score


__all__ = [
    "EMBED_BACKEND_NAME",
    "GENERATOR_BACKEND_NAME",
    "JUDGE_BACKEND_NAME",
    "ProgressCallback",
    "ScorerKind",
    "SUPERVISED_BACKEND_NAME",
    "SynthesisProgress",
    "SynthesisRunSummary",
    "TargetOutcome",
    "VALIDATOR_BACKEND_NAME",
    "synthesize_probes",
]
