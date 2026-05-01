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
from maimonedes.storage.synthesized_probes import (
    get_synthesized_probe,
    scores_for_synthesized_probe,
)


SYNTH_ANCHOR_PREFIX = "synth-"


def synth_anchor_id(synthesized_probe_id: int) -> str:
    """Namespaced anchor id used for perturbations of synthesized probes.

    Carrying `synth-{id}` in `perturbation_probes.anchor_id` keeps the
    existing per-anchor query path working for synthesized parents
    (Phase 2 dashboard, fragility, contrastive pair) without forcing
    every consumer to dual-mode on the FK column.
    """
    return f"{SYNTH_ANCHOR_PREFIX}{synthesized_probe_id}"


def parse_synth_anchor_id(anchor_id: str) -> int | None:
    """Extract the synthesized_probe_id from a namespaced anchor id, or None."""
    if not anchor_id.startswith(SYNTH_ANCHOR_PREFIX):
        return None
    try:
        return int(anchor_id[len(SYNTH_ANCHOR_PREFIX):])
    except ValueError:
        return None


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


def _build_synthesized_anchor(
    synthesized_probe_id: int, policy: Policy
) -> AnchorProbe:
    """Materialise a synthesized probe as an `AnchorProbe`-shaped parent.

    The perturbation generators only read `id`, `scenario`, and
    `policy_id` off the parent. Wrapping the synthesized probe as an
    `AnchorProbe` (with a synthetic `expected_baseline_compliance` of
    0.5 — never read by the generators) lets the existing pipeline
    work with no special-case branches.
    """
    synth = get_synthesized_probe(synthesized_probe_id)
    if synth is None:
        raise KeyError(f"synthesized_probe_id={synthesized_probe_id} not found")
    if synth.policy_id != policy.id:
        raise ValueError(
            f"synthesized probe {synthesized_probe_id} policy "
            f"{synth.policy_id!r} does not match loaded policy {policy.id!r}"
        )
    return AnchorProbe(
        id=synth_anchor_id(synthesized_probe_id),
        scenario=synth.scenario,
        policy_id=synth.policy_id,
        expected_baseline_compliance=0.5,
    )


def _baseline_for_synthesized(
    synthesized_probe_id: int,
) -> ComplianceScore | None:
    """Most-recent anchor-role score for a synthesized probe, or None."""
    scores = scores_for_synthesized_probe(synthesized_probe_id)
    anchor_scores = [s for s in scores if s.probe_role == "anchor"]
    if not anchor_scores:
        return None
    return max(anchor_scores, key=lambda s: (s.scored_at, s.llm_call_id or 0))


def run_perturbations(
    anchor_id: str | None = None,
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
    synthesized_probe_id: int | None = None,
) -> list[PerturbationOutcome]:
    """Generate, run, score, and persist perturbations for one parent.

    Two parent kinds are supported:

    - `anchor_id` set, `synthesized_probe_id` is `None` — perturb the
      curated anchor with the matching id from `anchors`. Default Phase 2
      behaviour.
    - `synthesized_probe_id` set, `anchor_id` is `None` — perturb the
      synthesized probe with that id from `synthesized_probes`. The
      probe's `scenario` text is loaded from the DB; persisted
      `perturbation_probes` rows carry `synthesized_probe_id` set
      and `anchor_id = f"synth-{id}"`.

    Exactly one of the two must be set.

    `on_outcome` fires once per probe, immediately after that probe's
    score is persisted (or its failure is recorded). `on_generation`
    fires once per generator with the count of probes that generator
    produced — useful for printing "generating..." progress before the
    scoring loop starts.

    When `replicates > 1`, the entire generate-and-score pipeline runs
    N times for the parent. Each replicate gets a fresh call into
    every generator (so paraphrase produces different rewrites each
    replicate at its own temperature=0.7) and fresh supervised + judge
    calls. All probes from a single `run_perturbations` invocation
    share a single `run_id` stamped into their `generator_metadata`,
    plus a per-replicate `replicate_index`. The Jacobian aggregator
    uses these to group replicates of the same `(parent, transform_label)`
    condition and report mean ± std.
    """
    if replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {replicates}")

    if (anchor_id is None) == (synthesized_probe_id is None):
        raise ValueError(
            "run_perturbations: exactly one of `anchor_id` or "
            "`synthesized_probe_id` must be set"
        )

    if synthesized_probe_id is not None:
        anchor = _build_synthesized_anchor(synthesized_probe_id, policy)
        baseline = _baseline_for_synthesized(synthesized_probe_id)
    else:
        assert anchor_id is not None  # narrow for type checker
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
                    },
                    "synthesized_probe_id": synthesized_probe_id,
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
    "SYNTH_ANCHOR_PREFIX",
    "parse_synth_anchor_id",
    "run_perturbations",
    "synth_anchor_id",
]
