"""Phase 1 orchestrator.

`run_once(anchor_id, ...)` is the end-to-end happy path: load anchor,
send it to the supervised system, score the output with the judge,
persist a `ComplianceScore`. The CLI subcommand `maimonedes run-once`
is a thin wrapper.

Both the supervised and judge calls go through their own
`RecordingClient`. They share an underlying backend (typically a
single `OllamaBackend` pointed at the GPU laptop, differing only in
the `model` argument), but the recording layer is per-call so the
audit trail clearly distinguishes the two roles.
"""
from __future__ import annotations

from collections.abc import Iterable

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy
from maimonedes.core.probe import AnchorProbe, get_anchor_by_id
from maimonedes.llm.client import LLMClient, Message
from maimonedes.llm.recording_client import RecordingClient
from maimonedes.scorer.judge import Judge
from maimonedes.storage.compliance import record_score


SUPERVISED_BACKEND_NAME = "ollama-supervised"
JUDGE_BACKEND_NAME = "ollama-judge"


def run_once(
    anchor_id: str,
    *,
    policy: Policy,
    anchors: Iterable[AnchorProbe],
    supervised_client: LLMClient,
    judge_client: LLMClient,
    supervised_model: str,
    judge_model: str,
    replay: bool = False,
    persist: bool = True,
    supervised_temperature: float = 0.0,
) -> ComplianceScore:
    """Run a single anchor through the Phase 1 pipeline.

    `persist=False` is used by the calibration harness, which calls the
    judge directly without polluting the `compliance_scores` table.
    """
    anchor = get_anchor_by_id(list(anchors), anchor_id)
    if anchor.policy_id != policy.id:
        raise ValueError(
            f"anchor {anchor.id!r} declares policy {anchor.policy_id!r} "
            f"but loaded policy is {policy.id!r}"
        )

    supervised_rc = RecordingClient(
        supervised_client, backend_name=SUPERVISED_BACKEND_NAME, replay=replay
    )
    judge_rc = RecordingClient(
        judge_client, backend_name=JUDGE_BACKEND_NAME, replay=replay
    )

    supervised_response = supervised_rc.chat_completion(
        [Message(role="user", content=anchor.scenario)],
        model=supervised_model,
        temperature=supervised_temperature,
    )

    judge = Judge(judge_rc, model=judge_model, supervised_model=supervised_model)
    score = judge.score(policy, anchor, supervised_response.content)

    if persist:
        record_score(score)
    return score


__all__ = ["JUDGE_BACKEND_NAME", "SUPERVISED_BACKEND_NAME", "run_once"]
