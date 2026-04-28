"""Stage-1 LLM-as-Judge.

`Judge.score(policy, anchor, supervised_output) -> ComplianceScore`
calls the wrapped `LLMClient` with a structured-JSON request, validates
the response against the rubric, normalises per-sub-condition scores
into [0, 1], and computes the rubric-weighted aggregate.

The framework — not the model — owns the aggregate computation. A
miscalibrated judge is a v1 risk; a miscalibrated weighted-sum is just
a bug, and bugs are testable.
"""
from __future__ import annotations

import json
from typing import Any

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy, SubCondition
from maimonedes.core.probe import AnchorProbe
from maimonedes.llm.client import LLMClient, LLMResponseError
from maimonedes.scorer.prompts import expected_response_format, judge_prompt


def _normalise(s: SubCondition, raw: Any) -> float:
    if s.scale == "boolean":
        if not isinstance(raw, bool):
            raise LLMResponseError(
                f"sub-condition {s.id!r}: expected boolean, got {type(raw).__name__}"
            )
        return 1.0 if raw else 0.0
    # 0-3 scale
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise LLMResponseError(
            f"sub-condition {s.id!r}: expected integer 0..3, got {type(raw).__name__}"
        )
    if not 0 <= raw <= 3:
        raise LLMResponseError(
            f"sub-condition {s.id!r}: integer out of [0,3], got {raw}"
        )
    return raw / 3.0


def _parse_judge_payload(policy: Policy, content: str) -> dict[str, float]:
    """Parse the judge's JSON content into per-sub-condition normalised scores."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"judge returned non-JSON content: {exc}") from exc
    if not isinstance(payload, dict) or "scores" not in payload:
        raise LLMResponseError("judge response missing top-level `scores` key")
    scores = payload["scores"]
    if not isinstance(scores, dict):
        raise LLMResponseError("judge response `scores` must be an object")

    normalised: dict[str, float] = {}
    for s in policy.rubric.sub_conditions:
        if s.id not in scores:
            raise LLMResponseError(f"judge response missing sub-condition {s.id!r}")
        normalised[s.id] = _normalise(s, scores[s.id])
    extras = set(scores) - {s.id for s in policy.rubric.sub_conditions}
    if extras:
        raise LLMResponseError(
            f"judge response contains unknown sub-conditions: {sorted(extras)}"
        )
    return normalised


def _aggregate(policy: Policy, normalised: dict[str, float]) -> float:
    total = sum(s.weight * normalised[s.id] for s in policy.rubric.sub_conditions)
    # Numerical hygiene — sums of 0.2+0.15+... can drift outside [0,1]
    # by a few ulps and trip the ComplianceScore validator.
    return min(1.0, max(0.0, total))


class Judge:
    """Stage-1 LLM-as-Judge."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        supervised_model: str,
        temperature: float = 0.0,
    ) -> None:
        self._client = client
        self._model = model
        self._supervised_model = supervised_model
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    def score(
        self,
        policy: Policy,
        anchor: AnchorProbe,
        supervised_output: str,
        *,
        llm_call_id: int | None = None,
    ) -> ComplianceScore:
        messages = judge_prompt(policy, anchor.scenario, supervised_output)
        response = self._client.chat_completion(
            messages,
            model=self._model,
            temperature=self._temperature,
            response_format=expected_response_format(policy),
        )
        normalised = _parse_judge_payload(policy, response.content)
        aggregate = _aggregate(policy, normalised)
        return ComplianceScore(
            anchor_id=anchor.id,
            policy_id=policy.id,
            per_sub_condition=normalised,
            aggregate=aggregate,
            judge_model=self._model,
            supervised_model=self._supervised_model,
            llm_call_id=llm_call_id,
        )


__all__ = ["Judge"]
