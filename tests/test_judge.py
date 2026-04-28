"""Tests for the Stage-1 LLM-as-Judge."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from maimonedes.core.policy import Policy, load_policy
from maimonedes.core.probe import AnchorProbe
from maimonedes.llm.client import ChatResponse, LLMResponseError
from maimonedes.scorer.judge import Judge
from maimonedes.scorer.prompts import expected_response_format, judge_prompt
from tests.fakes import FakeLLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"


@pytest.fixture
def policy() -> Policy:
    return load_policy(POLICY_PATH, RUBRIC_PATH)


@pytest.fixture
def anchor() -> AnchorProbe:
    return AnchorProbe(
        id="A1",
        scenario="Stable angina patient asking about lifestyle modifications.",
        expected_baseline_compliance=0.95,
    )


def _well_formed_response(policy: Policy) -> str:
    """Build a fully-compliant (all-true / all-3) judge JSON response."""
    scores = {}
    for s in policy.rubric.sub_conditions:
        scores[s.id] = True if s.scale == "boolean" else 3
    return json.dumps({"scores": scores})


def _wrap_response(content: str) -> ChatResponse:
    return ChatResponse(content=content, model="judge:test", latency_ms=1.0)


# ---- prompt construction ---------------------------------------------------


def test_judge_prompt_has_system_and_user_with_rubric(policy: Policy, anchor: AnchorProbe) -> None:
    messages = judge_prompt(policy, anchor.scenario, "supervised says X")
    assert [m.role for m in messages] == ["system", "user"]
    user_text = messages[1].content
    for s in policy.rubric.sub_conditions:
        assert s.id in user_text
    assert anchor.scenario.strip() in user_text
    assert "supervised says X" in user_text


def test_response_format_schema_lists_every_sub_condition(policy: Policy) -> None:
    fmt = expected_response_format(policy)
    assert fmt["type"] == "json_schema"
    schema = fmt["json_schema"]["schema"]
    score_props = schema["properties"]["scores"]["properties"]
    assert set(score_props) == {s.id for s in policy.rubric.sub_conditions}


# ---- happy path ------------------------------------------------------------


def test_well_formed_response_yields_aggregate_one(policy: Policy, anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(responses=[_wrap_response(_well_formed_response(policy))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    score = judge.score(policy, anchor, "every recommendation is in scope")
    assert score.aggregate == pytest.approx(1.0)
    assert score.anchor_id == "A1"
    assert score.policy_id == policy.id
    assert score.judge_model == "judge:test"
    assert score.supervised_model == "llama:test"
    # All boolean -> 1.0; all 0-3 at 3 -> 1.0
    assert all(v == pytest.approx(1.0) for v in score.per_sub_condition.values())


def test_mixed_scale_aggregate_matches_hand_computed(
    policy: Policy, anchor: AnchorProbe
) -> None:
    # Build a deliberate mix: boolean cond half right, 0-3 cond at value 1
    scores: dict[str, bool | int] = {}
    expected_per: dict[str, float] = {}
    for s in policy.rubric.sub_conditions:
        if s.scale == "boolean":
            scores[s.id] = True
            expected_per[s.id] = 1.0
        else:
            scores[s.id] = 1
            expected_per[s.id] = 1.0 / 3.0
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    score = judge.score(policy, anchor, "...")

    expected_aggregate = sum(
        s.weight * expected_per[s.id] for s in policy.rubric.sub_conditions
    )
    assert score.aggregate == pytest.approx(expected_aggregate)
    for sub_id, expected in expected_per.items():
        assert score.per_sub_condition[sub_id] == pytest.approx(expected)


def test_judge_passes_response_format_through_to_client(
    policy: Policy, anchor: AnchorProbe
) -> None:
    fake = FakeLLMClient(responses=[_wrap_response(_well_formed_response(policy))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    judge.score(policy, anchor, "...")
    assert len(fake.calls) == 1
    extra = fake.calls[0].extra
    assert "response_format" in extra
    assert extra["response_format"]["type"] == "json_schema"


def test_judge_uses_temperature_zero_by_default(policy: Policy, anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(responses=[_wrap_response(_well_formed_response(policy))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    judge.score(policy, anchor, "...")
    assert fake.calls[0].temperature == 0.0


# ---- error paths -----------------------------------------------------------


def test_malformed_json_raises_response_error(policy: Policy, anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(responses=[_wrap_response("not-json")])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError):
        judge.score(policy, anchor, "...")


def test_missing_top_level_scores_key_raises(policy: Policy, anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"results": {}}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="`scores` key"):
        judge.score(policy, anchor, "...")


def test_missing_sub_condition_raises(policy: Policy, anchor: AnchorProbe) -> None:
    scores: dict[str, bool | int] = {}
    for s in policy.rubric.sub_conditions[:-1]:  # drop last sub-condition
        scores[s.id] = True if s.scale == "boolean" else 2
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="missing sub-condition"):
        judge.score(policy, anchor, "...")


def test_unknown_sub_condition_in_response_raises(
    policy: Policy, anchor: AnchorProbe
) -> None:
    scores: dict[str, bool | int] = {}
    for s in policy.rubric.sub_conditions:
        scores[s.id] = True if s.scale == "boolean" else 2
    scores["surprise_extra"] = True
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="unknown sub-conditions"):
        judge.score(policy, anchor, "...")


def test_scale_violation_raises(policy: Policy, anchor: AnchorProbe) -> None:
    scores: dict[str, bool | int] = {}
    for s in policy.rubric.sub_conditions:
        scores[s.id] = True if s.scale == "boolean" else 2
    # Pick the first 0-3 sub-condition and set it to an out-of-range value
    target = next(s for s in policy.rubric.sub_conditions if s.scale == "0-3")
    scores[target.id] = 5
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="out of"):
        judge.score(policy, anchor, "...")


def test_wrong_type_for_boolean_raises(policy: Policy, anchor: AnchorProbe) -> None:
    scores: dict[str, bool | int] = {}
    for s in policy.rubric.sub_conditions:
        scores[s.id] = True if s.scale == "boolean" else 2
    target = next(s for s in policy.rubric.sub_conditions if s.scale == "boolean")
    scores[target.id] = 1  # int, not bool
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="boolean"):
        judge.score(policy, anchor, "...")
