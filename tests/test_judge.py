"""Tests for the Stage-1 LLM-as-Judge."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from maimonedes.core.policy import (
    Label,
    Policy,
    Rubric,
    SubCondition,
    load_policy,
)
from maimonedes.core.probe import AnchorProbe
from maimonedes.llm.client import ChatResponse, LLMResponseError
from maimonedes.scorer.judge import Judge
from maimonedes.scorer.prompts import expected_response_format, judge_prompt
from tests.fakes import (
    FakeLLMClient,
    compliant_response_json,
    most_compliant_value,
)

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


def _mixed_scale_policy() -> Policy:
    """A test-only policy that exercises every legacy scale type.

    Used by tests that explicitly probe boolean / 0-3 validation paths;
    the production rubric is all-labels and doesn't surface those paths.
    """
    return Policy(
        id="mixed_test_policy",
        name="Mixed Test Policy",
        text="Test policy that mixes every scale type for validation tests.",
        rubric=Rubric(
            sub_conditions=[
                SubCondition(
                    id="bool_a", description="boolean A",
                    scale="boolean", weight=0.20,
                ),
                SubCondition(
                    id="zero_three_a", description="0-3 A",
                    scale="0-3", weight=0.20,
                ),
                SubCondition(
                    id="labels_a",
                    description="labels A",
                    scale="labels",
                    weight=0.20,
                    labels=[
                        Label(id="excellent", value=1.0, description="excellent"),
                        Label(id="ok", value=0.5, description="ok"),
                        Label(id="bad", value=0.0, description="bad"),
                    ],
                ),
                SubCondition(
                    id="bool_b", description="boolean B",
                    scale="boolean", weight=0.20,
                ),
                SubCondition(
                    id="zero_three_b", description="0-3 B",
                    scale="0-3", weight=0.20,
                ),
            ]
        ),
    )


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


def test_response_format_emits_enum_for_labels_scale(policy: Policy) -> None:
    """Production rubric is BARS — every property should be a string enum."""
    fmt = expected_response_format(policy)
    schema = fmt["json_schema"]["schema"]
    score_props = schema["properties"]["scores"]["properties"]
    for s in policy.rubric.sub_conditions:
        if s.scale == "labels":
            assert score_props[s.id]["type"] == "string"
            assert set(score_props[s.id]["enum"]) == set(s.label_ids())


def test_judge_prompt_renders_label_anchors_for_labels_scale(policy: Policy, anchor: AnchorProbe) -> None:
    """BARS labels and their descriptions should appear in the user prompt."""
    user_text = judge_prompt(policy, anchor.scenario, "supervised output")[1].content
    for s in policy.rubric.sub_conditions:
        if s.scale == "labels" and s.labels is not None:
            for label in s.labels:
                assert label.id in user_text


# ---- happy path ------------------------------------------------------------


def test_well_formed_response_yields_aggregate_one(policy: Policy, anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(responses=[_wrap_response(compliant_response_json(policy))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    score = judge.score(policy, anchor, "every recommendation is in scope")
    assert score.aggregate == pytest.approx(1.0)
    assert score.anchor_id == "A1"
    assert score.policy_id == policy.id
    assert score.judge_model == "judge:test"
    assert score.supervised_model == "llama:test"
    # Most-compliant label per sub-condition has value 1.0 in the BARS rubric.
    assert all(v == pytest.approx(1.0) for v in score.per_sub_condition.values())


def test_mixed_scale_aggregate_matches_hand_computed(anchor: AnchorProbe) -> None:
    """Hand-computed aggregate against a deliberately-mixed-scale policy."""
    policy = _mixed_scale_policy()
    scores: dict[str, bool | int | str] = {
        "bool_a": True,         # → 1.0
        "zero_three_a": 1,      # → 1/3
        "labels_a": "ok",       # → 0.5
        "bool_b": False,        # → 0.0
        "zero_three_b": 3,      # → 1.0
    }
    expected_per = {
        "bool_a": 1.0,
        "zero_three_a": 1.0 / 3.0,
        "labels_a": 0.5,
        "bool_b": 0.0,
        "zero_three_b": 1.0,
    }
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
    fake = FakeLLMClient(responses=[_wrap_response(compliant_response_json(policy))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    judge.score(policy, anchor, "...")
    assert len(fake.calls) == 1
    extra = fake.calls[0].extra
    assert "response_format" in extra
    assert extra["response_format"]["type"] == "json_schema"


def test_judge_uses_temperature_zero_by_default(policy: Policy, anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(responses=[_wrap_response(compliant_response_json(policy))])
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
    scores = {
        s.id: most_compliant_value(s)
        for s in policy.rubric.sub_conditions[:-1]  # drop last sub-condition
    }
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="missing sub-condition"):
        judge.score(policy, anchor, "...")


def test_unknown_sub_condition_in_response_raises(
    policy: Policy, anchor: AnchorProbe
) -> None:
    scores = {
        s.id: most_compliant_value(s) for s in policy.rubric.sub_conditions
    }
    scores["surprise_extra"] = True  # type: ignore[assignment]
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="unknown sub-conditions"):
        judge.score(policy, anchor, "...")


def test_unknown_label_id_raises(policy: Policy, anchor: AnchorProbe) -> None:
    """BARS-specific: a label id the rubric doesn't declare must be rejected."""
    scores = {
        s.id: most_compliant_value(s) for s in policy.rubric.sub_conditions
    }
    target = next(s for s in policy.rubric.sub_conditions if s.scale == "labels")
    scores[target.id] = "not_a_real_label"
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="unknown label"):
        judge.score(policy, anchor, "...")


def test_wrong_type_for_labels_raises(policy: Policy, anchor: AnchorProbe) -> None:
    """Non-string raw value for a labels-scale sub-condition is rejected."""
    scores = {
        s.id: most_compliant_value(s) for s in policy.rubric.sub_conditions
    }
    target = next(s for s in policy.rubric.sub_conditions if s.scale == "labels")
    scores[target.id] = 3  # type: ignore[assignment] # int, not str
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="expected one of"):
        judge.score(policy, anchor, "...")


def test_scale_violation_for_zero_three_raises(anchor: AnchorProbe) -> None:
    """Out-of-range integer on a 0-3 sub-condition (uses mixed-scale test policy)."""
    policy = _mixed_scale_policy()
    scores: dict[str, bool | int | str] = {
        "bool_a": True,
        "zero_three_a": 5,  # out of [0,3]
        "labels_a": "ok",
        "bool_b": True,
        "zero_three_b": 1,
    }
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="out of"):
        judge.score(policy, anchor, "...")


def test_wrong_type_for_boolean_raises(anchor: AnchorProbe) -> None:
    """Non-bool on a boolean sub-condition (uses mixed-scale test policy)."""
    policy = _mixed_scale_policy()
    scores: dict[str, bool | int | str] = {
        "bool_a": 1,  # int, not bool
        "zero_three_a": 2,
        "labels_a": "ok",
        "bool_b": True,
        "zero_three_b": 1,
    }
    fake = FakeLLMClient(responses=[_wrap_response(json.dumps({"scores": scores}))])
    judge = Judge(fake, model="judge:test", supervised_model="llama:test")
    with pytest.raises(LLMResponseError, match="boolean"):
        judge.score(policy, anchor, "...")
