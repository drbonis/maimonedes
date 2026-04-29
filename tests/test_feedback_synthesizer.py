"""Unit tests for the Phase 4 feedback synthesizer."""
from __future__ import annotations

from pathlib import Path

import pytest

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.feedback import ContrastivePair
from maimonedes.core.policy import Policy, load_policy
from maimonedes.feedback.prompts import (
    EXAMPLE_OUTPUT,
    SYSTEM_PROMPT,
    feedback_prompt,
)
from maimonedes.feedback.synthesizer import (
    MAX_FEEDBACK_CHARS,
    FeedbackSynthesizer,
)
from maimonedes.llm.client import ChatResponse
from tests.fakes import FakeLLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"


@pytest.fixture
def policy() -> Policy:
    return load_policy(POLICY_PATH, RUBRIC_PATH)


def _make_score(aggregate: float, per_sub: dict[str, float]) -> ComplianceScore:
    return ComplianceScore(
        anchor_id="A1",
        policy_id="scope_of_practice",
        per_sub_condition=per_sub,
        aggregate=aggregate,
        judge_model="judge:test",
        supervised_model="llama:test",
    )


@pytest.fixture
def pair_temporal() -> ContrastivePair:
    safe_per_sub = {
        "flags_physician_review": 1.0,
        "expresses_uncertainty": 1.0,
        "avoids_unqualified_prescriptive_language": 1.0,
        "recommendation_within_scope": 1.0,
        "recommendation_appropriate_specificity": 1.0,
    }
    near_per_sub = {
        "flags_physician_review": 0.0,
        "expresses_uncertainty": 0.5,
        "avoids_unqualified_prescriptive_language": 0.5,
        "recommendation_within_scope": 1.0,
        "recommendation_appropriate_specificity": 1.0,
    }
    return ContrastivePair(
        anchor_id="A1",
        kind="temporal",
        safe_text="This requires physician review.",
        safe_score=_make_score(1.0, safe_per_sub),
        near_boundary_text="Take 10mg of metformin daily.",
        near_boundary_score=_make_score(0.4, near_per_sub),
        dropoff_axes=[
            "flags_physician_review",
            "expresses_uncertainty",
            "avoids_unqualified_prescriptive_language",
        ],
    )


@pytest.fixture
def pair_fragility(pair_temporal: ContrastivePair) -> ContrastivePair:
    return pair_temporal.model_copy(update={"kind": "fragility"})


# ---- prompt template -------------------------------------------------------


def test_feedback_prompt_includes_both_texts_and_dropoff_hint(
    policy: Policy, pair_temporal: ContrastivePair
) -> None:
    messages = feedback_prompt(policy, pair_temporal)
    assert messages[0].role == "system"
    assert messages[0].content == SYSTEM_PROMPT
    assert messages[1].role == "user"
    body = messages[1].content
    assert "This requires physician review." in body
    assert "Take 10mg of metformin daily." in body
    assert "flags_physician_review" in body  # dropoff hint
    assert "Compliant aggregate: 1.000" in body
    assert "Near-boundary aggregate: 0.400" in body
    assert EXAMPLE_OUTPUT in body
    assert "scope_of_practice" not in body  # policy.id is not exposed; policy.text is


def test_feedback_prompt_renders_fragility_kind(
    policy: Policy, pair_fragility: ContrastivePair
) -> None:
    messages = feedback_prompt(policy, pair_fragility)
    assert "Contrastive scenario: fragility" in messages[1].content


def test_feedback_prompt_is_deterministic(
    policy: Policy, pair_temporal: ContrastivePair
) -> None:
    """Same `(policy, pair)` → same messages so RecordingClient replay holds."""
    a = feedback_prompt(policy, pair_temporal)
    b = feedback_prompt(policy, pair_temporal)
    assert [m.model_dump() for m in a] == [m.model_dump() for m in b]


# ---- synthesizer happy path ------------------------------------------------


def test_synthesize_returns_clean_text(
    policy: Policy, pair_temporal: ContrastivePair
) -> None:
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content=(
                    "Always defer prescribing decisions to the supervising "
                    "physician. Express uncertainty when outside scope."
                ),
                model="judge:test",
                latency_ms=1.0,
            )
        ]
    )
    synth = FeedbackSynthesizer(fake, model="judge:test")
    result = synth.synthesize(policy, pair_temporal)
    assert result.text.startswith("Always defer prescribing")
    assert "[user]" in result.prompt
    assert "Compliant aggregate: 1.000" in result.prompt


# ---- validation ------------------------------------------------------------


def test_synthesize_strips_leading_and_trailing_fences(
    policy: Policy, pair_temporal: ContrastivePair
) -> None:
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content="```\nDefer to physician.\n```",
                model="judge:test",
                latency_ms=1.0,
            )
        ]
    )
    synth = FeedbackSynthesizer(fake, model="judge:test")
    result = synth.synthesize(policy, pair_temporal)
    assert result.text == "Defer to physician."


def test_synthesize_strips_surrounding_quotes(
    policy: Policy, pair_temporal: ContrastivePair
) -> None:
    fake = FakeLLMClient(
        responses=[
            ChatResponse(
                content='"Defer to physician always."',
                model="judge:test",
                latency_ms=1.0,
            )
        ]
    )
    synth = FeedbackSynthesizer(fake, model="judge:test")
    result = synth.synthesize(policy, pair_temporal)
    assert result.text == "Defer to physician always."


def test_synthesize_rejects_empty_response(
    policy: Policy, pair_temporal: ContrastivePair
) -> None:
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content="   \n  ", model="judge:test", latency_ms=1.0)
        ]
    )
    synth = FeedbackSynthesizer(fake, model="judge:test")
    with pytest.raises(ValueError, match="empty"):
        synth.synthesize(policy, pair_temporal)


def test_synthesize_rejects_only_fences(
    policy: Policy, pair_temporal: ContrastivePair
) -> None:
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content="```\n```", model="judge:test", latency_ms=1.0)
        ]
    )
    synth = FeedbackSynthesizer(fake, model="judge:test")
    with pytest.raises(ValueError, match="fences/quotes"):
        synth.synthesize(policy, pair_temporal)


def test_synthesize_rejects_overlong_response(
    policy: Policy, pair_temporal: ContrastivePair
) -> None:
    long_text = "Defer. " * 200  # ~1400 chars
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content=long_text, model="judge:test", latency_ms=1.0)
        ]
    )
    synth = FeedbackSynthesizer(fake, model="judge:test")
    with pytest.raises(ValueError, match="exceeds"):
        synth.synthesize(policy, pair_temporal)


def test_max_feedback_chars_constant_is_600() -> None:
    assert MAX_FEEDBACK_CHARS == 600


def test_synthesize_uses_configured_temperature(
    policy: Policy, pair_temporal: ContrastivePair
) -> None:
    fake = FakeLLMClient(
        responses=[
            ChatResponse(content="Defer.", model="judge:test", latency_ms=1.0)
        ]
    )
    synth = FeedbackSynthesizer(fake, model="judge:test", temperature=0.7)
    synth.synthesize(policy, pair_temporal)
    assert fake.calls[-1].temperature == 0.7
