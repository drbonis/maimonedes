"""Tests for the K-NN exemplar probe synthesizer."""
from __future__ import annotations

from pathlib import Path

import pytest

from maimonedes.core.policy import Policy, load_policy
from maimonedes.core.probe import load_anchors
from maimonedes.feedback.probe_prompts import (
    SYNTHESIS_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    synthesis_prompt,
    validator_prompt,
)
from maimonedes.feedback.probe_synthesis import (
    DEFAULT_K,
    DEFAULT_MAX_RETRIES,
    DEFAULT_TAU,
    KnnExemplarSynthesizer,
    SynthesisResult,
    _cosine_similarity,
    _missing_structural_anchors,
    _parse_validator,
    _strip_text,
    to_synthesized_probe,
)
from maimonedes.llm.client import ChatResponse
from maimonedes.llm.embed_client import EmbedResponse
from tests.fakes import FakeEmbedClient, FakeLLMClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"
PROBES_PATH = PROJECT_ROOT / "config" / "probes" / "anchors_v1.yaml"


@pytest.fixture
def policy() -> Policy:
    return load_policy(POLICY_PATH, RUBRIC_PATH)


@pytest.fixture
def anchors():
    return load_anchors(PROBES_PATH)


# ---- helpers ---------------------------------------------------------------


def test_cosine_similarity_known_values() -> None:
    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert _cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_norm_returns_zero() -> None:
    assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0


def test_strip_text_removes_fences_and_quotes() -> None:
    assert _strip_text("```\nhello\n```") == "hello"
    assert _strip_text('"hello"') == "hello"
    assert _strip_text("  spaced  ") == "spaced"


def test_strip_text_removes_okay_i_understand_prefix() -> None:
    raw = "Okay, I understand the rules. Hi, I'm John, a 65-year-old man."
    assert _strip_text(raw) == "Hi, I'm John, a 65-year-old man."


def test_strip_text_removes_new_scenario_header() -> None:
    raw = "**New Scenario:**\n\nHi, I'm John, a 58-year-old man working as a teacher."
    assert _strip_text(raw).startswith("Hi, I'm John")


def test_strip_text_removes_output_header() -> None:
    raw = "**Output:** Hi, I'm John, a 58-year-old man."
    assert _strip_text(raw) == "Hi, I'm John, a 58-year-old man."


def test_strip_text_removes_numbered_list_prefix() -> None:
    raw = "9. (A9) Hi, I'm John, a 60-year-old man working as a teacher."
    assert _strip_text(raw).startswith("Hi, I'm John")


def test_strip_text_removes_anchor_id_prefix() -> None:
    raw = "(A1) Hi, I'm John, a 58-year-old man working as a teacher."
    assert _strip_text(raw).startswith("Hi, I'm John")


def test_strip_text_strips_stacked_prefixes() -> None:
    raw = (
        "Okay, I understand the rules. **New Scenario:** "
        "Hi, I'm John, a 65-year-old man."
    )
    assert _strip_text(raw).startswith("Hi, I'm John")


def test_strip_text_passes_through_clean_scenario() -> None:
    raw = "Hi, I'm Maria, a 52-year-old female nurse. ..."
    assert _strip_text(raw) == raw


def test_parse_validator_yes_approved() -> None:
    approved, reason = _parse_validator("yes: this is realistic.")
    assert approved is True
    assert "realistic" in reason


def test_parse_validator_no_rejected() -> None:
    approved, reason = _parse_validator("no: too vague.")
    assert approved is False
    assert "vague" in reason


def test_parse_validator_markdown_bolded_yes_approved() -> None:
    """medgemma sometimes returns `**yes**: ...`; the parser must accept it."""
    approved, reason = _parse_validator(
        "**yes**: clinically realistic patient query"
    )
    assert approved is True
    assert "realistic" in reason


def test_parse_validator_markdown_bolded_yes_without_colon() -> None:
    approved, _reason = _parse_validator(
        "**yes**\nThis is a clinically realistic patient query."
    )
    assert approved is True


def test_parse_validator_markdown_bolded_no_rejected() -> None:
    approved, reason = _parse_validator("**no**: missing demographic preamble.")
    assert approved is False
    assert "preamble" in reason


def test_parse_validator_ambiguous_treated_as_rejected() -> None:
    approved, reason = _parse_validator("Maybe — depends on context.")
    assert approved is False
    assert "Maybe" in reason


def test_parse_validator_empty_treated_as_rejected() -> None:
    approved, reason = _parse_validator("   ")
    assert approved is False
    assert "empty" in reason


def test_parse_validator_internal_monologue_rejected() -> None:
    """Validator sometimes dumps thinking-process notes instead of yes/no.
    First word isn't 'yes' — should be rejected."""
    approved, _ = _parse_validator(
        "**Thinking Process:**\n\n1.  Analyze the Policy: ..."
    )
    assert approved is False


# ---- prompt template -------------------------------------------------------


def test_synthesis_prompt_includes_exemplars(policy: Policy, anchors) -> None:
    selected = list(anchors)[:3]
    messages = synthesis_prompt(policy, selected)
    assert messages[0].content == SYNTHESIS_SYSTEM_PROMPT
    body = messages[1].content
    for a in selected:
        assert a.id in body
        # First few words of each scenario should appear.
        assert a.scenario.split()[0] in body


def test_synthesis_prompt_includes_retry_nudge(policy: Policy, anchors) -> None:
    messages = synthesis_prompt(
        policy, list(anchors)[:2], retry_nudge="Try harder."
    )
    assert "Try harder." in messages[1].content


def test_validator_prompt_round_trip(policy: Policy) -> None:
    messages = validator_prompt(policy, "Test scenario.")
    assert messages[0].content == VALIDATOR_SYSTEM_PROMPT
    assert "Test scenario." in messages[1].content


# ---- synthesizer -----------------------------------------------------------


def _make_synthesizer(
    policy: Policy,
    anchors,
    *,
    library_dim: int = 8,
    generator_responses: list[ChatResponse] | None = None,
    validator_responses: list[ChatResponse] | None = None,
    embed_responses: list[EmbedResponse] | None = None,
    k: int = 3,
    tau: float = 0.7,
    max_retries: int = 2,
    n_library: int = 5,
):
    embed_fake = FakeEmbedClient(default_dim=library_dim)
    # Library anchors are embedded at synthesizer init — give them
    # arbitrary (non-overlapping with target) embeddings first, then
    # queue the synthesizer's re-embed responses behind them.
    # Non-zero library embeddings so cosine similarity is defined and
    # KNN search returns non-empty `parent_anchor_ids`.
    library_zeros = [
        EmbedResponse(
            embedding=[0.1] * library_dim,
            model="fake-embed",
            latency_ms=0.0,
        )
        for _ in range(n_library)
    ]
    for r in library_zeros + (embed_responses or []):
        embed_fake.queue(r)
    return (
        KnnExemplarSynthesizer(
            generator_client=FakeLLMClient(responses=generator_responses or []),
            validator_client=FakeLLMClient(responses=validator_responses or []),
            embed_client=embed_fake,
            generator_model="gen:test",
            validator_model="val:test",
            library_anchors=list(anchors)[:n_library],
            policy=policy,
            embedding_model="fake-embed",
            k=k,
            tau=tau,
            max_retries=max_retries,
        ),
        embed_fake,
    )


def test_synthesize_happy_path_approved(policy: Policy, anchors) -> None:
    target_dim = 8
    target = [1.0] + [0.0] * (target_dim - 1)
    # Library anchors get one-hot embeddings keyed by hash; we don't
    # control which one — but the achieved embedding will match the
    # target (we control it via embed queue) so tau passes.
    embed_responses = [
        EmbedResponse(embedding=target, model="fake-embed", latency_ms=0.0)
    ]
    synth, _ = _make_synthesizer(
        policy,
        anchors,
        library_dim=target_dim,
        generator_responses=[
            ChatResponse(content="Hi, I'm John, a 65-year-old man working as a teacher. New patient query.", model="gen:test", latency_ms=0.0),
        ],
        validator_responses=[
            ChatResponse(content="yes: looks realistic.", model="val:test", latency_ms=0.0),
        ],
        embed_responses=embed_responses,
    )
    result = synth.synthesize(target)
    assert result.status == "approved"
    assert result.quality_status == "approved"
    assert result.tau_distance == pytest.approx(1.0, abs=1e-6)
    assert "I'm John" in result.scenario
    assert result.parent_anchor_ids


def test_synthesize_below_tau_retries_then_succeeds(
    policy: Policy, anchors
) -> None:
    target_dim = 8
    target = [1.0] + [0.0] * (target_dim - 1)
    far = [0.0] * target_dim
    far[5] = 1.0  # cosine 0 vs target

    embed_responses = [
        EmbedResponse(embedding=far, model="fake-embed", latency_ms=0.0),
        EmbedResponse(embedding=target, model="fake-embed", latency_ms=0.0),
    ]
    synth, _ = _make_synthesizer(
        policy,
        anchors,
        library_dim=target_dim,
        generator_responses=[
            ChatResponse(
                content="Hi, I'm John, a 60-year-old man working as a teacher. Attempt 1.",
                model="gen:test",
                latency_ms=0.0,
            ),
            ChatResponse(
                content="Hi, I'm John, a 60-year-old man working as a teacher. Attempt 2.",
                model="gen:test",
                latency_ms=0.0,
            ),
        ],
        validator_responses=[
            ChatResponse(content="yes: ok.", model="val:test", latency_ms=0.0),
        ],
        embed_responses=embed_responses,
        max_retries=2,
    )
    result = synth.synthesize(target)
    assert result.status == "approved"
    assert result.retries_used == 1
    assert "Attempt 2" in result.scenario


def test_synthesize_validator_rejects_no_retry(
    policy: Policy, anchors
) -> None:
    target_dim = 8
    target = [1.0] + [0.0] * (target_dim - 1)
    embed_responses = [
        EmbedResponse(embedding=target, model="fake-embed", latency_ms=0.0)
    ]
    synth, _ = _make_synthesizer(
        policy,
        anchors,
        library_dim=target_dim,
        generator_responses=[
            ChatResponse(
                content="Hi, I'm John, a 60-year-old man working as a teacher. Borderline.",
                model="gen:test",
                latency_ms=0.0,
            ),
        ],
        validator_responses=[
            ChatResponse(
                content="no: scenario lacks clinical detail.",
                model="val:test",
                latency_ms=0.0,
            ),
        ],
        embed_responses=embed_responses,
        max_retries=2,
    )
    result = synth.synthesize(target)
    assert result.status == "rejected_validator"
    assert result.quality_status == "rejected"
    assert "lacks" in (result.quality_reason or "")
    assert result.retries_used == 0  # Validator rejection is sticky.


def test_synthesize_max_retries_exhausted(policy: Policy, anchors) -> None:
    target_dim = 8
    target = [1.0] + [0.0] * (target_dim - 1)
    far = [0.0] * target_dim
    far[5] = 1.0

    embed_responses = [
        EmbedResponse(embedding=far, model="fake-embed", latency_ms=0.0),
        EmbedResponse(embedding=far, model="fake-embed", latency_ms=0.0),
        EmbedResponse(embedding=far, model="fake-embed", latency_ms=0.0),
    ]
    synth, _ = _make_synthesizer(
        policy,
        anchors,
        library_dim=target_dim,
        generator_responses=[
            ChatResponse(
                content="Hi, I'm John, a 60-year-old man working as a teacher. Attempt 1.",
                model="gen:test",
                latency_ms=0.0,
            ),
            ChatResponse(
                content="Hi, I'm John, a 60-year-old man working as a teacher. Attempt 2.",
                model="gen:test",
                latency_ms=0.0,
            ),
            ChatResponse(
                content="Hi, I'm John, a 60-year-old man working as a teacher. Attempt 3.",
                model="gen:test",
                latency_ms=0.0,
            ),
        ],
        validator_responses=[],
        embed_responses=embed_responses,
        max_retries=2,
    )
    result = synth.synthesize(target)
    assert result.status == "rejected_max_retries"
    assert result.quality_status == "rejected"
    assert "max_retries_below_tau" in (result.quality_reason or "")


def test_synthesize_zero_max_retries_returns_below_tau_status(
    policy: Policy, anchors
) -> None:
    target_dim = 8
    target = [1.0] + [0.0] * (target_dim - 1)
    far = [0.0] * target_dim
    far[5] = 1.0
    embed_responses = [
        EmbedResponse(embedding=far, model="fake-embed", latency_ms=0.0)
    ]
    synth, _ = _make_synthesizer(
        policy,
        anchors,
        library_dim=target_dim,
        generator_responses=[
            ChatResponse(
                content="Hi, I'm John, a 60-year-old man working as a teacher. Once.",
                model="gen:test",
                latency_ms=0.0,
            ),
        ],
        validator_responses=[],
        embed_responses=embed_responses,
        max_retries=0,
    )
    result = synth.synthesize(target)
    assert result.status == "rejected_below_tau"
    assert result.retries_used == 0


def test_to_synthesized_probe_round_trips_status(
    policy: Policy, anchors
) -> None:
    target_dim = 8
    target = [1.0] + [0.0] * (target_dim - 1)
    embed_responses = [
        EmbedResponse(embedding=target, model="fake-embed", latency_ms=0.0)
    ]
    synth, _ = _make_synthesizer(
        policy,
        anchors,
        library_dim=target_dim,
        generator_responses=[
            ChatResponse(
                content="Hi, I'm John, a 60-year-old man working as a teacher. A scenario.",
                model="gen:test",
                latency_ms=0.0,
            ),
        ],
        validator_responses=[
            ChatResponse(content="yes: realistic.", model="val:test", latency_ms=0.0),
        ],
        embed_responses=embed_responses,
    )
    result = synth.synthesize(target)
    probe = to_synthesized_probe(result, policy_id=policy.id, gp_fit_id=42)
    assert probe.policy_id == policy.id
    assert probe.gp_fit_id == 42
    assert probe.quality_status == "approved"
    assert probe.parent_anchor_ids


def test_synthesizer_rejects_empty_library(policy: Policy) -> None:
    with pytest.raises(ValueError, match="library_anchors empty"):
        KnnExemplarSynthesizer(
            generator_client=FakeLLMClient(),
            validator_client=FakeLLMClient(),
            embed_client=FakeEmbedClient(),
            generator_model="gen",
            validator_model="val",
            library_anchors=[],
            policy=policy,
        )


def test_default_constants_match_locked_decisions() -> None:
    assert DEFAULT_K == 5
    assert DEFAULT_TAU == 0.7
    assert DEFAULT_MAX_RETRIES == 3


# ---- structural anchor check (Option B: regex pre-tau) --------------------


def test_missing_structural_anchors_detects_each_token() -> None:
    """`John`, `N-year-old`, `teacher` are required substitution targets."""
    full = "Hi, I'm John, a 65-year-old man working as a teacher. Question?"
    assert _missing_structural_anchors(full) == []

    no_name = "Hi, a 65-year-old man working as a teacher. Question?"
    assert _missing_structural_anchors(no_name) == ["first name 'John'"]

    no_age = "Hi, I'm John, a man working as a teacher. Question?"
    assert _missing_structural_anchors(no_age) == ["'N-year-old' age token"]

    no_occupation = "Hi, I'm John, a 65-year-old man asking a question."
    assert _missing_structural_anchors(no_occupation) == ["occupation 'teacher'"]

    none = "Hi, I'm Maria, a 52-year-old female nurse asking."
    missing = _missing_structural_anchors(none)
    assert "first name 'John'" in missing
    assert "occupation 'teacher'" in missing


def test_synthesize_retries_when_structural_anchors_missing(
    policy: Policy, anchors
) -> None:
    """First attempt lacks 'John' / 'teacher'; second attempt is well-formed.

    The structural retry happens BEFORE the embed call, so the embed
    client only sees one re-embed request (for the well-formed second
    attempt) — proves the retry is on the cheap path."""
    target_dim = 8
    target = [1.0] + [0.0] * (target_dim - 1)
    embed_responses = [
        EmbedResponse(embedding=target, model="fake-embed", latency_ms=0.0),
    ]
    synth, embed_fake = _make_synthesizer(
        policy,
        anchors,
        library_dim=target_dim,
        generator_responses=[
            ChatResponse(
                content="A vague clinical scenario without anchors.",
                model="gen:test",
                latency_ms=0.0,
            ),
            ChatResponse(
                content=(
                    "Hi, I'm John, a 60-year-old man working as a teacher. "
                    "Should I take more metformin?"
                ),
                model="gen:test",
                latency_ms=0.0,
            ),
        ],
        validator_responses=[
            ChatResponse(content="yes: realistic.", model="val:test", latency_ms=0.0),
        ],
        embed_responses=embed_responses,
        max_retries=2,
    )
    result = synth.synthesize(target)
    assert result.status == "approved"
    assert result.retries_used == 1, (
        "expected exactly one structural retry before approval"
    )
    assert "I'm John" in result.scenario


def test_synthesize_max_retries_on_structural_failure(
    policy: Policy, anchors
) -> None:
    """Generator never produces required anchors → rejected_max_retries
    with a quality_reason that names the structural gap (so the operator
    can tell this from a τ-failure run)."""
    target_dim = 8
    target = [1.0] + [0.0] * (target_dim - 1)
    synth, embed_fake = _make_synthesizer(
        policy,
        anchors,
        library_dim=target_dim,
        generator_responses=[
            ChatResponse(content="No anchors 1.", model="gen:test", latency_ms=0.0),
            ChatResponse(content="No anchors 2.", model="gen:test", latency_ms=0.0),
            ChatResponse(content="No anchors 3.", model="gen:test", latency_ms=0.0),
        ],
        validator_responses=[],
        embed_responses=[],  # No embeds expected — structural check trips first.
        max_retries=2,
    )
    result = synth.synthesize(target)
    assert result.status == "rejected_max_retries"
    assert "missing_structural_anchors" in (result.quality_reason or "")
    # No embed calls should have happened — the structural check is on
    # the cheap path BEFORE re-embedding.
    assert len(embed_fake.calls) == 5  # 5 library anchors at init only
