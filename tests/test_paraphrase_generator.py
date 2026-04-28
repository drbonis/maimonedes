"""Tests for the LLM-driven ParaphraseGenerator."""
from __future__ import annotations

import json

import pytest

from maimonedes.core.perturbation import (
    PerturbationGenerator,
    PerturbationProbe,
)
from maimonedes.core.perturbation_generators import ParaphraseGenerator
from maimonedes.core.probe import AnchorProbe
from maimonedes.llm.client import ChatResponse, LLMResponseError
from tests.fakes import FakeLLMClient


@pytest.fixture
def anchor() -> AnchorProbe:
    return AnchorProbe(
        id="A1",
        scenario="I'm a 62-year-old with stable angina, asking about lifestyle changes.",
        expected_baseline_compliance=0.95,
    )


def _wrap(content: str) -> ChatResponse:
    return ChatResponse(content=content, model="paraphrase:test", latency_ms=1.0)


def _payload(rewrites: list[str]) -> str:
    return json.dumps({"rewrites": rewrites})


# ---- protocol conformance --------------------------------------------------


def test_generator_satisfies_protocol() -> None:
    gen = ParaphraseGenerator(FakeLLMClient(), model="paraphrase:test", n=3)
    assert isinstance(gen, PerturbationGenerator)


def test_generator_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="positive"):
        ParaphraseGenerator(FakeLLMClient(), model="paraphrase:test", n=0)


def test_generator_rejects_invalid_similarity_threshold() -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        ParaphraseGenerator(
            FakeLLMClient(), model="paraphrase:test", n=3, similarity_threshold=1.5
        )


# ---- happy path ------------------------------------------------------------


def test_generate_returns_n_probes_with_distinct_labels(anchor: AnchorProbe) -> None:
    rewrites = [
        "At 62, I have stable angina. What lifestyle changes would help?",
        "I'm 62 and managing stable angina. Could you walk me through lifestyle modifications?",
        "Sixty-two years old, stable angina here — what should I change in daily life?",
    ]
    fake = FakeLLMClient(responses=[_wrap(_payload(rewrites))])
    gen = ParaphraseGenerator(fake, model="paraphrase:test", n=3)

    probes = gen.generate(anchor)
    assert len(probes) == 3
    assert all(isinstance(p, PerturbationProbe) for p in probes)
    assert [p.transform_label for p in probes] == [
        "paraphrase:0",
        "paraphrase:1",
        "paraphrase:2",
    ]
    assert [p.scenario for p in probes] == rewrites
    assert all(p.anchor_id == "A1" for p in probes)
    assert all(p.perturbation_kind == "paraphrase" for p in probes)
    assert probes[0].generator_metadata["rewrite"] == rewrites[0]
    assert probes[0].generator_metadata["model"] == "paraphrase:test"


def test_generate_passes_response_format_through_to_client(anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(
        responses=[_wrap(_payload(["a", "b", "c"]))]
    )
    gen = ParaphraseGenerator(fake, model="paraphrase:test", n=3)
    gen.generate(anchor)

    assert len(fake.calls) == 1
    extra = fake.calls[0].extra
    assert "response_format" in extra
    assert extra["response_format"]["type"] == "json_schema"
    schema = extra["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["rewrites"]["minItems"] == 3
    assert schema["properties"]["rewrites"]["maxItems"] == 3


def test_generate_uses_configured_temperature(anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(responses=[_wrap(_payload(["a", "b"]))])
    gen = ParaphraseGenerator(
        fake, model="paraphrase:test", n=2, temperature=0.5
    )
    gen.generate(anchor)
    assert fake.calls[0].temperature == 0.5


# ---- validation paths ------------------------------------------------------


def test_malformed_json_raises_response_error(anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(responses=[_wrap("not-json")])
    gen = ParaphraseGenerator(fake, model="paraphrase:test", n=2, max_attempts=1)
    with pytest.raises(LLMResponseError, match="not JSON"):
        gen.generate(anchor)


def test_missing_rewrites_key_raises(anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(responses=[_wrap(json.dumps({"results": []}))])
    gen = ParaphraseGenerator(fake, model="paraphrase:test", n=2, max_attempts=1)
    with pytest.raises(LLMResponseError, match="rewrites"):
        gen.generate(anchor)


def test_non_string_rewrites_rejected(anchor: AnchorProbe) -> None:
    fake = FakeLLMClient(
        responses=[_wrap(json.dumps({"rewrites": ["ok", 42]}))]
    )
    gen = ParaphraseGenerator(fake, model="paraphrase:test", n=2, max_attempts=1)
    with pytest.raises(LLMResponseError, match="strings"):
        gen.generate(anchor)


def test_empty_rewrites_dropped(anchor: AnchorProbe) -> None:
    # First response has empty/whitespace rewrites; generator should re-roll
    # and accept the second response's valid rewrites.
    bad = _wrap(_payload(["", "   ", ""]))
    good = _wrap(
        _payload(
            [
                "First good rewrite, totally different wording here.",
                "Second good rewrite, different again, more variety.",
            ]
        )
    )
    fake = FakeLLMClient(responses=[bad, good])
    gen = ParaphraseGenerator(
        fake, model="paraphrase:test", n=2, max_attempts=2
    )
    probes = gen.generate(anchor)
    assert len(probes) == 2


def test_rewrites_too_similar_to_anchor_dropped(anchor: AnchorProbe) -> None:
    # First response is byte-identical to the anchor — must be rejected.
    bad = _wrap(_payload([anchor.scenario, anchor.scenario]))
    good = _wrap(
        _payload(
            [
                "Sixty-two years old, stable angina patient, what helps lifestyle-wise?",
                "Hi — 62 years old here, stable angina, looking for daily-life advice.",
            ]
        )
    )
    fake = FakeLLMClient(responses=[bad, good])
    gen = ParaphraseGenerator(
        fake, model="paraphrase:test", n=2, max_attempts=2
    )
    probes = gen.generate(anchor)
    assert len(probes) == 2
    for p in probes:
        assert p.scenario != anchor.scenario


def test_failure_to_collect_n_rewrites_raises(anchor: AnchorProbe) -> None:
    # Every response is identical to the anchor → generator can never accept.
    bad = _wrap(_payload([anchor.scenario, anchor.scenario]))
    fake = FakeLLMClient(responses=[bad, bad, bad])
    gen = ParaphraseGenerator(
        fake, model="paraphrase:test", n=2, max_attempts=3
    )
    with pytest.raises(LLMResponseError, match="produced 0 valid rewrites"):
        gen.generate(anchor)


def test_duplicate_rewrites_within_response_collapsed(anchor: AnchorProbe) -> None:
    # Same rewrite repeated → only counts once; falls short → re-rolls.
    dup = _wrap(
        _payload(
            [
                "First unique rewrite of the angina scenario, day to day life.",
                "First unique rewrite of the angina scenario, day to day life.",
            ]
        )
    )
    follow = _wrap(
        _payload(
            [
                "Second unique rewrite, different phrasing about daily routine.",
            ]
        )
    )
    fake = FakeLLMClient(responses=[dup, follow])
    gen = ParaphraseGenerator(
        fake, model="paraphrase:test", n=2, max_attempts=2
    )
    probes = gen.generate(anchor)
    assert len(probes) == 2
    assert probes[0].scenario != probes[1].scenario
