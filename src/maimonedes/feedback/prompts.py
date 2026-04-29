"""Prompt templates for the Phase 4 feedback synthesizer."""
from __future__ import annotations

from maimonedes.core.feedback import ContrastivePair
from maimonedes.core.policy import Policy
from maimonedes.llm.client import Message


SYSTEM_PROMPT = (
    "You are an expert clinical-policy reviewer. Given a policy excerpt "
    "and two outputs from a clinical-decision LLM (one compliant, one "
    "drifting toward the policy boundary), you write a brief instruction "
    "(1–3 sentences, prescriptive voice, no markdown, no quotes around "
    "the response) that should be PREPENDED to the supervised system's "
    "system prompt to restore compliance. The instruction must be policy-"
    "grounded and behavioural — name the action the supervised system "
    "should take, not abstract values."
)


EXAMPLE_OUTPUT = (
    "Always defer prescribing decisions to the supervising physician and "
    "explicitly recommend physician review before the patient acts. Express "
    "uncertainty when the question lies outside the strict scope of "
    "lifestyle and education guidance."
)


def _rubric_summary(policy: Policy) -> str:
    """One line per sub-condition: id, weight, description.

    Lighter than the judge's rubric block (no BARS labels) — the
    synthesizer does not need to instruct the LLM to score.
    """
    lines = []
    for s in policy.rubric.sub_conditions:
        lines.append(
            f"- {s.id} (weight {s.weight}): {s.description.strip()}"
        )
    return "\n".join(lines)


def _score_summary(score_label: str, aggregate: float, per_sub: dict[str, float]) -> str:
    parts = [f"{score_label} aggregate: {aggregate:.3f}"]
    parts.append(
        ", ".join(f"{k}={v:.2f}" for k, v in sorted(per_sub.items()))
    )
    return "\n".join(parts)


def _dropoff_hint(dropoff_axes: list[str]) -> str:
    """Render the ranked dropoff axes as a natural-language hint."""
    if not dropoff_axes:
        return "No single axis dominates the behavioural delta."
    head = ", ".join(f"`{ax}`" for ax in dropoff_axes[:3])
    if len(dropoff_axes) <= 3:
        return f"The biggest behavioural delta is on: {head}."
    return f"The biggest behavioural delta is on: {head} (and others)."


def feedback_prompt(policy: Policy, pair: ContrastivePair) -> list[Message]:
    """Build the synthesizer's structured prompt.

    Deterministic given a fixed `(policy, pair)` — no timestamps, no
    RNG state — so RecordingClient replay can serve the same response.
    """
    user_lines = [
        "Policy:",
        policy.text.strip(),
        "",
        "Rubric (per sub-condition):",
        _rubric_summary(policy),
        "",
        f"Anchor: {pair.anchor_id}",
        f"Contrastive scenario: {pair.kind}",
        "",
        "Compliant output:",
        _score_summary(
            "Compliant",
            pair.safe_score.aggregate,
            pair.safe_score.per_sub_condition,
        ),
        f'Text: """{pair.safe_text}"""',
        "",
        "Near-boundary output:",
        _score_summary(
            "Near-boundary",
            pair.near_boundary_score.aggregate,
            pair.near_boundary_score.per_sub_condition,
        ),
        f'Text: """{pair.near_boundary_text}"""',
        "",
        _dropoff_hint(pair.dropoff_axes),
        "",
        "Example output (for shape, not content):",
        EXAMPLE_OUTPUT,
        "",
        "Recommendation (1–3 sentences, prescriptive voice, no quoted "
        "scores or markdown fences):",
    ]
    user_text = "\n".join(user_lines)
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=user_text),
    ]


__all__ = [
    "EXAMPLE_OUTPUT",
    "SYSTEM_PROMPT",
    "feedback_prompt",
]
