"""Phase 4 feedback synthesizer.

LLM call against the judge model (or any compatible client) that
turns a `ContrastivePair` into a short prescriptive recommendation
suitable for prepending to the supervised system's prompt.

The output is free text — no JSON. Validation belongs to the wrapper
(non-empty, length-capped, no markdown fences) so a runaway judge
can't poison the supervised system prompt with a wall of fenced
markdown.
"""
from __future__ import annotations

from dataclasses import dataclass

from maimonedes.core.feedback import ContrastivePair
from maimonedes.core.policy import Policy
from maimonedes.feedback.prompts import feedback_prompt
from maimonedes.llm.client import LLMClient


MAX_FEEDBACK_CHARS = 600
FENCE_CHARS = ("```", '"""', "'''")


@dataclass(frozen=True)
class SynthesizedFeedback:
    """The synthesizer's return shape — text + audit-trail id + the prompt."""

    text: str
    llm_call_id: int | None
    prompt: str  # joined messages, useful for tests + debugging


class FeedbackSynthesizer:
    """Builds prescriptive recommendations from contrastive pairs."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        temperature: float = 0.3,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    @property
    def model(self) -> str:
        return self._model

    def synthesize(
        self,
        policy: Policy,
        pair: ContrastivePair,
    ) -> SynthesizedFeedback:
        messages = feedback_prompt(policy, pair)
        response = self._client.chat_completion(
            messages,
            model=self._model,
            temperature=self._temperature,
        )
        text = self._validate(response.content)
        prompt_text = "\n\n".join(
            f"[{m.role}]\n{m.content}" for m in messages
        )
        llm_call_id = getattr(response, "llm_call_id", None)
        return SynthesizedFeedback(
            text=text,
            llm_call_id=llm_call_id,
            prompt=prompt_text,
        )

    @staticmethod
    def _validate(raw: str) -> str:
        text = raw.strip()
        if not text:
            raise ValueError("synthesized feedback is empty")
        # Strip a single pair of leading/trailing fences if present.
        for fence in FENCE_CHARS:
            if text.startswith(fence):
                text = text[len(fence) :].lstrip("\n")
            if text.endswith(fence):
                text = text[: -len(fence)].rstrip("\n")
        text = text.strip().strip('"').strip("'").strip()
        if not text:
            raise ValueError("synthesized feedback was only fences/quotes")
        if len(text) > MAX_FEEDBACK_CHARS:
            raise ValueError(
                f"synthesized feedback exceeds {MAX_FEEDBACK_CHARS} chars "
                f"({len(text)} chars); refusing to deliver"
            )
        return text


__all__ = [
    "FENCE_CHARS",
    "FeedbackSynthesizer",
    "MAX_FEEDBACK_CHARS",
    "SynthesizedFeedback",
]
