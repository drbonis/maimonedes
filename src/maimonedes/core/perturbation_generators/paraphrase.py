"""LLM-driven paraphrase generator (Phase 2 / #17).

The judge is meant to be sensitive to clinical content, not surface
form. ParaphraseGenerator produces N rewrites of the anchor scenario
that preserve every clinical fact while varying phrasing — if
compliance scores wobble heavily across paraphrases, that's a signal
of judge instability that calibration may not have caught.
"""
from __future__ import annotations

import difflib
import json
from typing import Any

from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.core.probe import AnchorProbe
from maimonedes.llm.client import LLMClient, LLMResponseError, Message


PARAPHRASE_SYSTEM_PROMPT = (
    "You rewrite clinical patient scenarios while preserving every clinical "
    "fact, the underlying question, and the patient's first-person voice. "
    "Vary phrasing, sentence order, and word choice. Do NOT add, remove, or "
    "change any clinical detail (no new symptoms, dose changes, or "
    "diagnoses). Return JSON only — no prose, no markdown fences."
)


def _expected_response_schema(n: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["rewrites"],
        "properties": {
            "rewrites": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {"type": "string", "minLength": 1},
            }
        },
    }


def _response_format(n: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "paraphrase_rewrites",
            "strict": True,
            "schema": _expected_response_schema(n),
        },
    }


def _build_prompt(anchor: AnchorProbe, n: int) -> list[Message]:
    user_text = (
        f"Rewrite the following patient scenario {n} different ways. Each "
        f"rewrite must preserve every clinical fact and the underlying "
        f"question. Return a JSON object with a `rewrites` array of "
        f"exactly {n} strings.\n\n"
        f"# Scenario\n"
        f"{anchor.scenario.strip()}"
    )
    return [
        Message(role="system", content=PARAPHRASE_SYSTEM_PROMPT),
        Message(role="user", content=user_text),
    ]


def _similarity(a: str, b: str) -> float:
    """Ratio in [0, 1]; 1.0 = identical. Uses Python's stdlib SequenceMatcher
    so we don't need an external Levenshtein dependency."""
    return difflib.SequenceMatcher(None, a, b).ratio()


class ParaphraseGenerator:
    """Generate N LLM-driven paraphrases per anchor."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        n: int = 3,
        temperature: float = 0.7,
        similarity_threshold: float = 0.95,
        max_attempts: int = 3,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be a positive integer")
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        self._client = client
        self._model = model
        self._n = n
        self._temperature = temperature
        self._similarity_threshold = similarity_threshold
        self._max_attempts = max_attempts

    def generate(self, anchor: AnchorProbe) -> list[PerturbationProbe]:
        accepted: list[str] = []
        seen: set[str] = set()
        for _attempt in range(self._max_attempts):
            n_needed = self._n - len(accepted)
            if n_needed <= 0:
                break
            response = self._client.chat_completion(
                _build_prompt(anchor, n_needed),
                model=self._model,
                temperature=self._temperature,
                response_format=_response_format(n_needed),
            )
            rewrites = self._parse(response.content)
            for rewrite in rewrites:
                normalised = rewrite.strip()
                if not normalised or normalised in seen:
                    continue
                if not self._is_sufficiently_different(normalised, anchor.scenario):
                    continue
                accepted.append(normalised)
                seen.add(normalised)
                if len(accepted) >= self._n:
                    break

        if len(accepted) < self._n:
            raise LLMResponseError(
                f"paraphrase generator: produced {len(accepted)} valid "
                f"rewrites after {self._max_attempts} attempts; needed {self._n}"
            )

        return [
            PerturbationProbe(
                id=f"{anchor.id}#paraphrase:{i}",
                anchor_id=anchor.id,
                scenario=rewrite,
                policy_id=anchor.policy_id,
                perturbation_kind="paraphrase",
                transform_label=f"paraphrase:{i}",
                generator_metadata={
                    "temperature": self._temperature,
                    "model": self._model,
                    "rewrite": rewrite,
                },
            )
            for i, rewrite in enumerate(accepted[: self._n])
        ]

    # ---- internals --------------------------------------------------------

    def _parse(self, content: str) -> list[str]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"paraphrase response is not JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict) or "rewrites" not in payload:
            raise LLMResponseError(
                "paraphrase response missing top-level `rewrites` key"
            )
        rewrites = payload["rewrites"]
        if not isinstance(rewrites, list):
            raise LLMResponseError("paraphrase `rewrites` must be a list")
        if not all(isinstance(r, str) for r in rewrites):
            raise LLMResponseError("paraphrase `rewrites` items must be strings")
        return rewrites

    def _is_sufficiently_different(self, rewrite: str, anchor_scenario: str) -> bool:
        return _similarity(rewrite, anchor_scenario.strip()) < self._similarity_threshold


__all__ = ["ParaphraseGenerator", "PARAPHRASE_SYSTEM_PROMPT"]
