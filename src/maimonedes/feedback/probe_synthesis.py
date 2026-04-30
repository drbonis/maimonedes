"""Phase 5 K-NN exemplar probe synthesizer.

Per the locked decisions in the planning thread:
- Strategy: K-NN exemplar prompting (find K nearest library anchors
  to the GP target, ask the generator LLM to synthesize a NEW
  scenario clinically similar but exploring different territory).
- Verification: re-embed the generated scenario and compare to the
  GP target via cosine similarity; reject if below tau (default 0.7).
- Retry: up to `max_retries` (default 3) on tau failures, with a
  small instruction nudge to break local minima.
- Quality gating: LLM-as-validator with a configurable model;
  validator rejections are sticky (no retry), tau rejections are not.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from maimonedes.core.policy import Policy
from maimonedes.core.probe import AnchorProbe
from maimonedes.core.synthesized_probe import (
    GenerationMethod,
    QualityStatus,
    SynthesizedProbe,
)
from maimonedes.feedback.probe_prompts import (
    synthesis_prompt,
    validator_prompt,
)
from maimonedes.llm.client import LLMClient
from maimonedes.llm.embed_client import EmbedClient


log = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_TAU = 0.7
DEFAULT_MAX_RETRIES = 3
MAX_SCENARIO_CHARS = 500
FENCE_CHARS = ("```", '"""', "'''")

# Meta-prefix patterns the generator LLM emits despite the prompt's
# "no markdown, no commentary" instruction. medgemma 4B in particular
# tends to wrap its scenario in markdown headers or open with a
# rule-acknowledgement preamble. We strip these so the cleaned text
# starts at the actual scenario.
_META_PREFIX_RE = re.compile(
    r"""
    ^(?:
        # "Okay, I understand the rules. I will generate ..." up to the
        # first sentence-ending punctuation.
        okay[^.]*\.\s*
        | # "**New Scenario:**", "**Output:**", "**Scenario:**", etc.
        \**\s*(new\s+scenario|output|scenario|new\s+scenario\s*\d*)\s*:?\s*\**\s*\n?
        | # Numbered list-item prefix like "9. (A9) "
        \d+\.\s*\([A-Z]\d+\)\s*
        | # Bare list-item prefix like "(A1) "
        \(\s*[A-Z]\d+\s*\)\s*
        | # "**Thinking Process:**" / "**Constraint Checklist:**" — these
          # almost always indicate a generator failure rather than a
          # recoverable scenario, but we still try to extract what's after
          # them; if nothing remains, the validator catches it.
        \**\s*(thinking\s+process|constraint\s+checklist)\s*:?\s*\**\s*\n?
    )
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


SynthesisStatus = Literal[
    "approved",
    "rejected_below_tau",
    "rejected_validator",
    "rejected_max_retries",
]


@dataclass(frozen=True)
class SynthesisResult:
    """One synthesizer run's outcome."""

    scenario: str
    target_embedding: list[float]
    achieved_embedding: list[float]
    tau_distance: float  # cosine similarity in [-1, 1]
    parent_anchor_ids: list[str]
    generator_llm_call_id: int | None
    validator_llm_call_id: int | None
    status: SynthesisStatus
    quality_reason: str | None
    retries_used: int

    @property
    def quality_status(self) -> QualityStatus:
        return "approved" if self.status == "approved" else "rejected"


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _strip_text(raw: str) -> str:
    text = raw.strip()
    # Strip code/quote fences first.
    for fence in FENCE_CHARS:
        if text.startswith(fence):
            text = text[len(fence) :].lstrip("\n")
        if text.endswith(fence):
            text = text[: -len(fence)].rstrip("\n")
    text = text.strip().strip('"').strip("'").strip()
    # Strip leading meta-prefixes the generator emits despite the prompt.
    # Apply iteratively in case multiple prefixes stack ("Okay, I
    # understand. **New Scenario:** ...").
    for _ in range(4):
        match = _META_PREFIX_RE.match(text)
        if not match:
            break
        text = text[match.end() :].strip()
    return text


def _parse_validator(reply: str) -> tuple[bool, str]:
    """Return `(approved, reason)`. Defaults to rejected on ambiguous output.

    Robust to markdown decoration: medgemma sometimes returns
    `**yes**: clinically realistic ...` instead of plain `yes: ...`.
    Strip `*` / `_` / `` ` `` from the verdict before checking the
    first word, but preserve the original `reply` for the reason.
    """
    text = reply.strip()
    if not text:
        return False, "validator returned empty response"
    # Strip markdown bold/italic/code from the head only — the reason
    # text after the colon is shown to operators verbatim, so we keep
    # its decoration intact.
    head_raw, _, tail = text.partition(":")
    head_clean = re.sub(r"[*_`#]+", "", head_raw).strip().lower()
    first_word = head_clean.split()[0] if head_clean else ""
    reason = tail.strip() if tail else text
    if first_word == "yes":
        return True, reason
    return False, reason


class KnnExemplarSynthesizer:
    """Generate, re-embed, gate. One target → one synthesizer run."""

    generation_method: GenerationMethod = "knn_exemplar"

    def __init__(
        self,
        *,
        generator_client: LLMClient,
        validator_client: LLMClient,
        embed_client: EmbedClient,
        generator_model: str,
        validator_model: str,
        library_anchors: Sequence[AnchorProbe],
        policy: Policy,
        embedding_model: str = "bio_clinicalbert",
        k: int = DEFAULT_K,
        tau: float = DEFAULT_TAU,
        max_retries: int = DEFAULT_MAX_RETRIES,
        generator_temperature: float = 0.7,
        validator_temperature: float = 0.0,
    ) -> None:
        if not library_anchors:
            raise ValueError("KnnExemplarSynthesizer: library_anchors empty")
        if k < 1:
            raise ValueError("k must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._generator = generator_client
        self._validator = validator_client
        self._embed = embed_client
        self._generator_model = generator_model
        self._validator_model = validator_model
        self._policy = policy
        self._anchors = list(library_anchors)
        self._embedding_model = embedding_model
        self._k = k
        self._tau = tau
        self._max_retries = max_retries
        self._gen_temperature = generator_temperature
        self._val_temperature = validator_temperature
        # Cache library-anchor embeddings on init so KNN search is in-memory.
        self._library_embeddings: dict[str, list[float]] = {}
        for anchor in self._anchors:
            resp = self._embed.embed(anchor.scenario, model=embedding_model)
            self._library_embeddings[anchor.id] = resp.embedding

    @property
    def k(self) -> int:
        return self._k

    @property
    def tau(self) -> float:
        return self._tau

    def synthesize(self, target_embedding: Sequence[float]) -> SynthesisResult:
        if len(target_embedding) == 0:
            raise ValueError("target_embedding must be non-empty")

        nearest_ids = self._k_nearest(target_embedding)
        exemplars = [a for a in self._anchors if a.id in nearest_ids]
        # Preserve KNN order (closest first).
        exemplars.sort(key=lambda a: nearest_ids.index(a.id))

        retry_nudge: str | None = None
        retries_used = 0
        last_text = ""
        last_achieved: list[float] = []
        last_tau = 0.0
        last_call_id: int | None = None

        for attempt in range(self._max_retries + 1):
            messages = synthesis_prompt(
                self._policy, exemplars, retry_nudge=retry_nudge
            )
            response = self._generator.chat_completion(
                messages,
                model=self._generator_model,
                temperature=self._gen_temperature,
            )
            text = _strip_text(response.content)
            if not text:
                retry_nudge = (
                    "Your previous attempt was empty. Output a single "
                    "1–2 sentence scenario as plain text."
                )
                retries_used = attempt
                continue
            if len(text) > MAX_SCENARIO_CHARS:
                text = text[:MAX_SCENARIO_CHARS].rstrip()
                # Length truncation is tolerated, not a tau-retry trigger.

            embed_response = self._embed.embed(text, model=self._embedding_model)
            achieved = embed_response.embedding
            tau_distance = _cosine_similarity(target_embedding, achieved)
            last_text = text
            last_achieved = achieved
            last_tau = tau_distance
            last_call_id = getattr(response, "llm_call_id", None)

            if tau_distance >= self._tau:
                validator_call_id, approved, reason = self._validate(text)
                if approved:
                    return SynthesisResult(
                        scenario=text,
                        target_embedding=list(target_embedding),
                        achieved_embedding=achieved,
                        tau_distance=tau_distance,
                        parent_anchor_ids=nearest_ids,
                        generator_llm_call_id=last_call_id,
                        validator_llm_call_id=validator_call_id,
                        status="approved",
                        quality_reason=reason,
                        retries_used=attempt,
                    )
                return SynthesisResult(
                    scenario=text,
                    target_embedding=list(target_embedding),
                    achieved_embedding=achieved,
                    tau_distance=tau_distance,
                    parent_anchor_ids=nearest_ids,
                    generator_llm_call_id=last_call_id,
                    validator_llm_call_id=validator_call_id,
                    status="rejected_validator",
                    quality_reason=reason,
                    retries_used=attempt,
                )
            retries_used = attempt
            retry_nudge = (
                f"Your previous attempt landed at cosine={tau_distance:.3f} "
                f"vs the target embedding. Lean further away from the "
                f"exemplars; emphasise a different clinical specifier "
                f"while keeping the scenario realistic."
            )

        status: SynthesisStatus = (
            "rejected_max_retries"
            if self._max_retries > 0
            else "rejected_below_tau"
        )
        return SynthesisResult(
            scenario=last_text,
            target_embedding=list(target_embedding),
            achieved_embedding=last_achieved,
            tau_distance=last_tau,
            parent_anchor_ids=nearest_ids,
            generator_llm_call_id=last_call_id,
            validator_llm_call_id=None,
            status=status,
            quality_reason=f"max_retries_below_tau (last cos={last_tau:.3f})",
            retries_used=retries_used,
        )

    # ---- internals ---------------------------------------------------------

    def _k_nearest(self, target: Sequence[float]) -> list[str]:
        target_arr = np.asarray(target, dtype=float)
        target_norm = float(np.linalg.norm(target_arr))
        if target_norm == 0:
            return [a.id for a in self._anchors[: self._k]]
        scored: list[tuple[str, float]] = []
        for anchor_id, vec in self._library_embeddings.items():
            v = np.asarray(vec, dtype=float)
            n = float(np.linalg.norm(v))
            if n == 0:
                continue
            cos = float(np.dot(target_arr, v) / (target_norm * n))
            scored.append((anchor_id, cos))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return [aid for aid, _cos in scored[: self._k]]

    def _validate(self, scenario: str) -> tuple[int | None, bool, str]:
        messages = validator_prompt(self._policy, scenario)
        response = self._validator.chat_completion(
            messages,
            model=self._validator_model,
            temperature=self._val_temperature,
        )
        approved, reason = _parse_validator(response.content)
        call_id = getattr(response, "llm_call_id", None)
        return call_id, approved, reason


def to_synthesized_probe(
    result: SynthesisResult,
    *,
    policy_id: str,
    gp_fit_id: int | None,
) -> SynthesizedProbe:
    """Convenience: shape a `SynthesisResult` into a persistable row."""
    return SynthesizedProbe(
        policy_id=policy_id,
        scenario=result.scenario,
        generation_method="knn_exemplar",
        target_embedding=result.target_embedding,
        achieved_embedding=result.achieved_embedding,
        tau_distance=result.tau_distance,
        parent_anchor_ids=result.parent_anchor_ids,
        synthesizer_llm_call_id=result.generator_llm_call_id,
        validator_llm_call_id=result.validator_llm_call_id,
        quality_status=result.quality_status,
        quality_reason=result.quality_reason,
        gp_fit_id=gp_fit_id,
    )


__all__ = [
    "DEFAULT_K",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_TAU",
    "KnnExemplarSynthesizer",
    "MAX_SCENARIO_CHARS",
    "SynthesisResult",
    "SynthesisStatus",
    "to_synthesized_probe",
]
