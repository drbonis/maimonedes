"""Phase 5 Stage-2 online scorer.

Loads a trained `Stage2Model` (per-axis Ridge heads on
Bio_ClinicalBERT embeddings) and scores supervised outputs at near-
zero cost: one HTTP embed call + a microsecond-scale linear forward
pass per axis. Aggregation matches `Judge._aggregate` so Stage-1
and Stage-2 outputs are directly comparable.

`judge_model` on the resulting `ComplianceScore` is `"stage2:<path>"`
so downstream queries can distinguish Stage-1 vs Stage-2 rows.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy
from maimonedes.core.probe import AnchorProbe
from maimonedes.llm.embed_client import EmbedClient
from maimonedes.models.stage2 import Stage2Model


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Stage2Scorer:
    """Stage-2 inference: embed once, predict per axis, aggregate."""

    def __init__(
        self,
        model: Stage2Model,
        embed_client: EmbedClient,
        policy: Policy,
        *,
        model_path: str | None = None,
    ) -> None:
        if model.policy_id != policy.id:
            raise ValueError(
                f"Stage2Scorer: model policy {model.policy_id!r} does not "
                f"match runtime policy {policy.id!r}"
            )
        self._model = model
        self._embed = embed_client
        self._policy = policy
        # Tag rows with the path so a downstream query can recover which
        # model produced them, even after the in-memory artefact is gone.
        self._tag = f"stage2:{Path(model_path).name}" if model_path else "stage2"

    @property
    def model(self) -> Stage2Model:
        return self._model

    @property
    def tag(self) -> str:
        return self._tag

    def score(
        self,
        anchor: AnchorProbe,
        supervised_output: str,
        *,
        supervised_model: str = "stage2-input",
        llm_call_id: int | None = None,
    ) -> ComplianceScore:
        """Embed → predict → return a ComplianceScore.

        `llm_call_id` lets callers thread the supervised LLMCall FK
        through so audit-stage2 can later recover the source text by
        joining back to llm_calls. Pre-#56 callers omit it; the
        resulting score row will have a NULL FK and will be skipped by
        the audit (which has no other way to recover the supervised
        text).
        """
        if anchor.policy_id != self._policy.id:
            raise ValueError(
                f"Stage2Scorer: anchor {anchor.id!r} declares policy "
                f"{anchor.policy_id!r}; expected {self._policy.id!r}"
            )
        response = self._embed.embed(
            supervised_output, model=self._model.embedding_model
        )
        aggregate, per_sub = self._model.predict_aggregate(
            response.embedding, policy=self._policy
        )
        return ComplianceScore(
            anchor_id=anchor.id,
            policy_id=self._policy.id,
            per_sub_condition=per_sub,
            aggregate=aggregate,
            judge_model=self._tag,
            supervised_model=supervised_model,
            scored_at=_utcnow(),
            llm_call_id=llm_call_id,
        )


__all__ = ["Stage2Scorer"]
