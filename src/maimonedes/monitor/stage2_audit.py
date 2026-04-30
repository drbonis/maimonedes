"""Phase 5 Stage-2 audit detector.

Hybrid trigger (per the locked decision in the planning thread):
fires whenever EITHER `n_scores` Stage-2 evaluations have accumulated
since the last audit OR `hours` of wall-clock time have elapsed,
whichever comes first. On audit, re-routes the same supervised
outputs through the Stage-1 judge and computes per-axis MAE +
Spearman ρ vs Stage-2's predictions to detect drift.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import select

from maimonedes.core.policy import Policy
from maimonedes.core.probe import AnchorProbe, get_anchor_by_id
from maimonedes.llm.client import LLMClient
from maimonedes.models.stage2 import (
    AxisMetrics,
    Stage2Model,
    _grade_agreement,
    _spearman_rho,
)
from maimonedes.scorer.judge import Judge
from maimonedes.storage.audit_runs import AuditRunRow
from maimonedes.storage.compliance import ComplianceScoreRow, row_to_score
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import get_session


log = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AuditReport:
    audit_run_id: int
    stage2_model_id: int
    n_samples: int
    metrics: list[AxisMetrics]
    agreement_status: str
    trigger_reason: str


@dataclass
class Stage2AuditDetector:
    """Hybrid (N-scores OR K-hours) trigger for Stage-2 audit."""

    stage2_model_id: int
    n_scores: int = 10
    hours: float = 24.0

    def last_audit_ended_at(self) -> datetime | None:
        with get_session() as session:
            row = session.execute(
                select(AuditRunRow)
                .where(AuditRunRow.stage2_model_id == self.stage2_model_id)
                .order_by(AuditRunRow.id.desc())
                .limit(1)
            ).scalar_one_or_none()
            if row is None:
                return None
            ended = row.ended_at or row.started_at
            if ended.tzinfo is None:
                ended = ended.replace(tzinfo=timezone.utc)
            return ended

    def scores_since_last(
        self, model_tag: str, last_ended_at: datetime | None
    ) -> list[ComplianceScoreRow]:
        with get_session() as session:
            stmt = (
                select(ComplianceScoreRow)
                .where(ComplianceScoreRow.judge_model == model_tag)
                .order_by(ComplianceScoreRow.id.asc())
            )
            if last_ended_at is not None:
                stmt = stmt.where(ComplianceScoreRow.scored_at > last_ended_at)
            rows = list(session.execute(stmt).scalars().all())
            for r in rows:
                session.expunge(r)
            return rows

    def should_audit(
        self,
        *,
        now: datetime,
        scores_since_last: int,
        last_ended_at: datetime | None,
    ) -> tuple[bool, str]:
        """Return `(should_audit, trigger_reason)`. First-ever audit is forced."""
        if last_ended_at is None:
            return True, "first_ever"
        if scores_since_last >= self.n_scores:
            return True, "n_scores"
        elapsed_h = (now - last_ended_at).total_seconds() / 3600.0
        if elapsed_h >= self.hours:
            return True, "hours_elapsed"
        return False, "below_thresholds"

    def run_audit(
        self,
        *,
        stage2_model: Stage2Model,
        policy: Policy,
        anchors: Iterable[AnchorProbe],
        judge_client: LLMClient,
        judge_model: str,
        supervised_model: str,
        score_rows: list[ComplianceScoreRow],
        trigger_reason: str,
    ) -> AuditReport:
        """Re-route each Stage-2 row through Stage-1; compute agreement metrics."""
        anchors_list = list(anchors)
        anchor_by_id = {a.id: a for a in anchors_list}
        judge = Judge(
            judge_client, model=judge_model, supervised_model=supervised_model
        )

        per_axis_truth: dict[str, list[float]] = {
            s.id: [] for s in policy.rubric.sub_conditions
        }
        per_axis_pred: dict[str, list[float]] = {
            s.id: [] for s in policy.rubric.sub_conditions
        }

        with get_session() as session:
            audit_row = AuditRunRow(
                stage2_model_id=self.stage2_model_id,
                n_samples=0,
                mae_per_axis_json="{}",
                spearman_per_axis_json="{}",
                agreement_status="pending",
                trigger_reason=trigger_reason,
            )
            session.add(audit_row)
            session.flush()
            audit_run_id = audit_row.id

        n_used = 0
        for row in score_rows:
            if row.llm_call_id is None:
                continue
            anchor = anchor_by_id.get(row.anchor_id)
            if anchor is None:
                continue
            with get_session() as session:
                llm = session.get(LLMCall, row.llm_call_id)
                if llm is None:
                    continue
                text = llm.response_content

            try:
                truth = judge.score(policy, anchor, text)
            except Exception as exc:
                log.warning(
                    "stage2_audit.judge_failed",
                    extra={
                        "anchor_id": anchor.id,
                        "score_id": row.id,
                        "error": str(exc),
                    },
                )
                continue

            stage2_pred = row_to_score(row)
            for sub in policy.rubric.sub_conditions:
                t = float(truth.per_sub_condition.get(sub.id, 0.0))
                p = float(stage2_pred.per_sub_condition.get(sub.id, 0.0))
                per_axis_truth[sub.id].append(t)
                per_axis_pred[sub.id].append(p)
            n_used += 1

        metrics: list[AxisMetrics] = []
        for sub in policy.rubric.sub_conditions:
            ys = per_axis_truth[sub.id]
            ps = per_axis_pred[sub.id]
            if not ys:
                metrics.append(AxisMetrics(sub_id=sub.id, mae=float("nan"), spearman_rho=float("nan")))
                continue
            mae = float(np.mean(np.abs(np.asarray(ys) - np.asarray(ps))))
            rho = _spearman_rho(ys, ps)
            metrics.append(AxisMetrics(sub_id=sub.id, mae=mae, spearman_rho=rho))

        status = _grade_agreement(metrics) if n_used > 0 else "red"

        with get_session() as session:
            audit_row = session.get(AuditRunRow, audit_run_id)
            assert audit_row is not None
            audit_row.n_samples = n_used
            audit_row.mae_per_axis_json = json.dumps(
                {m.sub_id: m.mae for m in metrics}, sort_keys=True
            )
            audit_row.spearman_per_axis_json = json.dumps(
                {m.sub_id: m.spearman_rho for m in metrics},
                sort_keys=True,
            )
            audit_row.agreement_status = status
            audit_row.ended_at = _utcnow()

        previous = stage2_model.agreement_status
        if previous in {"green"} and status in {"amber", "red"}:
            log.warning(
                "stage2_audit.agreement_dropped",
                extra={
                    "stage2_model_id": self.stage2_model_id,
                    "previous": previous,
                    "current": status,
                },
            )
        return AuditReport(
            audit_run_id=audit_run_id,
            stage2_model_id=self.stage2_model_id,
            n_samples=n_used,
            metrics=metrics,
            agreement_status=status,
            trigger_reason=trigger_reason,
        )


def get_anchor_for_audit(
    anchors: Iterable[AnchorProbe], anchor_id: str
) -> AnchorProbe | None:
    """Convenience used by tests + CLI; mirrors `core.probe.get_anchor_by_id`."""
    try:
        return get_anchor_by_id(list(anchors), anchor_id)
    except KeyError:
        return None


__all__ = [
    "AuditReport",
    "Stage2AuditDetector",
    "get_anchor_for_audit",
]
