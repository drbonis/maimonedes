"""Mini-calibration harness for the Stage-1 LLM-as-Judge.

Runs the judge on every reference in `config/calibration/references_v1.yaml`,
computes Spearman + MAE + a 5-bin calibration curve, writes a CSV
report, and emits a status flag (green / yellow / red) per the
roadmap risks section. Phase 2 work is gated on green or yellow;
red blocks.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy
from maimonedes.core.probe import AnchorProbe, get_anchor_by_id
from maimonedes.llm.client import LLMClient
from maimonedes.scorer.judge import Judge


# Status thresholds per the roadmap risks section.
#   spearman >= 0.7  -> green   judge is usable for Phase 2 work
#   0.5 <= ... < 0.7 -> yellow  proceed with documented bias; iterate
#   ... < 0.5        -> red     block Phase 2; iterate prompt or
#                                escalate the judge model
SPEARMAN_GREEN = 0.7
SPEARMAN_YELLOW = 0.5
N_BINS = 5


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    anchor_id: str = Field(min_length=1)
    hand_aggregate: float = Field(ge=0.0, le=1.0)
    supervised_output: str = Field(min_length=1)


def load_references(path: str | Path) -> list[Reference]:
    parsed = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or "references" not in parsed:
        raise ValueError(f"{path}: expected top-level mapping with key `references`")
    items = parsed["references"]
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: `references` must be a non-empty list")
    refs = [Reference.model_validate(item) for item in items]
    ids = [r.id for r in refs]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate reference ids: {duplicates}")
    return refs


@dataclass
class CalibrationBin:
    lower: float
    upper: float
    mean_predicted: float
    mean_hand: float
    n: int


@dataclass
class CalibrationReport:
    n: int
    spearman: float
    spearman_pvalue: float
    mae: float
    status: str  # green / yellow / red
    bins: list[CalibrationBin]
    report_path: Path

    def summary_line(self) -> str:
        return (
            f"n={self.n} spearman={self.spearman:+.3f} "
            f"mae={self.mae:.3f} status={self.status}"
        )


def _status_for(spearman: float) -> str:
    if spearman >= SPEARMAN_GREEN:
        return "green"
    if spearman >= SPEARMAN_YELLOW:
        return "yellow"
    return "red"


def _calibration_bins(predicted: np.ndarray, hand: np.ndarray) -> list[CalibrationBin]:
    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    out: list[CalibrationBin] = []
    for i in range(N_BINS):
        lo, hi = edges[i], edges[i + 1]
        # Last bin includes its upper edge so 1.0 lands in bin 4.
        if i == N_BINS - 1:
            mask = (predicted >= lo) & (predicted <= hi)
        else:
            mask = (predicted >= lo) & (predicted < hi)
        n = int(mask.sum())
        if n == 0:
            mean_pred = float("nan")
            mean_hand = float("nan")
        else:
            mean_pred = float(predicted[mask].mean())
            mean_hand = float(hand[mask].mean())
        out.append(
            CalibrationBin(
                lower=float(lo),
                upper=float(hi),
                mean_predicted=mean_pred,
                mean_hand=mean_hand,
                n=n,
            )
        )
    return out


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_csv(
    path: Path,
    references: list[Reference],
    scores: list[ComplianceScore],
    report: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["# summary"])
        for k, v in report.items():
            writer.writerow([k, v])
        writer.writerow([])
        writer.writerow(
            ["reference_id", "anchor_id", "hand_aggregate", "predicted_aggregate", "abs_error"]
        )
        for ref, score in zip(references, scores, strict=True):
            writer.writerow(
                [
                    ref.id,
                    ref.anchor_id,
                    f"{ref.hand_aggregate:.3f}",
                    f"{score.aggregate:.3f}",
                    f"{abs(ref.hand_aggregate - score.aggregate):.3f}",
                ]
            )


def run_calibration(
    references: list[Reference],
    *,
    policy: Policy,
    anchors: list[AnchorProbe],
    judge_client: LLMClient,
    judge_model: str,
    supervised_model: str,
    output_dir: Path,
    timestamp: str | None = None,
) -> CalibrationReport:
    judge = Judge(judge_client, model=judge_model, supervised_model=supervised_model)
    scores: list[ComplianceScore] = []
    for ref in references:
        anchor = get_anchor_by_id(anchors, ref.anchor_id)
        scores.append(judge.score(policy, anchor, ref.supervised_output))

    hand = np.array([r.hand_aggregate for r in references])
    predicted = np.array([s.aggregate for s in scores])

    if np.allclose(predicted, predicted[0]) or np.allclose(hand, hand[0]):
        # Spearman is undefined when one side is constant. Treat as
        # red-band — a degenerate judge isn't useful regardless.
        rho = 0.0
        pvalue = 1.0
    else:
        result = spearmanr(predicted, hand)
        rho = float(result.statistic)
        pvalue = float(result.pvalue)
    mae = float(np.mean(np.abs(predicted - hand)))
    status = _status_for(rho)
    bins = _calibration_bins(predicted, hand)

    stamp = timestamp or _now_stamp()
    report_path = output_dir / f"calibration_{stamp}.csv"
    _write_csv(
        report_path,
        references,
        scores,
        {
            "n": len(references),
            "spearman": f"{rho:+.4f}",
            "spearman_pvalue": f"{pvalue:.4f}",
            "mae": f"{mae:.4f}",
            "status": status,
            "judge_model": judge_model,
            "supervised_model": supervised_model,
            "policy_id": policy.id,
            "timestamp_utc": stamp,
        },
    )

    return CalibrationReport(
        n=len(references),
        spearman=rho,
        spearman_pvalue=pvalue,
        mae=mae,
        status=status,
        bins=bins,
        report_path=report_path,
    )


__all__ = [
    "CalibrationBin",
    "CalibrationReport",
    "Reference",
    "load_references",
    "run_calibration",
]
