"""Policy + Rubric data model.

Policies (clinical scope, epistemic calibration, demographic invariance,
...) are versioned YAML files; each one declares prose policy text plus
a rubric of 5–7 sub-conditions the judge scores. Phase 1 wires up the
scope-of-practice policy.

Sub-condition scales:
- "boolean": judge emits true/false  → 1.0 / 0.0
- "0-3":     judge emits int 0..3    → val / 3.0
- "labels":  judge picks one label_id → label.value (BARS edition; the
            label set is sub-condition-specific and each label carries
            a behavior-anchored description plus a normalised value
            in [0, 1]). See `docs/blackbox_supervision_architecture.md`
            §5.2 for context.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


Scale = Literal["boolean", "0-3", "labels"]
WEIGHT_SUM_TOLERANCE = 1e-6


class Label(BaseModel):
    """One BARS rating step for a labels-scale sub-condition.

    `value` is the normalised score the framework records when the
    judge picks this label; `description` is the behavior anchor the
    judge sees in the prompt.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    value: float = Field(ge=0.0, le=1.0)
    description: str = Field(min_length=1)


class SubCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    scale: Scale
    weight: float = Field(ge=0.0, le=1.0)
    # Required when `scale == "labels"`; must be omitted otherwise.
    labels: list[Label] | None = None

    @model_validator(mode="after")
    def _check_scale_consistency(self) -> "SubCondition":
        if self.scale == "labels":
            if not self.labels or len(self.labels) < 2:
                raise ValueError(
                    f"sub-condition {self.id!r}: scale='labels' requires at "
                    f"least 2 labels"
                )
            ids = [label.id for label in self.labels]
            if len(set(ids)) != len(ids):
                duplicates = sorted({i for i in ids if ids.count(i) > 1})
                raise ValueError(
                    f"sub-condition {self.id!r}: duplicate label ids: {duplicates}"
                )
        elif self.labels is not None:
            raise ValueError(
                f"sub-condition {self.id!r}: `labels` field is only valid "
                f"when scale='labels'"
            )
        return self

    def label_ids(self) -> list[str]:
        """Return the list of label ids in YAML order. Labels-scale only."""
        if self.labels is None:
            raise ValueError(
                f"sub-condition {self.id!r}: scale={self.scale!r} has no labels"
            )
        return [label.id for label in self.labels]

    def value_for_label(self, label_id: str) -> float:
        """Return the normalised value for `label_id`. Labels-scale only.

        Raises `KeyError` if `label_id` isn't part of this sub-condition.
        """
        if self.labels is None:
            raise ValueError(
                f"sub-condition {self.id!r}: scale={self.scale!r} has no labels"
            )
        for label in self.labels:
            if label.id == label_id:
                return label.value
        raise KeyError(label_id)


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sub_conditions: list[SubCondition]

    @model_validator(mode="after")
    def _check_invariants(self) -> "Rubric":
        if not (5 <= len(self.sub_conditions) <= 7):
            raise ValueError(
                f"rubric must have 5–7 sub-conditions, got {len(self.sub_conditions)}"
            )
        ids = [s.id for s in self.sub_conditions]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate sub-condition ids: {duplicates}")
        total = sum(s.weight for s in self.sub_conditions)
        if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(
                f"sub-condition weights must sum to 1.0; got {total:.6f}"
            )
        return self

    def by_id(self, sub_id: str) -> SubCondition:
        for s in self.sub_conditions:
            if s.id == sub_id:
                return s
        raise KeyError(sub_id)


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    text: str = Field(min_length=1)
    rubric: Rubric


def load_policy(policy_path: str | Path, rubric_path: str | Path | None = None) -> Policy:
    """Load a policy from YAML.

    The rubric can either be inlined under `rubric:` in the policy file
    or kept in a separate file referenced by `rubric_path`. Splitting
    makes the rubric independently editable by clinicians.
    """
    policy_data = _load_yaml(policy_path)
    if rubric_path is not None:
        rubric_data = _load_yaml(rubric_path)
        policy_data["rubric"] = rubric_data
    return Policy.model_validate(policy_data)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(parsed).__name__}")
    return parsed


__all__ = [
    "Label",
    "Policy",
    "Rubric",
    "Scale",
    "SubCondition",
    "load_policy",
]
