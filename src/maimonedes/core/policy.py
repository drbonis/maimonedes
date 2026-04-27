"""Policy + Rubric data model.

Policies (clinical scope, epistemic calibration, demographic invariance,
...) are versioned YAML files; each one declares prose policy text plus
a rubric of 5–7 sub-conditions the judge scores. Phase 1 wires up the
scope-of-practice policy. The §5.2 rubric in
`docs/blackbox_supervision_architecture.md` is the seed.

Sub-condition scales:
- "boolean": judge emits true/false → 1.0 / 0.0 in code
- "0-3": judge emits int 0..3 → val / 3.0 in code
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


Scale = Literal["boolean", "0-3"]
WEIGHT_SUM_TOLERANCE = 1e-6


class SubCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    scale: Scale
    weight: float = Field(ge=0.0, le=1.0)


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


__all__ = ["Policy", "Rubric", "Scale", "SubCondition", "load_policy"]
