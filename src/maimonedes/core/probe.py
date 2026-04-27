"""Probe data model + YAML loader for the anchor library.

A `Probe` is anything that produces a `user` message for the
supervised system. The base class exists so Phase 2's perturbation
probes can inherit a common shape without changing how anchors are
loaded or scored.

Phase 1 only needs anchors. The eight v1 anchors come from
`docs/roadmap.md` (table "Probe library v1") and live in
`config/probes/anchors_v1.yaml`.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Probe(BaseModel):
    """Base type for any user-message generator (anchor or perturbation)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    policy_id: str = Field(min_length=1, default="scope_of_practice")


class AnchorProbe(Probe):
    """A fixed clinical scenario from the v1 probe library."""

    kind: Literal["anchor"] = "anchor"
    expected_baseline_compliance: float = Field(ge=0.0, le=1.0)


def load_anchors(path: str | Path) -> list[AnchorProbe]:
    """Load the anchor library from a YAML file."""
    raw = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict) or "anchors" not in parsed:
        raise ValueError(f"{path}: expected a top-level mapping with key `anchors`")
    items = parsed["anchors"]
    if not isinstance(items, list):
        raise ValueError(f"{path}: `anchors` must be a list")
    anchors = [AnchorProbe.model_validate(item) for item in items]
    _check_unique_ids(anchors)
    return anchors


def get_anchor_by_id(anchors: Iterable[AnchorProbe], anchor_id: str) -> AnchorProbe:
    for a in anchors:
        if a.id == anchor_id:
            return a
    raise KeyError(anchor_id)


def _check_unique_ids(anchors: list[AnchorProbe]) -> None:
    ids = [a.id for a in anchors]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate anchor ids: {duplicates}")


__all__ = ["AnchorProbe", "Probe", "get_anchor_by_id", "load_anchors"]
