"""Drift schedule primitives.

The drift study contaminates the supervised system's prompt with a
stage-by-stage suffix. Five stages, each contributing one accretive
sentence; the labels are fixed across the codebase so detector
wrappers, dashboard, and CLI can rely on a stable vocabulary.

The full `DriftSchedule` loader lives alongside the orchestrator
in #25; the Literal alone is exported here so the storage layer
(#22) can pin a stable column type without depending on YAML config.
"""
from __future__ import annotations

from typing import Literal, get_args


StageLabel = Literal["baseline", "concise", "actionable", "no_caveats", "trust"]


STAGE_LABELS: tuple[StageLabel, ...] = get_args(StageLabel)


__all__ = ["StageLabel", "STAGE_LABELS"]
