"""Drift schedule primitives.

The drift study contaminates the supervised system's prompt with a
stage-by-stage suffix. Five stages, each contributing one accretive
sentence; the labels are fixed across the codebase so detector
wrappers, dashboard, and CLI can rely on a stable vocabulary.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import yaml


StageLabel = Literal["baseline", "concise", "actionable", "no_caveats", "trust"]


STAGE_LABELS: tuple[StageLabel, ...] = get_args(StageLabel)


@dataclass(frozen=True)
class DriftStage:
    label: StageLabel
    sessions: int
    suffix: str


@dataclass(frozen=True)
class DriftSchedule:
    """Ordered list of stages, each running for N consecutive sessions.

    `iter_sessions()` flattens the schedule into one tuple per
    `session_index`, in stage order. A 5-stage × 10-session schedule
    yields 50 items.
    """

    stages: tuple[DriftStage, ...]

    @property
    def total_sessions(self) -> int:
        return sum(stage.sessions for stage in self.stages)

    def iter_sessions(self) -> Iterator[tuple[int, StageLabel, str]]:
        idx = 0
        for stage in self.stages:
            for _ in range(stage.sessions):
                yield idx, stage.label, stage.suffix
                idx += 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DriftSchedule":
        raw = Path(path).read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, dict) or "stages" not in parsed:
            raise ValueError(
                f"{path}: expected top-level mapping with key `stages`"
            )
        items = parsed["stages"]
        if not isinstance(items, list) or not items:
            raise ValueError(f"{path}: `stages` must be a non-empty list")
        stages: list[DriftStage] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: stage[{i}] must be a mapping")
            label = item.get("label")
            if label not in STAGE_LABELS:
                raise ValueError(
                    f"{path}: stage[{i}] label {label!r} not in {STAGE_LABELS}"
                )
            sessions = item.get("sessions")
            if not isinstance(sessions, int) or sessions <= 0:
                raise ValueError(
                    f"{path}: stage[{i}] sessions must be a positive int"
                )
            suffix = item.get("suffix", "")
            if not isinstance(suffix, str):
                raise ValueError(f"{path}: stage[{i}] suffix must be a string")
            stages.append(DriftStage(label=label, sessions=sessions, suffix=suffix))
        return cls(stages=tuple(stages))


__all__ = [
    "DriftSchedule",
    "DriftStage",
    "STAGE_LABELS",
    "StageLabel",
]
