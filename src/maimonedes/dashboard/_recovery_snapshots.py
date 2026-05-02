"""Picklable snapshot types for the recovery page's `st.cache_data` callers.

Same caching gotcha as `_drift_snapshots`: page filenames under
`dashboard/pages/` start with digits, so classes defined inside the
page module aren't picklable through `st.cache_data`. Define them
here instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RecoveryRunSnapshot:
    """Picklable summary used by the recovery-run dropdown."""

    recovery_run_id: int
    parent_drift_run_id: int
    started_at: datetime | None
    notes: str | None
    contrastive_kind: str
    label: str
    contamination_mode: str = "clean"
    contamination_stage_label: str | None = None


@dataclass
class AnchorBarsSnapshot:
    """Per-anchor bar trio: baseline → drift-low → recovery."""

    anchor_id: str
    baseline: float | None
    drift_low: float | None
    recovery: float | None
    feedback_text: str | None
    safe_text: str | None
    near_boundary_text: str | None


__all__ = ["AnchorBarsSnapshot", "RecoveryRunSnapshot"]
