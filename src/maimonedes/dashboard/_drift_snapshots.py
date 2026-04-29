"""Picklable snapshot types for the drift page's `st.cache_data` callers.

Streamlit pages live under `dashboard/pages/` with filenames that start
with a digit (e.g. `03_drift.py`). The synthesized module name is not a
valid Python identifier, so `pickle.dumps` cannot resolve a class
defined inside that page back through `import`. `st.cache_data` uses
pickle, so any custom class returned by a cached function in a page
must be defined in a stably-importable module — like this one.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class RunSnapshot:
    """Picklable summary used by `@st.cache_data` keys + dropdowns."""

    run_id: int
    started_at: datetime | None
    notes: str | None
    label: str


@dataclass
class AnchorTrace:
    """All data needed to render one anchor's plot, packaged for caching."""

    anchor_id: str
    session_indices: list[int]
    stage_labels: list[str]
    aggregates: list[float]
    cusum_values: list[float]
    cusum_threshold: float
    cusum_first_fire: int | None
    ewma_values: list[float]
    ewma_lcls: list[float]
    ewma_asymptotic_lcl: float
    ewma_first_fire: int | None
    first_violation: int | None
    detector_skipped: bool


__all__ = ["AnchorTrace", "RunSnapshot"]
