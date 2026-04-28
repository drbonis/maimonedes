"""Multi-page dashboard entry point.

Streamlit auto-mounts every `.py` file under `dashboard/pages/` as a
sidebar entry. This module is the landing page; the actual content
lives in:

- pages/01_compliance_scores.py  — Phase 1 per-anchor score table
- pages/02_fragility.py           — Phase 2 Jacobian + fragility table

Run with: `streamlit run src/maimonedes/dashboard/app.py`
"""
from __future__ import annotations

import streamlit as st

PAGE_TITLE = "maimonedes dashboard"


def _render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.markdown(
        "Black-box behavioral supervision for clinical-decision LLMs.\n\n"
        "Use the sidebar to navigate:\n\n"
        "- **Compliance scores — Phase 1**: latest aggregate per anchor, "
        "per-sub-condition detail, recent-aggregate history.\n"
        "- **Fragility — Phase 2**: per-anchor Jacobian heatmap and "
        "aggregated fragility table across the eight anchors."
    )


_render()
