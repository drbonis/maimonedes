"""Multi-page dashboard entry point.

Streamlit auto-mounts every `.py` file under `dashboard/pages/` as a
sidebar entry. This module is the landing page; the actual content
lives in:

- pages/01_compliance_scores.py  — Phase 1 per-anchor score table
- pages/02_fragility.py           — Phase 2 Jacobian + fragility table
- pages/03_drift.py               — Phase 3 drift timeline + detector
- pages/04_recovery.py            — Phase 4 before/after + feedback
- pages/05_gp.py                  — Phase 5 GP uncertainty surface + targets
- pages/06_metric.py              — Phase 5 Riemannian metric explorer
- pages/07_curvature.py           — Phase 5 curvature alarm (κ change between fits)
- pages/08_decoupling.py          — Phase 5 decoupling alarm (per-anchor covariance shift)

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
        "aggregated fragility table across the eight anchors.\n"
        "- **Drift — Phase 3**: timeline plot of raw aggregate, EWMA, "
        "and CUSUM across a synthetic drift run with detection-latency "
        "annotations.\n"
        "- **Recovery — Phase 4**: before/after compliance bars plus the "
        "synthesized feedback text rendered as a quote block, one panel "
        "per affected anchor.\n"
        "- **GP — Phase 5**: PCA-projected scatter of training embeddings + "
        "library anchors + proposed targets, ranked by uncertainty × "
        "boundary risk.\n"
        "- **Metric — Phase 5**: Riemannian metric tensor field as a 2D "
        "compliance-plane slice with √det(g) heatmap, metric ellipses, "
        "anchor overlay, and a Euclidean-vs-Riemannian distance comparator "
        "with §4.6 Example 1 / 3 / 5 reproductions.\n"
        "- **Curvature — Phase 5**: per-anchor κ comparison between two "
        "metric_fits, eigenvalue-spectrum bars, ellipse-pair view, and the "
        "persisted `structural_signals` curvature alarm feed.\n"
        "- **Decoupling — Phase 5**: per-anchor covariance shift — baseline "
        "vs current k×k covariance heatmaps with sign-flip cells "
        "highlighted, axis-pair scatter rotation with PC1 lines, and the "
        "persisted decoupling alarm feed."
    )


_render()
