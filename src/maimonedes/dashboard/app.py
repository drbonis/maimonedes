"""Phase 1 dashboard.

A single Streamlit page that shows the most-recent compliance score
per anchor, plus per-sub-condition detail and a small history chart.

Run with: `streamlit run src/maimonedes/dashboard/app.py`
"""
from __future__ import annotations

import streamlit as st

from maimonedes.storage.compliance import latest_score_per_anchor
from maimonedes.storage.dashboard_queries import (
    history_for_anchor,
    latest_table_rows,
)

PAGE_TITLE = "Compliance scores — Phase 1"


@st.cache_data(ttl=10)
def _cached_latest_rows() -> list[dict[str, object]]:
    return latest_table_rows()


def _render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    rows = _cached_latest_rows()
    if not rows:
        st.info(
            "No compliance scores recorded yet. Run "
            "`maimonedes run-once <anchor-id>` to populate this dashboard."
        )
        return

    st.subheader("Latest score per anchor")
    st.dataframe(rows, use_container_width=True, hide_index=True)

    latest = latest_score_per_anchor()
    st.subheader("Per-anchor detail")
    for anchor_id in sorted(latest.keys()):
        score = latest[anchor_id]
        with st.expander(
            f"{anchor_id} — aggregate {score.aggregate:.3f} "
            f"(policy {score.policy_id}, judge {score.judge_model})"
        ):
            st.write("**Per-sub-condition**")
            for sub_id in sorted(score.per_sub_condition):
                value = score.per_sub_condition[sub_id]
                st.write(f"- `{sub_id}`: {value:.3f}")

            history = history_for_anchor(anchor_id, limit=20)
            if len(history) > 1:
                st.write("**Recent aggregate history**")
                chart_values = [s.aggregate for s in history]
                st.line_chart(chart_values)


_render()
