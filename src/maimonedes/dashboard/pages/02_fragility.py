"""Phase 2 page: Jacobian heatmap + aggregated fragility table.

Per the §4.4 prediction in `docs/blackbox_supervision_architecture.md`,
the (authority, *) and (boundary, *) rows of the aggregated table
should be visibly redder than (demographic, *). This page is where
that prediction is read against the empirical data.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from maimonedes.monitor.fragility import (
    aggregated_fragility,
    all_jacobians,
)

PAGE_TITLE = "Fragility — Phase 2"
EMPTY_STATE_MSG = (
    "No perturbation data yet. Run `maimonedes perturb <anchor-id>` "
    "(or `--all-anchors`) and then `maimonedes run-once` for at least "
    "one anchor so the Jacobian has a baseline to subtract from."
)


@st.cache_data(ttl=10)
def _cached_jacobians() -> dict[str, dict[str, object]]:
    """Return jacobians as plain dicts (cache_data needs picklable values)."""
    return {
        anchor_id: {
            "baseline_aggregate": jac.baseline_aggregate,
            "columns": list(jac.columns),
            "rows": [
                {
                    "transform_label": r.transform_label,
                    "perturbation_kind": r.perturbation_kind,
                    "deltas": dict(r.deltas),
                }
                for r in jac.rows
            ],
        }
        for anchor_id, jac in all_jacobians().items()
    }


@st.cache_data(ttl=10)
def _cached_aggregated() -> dict[str, object]:
    table = aggregated_fragility()
    return {
        "perturbation_kinds": list(table.perturbation_kinds),
        "columns": list(table.columns),
        "cells": [
            {
                "perturbation_kind": c.perturbation_kind,
                "column": c.column,
                "mean_delta": c.mean_delta,
                "count": c.count,
            }
            for c in table.cells
        ],
    }


def _jacobian_dataframe(jac: dict[str, object]) -> pd.DataFrame:
    columns = jac["columns"]  # type: ignore[assignment]
    rows = jac["rows"]  # type: ignore[assignment]
    if not rows:
        return pd.DataFrame(columns=list(columns))  # type: ignore[arg-type]
    data = []
    index = []
    for r in rows:
        index.append(r["transform_label"])
        data.append([r["deltas"].get(col, 0.0) for col in columns])  # type: ignore[union-attr]
    df = pd.DataFrame(data, index=index, columns=list(columns))  # type: ignore[arg-type]
    df.index.name = "transform_label"
    # Defense-in-depth: pandas Styler refuses to render with duplicate
    # indexes. The orchestrator should already have deduped per
    # transform_label, but if a stale upstream layer ever sneaks dupes
    # past, keep the first occurrence rather than crashing the page.
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="first")]
    return df


def _aggregated_dataframe(table: dict[str, object]) -> pd.DataFrame:
    kinds = list(table["perturbation_kinds"])  # type: ignore[arg-type]
    columns = list(table["columns"])  # type: ignore[arg-type]
    grid = {(c["perturbation_kind"], c["column"]): c["mean_delta"] for c in table["cells"]}  # type: ignore[union-attr]
    if not kinds:
        return pd.DataFrame(columns=columns)
    data = [
        [grid.get((kind, col)) for col in columns] for kind in kinds
    ]
    df = pd.DataFrame(data, index=kinds, columns=columns)
    df.index.name = "perturbation_kind"
    return df


def _delta_color(val: object) -> str:
    """Hand-rolled red/blue gradient (avoids the matplotlib dep).

    Δ < 0 → red (compliance loss); Δ > 0 → blue (compliance gain);
    Δ ≈ 0 / NaN → no fill.
    """
    if val is None or (isinstance(val, float) and val != val):  # NaN
        return ""
    if not isinstance(val, (int, float)):
        return ""
    if abs(val) < 1e-3:
        return ""
    intensity = max(0.0, min(1.0, abs(float(val))))
    if val < 0:
        return (
            f"background-color: rgba(220, 80, 80, {0.15 + 0.55 * intensity:.3f})"
        )
    return (
        f"background-color: rgba(80, 120, 220, {0.15 + 0.55 * intensity:.3f})"
    )


def _heatmap(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Red = compliance loss (Δ < 0), blue = compliance gain (Δ > 0)."""
    return df.style.format("{:+.3f}", na_rep="—").map(_delta_color)


def _render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    jacobians = _cached_jacobians()
    aggregated = _cached_aggregated()

    if not jacobians and not aggregated["cells"]:  # type: ignore[index]
        st.info(EMPTY_STATE_MSG)
        return

    st.subheader("Aggregated fragility (mean Δ across anchors)")
    if aggregated["cells"]:  # type: ignore[index]
        agg_df = _aggregated_dataframe(aggregated)
        st.dataframe(_heatmap(agg_df), use_container_width=True)
    else:
        st.write("(no aggregated data yet)")

    st.subheader("Per-anchor Jacobian")
    if not jacobians:
        st.write("(no anchor has both a baseline and perturbations yet)")
        return

    anchor_id = st.selectbox(
        "Anchor",
        options=sorted(jacobians.keys()),
        index=0,
    )
    jac = jacobians[anchor_id]
    st.caption(
        f"Baseline aggregate for **{anchor_id}**: "
        f"{jac['baseline_aggregate']:.3f}"  # type: ignore[index,str-format]
    )
    df = _jacobian_dataframe(jac)
    if df.empty:
        st.write("(no perturbations recorded for this anchor)")
        return
    st.dataframe(_heatmap(df), use_container_width=True)


_render()
