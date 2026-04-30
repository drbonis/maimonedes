"""Phase 5 page: GP uncertainty surface + proposed-target visualization.

Shows the embedding-space distribution of training data, the GP's
proposed targets overlaid, and a side panel with the ranked targets.

The 2D projection prefers UMAP (better local-neighborhood
preservation, exposes clinical-domain clusters and
compliance-gradient sub-structure that PCA washes out). Falls back
to PCA when UMAP is unavailable or the point count is too small for
UMAP's `n_neighbors=15` default. The PCA-only modeling layer in
`monitor.gp_layer` is unaffected — that one needs out-of-sample
transform + inverse transform, which UMAP can't reliably provide.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from maimonedes.dashboard._gp_snapshots import (
    GPFitSnapshot,
    GPScatterPoint,
    GPSnapshot,
)
from maimonedes.monitor.gp_layer import ComplianceGP, propose_targets
from maimonedes.storage.gp_fits import get_gp_fit, list_gp_fits


PAGE_TITLE = "GP — Phase 5"
EMPTY_STATE_MSG = (
    "No GP fits recorded yet. Run `maimonedes fit-gp` to populate this dashboard."
)
UMAP_N_NEIGHBORS = 15  # UMAP requires at least n_neighbors+1 points to fit


def _project_2d(combined: np.ndarray) -> tuple[np.ndarray, str]:
    """Project the combined embedding stack to 2D for the scatter plot.

    Tries UMAP first (preserves local neighborhood structure on
    Bio_ClinicalBERT embeddings far better than PCA's variance-axis
    projection). Falls back to PCA when:
    - `umap-learn` is not installed in the environment, or
    - the combined matrix has too few points for UMAP's default
      `n_neighbors=15` neighborhood.

    Returns `(coords_n_x_2, method_label)` where `method_label` is
    `"UMAP"` or `"PCA"` and gets shown in the page caption so the
    operator knows which projection produced the scatter.
    """
    n_points = combined.shape[0]
    use_umap = n_points >= UMAP_N_NEIGHBORS + 1
    if use_umap:
        try:
            import umap  # noqa: PLC0415 — gated import for optional dep

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=0.1,
                random_state=0,
            )
            coords = reducer.fit_transform(combined)
            return coords, "UMAP"
        except ImportError:
            pass
    n_components = min(2, combined.shape[1], max(1, n_points - 1))
    pca = PCA(n_components=2 if n_components >= 2 else 1)
    coords = pca.fit_transform(combined)
    if coords.shape[1] == 1:
        # Pad so plotting code can rely on (x, y).
        coords = np.hstack([coords, np.zeros_like(coords)])
    return coords, "PCA"


@st.cache_data(ttl=10)
def _cached_fits() -> list[GPFitSnapshot]:
    snapshots: list[GPFitSnapshot] = []
    for row in list_gp_fits():
        trained = row.trained_at
        ts_str = (
            trained.strftime("%Y-%m-%d %H:%M")
            if isinstance(trained, datetime)
            else "?"
        )
        snapshots.append(
            GPFitSnapshot(
                fit_id=row.id,
                path=row.path,
                policy_id=row.policy_id,
                n_samples=row.n_samples,
                log_marginal_likelihood=float(row.log_marginal_likelihood),
                kernel_name=row.kernel_name,
                embedding_model=row.embedding_model,
                trained_at=trained if isinstance(trained, datetime) else None,
                label=(
                    f"fit #{row.id}  ·  {ts_str}  ·  n={row.n_samples}  ·  "
                    f"lml={row.log_marginal_likelihood:.2f}"
                ),
            )
        )
    return snapshots


def _nearest_library_anchor(
    target: list[float], library: dict[str, list[float]]
) -> tuple[str, float] | None:
    if not library:
        return None
    target_arr = np.asarray(target, dtype=float)
    target_norm = np.linalg.norm(target_arr)
    if target_norm == 0:
        return None
    best: tuple[str, float] | None = None
    for anchor_id, vec in library.items():
        v = np.asarray(vec, dtype=float)
        n = np.linalg.norm(v)
        if n == 0:
            continue
        cos = float(np.dot(target_arr, v) / (target_norm * n))
        if best is None or cos > best[1]:
            best = (anchor_id, cos)
    return best


@st.cache_data(ttl=10)
def _cached_snapshot(fit_id: int) -> GPSnapshot | None:
    fit_row = get_gp_fit(fit_id)
    if fit_row is None:
        return None
    try:
        gp = ComplianceGP.load(fit_row.path)
    except (ValueError, FileNotFoundError):
        return None

    if gp.training_embeddings.shape[0] == 0:
        return GPSnapshot(fit_id=fit_id)

    targets = propose_targets(gp, n_targets=10, candidate_pool_size=500, seed=0)
    target_embeddings = np.asarray([t.embedding for t in targets]) if targets else np.zeros((0, gp.feature_dim))

    library_keys = list(gp.library_anchor_embeddings.keys())
    library_array = np.asarray(
        [gp.library_anchor_embeddings[k] for k in library_keys]
    ) if library_keys else np.zeros((0, gp.feature_dim))

    # Stack for one fit so all three sources share a common 2D basis.
    stack: list[np.ndarray] = [gp.training_embeddings]
    if target_embeddings.shape[0]:
        stack.append(target_embeddings)
    if library_array.shape[0]:
        stack.append(library_array)
    combined = np.vstack(stack)
    coords, projection_method = _project_2d(combined)

    points: list[GPScatterPoint] = []
    n_train = gp.training_embeddings.shape[0]
    for i in range(n_train):
        agg = float(gp.training_aggregates[i])
        points.append(
            GPScatterPoint(
                kind="training",
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                aggregate=agg,
                score=None,
                label=f"train[{i}] aggregate={agg:.3f}",
            )
        )

    n_used = n_train
    if target_embeddings.shape[0]:
        for j, target in enumerate(targets):
            x = float(coords[n_used, 0])
            y = float(coords[n_used, 1])
            n_used += 1
            points.append(
                GPScatterPoint(
                    kind="target",
                    x=x,
                    y=y,
                    aggregate=None,
                    score=target.score,
                    uncertainty=target.uncertainty,
                    label=(
                        f"target[{j+1}] score={target.score:.3f} "
                        f"σ={target.uncertainty:.3f} μ={target.expected_score:.3f}"
                    ),
                )
            )

    if library_array.shape[0]:
        for k, anchor_id in enumerate(library_keys):
            x = float(coords[n_used, 0])
            y = float(coords[n_used, 1])
            n_used += 1
            points.append(
                GPScatterPoint(
                    kind="anchor",
                    x=x,
                    y=y,
                    aggregate=None,
                    score=None,
                    label=f"library_anchor={anchor_id}",
                )
            )

    target_rows: list[dict] = []
    for i, target in enumerate(targets, start=1):
        nearest = _nearest_library_anchor(
            target.embedding, gp.library_anchor_embeddings
        )
        target_rows.append(
            {
                "rank": i,
                "uncertainty": round(target.uncertainty, 4),
                "expected_score": round(target.expected_score, 4),
                "target_score": round(target.score, 4),
                "nearest_library_anchor": (
                    f"{nearest[0]} (cos={nearest[1]:.3f})"
                    if nearest is not None
                    else "—"
                ),
            }
        )

    return GPSnapshot(
        fit_id=fit_id,
        points=points,
        target_rows=target_rows,
        projection_method=projection_method,
    )


def _build_figure(snapshot: GPSnapshot) -> go.Figure:
    fig = go.Figure()
    training = [p for p in snapshot.points if p.kind == "training"]
    targets = [p for p in snapshot.points if p.kind == "target"]
    anchors = [p for p in snapshot.points if p.kind == "anchor"]

    if training:
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in training],
                y=[p.y for p in training],
                mode="markers",
                name="training",
                marker=dict(
                    size=8,
                    color=[p.aggregate or 0.0 for p in training],
                    colorscale="RdYlGn",
                    cmin=0.0,
                    cmax=1.0,
                    colorbar=dict(title="aggregate"),
                ),
                text=[p.label for p in training],
                hoverinfo="text",
            )
        )
    if anchors:
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in anchors],
                y=[p.y for p in anchors],
                mode="markers",
                name="library_anchors",
                marker=dict(size=12, color="#444", symbol="diamond"),
                text=[p.label for p in anchors],
                hoverinfo="text",
            )
        )
    if targets:
        fig.add_trace(
            go.Scatter(
                x=[p.x for p in targets],
                y=[p.y for p in targets],
                mode="markers",
                name="proposed_targets",
                marker=dict(
                    size=14,
                    color=[p.score or 0.0 for p in targets],
                    colorscale="Viridis",
                    symbol="star",
                ),
                text=[p.label for p in targets],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        xaxis=dict(title="PC1"),
        yaxis=dict(title="PC2"),
        legend=dict(orientation="h", y=-0.2),
        height=520,
        margin=dict(l=40, r=40, t=40, b=80),
    )
    return fig


def _render() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)

    fits = _cached_fits()
    if not fits:
        st.info(EMPTY_STATE_MSG)
        return

    labels = {f.label: f for f in fits}
    chosen = st.selectbox("GP fit", options=list(labels.keys()), index=0)
    fit = labels[chosen]

    snapshot = _cached_snapshot(fit.fit_id)
    if snapshot is None:
        st.warning(
            "GP fit row exists but the artefact could not be loaded "
            f"(path={fit.path})."
        )
        return
    if not snapshot.points:
        st.warning("This GP has no training data to plot.")
        return

    st.plotly_chart(_build_figure(snapshot), use_container_width=True)
    st.caption(
        f"{snapshot.projection_method} projection of training embeddings "
        f"(aggregate score colormap, low → high). Library anchors as "
        f"black diamonds; proposed targets as stars colored by target "
        f"score."
    )

    st.subheader(f"Top-{len(snapshot.target_rows)} proposed targets")
    if snapshot.target_rows:
        st.dataframe(
            snapshot.target_rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("(no targets proposed for this fit)")


_render()
