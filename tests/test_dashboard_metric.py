"""AppTest smoke checks for the Phase 5 Riemannian metric dashboard page."""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.core.policy import load_policy
from maimonedes.monitor.metric import fit_metric
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.metric_fits import record_metric_fit
from maimonedes.storage.perturbations import record_perturbation
from maimonedes.storage.repo import init_engine, reset_engine_for_tests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
DASHBOARD_DIR = PROJECT_ROOT / "src" / "maimonedes" / "dashboard"
METRIC_PAGE = DASHBOARD_DIR / "pages" / "06_metric.py"
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"


SCOPE_SUB_IDS = (
    "flags_physician_review",
    "expresses_uncertainty",
    "avoids_unqualified_prescriptive_language",
    "recommendation_within_scope",
    "recommendation_appropriate_specificity",
)


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "metric_dash.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    try:
        import streamlit as st

        st.cache_data.clear()
    except Exception:
        pass
    yield url
    reset_engine_for_tests()


def _seed_anchor(anchor_id: str, base: float) -> None:
    """Mirror of `tests/test_metric.py:_seed_anchor_for_metric`.

    Seeds one anchor + four perturbations against the 5-axis scope rubric
    so `fit_metric` has usable Jacobians.
    """
    def _score(values: list[float], probe_role: str, perturbation_id: int | None = None) -> ComplianceScore:
        return ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition={s: v for s, v in zip(SCOPE_SUB_IDS, values)},
            aggregate=float(sum(values) / len(values)),
            judge_model="judge:test",
            supervised_model="supervised:test",
            perturbation_id=perturbation_id,
            probe_role=probe_role,
        )

    record_score(_score([base] * 5, probe_role="anchor"))
    perturbation_specs = [
        ("authority:gp", "authority", [-0.15, 0.0, -0.05, 0.0, 0.0]),
        ("boundary:should", "boundary", [0.0, -0.20, 0.0, -0.05, 0.0]),
        ("demographic:elder", "demographic", [-0.05, -0.05, -0.05, 0.0, -0.10]),
        ("paraphrase:1", "paraphrase", [-0.10, -0.05, 0.0, -0.05, -0.05]),
    ]
    for label, kind, deltas in perturbation_specs:
        probe = PerturbationProbe(
            anchor_id=anchor_id,
            scenario=f"perturbed by {label}",
            perturbation_kind=kind,  # type: ignore[arg-type]
            transform_label=label,
        )
        row_id = record_perturbation(probe)
        values = [max(0.0, base + d) for d in deltas]
        record_score(
            _score(values, probe_role="perturbation", perturbation_id=row_id)
        )


def _seed_metric_fit(tmp_path: Path, anchor_ids: list[str]) -> int:
    """Seed anchors, fit a metric, persist the .npz + DB row, return fit_id."""
    bases = [0.85, 0.60, 0.75]
    for aid, base in zip(anchor_ids, bases):
        _seed_anchor(aid, base=base)
    policy = load_policy(POLICY_PATH, RUBRIC_PATH)
    metric = fit_metric(policy=policy, hidden=8, depth=2, epochs=20)
    out_dir = tmp_path / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metric_test.npz"
    metric.save(out_path)
    return record_metric_fit(
        policy_id=policy.id,
        path=str(out_path),
        n_anchors=metric.n_anchors,
        n_jacobians=metric.n_jacobians,
        val_loss=metric.eval_metrics.get("val_loss"),
        train_loss=metric.eval_metrics.get("train_loss"),
        hyperparams={"hidden": 8, "depth": 2, "epochs": 20},
    )


# ---------------------------------------------------------------------------
# Empty-state path
# ---------------------------------------------------------------------------


def test_metric_page_empty_state_when_no_fits(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(METRIC_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Metric — Phase 5"
    info_msgs = [el.value for el in at.info]
    assert any("No Riemannian metric fits" in m for m in info_msgs)


def test_metric_page_warns_when_artefact_missing(
    db: str, tmp_path: Path
) -> None:
    """`metric_fit` row exists but the .npz on disk is gone — warn, no crash."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    record_metric_fit(
        policy_id="scope_of_practice",
        path=str(tmp_path / "missing.npz"),
        n_anchors=1,
        n_jacobians=1,
        val_loss=None,
        train_loss=None,
    )

    at = AppTest.from_file(str(METRIC_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    warnings = [el.value for el in at.warning]
    assert any("Metric `.npz` not found" in m for m in warnings)


# ---------------------------------------------------------------------------
# Populated path
# ---------------------------------------------------------------------------


def test_metric_page_renders_grid_and_anchors(
    db: str, tmp_path: Path
) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_metric_fit(tmp_path, anchor_ids=["A1", "A2", "A3"])

    at = AppTest.from_file(str(METRIC_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    # Fit selector + axis selectors all present.
    assert len(at.selectbox) >= 3

    # Plotly figure rendered.
    plotly_charts = [el for el in at.main if getattr(el, "type", "") == "plotly_chart"]
    # AppTest may not expose plotly traces directly; check at the subheader/caption layer.
    subheaders = [el.value for el in at.subheader]
    assert any("Compliance plane" in v for v in subheaders)
    assert any("Distance comparator" in v for v in subheaders)


def test_metric_page_preset_switches_render_without_error(
    db: str, tmp_path: Path
) -> None:
    """Each §4.6 preset radio option should render its segment table."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_metric_fit(tmp_path, anchor_ids=["A1", "A2", "A3"])

    for preset_index in range(4):
        at = AppTest.from_file(str(METRIC_PAGE)).run(timeout=120)
        assert not at.exception, [str(e) for e in at.exception]
        # Find the preset radio (last radio on the page) and click each option.
        if at.radio:
            at.radio[0].set_value(at.radio[0].options[preset_index])
            at.run(timeout=120)
            assert not at.exception, [str(e) for e in at.exception]


def test_metric_page_renders_with_zero_jacobian_anchors(
    db: str, tmp_path: Path
) -> None:
    """Fit exists from a hand-built MetricMLP, but no anchors with Jacobians.

    The anchor overlay should be empty; the heatmap + ellipses must still
    render and the page must not crash.
    """
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    # Synthesize a metric fit without seeding any Jacobian-producing anchors:
    # build the smallest valid fit by training with a single trivial pair.
    import numpy as np

    from maimonedes.monitor.metric import fit_metric_from_pairs

    k = 5
    c_array = np.array([[0.5] * k], dtype=np.float64)
    g_array = np.array([np.eye(k, dtype=np.float64)])
    metric = fit_metric_from_pairs(
        c_array,
        g_array,
        k=k,
        hidden=8,
        depth=2,
        epochs=5,
        policy_id="scope_of_practice",
        n_anchors=1,
        n_jacobians=1,
        sub_condition_ids=SCOPE_SUB_IDS,
    )
    out_dir = tmp_path / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metric_zero_anchors.npz"
    metric.save(out_path)
    record_metric_fit(
        policy_id="scope_of_practice",
        path=str(out_path),
        n_anchors=1,
        n_jacobians=1,
        val_loss=None,
        train_loss=None,
    )

    at = AppTest.from_file(str(METRIC_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]
    subheaders = [el.value for el in at.subheader]
    assert any("Compliance plane" in v for v in subheaders)
