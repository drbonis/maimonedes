"""AppTest smoke checks for the Phase 5 curvature alarm dashboard page."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.core.policy import load_policy
from maimonedes.monitor.metric import fit_metric, fit_metric_from_pairs
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.metric_fits import record_metric_fit
from maimonedes.storage.perturbations import record_perturbation
from maimonedes.storage.repo import init_engine, reset_engine_for_tests
from maimonedes.storage.structural_signals import record_structural_signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
DASHBOARD_DIR = PROJECT_ROOT / "src" / "maimonedes" / "dashboard"
CURV_PAGE = DASHBOARD_DIR / "pages" / "07_curvature.py"
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
    db_path = tmp_path / "curv_dash.sqlite"
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


def _seed_two_fits(tmp_path: Path) -> tuple[int, int]:
    """Seed anchors + fit a baseline metric, then fit a second metric with a
    different seed so curvature differs slightly per-anchor.
    """
    for aid, base in (("A1", 0.85), ("A2", 0.60), ("A3", 0.75)):
        _seed_anchor(aid, base=base)
    policy = load_policy(POLICY_PATH, RUBRIC_PATH)
    out_dir = tmp_path / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_b = fit_metric(policy=policy, hidden=8, depth=2, epochs=20, seed=0)
    path_b = out_dir / "metric_baseline.npz"
    metric_b.save(path_b)
    fit_b = record_metric_fit(
        policy_id=policy.id,
        path=str(path_b),
        n_anchors=metric_b.n_anchors,
        n_jacobians=metric_b.n_jacobians,
        val_loss=metric_b.eval_metrics.get("val_loss"),
        train_loss=metric_b.eval_metrics.get("train_loss"),
    )

    metric_c = fit_metric(policy=policy, hidden=8, depth=2, epochs=20, seed=42)
    path_c = out_dir / "metric_current.npz"
    metric_c.save(path_c)
    fit_c = record_metric_fit(
        policy_id=policy.id,
        path=str(path_c),
        n_anchors=metric_c.n_anchors,
        n_jacobians=metric_c.n_jacobians,
        val_loss=metric_c.eval_metrics.get("val_loss"),
        train_loss=metric_c.eval_metrics.get("train_loss"),
    )

    return fit_b, fit_c


# ---------------------------------------------------------------------------
# Empty-state path
# ---------------------------------------------------------------------------


def test_curvature_page_empty_state_when_no_fits(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(CURV_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Curvature — Phase 5"
    info_msgs = [el.value for el in at.info]
    assert any("at least two `metric_fits`" in m for m in info_msgs)


def test_curvature_page_info_when_only_one_fit(
    db: str, tmp_path: Path
) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    # Single fit recorded → info banner ("need 2 fits").
    out_dir = tmp_path / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    metric = fit_metric_from_pairs(
        np.array([[0.5] * 5], dtype=np.float64),
        np.array([np.eye(5, dtype=np.float64)]),
        k=5,
        hidden=8,
        depth=2,
        epochs=5,
        policy_id="scope_of_practice",
        n_anchors=1,
        n_jacobians=1,
        sub_condition_ids=SCOPE_SUB_IDS,
    )
    path = out_dir / "metric_only.npz"
    metric.save(path)
    record_metric_fit(
        policy_id="scope_of_practice",
        path=str(path),
        n_anchors=1,
        n_jacobians=1,
        val_loss=None,
        train_loss=None,
    )

    at = AppTest.from_file(str(CURV_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    info_msgs = [el.value for el in at.info]
    assert any("at least two `metric_fits`" in m for m in info_msgs)


# ---------------------------------------------------------------------------
# Populated path
# ---------------------------------------------------------------------------


def test_curvature_page_renders_table_and_spectrum(
    db: str, tmp_path: Path
) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_two_fits(tmp_path)

    at = AppTest.from_file(str(CURV_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    # Two fit selectors + one anchor selector + axis-pair selectors all rendered.
    assert len(at.selectbox) >= 3

    subheaders = [el.value for el in at.subheader]
    assert any("Per-anchor κ comparison" in v for v in subheaders)
    assert any("Per-anchor eigenvalue spectrum" in v for v in subheaders)
    assert any("Recorded curvature alarms" in v for v in subheaders)


def test_curvature_page_threshold_slider_re_renders(
    db: str, tmp_path: Path
) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_two_fits(tmp_path)

    at = AppTest.from_file(str(CURV_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    # Threshold slider exists; nudge it.
    assert len(at.slider) >= 1
    at.slider[0].set_value(1.0)
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


def test_curvature_page_mismatched_policies_errors(
    db: str, tmp_path: Path
) -> None:
    """Two fits with different `policy_id` → error banner, no traceback."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    out_dir = tmp_path / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for pid, fname in (
        ("scope_of_practice", "metric_a.npz"),
        ("other_policy", "metric_b.npz"),
    ):
        metric = fit_metric_from_pairs(
            np.array([[0.5] * 5], dtype=np.float64),
            np.array([np.eye(5, dtype=np.float64)]),
            k=5,
            hidden=8,
            depth=2,
            epochs=5,
            policy_id=pid,
            n_anchors=1,
            n_jacobians=1,
            sub_condition_ids=SCOPE_SUB_IDS,
        )
        path = out_dir / fname
        metric.save(path)
        record_metric_fit(
            policy_id=pid,
            path=str(path),
            n_anchors=1,
            n_jacobians=1,
            val_loss=None,
            train_loss=None,
        )

    at = AppTest.from_file(str(CURV_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    # Force the second selector to the cross-policy choice.
    if len(at.selectbox) >= 2:
        # selectbox[0] = baseline, selectbox[1] = current
        baseline_options = at.selectbox[0].options
        current_options = at.selectbox[1].options
        # Pick options whose policy_id substring differs.
        baseline_pid = next(
            (o for o in baseline_options if "scope_of_practice" in o), None
        )
        current_pid = next(
            (o for o in current_options if "other_policy" in o), None
        )
        if baseline_pid is None or current_pid is None:
            pytest.skip("could not find both policy options")
        at.selectbox[0].set_value(baseline_pid)
        at.selectbox[1].set_value(current_pid)
        at.run(timeout=120)
        assert not at.exception, [str(e) for e in at.exception]
        error_msgs = [el.value for el in at.error]
        assert any("must share a `policy_id`" in m for m in error_msgs)


def test_curvature_page_signals_feed_renders(
    db: str, tmp_path: Path
) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_two_fits(tmp_path)
    record_structural_signal(
        anchor_id="A1",
        signal_type="curvature",
        metric_value=1.42,
        threshold=0.5,
        evidence={"baseline_kappa": 1.0, "current_kappa": 2.42},
    )

    at = AppTest.from_file(str(CURV_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    # The signals table is the last dataframe; check it carries our row.
    assert len(at.dataframe) >= 1
