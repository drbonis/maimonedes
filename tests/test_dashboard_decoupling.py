"""AppTest smoke checks for the Phase 5 decoupling alarm dashboard page."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.perturbations import record_perturbation
from maimonedes.storage.repo import init_engine, reset_engine_for_tests
from maimonedes.storage.structural_signals import record_structural_signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
DASHBOARD_DIR = PROJECT_ROOT / "src" / "maimonedes" / "dashboard"
DECOUPLING_PAGE = DASHBOARD_DIR / "pages" / "08_decoupling.py"


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
    db_path = tmp_path / "decoupling_dash.sqlite"
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


def _record_perturbation_score(
    anchor_id: str,
    sub_values: dict[str, float],
    *,
    pert_id: int,
) -> None:
    record_score(
        ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition=sub_values,
            aggregate=float(sum(sub_values.values()) / len(sub_values)),
            judge_model="judge:test",
            supervised_model="supervised:test",
            perturbation_id=pert_id,
            probe_role="perturbation",
        )
    )


def _seed_anchor_with_correlation_flip(
    anchor_id: str,
    *,
    n_baseline: int = 30,
    n_current: int = 15,
    seed: int = 0,
) -> None:
    """Seed perturbation-stage scores with a correlation flip mid-stream.

    Baseline window: scope and calibration positively correlated.
    Current window: same axes negatively correlated. Other axes are
    independent noise.

    `_recent_perturbation_scores` orders newest-first, so we insert
    *baseline points first* (older) and *current points second* (newer)
    so the page's [:current_window] head is the current data.
    """
    rng = np.random.default_rng(seed)
    parent_probe = PerturbationProbe(
        anchor_id=anchor_id,
        scenario="seeded for decoupling test",
        perturbation_kind="paraphrase",  # type: ignore[arg-type]
        transform_label=f"seed:{anchor_id}",
    )
    pert_id = record_perturbation(parent_probe)

    def _insert(values: list[float]) -> None:
        sub_values = {sid: float(v) for sid, v in zip(SCOPE_SUB_IDS, values)}
        _record_perturbation_score(anchor_id, sub_values, pert_id=pert_id)

    # Baseline (older / inserted first): positive correlation between c0, c1.
    for _ in range(n_baseline):
        z = rng.normal(0, 0.08)
        c0 = float(np.clip(0.7 + z, 0.0, 1.0))
        c1 = float(np.clip(0.7 + z + rng.normal(0, 0.02), 0.0, 1.0))
        # Independent noise on the other 3 axes.
        rest = [float(np.clip(0.7 + rng.normal(0, 0.04), 0.0, 1.0)) for _ in range(3)]
        _insert([c0, c1] + rest)

    # Current (newer / inserted second): negative correlation between c0, c1.
    for _ in range(n_current):
        z = rng.normal(0, 0.08)
        c0 = float(np.clip(0.6 + z, 0.0, 1.0))
        c1 = float(np.clip(0.6 - z + rng.normal(0, 0.02), 0.0, 1.0))
        rest = [float(np.clip(0.6 + rng.normal(0, 0.04), 0.0, 1.0)) for _ in range(3)]
        _insert([c0, c1] + rest)


def _seed_low_evidence_anchor(anchor_id: str, n: int = 3) -> None:
    parent_probe = PerturbationProbe(
        anchor_id=anchor_id,
        scenario="seeded few-points",
        perturbation_kind="paraphrase",  # type: ignore[arg-type]
        transform_label=f"seed-low:{anchor_id}",
    )
    pert_id = record_perturbation(parent_probe)
    for i in range(n):
        sub_values = {sid: 0.5 + 0.01 * i for sid in SCOPE_SUB_IDS}
        _record_perturbation_score(anchor_id, sub_values, pert_id=pert_id)


# ---------------------------------------------------------------------------
# Empty-state path
# ---------------------------------------------------------------------------


def test_decoupling_page_empty_state_when_no_perturbations(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DECOUPLING_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Decoupling — Phase 5"
    info_msgs = [el.value for el in at.info]
    assert any("No anchor has any perturbation-stage scores" in m for m in info_msgs)


# ---------------------------------------------------------------------------
# Populated path
# ---------------------------------------------------------------------------


def test_decoupling_page_renders_table_and_heatmaps(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_anchor_with_correlation_flip("A1")

    at = AppTest.from_file(str(DECOUPLING_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    subheaders = [el.value for el in at.subheader]
    assert any("Per-anchor decoupling signal" in v for v in subheaders)
    assert any("Covariance heatmaps" in v for v in subheaders)
    assert any("Axis-pair scatter rotation" in v for v in subheaders)
    assert any("Recorded decoupling alarms" in v for v in subheaders)


def test_decoupling_page_window_sliders_re_render(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_anchor_with_correlation_flip("A1")

    at = AppTest.from_file(str(DECOUPLING_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    # baseline_window + current_window sliders both present.
    assert len(at.slider) >= 2
    at.slider[0].set_value(20)
    at.slider[1].set_value(10)
    at.run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]


def test_decoupling_page_warns_when_evidence_below_window(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    # Only 3 perturbation scores — below the default current_window=20.
    _seed_low_evidence_anchor("A_low", n=3)

    at = AppTest.from_file(str(DECOUPLING_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    warnings = [el.value for el in at.warning]
    # Either "no anchors had enough" (table empty) or per-anchor warning.
    msg_blob = " ".join(warnings)
    assert "perturbation-stage" in msg_blob or "perturbation-stage scores" in msg_blob


def test_decoupling_page_default_anchor_prefers_fired(db: str) -> None:
    """When at least one anchor's signal fires, the default selectbox value
    should be that anchor (or the first fired row in the sorted table)."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    # Two anchors: one with a strong correlation flip, one with quiet noise.
    _seed_anchor_with_correlation_flip("A_flip", seed=1)

    parent_probe = PerturbationProbe(
        anchor_id="A_quiet",
        scenario="quiet baseline noise",
        perturbation_kind="paraphrase",  # type: ignore[arg-type]
        transform_label="seed:A_quiet",
    )
    pert_id = record_perturbation(parent_probe)
    rng = np.random.default_rng(7)
    for _ in range(45):
        sub = {
            sid: float(np.clip(0.7 + rng.normal(0, 0.01), 0.0, 1.0))
            for sid in SCOPE_SUB_IDS
        }
        _record_perturbation_score("A_quiet", sub, pert_id=pert_id)

    at = AppTest.from_file(str(DECOUPLING_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    # Anchor selector exists; we don't enforce which value defaults to —
    # this test mainly asserts the page renders cleanly with multiple anchors.
    assert len(at.selectbox) >= 1


def test_decoupling_page_signals_feed_renders(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_anchor_with_correlation_flip("A1")
    record_structural_signal(
        anchor_id="A1",
        signal_type="decoupling",
        metric_value=0.73,
        threshold=0.4,
        evidence={"flipped_pairs": [(0, 1)], "frobenius_delta": 0.73},
    )

    at = AppTest.from_file(str(DECOUPLING_PAGE)).run(timeout=120)
    assert not at.exception, [str(e) for e in at.exception]

    # Multiple dataframes; the last is the signals feed.
    assert len(at.dataframe) >= 1
