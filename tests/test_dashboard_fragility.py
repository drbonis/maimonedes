"""AppTest smoke checks for the Phase 2 fragility page."""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.perturbations import record_perturbation
from maimonedes.storage.repo import init_engine, reset_engine_for_tests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
DASHBOARD_DIR = PROJECT_ROOT / "src" / "maimonedes" / "dashboard"
ENTRY_APP = DASHBOARD_DIR / "app.py"
FRAGILITY_PAGE = DASHBOARD_DIR / "pages" / "02_fragility.py"


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "frag_dash.sqlite"
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


def _seed_anchor(anchor: str, aggregate: float) -> None:
    record_score(
        ComplianceScore(
            anchor_id=anchor,
            policy_id="scope_of_practice",
            per_sub_condition={
                "flags_physician_review": 1.0,
                "expresses_uncertainty": 0.667,
            },
            aggregate=aggregate,
            judge_model="judge:test",
            supervised_model="supervised:test",
        )
    )


def _seed_perturbation(
    anchor: str,
    label: str,
    kind: str,
    aggregate: float,
) -> int:
    probe = PerturbationProbe(
        anchor_id=anchor,
        scenario=f"perturbed:{label}",
        perturbation_kind=kind,  # type: ignore[arg-type]
        transform_label=label,
        generator_metadata={},
    )
    row_id = record_perturbation(probe)
    record_score(
        ComplianceScore(
            anchor_id=anchor,
            policy_id="scope_of_practice",
            per_sub_condition={
                "flags_physician_review": 0.0,
                "expresses_uncertainty": 0.667,
            },
            aggregate=aggregate,
            judge_model="judge:test",
            supervised_model="supervised:test",
            perturbation_id=row_id,
            probe_role="perturbation",
        )
    )
    return row_id


# ---- entry app -------------------------------------------------------------


def test_entry_app_renders_landing_with_navigation_hint(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ENTRY_APP)).run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "maimonedes dashboard"


# ---- fragility page --------------------------------------------------------


def test_fragility_page_empty_state_when_no_data(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(FRAGILITY_PAGE)).run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Fragility — Phase 2"
    info_msgs = [el.value for el in at.info]
    assert any("No perturbation data" in msg for msg in info_msgs)


def test_fragility_page_renders_aggregated_and_per_anchor_with_data(
    db: str,
) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_anchor("A1", 0.8)
    _seed_anchor("A2", 0.7)
    _seed_perturbation("A1", "authority:gp", "authority", 0.3)
    _seed_perturbation("A1", "boundary:should_to_will", "boundary", 0.5)
    _seed_perturbation("A2", "authority:gp", "authority", 0.4)

    at = AppTest.from_file(str(FRAGILITY_PAGE)).run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Fragility — Phase 2"

    # Two `st.subheader` blocks rendered (aggregated + per-anchor).
    subheader_values = [el.value for el in at.subheader]
    assert any("Aggregated fragility" in v for v in subheader_values)
    assert any("Per-anchor Jacobian" in v for v in subheader_values)

    # The anchor selector should expose both anchors with baselines.
    assert len(at.selectbox) >= 1
    options = list(at.selectbox[0].options)
    assert {"A1", "A2"}.issubset(set(options))

    # At least two dataframes were rendered (aggregated + per-anchor).
    assert len(at.dataframe) >= 2
