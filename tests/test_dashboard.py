"""Tests for the Phase 1 Streamlit dashboard.

The dashboard module is a thin shim: it calls into
`storage.dashboard_queries` and renders the results. We exercise both
layers — the query helpers directly (cheap, deterministic) and the
page via Streamlit's `AppTest` runner (smoke check that imports
resolve and the page emits a title without exploding).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.dashboard_queries import (
    history_for_anchor,
    latest_table_rows,
    per_sub_condition,
)
from maimonedes.storage.repo import init_engine, reset_engine_for_tests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
DASHBOARD_APP = PROJECT_ROOT / "src" / "maimonedes" / "dashboard" / "app.py"


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "dash.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    # `st.cache_data` is process-wide, so clear it per-test or the
    # first AppTest run's "empty DB" result poisons subsequent tests
    # that have populated the DB.
    try:
        import streamlit as st

        st.cache_data.clear()
    except Exception:
        pass
    yield url
    reset_engine_for_tests()


def _seed(anchor: str, aggregate: float) -> int:
    return record_score(
        ComplianceScore(
            anchor_id=anchor,
            policy_id="scope_of_practice",
            per_sub_condition={
                "flags_physician_review": 1.0,
                "expresses_uncertainty": 0.667,
            },
            aggregate=aggregate,
            judge_model="medgemma:test",
            supervised_model="llama:test",
        )
    )


# ---- query helpers ---------------------------------------------------------


def test_latest_table_rows_empty_db_returns_empty_list(db: str) -> None:
    assert latest_table_rows() == []


def test_latest_table_rows_one_row_per_anchor_sorted(db: str) -> None:
    _seed("A2", 0.7)
    time.sleep(0.01)
    _seed("A1", 0.4)
    time.sleep(0.01)
    _seed("A1", 0.6)  # newer A1 score should win

    rows = latest_table_rows()
    assert [r["anchor_id"] for r in rows] == ["A1", "A2"]
    assert rows[0]["aggregate"] == pytest.approx(0.6)
    assert rows[1]["aggregate"] == pytest.approx(0.7)
    # Required columns the page renders
    expected = {
        "anchor_id",
        "aggregate",
        "policy_id",
        "judge_model",
        "supervised_model",
        "scored_at",
    }
    assert expected.issubset(rows[0].keys())


def test_history_for_anchor_returns_oldest_first(db: str) -> None:
    _seed("A1", 0.4)
    time.sleep(0.01)
    _seed("A1", 0.5)
    time.sleep(0.01)
    _seed("A1", 0.6)

    history = history_for_anchor("A1", limit=10)
    assert [s.aggregate for s in history] == pytest.approx([0.4, 0.5, 0.6])


def test_per_sub_condition_flattens_score_map(db: str) -> None:
    from maimonedes.storage.compliance import latest_score_per_anchor

    _seed("A1", 0.6)
    flat = per_sub_condition(latest_score_per_anchor())
    assert "A1" in flat
    assert flat["A1"]["flags_physician_review"] == pytest.approx(1.0)


# ---- AppTest smoke ---------------------------------------------------------


def test_dashboard_renders_empty_state_when_db_is_empty(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASHBOARD_APP)).run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Compliance scores — Phase 1"
    # Empty-state message
    info_msgs = [el.value for el in at.info]
    assert any("No compliance scores" in msg for msg in info_msgs)


def test_dashboard_renders_table_when_populated(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed("A1", 0.6)
    _seed("A2", 0.4)

    at = AppTest.from_file(str(DASHBOARD_APP)).run(timeout=30)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Compliance scores — Phase 1"

    # `st.dataframe` lands in `at.dataframe`; assert at least one frame
    # was rendered with our two anchors.
    assert len(at.dataframe) >= 1
    rendered = at.dataframe[0].value
    # Streamlit returns a DataFrame-ish object; coerce to list of dicts.
    if hasattr(rendered, "to_dict"):
        records = rendered.to_dict(orient="records")
    else:
        records = list(rendered)
    anchor_ids = {row["anchor_id"] for row in records}
    assert {"A1", "A2"}.issubset(anchor_ids)
