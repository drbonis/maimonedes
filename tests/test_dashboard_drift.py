"""AppTest smoke checks for the Phase 3 drift dashboard page."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
)
from maimonedes.storage.repo import init_engine, reset_engine_for_tests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
DASHBOARD_DIR = PROJECT_ROOT / "src" / "maimonedes" / "dashboard"
DRIFT_PAGE = DASHBOARD_DIR / "pages" / "03_drift.py"


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "drift_dash.sqlite"
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


def _seed_full_run(
    *,
    n_baseline: int = 10,
    n_post: int = 20,
    baseline_mean: float = 0.9,
    post_mean: float = 0.4,
    rng_seed: int = 7,
) -> int:
    rng = np.random.default_rng(rng_seed)
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    for idx in range(n_baseline + n_post):
        stage = "baseline" if idx < n_baseline else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        for anchor in ("A1", "A2"):
            mean = baseline_mean if idx < n_baseline else post_mean
            agg = float(np.clip(mean + rng.normal(0, 0.02), 0.0, 1.0))
            record_score(
                ComplianceScore(
                    anchor_id=anchor,
                    policy_id="scope_of_practice",
                    per_sub_condition={"flags_physician_review": 1.0},
                    aggregate=agg,
                    judge_model="judge:test",
                    supervised_model="llama:test",
                    drift_session_id=sid,
                )
            )
    return run_id


def test_drift_page_empty_state_when_no_runs(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DRIFT_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Drift — Phase 3"
    info_msgs = [el.value for el in at.info]
    assert any("No drift runs recorded yet" in msg for msg in info_msgs)


def test_drift_page_renders_table_and_callout_with_full_run(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_full_run()

    at = AppTest.from_file(str(DRIFT_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Drift — Phase 3"

    # Run + anchor selectboxes both populated.
    assert len(at.selectbox) >= 2

    # Detection-latency subheader rendered.
    subheader_values = [el.value for el in at.subheader]
    assert any("Detection latency" in v for v in subheader_values)

    # Headline callout (Lead time) appears in markdown.
    markdown_blob = "\n".join(el.value for el in at.markdown)
    assert "Lead time" in markdown_blob


def test_drift_page_degraded_when_baseline_too_short(db: str) -> None:
    """A run with only 2 baseline samples per anchor → detector traces
    are absent, but the raw plot still renders and the warning shows.
    """
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    rng = np.random.default_rng(0)
    for idx in range(8):
        stage = "baseline" if idx < 2 else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        for anchor in ("A1",):
            agg = float(0.7 + rng.normal(0, 0.02))
            record_score(
                ComplianceScore(
                    anchor_id=anchor,
                    policy_id="scope_of_practice",
                    per_sub_condition={"flags_physician_review": 1.0},
                    aggregate=agg,
                    judge_model="judge:test",
                    supervised_model="llama:test",
                    drift_session_id=sid,
                )
            )

    at = AppTest.from_file(str(DRIFT_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]

    warning_msgs = [el.value for el in at.warning]
    assert any("baseline-stage samples" in msg for msg in warning_msgs)
