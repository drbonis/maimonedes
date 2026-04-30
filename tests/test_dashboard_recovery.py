"""AppTest smoke checks for the Phase 4 recovery dashboard page."""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.feedback import Feedback
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
)
from maimonedes.storage.recovery import (
    create_recovery_run,
    record_feedback,
)
from maimonedes.storage.repo import init_engine, reset_engine_for_tests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
DASHBOARD_DIR = PROJECT_ROOT / "src" / "maimonedes" / "dashboard"
RECOVERY_PAGE = DASHBOARD_DIR / "pages" / "04_recovery.py"


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "recovery_dash.sqlite"
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


def _seed_full_recovery() -> tuple[int, int]:
    drift_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    sid_baseline = create_drift_session(drift_id, 0, "baseline", "")
    sid_drift = create_drift_session(drift_id, 5, "concise", "x")
    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": 0.9},
            aggregate=0.9,
            judge_model="j",
            supervised_model="s",
            drift_session_id=sid_baseline,
        )
    )
    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": 0.3},
            aggregate=0.3,
            judge_model="j",
            supervised_model="s",
            drift_session_id=sid_drift,
        )
    )

    recovery_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="llama:test",
        judge_model="judge:test",
        contrastive_kind="temporal",
    )
    record_feedback(
        Feedback(
            recovery_run_id=recovery_id,
            parent_drift_run_id=drift_id,
            anchor_id="A1",
            contrastive_kind="temporal",
            feedback_text="Always defer to physician.",
        )
    )
    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": 0.85},
            aggregate=0.85,
            judge_model="j",
            supervised_model="s",
            recovery_run_id=recovery_id,
        )
    )
    return drift_id, recovery_id


def test_recovery_page_empty_state_when_no_runs(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(RECOVERY_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "Recovery — Phase 4"
    info_msgs = [el.value for el in at.info]
    assert any("No recovery runs" in msg for msg in info_msgs)


def test_recovery_page_renders_bars_and_feedback_with_full_run(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_full_recovery()

    at = AppTest.from_file(str(RECOVERY_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]

    # Recovery-run dropdown is populated.
    assert len(at.selectbox) >= 1

    # Subheader for the "Before / after compliance" section.
    subheader_values = [el.value for el in at.subheader]
    assert any("Before / after compliance" in v for v in subheader_values)
    assert any("Per-anchor detail" in v for v in subheader_values)

    # The synthesized feedback text appears as a markdown quote.
    md_blob = "\n".join(el.value for el in at.markdown)
    assert "Always defer to physician." in md_blob
    # Headline callout with verdict.
    assert "verdict:" in md_blob


def test_recovery_page_degraded_when_no_scored_anchors(db: str) -> None:
    """Recovery run with feedback but no compliance scores → still renders the
    feedback panel; warning about missing scores is shown."""
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    drift_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    recovery_id = create_recovery_run(
        parent_drift_run_id=drift_id,
        supervised_model="m",
        judge_model="j",
        contrastive_kind="temporal",
    )
    record_feedback(
        Feedback(
            recovery_run_id=recovery_id,
            parent_drift_run_id=drift_id,
            anchor_id="A1",
            contrastive_kind="temporal",
            feedback_text="Defer to physician.",
        )
    )

    at = AppTest.from_file(str(RECOVERY_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    warning_msgs = [el.value for el in at.warning]
    assert any("no scored anchors" in msg.lower() for msg in warning_msgs)
