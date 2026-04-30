"""AppTest smoke checks for the Phase 5 GP dashboard page."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.policy import Policy, load_policy
from maimonedes.monitor.gp_layer import fit_compliance_gp
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.gp_fits import record_gp_fit
from maimonedes.storage.llm_calls import LLMCall
from maimonedes.storage.repo import (
    get_session,
    init_engine,
    reset_engine_for_tests,
)
from tests.fakes import FakeEmbedClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"
DASHBOARD_DIR = PROJECT_ROOT / "src" / "maimonedes" / "dashboard"
GP_PAGE = DASHBOARD_DIR / "pages" / "05_gp.py"
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "gp_dash.sqlite"
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


@pytest.fixture
def policy() -> Policy:
    return load_policy(POLICY_PATH, RUBRIC_PATH)


def _seed_and_fit(policy: Policy, tmp_path: Path) -> int:
    sub_ids = [s.id for s in policy.rubric.sub_conditions]
    payloads = []
    with get_session() as session:
        for anchor_id in ("A1", "A2", "A3"):
            for i in range(8):
                aggregate = 0.5 + 0.4 * (i % 5 - 2) / 4.0
                aggregate = max(0.0, min(1.0, aggregate))
                text = f"{anchor_id}-output-{i}"
                llm = LLMCall(
                    backend_name="ollama-supervised",
                    model="llama:test",
                    request_messages_json=json.dumps([]),
                    response_content=text,
                    raw_response_json="{}",
                    prompt_tokens=10,
                    completion_tokens=10,
                    latency_ms=1.0,
                    request_hash=f"gp-dash-{anchor_id}-{i}",
                )
                session.add(llm)
                session.flush()
                payloads.append((anchor_id, llm.id, aggregate))
    for anchor_id, llm_id, aggregate in payloads:
        record_score(
            ComplianceScore(
                anchor_id=anchor_id,
                policy_id=policy.id,
                per_sub_condition={sid: aggregate for sid in sub_ids},
                aggregate=aggregate,
                judge_model="judge:test",
                supervised_model="llama:test",
                llm_call_id=llm_id,
            )
        )

    fake_embed = FakeEmbedClient(default_dim=8)
    gp = fit_compliance_gp(
        embed_client=fake_embed,
        policy=policy,
        embedding_model="fake-embed",
        min_samples=10,
        library_anchor_texts={"A1": "lifestyle question", "A2": "triage question"},
    )
    path = tmp_path / "gp_dash.pkl"
    gp.save(path)
    return record_gp_fit(
        path=str(path),
        policy_id=policy.id,
        n_samples=gp.n_samples,
        kernel_name=gp.kernel_repr,
        log_marginal_likelihood=gp.log_marginal_likelihood,
        embedding_model=gp.embedding_model,
    )


def test_gp_page_empty_state_when_no_fits(db: str) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(GP_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.title[0].value == "GP — Phase 5"
    info_msgs = [el.value for el in at.info]
    assert any("No GP fits" in m for m in info_msgs)


def test_gp_page_renders_scatter_and_targets_table(
    db: str, policy: Policy, tmp_path: Path
) -> None:
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    _seed_and_fit(policy, tmp_path)

    at = AppTest.from_file(str(GP_PAGE)).run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]

    # GP-fit dropdown rendered.
    assert len(at.selectbox) >= 1

    # Subheader for the targets table.
    subheader_values = [el.value for el in at.subheader]
    assert any("proposed targets" in v.lower() for v in subheader_values)

    # At least one dataframe (the targets table) rendered.
    assert len(at.dataframe) >= 1
