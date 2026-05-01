"""Tests for the Phase 5 synthesized_probes schema + helpers."""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.synthesized_probe import SynthesizedProbe
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.repo import (
    get_engine,
    init_engine,
    reset_engine_for_tests,
)
from maimonedes.storage.synthesized_probes import (
    get_synthesized_probe,
    list_synthesized_probes,
    record_synthesized_probe,
    scores_for_synthesized_probe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "synth.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


# ---- migration shape -------------------------------------------------------


def test_migration_creates_synthesized_probes_table(db: str) -> None:
    insp = inspect(get_engine())
    assert "synthesized_probes" in insp.get_table_names()
    indexes = {ix["name"] for ix in insp.get_indexes("synthesized_probes")}
    assert "ix_synthesized_probes_policy_id_created_at" in indexes
    assert "ix_synthesized_probes_quality_status" in indexes
    assert "ix_synthesized_probes_gp_fit_id" in indexes


def test_migration_extends_compliance_scores_with_synthesized_probe_id(
    db: str,
) -> None:
    insp = inspect(get_engine())
    cols = {c["name"] for c in insp.get_columns("compliance_scores")}
    assert "synthesized_probe_id" in cols
    indexes = {ix["name"] for ix in insp.get_indexes("compliance_scores")}
    assert "ix_compliance_scores_synthesized_probe_id" in indexes


def test_alembic_round_trip_clean(db: str) -> None:
    cfg = _alembic_cfg(db)
    command.downgrade(cfg, "0013_gp_fits")
    insp = inspect(get_engine())
    assert "synthesized_probes" not in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("compliance_scores")}
    assert "synthesized_probe_id" not in cols
    command.upgrade(cfg, "head")
    insp = inspect(get_engine())
    assert "synthesized_probes" in insp.get_table_names()


# ---- repository round-trip -------------------------------------------------


def _probe(
    *,
    scenario: str = "Patient with stable angina asks lifestyle questions.",
    tau: float = 0.85,
    quality_status: str = "approved",
    gp_fit_id: int | None = None,
) -> SynthesizedProbe:
    return SynthesizedProbe(
        policy_id="scope_of_practice",
        scenario=scenario,
        generation_method="knn_exemplar",
        target_embedding=[0.1] * 8,
        achieved_embedding=[0.12] * 8,
        tau_distance=tau,
        parent_anchor_ids=["A1", "A3"],
        synthesizer_llm_call_id=None,
        validator_llm_call_id=None,
        quality_status=quality_status,  # type: ignore[arg-type]
        quality_reason=None,
        gp_fit_id=gp_fit_id,
    )


def test_record_and_get_round_trip(db: str) -> None:
    probe_id = record_synthesized_probe(_probe())
    fetched = get_synthesized_probe(probe_id)
    assert fetched is not None
    assert fetched.id == probe_id
    assert fetched.policy_id == "scope_of_practice"
    assert fetched.parent_anchor_ids == ["A1", "A3"]
    assert fetched.tau_distance == pytest.approx(0.85)
    assert fetched.quality_status == "approved"


def test_get_synthesized_probe_missing_returns_none(db: str) -> None:
    assert get_synthesized_probe(999) is None


def test_list_synthesized_probes_filters(db: str) -> None:
    a = record_synthesized_probe(_probe(scenario="A", quality_status="approved"))
    b = record_synthesized_probe(_probe(scenario="B", quality_status="rejected"))
    all_probes = list_synthesized_probes()
    assert {p.id for p in all_probes} == {a, b}
    approved = list_synthesized_probes(quality_status="approved")
    assert {p.id for p in approved} == {a}


def test_scores_for_synthesized_probe_returns_in_order(db: str) -> None:
    probe_id = record_synthesized_probe(_probe())

    record_score(
        ComplianceScore(
            anchor_id="synthesized:1",
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": 0.9},
            aggregate=0.9,
            judge_model="judge:test",
            supervised_model="llama:test",
            synthesized_probe_id=probe_id,
        )
    )
    record_score(
        ComplianceScore(
            anchor_id="synthesized:1",
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": 0.5},
            aggregate=0.5,
            judge_model="judge:test",
            supervised_model="llama:test",
            synthesized_probe_id=probe_id,
        )
    )
    scores = scores_for_synthesized_probe(probe_id)
    assert len(scores) == 2
    aggregates = [s.aggregate for s in scores]
    assert aggregates[0] <= aggregates[-1] or aggregates[0] >= aggregates[-1]


def test_compliance_score_with_orphan_synthesized_probe_id_rejected(
    db: str,
) -> None:
    with pytest.raises(IntegrityError):
        record_score(
            ComplianceScore(
                anchor_id="synthesized:bogus",
                policy_id="scope_of_practice",
                per_sub_condition={"flags_physician_review": 0.5},
                aggregate=0.5,
                judge_model="judge:test",
                supervised_model="llama:test",
                synthesized_probe_id=999_999,
            )
        )
