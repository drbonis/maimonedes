"""Tests for ComplianceScore model + storage round-trip."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import inspect

from maimonedes.core.compliance import ComplianceScore
from maimonedes.settings import Settings
from maimonedes.storage.compliance import (
    latest_score_per_anchor,
    record_score,
    recent_scores,
)
from maimonedes.storage.repo import (
    get_engine,
    init_engine,
    reset_engine_for_tests,
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
    db_path = tmp_path / "score.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


def _make_score(
    anchor: str = "A1",
    aggregate: float = 0.8,
    per: dict[str, float] | None = None,
) -> ComplianceScore:
    return ComplianceScore(
        anchor_id=anchor,
        policy_id="scope_of_practice",
        per_sub_condition=per
        if per is not None
        else {"flags_physician_review": 1.0, "expresses_uncertainty": 0.667},
        aggregate=aggregate,
        judge_model="medgemma:test",
        supervised_model="llama:test",
    )


# ---- model invariants ------------------------------------------------------


def test_aggregate_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        _make_score(aggregate=1.5)
    with pytest.raises(ValidationError):
        _make_score(aggregate=-0.01)


def test_scored_at_defaults_to_now_utc() -> None:
    s = _make_score()
    assert s.scored_at.tzinfo is not None


# ---- migration -------------------------------------------------------------


def test_migration_creates_indexes(db: str) -> None:
    insp = inspect(get_engine())
    indexes = {ix["name"] for ix in insp.get_indexes("compliance_scores")}
    assert "ix_compliance_scores_anchor_scored_at" in indexes
    assert "ix_compliance_scores_policy_scored_at" in indexes


def test_alembic_downgrade_drops_table(db: str) -> None:
    cfg = _alembic_cfg(db)
    command.downgrade(cfg, "0002_llm_calls")
    insp = inspect(get_engine())
    assert "compliance_scores" not in insp.get_table_names()


# ---- repository round-trip -------------------------------------------------


def test_record_and_recent_score_round_trip(db: str) -> None:
    payload = {"flags_physician_review": 1.0, "expresses_uncertainty": 0.333}
    score = _make_score(per=payload, aggregate=0.6)
    row_id = record_score(score)
    assert row_id > 0

    rows = recent_scores("A1")
    assert len(rows) == 1
    out = rows[0]
    assert out.anchor_id == "A1"
    assert out.aggregate == pytest.approx(0.6)
    assert out.per_sub_condition == payload
    assert out.scored_at.tzinfo is not None  # tzinfo restored on read


def test_recent_scores_orders_newest_first(db: str) -> None:
    record_score(_make_score(aggregate=0.4))
    time.sleep(0.01)  # SQLite stores DATETIME at second/μs resolution
    record_score(_make_score(aggregate=0.5))
    time.sleep(0.01)
    record_score(_make_score(aggregate=0.6))

    rows = recent_scores("A1", limit=10)
    aggregates = [r.aggregate for r in rows]
    assert aggregates == sorted(aggregates, reverse=True)


def test_latest_score_per_anchor_returns_most_recent_per_id(db: str) -> None:
    record_score(_make_score(anchor="A1", aggregate=0.4))
    time.sleep(0.01)
    record_score(_make_score(anchor="A1", aggregate=0.6))
    record_score(_make_score(anchor="A2", aggregate=0.9))

    latest = latest_score_per_anchor()
    assert set(latest.keys()) == {"A1", "A2"}
    assert latest["A1"].aggregate == pytest.approx(0.6)
    assert latest["A2"].aggregate == pytest.approx(0.9)


def test_recent_scores_empty_for_unknown_anchor(db: str) -> None:
    assert recent_scores("UNKNOWN") == []
