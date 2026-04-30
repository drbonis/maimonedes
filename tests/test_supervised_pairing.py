"""Tests for the supervised ↔ compliance_scores chronological pairing helper."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.settings import Settings
from maimonedes.storage.compliance import (
    ComplianceScoreRow,
    record_score,
)
from maimonedes.storage.llm_calls import (
    LLMCall,
    SUPERVISED_TO_SCORE_MAX_SECONDS,
    pair_supervised_with_scores,
)
from maimonedes.storage.repo import (
    get_session,
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
    db_path = tmp_path / "pair.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


def _insert_supervised(
    *,
    response: str,
    timestamp: datetime,
    request_hash: str,
) -> int:
    """Insert one ollama-supervised row at the given timestamp; return id."""
    with get_session() as session:
        row = LLMCall(
            timestamp=timestamp,
            backend_name="ollama-supervised",
            model="llama:test",
            request_messages_json=json.dumps([]),
            response_content=response,
            raw_response_json="{}",
            prompt_tokens=10,
            completion_tokens=10,
            latency_ms=1.0,
            request_hash=request_hash,
        )
        session.add(row)
        session.flush()
        return row.id


def _insert_score(
    *,
    anchor_id: str,
    aggregate: float,
    scored_at: datetime,
) -> None:
    """Insert one compliance_scores row at the given scored_at."""
    record_score(
        ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": aggregate},
            aggregate=aggregate,
            judge_model="judge:test",
            supervised_model="llama:test",
            scored_at=scored_at,
        )
    )


# ---- happy path ------------------------------------------------------------


def test_pair_one_to_one_in_order(db: str) -> None:
    """One supervised call followed by one score → one pair, in order."""
    base = datetime.now(timezone.utc)
    _insert_supervised(
        response="text-A1",
        timestamp=base,
        request_hash="A1-sup",
    )
    _insert_score(anchor_id="A1", aggregate=0.9, scored_at=base + timedelta(seconds=2))
    _insert_supervised(
        response="text-A2",
        timestamp=base + timedelta(seconds=10),
        request_hash="A2-sup",
    )
    _insert_score(anchor_id="A2", aggregate=0.7, scored_at=base + timedelta(seconds=12))

    pairs = pair_supervised_with_scores("scope_of_practice")
    assert len(pairs) == 2
    assert pairs[0].anchor_id == "A1"
    assert pairs[0].supervised_text == "text-A1"
    assert pairs[1].anchor_id == "A2"
    assert pairs[1].supervised_text == "text-A2"


def test_score_with_no_preceding_supervised_is_skipped(db: str) -> None:
    base = datetime.now(timezone.utc)
    # Only the score; no supervised call before it.
    _insert_score(anchor_id="A1", aggregate=0.5, scored_at=base)
    pairs = pair_supervised_with_scores("scope_of_practice")
    assert pairs == []


def test_supervised_with_no_following_score_is_dropped(db: str) -> None:
    base = datetime.now(timezone.utc)
    _insert_supervised(
        response="orphan",
        timestamp=base,
        request_hash="orphan",
    )
    pairs = pair_supervised_with_scores("scope_of_practice")
    assert pairs == []


def test_orphaned_supervised_outside_window_is_skipped(db: str) -> None:
    """A sup call older than SUPERVISED_TO_SCORE_MAX_SECONDS gets dropped, not paired."""
    base = datetime.now(timezone.utc)
    # First sup is way before any score (orphan from a failed run).
    _insert_supervised(
        response="orphan",
        timestamp=base,
        request_hash="orphan",
    )
    # Second sup + score are a normal pair, much later.
    later = base + timedelta(seconds=SUPERVISED_TO_SCORE_MAX_SECONDS + 60)
    _insert_supervised(
        response="real",
        timestamp=later,
        request_hash="real",
    )
    _insert_score(anchor_id="A1", aggregate=0.8, scored_at=later + timedelta(seconds=2))

    pairs = pair_supervised_with_scores("scope_of_practice")
    assert len(pairs) == 1
    assert pairs[0].supervised_text == "real"


def test_filters_by_policy_id(db: str) -> None:
    base = datetime.now(timezone.utc)
    _insert_supervised(
        response="text",
        timestamp=base,
        request_hash="x",
    )
    record_score(
        ComplianceScore(
            anchor_id="A1",
            policy_id="other_policy",
            per_sub_condition={"x": 1.0},
            aggregate=0.9,
            judge_model="j",
            supervised_model="s",
            scored_at=base + timedelta(seconds=1),
        )
    )
    pairs = pair_supervised_with_scores("scope_of_practice")
    assert pairs == []
    pairs_other = pair_supervised_with_scores("other_policy")
    assert len(pairs_other) == 1


def test_filters_by_backend_prefix(db: str) -> None:
    """Only sup rows whose backend_name starts with the prefix are paired."""
    base = datetime.now(timezone.utc)
    # Insert a non-supervised call first; should NOT be picked.
    with get_session() as session:
        row = LLMCall(
            timestamp=base,
            backend_name="ollama-judge",
            model="judge:test",
            request_messages_json=json.dumps([]),
            response_content="judge-output",
            raw_response_json="{}",
            prompt_tokens=10,
            completion_tokens=10,
            latency_ms=1.0,
            request_hash="judge-only",
        )
        session.add(row)
    _insert_score(anchor_id="A1", aggregate=0.9, scored_at=base + timedelta(seconds=1))
    pairs = pair_supervised_with_scores("scope_of_practice")
    assert pairs == []


def test_pairing_is_chronological_not_per_anchor(db: str) -> None:
    """Cross-anchor interleaving still pairs in time order."""
    base = datetime.now(timezone.utc)
    _insert_supervised(
        response="text-A1",
        timestamp=base,
        request_hash="s1",
    )
    _insert_supervised(
        response="text-A2",
        timestamp=base + timedelta(seconds=5),
        request_hash="s2",
    )
    # A2 score comes back first, then A1 (out of insertion order).
    _insert_score(anchor_id="A2", aggregate=0.6, scored_at=base + timedelta(seconds=7))
    _insert_score(anchor_id="A1", aggregate=0.9, scored_at=base + timedelta(seconds=8))
    pairs = pair_supervised_with_scores("scope_of_practice")
    # First sup pairs with first score in time order: text-A1 → A2 score.
    # That's the heuristic's known limitation when scores arrive out
    # of supervised order. Document the behaviour rather than fighting it.
    assert len(pairs) == 2
    assert pairs[0].supervised_text == "text-A1"
    assert pairs[0].anchor_id == "A2"
    assert pairs[1].supervised_text == "text-A2"
    assert pairs[1].anchor_id == "A1"
