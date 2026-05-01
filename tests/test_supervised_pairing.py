"""Tests for the supervised ↔ compliance_scores chronological pairing helper."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from typer.testing import CliRunner

from maimonedes.cli import app
from maimonedes.core.compliance import ComplianceScore
from maimonedes.settings import Settings
from maimonedes.storage.compliance import (
    ComplianceScoreRow,
    record_score,
)
from maimonedes.storage.llm_calls import (
    LLMCall,
    SUPERVISED_TO_SCORE_MAX_SECONDS,
    backfill_llm_call_ids,
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


# ---- FK preference + backfill (#44) ---------------------------------------


def _insert_score_with_fk(
    *,
    anchor_id: str,
    aggregate: float,
    scored_at: datetime,
    llm_call_id: int | None,
) -> None:
    record_score(
        ComplianceScore(
            anchor_id=anchor_id,
            policy_id="scope_of_practice",
            per_sub_condition={"flags_physician_review": aggregate},
            aggregate=aggregate,
            judge_model="judge:test",
            supervised_model="llama:test",
            scored_at=scored_at,
            llm_call_id=llm_call_id,
        )
    )


def test_pair_uses_fk_when_set(db: str) -> None:
    """A compliance_scores row with llm_call_id set is paired by FK."""
    base = datetime.now(timezone.utc)
    sup_a = _insert_supervised(
        response="text-A",
        timestamp=base,
        request_hash="A",
    )
    sup_b = _insert_supervised(
        response="text-B",
        timestamp=base + timedelta(seconds=10),
        request_hash="B",
    )
    # Score B is scored before its sup chronologically (insert order
    # diverges); FK path must still pair B's score to sup_b.
    _insert_score_with_fk(
        anchor_id="B",
        aggregate=0.6,
        scored_at=base + timedelta(seconds=15),
        llm_call_id=sup_b,
    )
    _insert_score_with_fk(
        anchor_id="A",
        aggregate=0.9,
        scored_at=base + timedelta(seconds=20),
        llm_call_id=sup_a,
    )
    pairs = pair_supervised_with_scores("scope_of_practice")
    assert len(pairs) == 2
    by_anchor = {p.anchor_id: p.supervised_text for p in pairs}
    assert by_anchor == {"A": "text-A", "B": "text-B"}


def test_pair_falls_back_to_heuristic_when_fk_unset(db: str) -> None:
    """Mixed DB: FK rows use FK, NULL rows use chronological pairing."""
    base = datetime.now(timezone.utc)
    sup_fk = _insert_supervised(
        response="text-FK",
        timestamp=base,
        request_hash="fk",
    )
    sup_legacy = _insert_supervised(
        response="text-LEGACY",
        timestamp=base + timedelta(seconds=10),
        request_hash="legacy",
    )
    # FK-bearing score (links to sup_fk).
    _insert_score_with_fk(
        anchor_id="FK",
        aggregate=0.8,
        scored_at=base + timedelta(seconds=2),
        llm_call_id=sup_fk,
    )
    # NULL-FK score; pairs chronologically with sup_legacy.
    _insert_score_with_fk(
        anchor_id="LEG",
        aggregate=0.4,
        scored_at=base + timedelta(seconds=12),
        llm_call_id=None,
    )
    pairs = pair_supervised_with_scores("scope_of_practice")
    assert len(pairs) == 2
    by_anchor = {p.anchor_id: p.supervised_text for p in pairs}
    assert by_anchor == {"FK": "text-FK", "LEG": "text-LEGACY"}


def test_prefer_fk_when_set_false_forces_heuristic(db: str) -> None:
    """`prefer_fk_when_set=False` ignores FK; uses heuristic for all rows."""
    base = datetime.now(timezone.utc)
    sup_a = _insert_supervised(
        response="text-A",
        timestamp=base,
        request_hash="A",
    )
    sup_b = _insert_supervised(
        response="text-B",
        timestamp=base + timedelta(seconds=10),
        request_hash="B",
    )
    # FKs swap A and B; heuristic mode ignores them.
    _insert_score_with_fk(
        anchor_id="A",
        aggregate=0.9,
        scored_at=base + timedelta(seconds=2),
        llm_call_id=sup_b,
    )
    _insert_score_with_fk(
        anchor_id="B",
        aggregate=0.5,
        scored_at=base + timedelta(seconds=12),
        llm_call_id=sup_a,
    )
    pairs = pair_supervised_with_scores(
        "scope_of_practice", prefer_fk_when_set=False
    )
    assert len(pairs) == 2
    # In heuristic mode, time order pairs A first (text-A) and B second (text-B).
    by_anchor = {p.anchor_id: p.supervised_text for p in pairs}
    assert by_anchor == {"A": "text-A", "B": "text-B"}


def test_backfill_empty_db_returns_zero(db: str) -> None:
    assert backfill_llm_call_ids() == 0


def test_backfill_pairs_null_rows(db: str) -> None:
    base = datetime.now(timezone.utc)
    sup_a = _insert_supervised(
        response="text-A", timestamp=base, request_hash="A"
    )
    sup_b = _insert_supervised(
        response="text-B",
        timestamp=base + timedelta(seconds=10),
        request_hash="B",
    )
    _insert_score_with_fk(
        anchor_id="A",
        aggregate=0.9,
        scored_at=base + timedelta(seconds=2),
        llm_call_id=None,
    )
    _insert_score_with_fk(
        anchor_id="B",
        aggregate=0.7,
        scored_at=base + timedelta(seconds=12),
        llm_call_id=None,
    )

    n = backfill_llm_call_ids()
    assert n == 2
    with get_session() as session:
        rows = (
            session.query(ComplianceScoreRow)
            .order_by(ComplianceScoreRow.id)
            .all()
        )
        assert rows[0].llm_call_id == sup_a
        assert rows[1].llm_call_id == sup_b


def test_backfill_idempotent_skips_already_set_rows(db: str) -> None:
    base = datetime.now(timezone.utc)
    sup_a = _insert_supervised(
        response="text-A", timestamp=base, request_hash="A"
    )
    sup_b = _insert_supervised(
        response="text-B",
        timestamp=base + timedelta(seconds=10),
        request_hash="B",
    )
    _insert_score_with_fk(
        anchor_id="A",
        aggregate=0.9,
        scored_at=base + timedelta(seconds=2),
        llm_call_id=sup_a,
    )
    _insert_score_with_fk(
        anchor_id="B",
        aggregate=0.7,
        scored_at=base + timedelta(seconds=12),
        llm_call_id=None,
    )
    # First pass: only the NULL row gets set.
    assert backfill_llm_call_ids() == 1
    # Second pass: nothing left to do.
    assert backfill_llm_call_ids() == 0
    with get_session() as session:
        rows = (
            session.query(ComplianceScoreRow)
            .order_by(ComplianceScoreRow.id)
            .all()
        )
        assert rows[0].llm_call_id == sup_a
        assert rows[1].llm_call_id == sup_b


def test_backfill_dry_run_does_not_write(db: str) -> None:
    base = datetime.now(timezone.utc)
    _insert_supervised(
        response="text-A", timestamp=base, request_hash="A"
    )
    _insert_score_with_fk(
        anchor_id="A",
        aggregate=0.9,
        scored_at=base + timedelta(seconds=2),
        llm_call_id=None,
    )
    n_dry = backfill_llm_call_ids(dry_run=True)
    assert n_dry == 1
    with get_session() as session:
        row = session.query(ComplianceScoreRow).one()
        assert row.llm_call_id is None


def test_backfill_does_not_double_pair_already_used_supervised(db: str) -> None:
    """A supervised row already claimed by an FK is not reused."""
    base = datetime.now(timezone.utc)
    sup_a = _insert_supervised(
        response="text-A", timestamp=base, request_hash="A"
    )
    sup_b = _insert_supervised(
        response="text-B",
        timestamp=base + timedelta(seconds=10),
        request_hash="B",
    )
    # Score 1 → FK to sup_a.
    _insert_score_with_fk(
        anchor_id="A",
        aggregate=0.9,
        scored_at=base + timedelta(seconds=12),
        llm_call_id=sup_a,
    )
    # Score 2 → NULL; backfill should pair to sup_b (NOT reuse sup_a).
    _insert_score_with_fk(
        anchor_id="B",
        aggregate=0.7,
        scored_at=base + timedelta(seconds=14),
        llm_call_id=None,
    )
    n = backfill_llm_call_ids()
    assert n == 1
    with get_session() as session:
        rows = (
            session.query(ComplianceScoreRow)
            .order_by(ComplianceScoreRow.id)
            .all()
        )
        assert rows[0].llm_call_id == sup_a
        assert rows[1].llm_call_id == sup_b


def test_cli_backfill_llm_call_ids_runs_against_seeded_db(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    base = datetime.now(timezone.utc)
    _insert_supervised(
        response="text-A", timestamp=base, request_hash="A"
    )
    _insert_score_with_fk(
        anchor_id="A",
        aggregate=0.9,
        scored_at=base + timedelta(seconds=2),
        llm_call_id=None,
    )

    monkeypatch.setenv("DATABASE_URL", db)
    dry = runner.invoke(app, ["backfill-llm-call-ids", "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "would update 1" in dry.output

    real = runner.invoke(app, ["backfill-llm-call-ids"])
    assert real.exit_code == 0, real.output
    assert "updated 1" in real.output

    # Idempotent: second run is a no-op.
    again = runner.invoke(app, ["backfill-llm-call-ids"])
    assert again.exit_code == 0
    assert "updated 0" in again.output


# ---- existing heuristic-only behaviour (unchanged) ------------------------


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
