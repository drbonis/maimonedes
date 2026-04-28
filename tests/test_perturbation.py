"""Tests for PerturbationProbe model + storage round-trip + migrations."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from maimonedes.core.compliance import ComplianceScore
from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.settings import Settings
from maimonedes.storage.compliance import (
    latest_anchor_baseline,
    latest_score_per_anchor,
    record_score,
)
from maimonedes.storage.perturbations import (
    get_perturbation_row_id,
    record_perturbation,
    recent_perturbations,
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
    db_path = tmp_path / "p2.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


# ---- model invariants ------------------------------------------------------


def test_perturbation_probe_id_auto_derived_from_anchor_and_label() -> None:
    p = PerturbationProbe(
        anchor_id="A1",
        scenario="x",
        perturbation_kind="authority",
        transform_label="authority:senior_cardiologist",
    )
    assert p.id == "A1#authority:senior_cardiologist"
    assert p.kind == "perturbation"
    assert p.policy_id == "scope_of_practice"


def test_perturbation_probe_explicit_id_wins() -> None:
    p = PerturbationProbe(
        id="custom-id",
        anchor_id="A1",
        scenario="x",
        perturbation_kind="paraphrase",
        transform_label="paraphrase:0",
    )
    assert p.id == "custom-id"


def test_perturbation_probe_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        PerturbationProbe(
            anchor_id="A1",
            scenario="x",
            perturbation_kind="cosmic_ray",  # type: ignore[arg-type]
            transform_label="x",
        )


def test_perturbation_probe_is_frozen() -> None:
    p = PerturbationProbe(
        anchor_id="A1",
        scenario="x",
        perturbation_kind="paraphrase",
        transform_label="paraphrase:0",
    )
    with pytest.raises(Exception):
        p.scenario = "mutated"  # type: ignore[misc]


# ---- migration shape -------------------------------------------------------


def test_migration_creates_perturbation_probes_with_indexes(db: str) -> None:
    insp = inspect(get_engine())
    assert "perturbation_probes" in insp.get_table_names()
    indexes = {ix["name"] for ix in insp.get_indexes("perturbation_probes")}
    assert "ix_perturbation_probes_anchor_kind" in indexes
    assert "ix_perturbation_probes_anchor_label" in indexes


def test_migration_extends_compliance_scores_with_perturbation_id(db: str) -> None:
    insp = inspect(get_engine())
    cols = {c["name"] for c in insp.get_columns("compliance_scores")}
    assert "perturbation_id" in cols
    assert "probe_role" in cols
    indexes = {ix["name"] for ix in insp.get_indexes("compliance_scores")}
    assert "ix_compliance_scores_perturbation_id" in indexes
    assert "ix_compliance_scores_role_anchor" in indexes


def test_alembic_downgrade_drops_phase_2_artefacts(db: str) -> None:
    cfg = _alembic_cfg(db)
    command.downgrade(cfg, "0003_compliance_scores")
    insp = inspect(get_engine())
    assert "perturbation_probes" not in insp.get_table_names()
    cols = {c["name"] for c in insp.get_columns("compliance_scores")}
    assert "perturbation_id" not in cols
    assert "probe_role" not in cols


# ---- repository round-trip -------------------------------------------------


def _make_perturbation(
    anchor: str = "A1",
    label: str = "authority:senior_cardiologist",
) -> PerturbationProbe:
    return PerturbationProbe(
        anchor_id=anchor,
        scenario=f"perturbed scenario for {label}",
        perturbation_kind="authority",
        transform_label=label,
        generator_metadata={"prefix": "The senior cardiologist requests:"},
    )


def test_record_and_recent_perturbation_round_trip_preserves_metadata(
    db: str,
) -> None:
    probe = _make_perturbation()
    row_id = record_perturbation(probe)
    assert row_id > 0

    rows = recent_perturbations("A1")
    assert len(rows) == 1
    out = rows[0]
    assert out.anchor_id == "A1"
    assert out.transform_label == "authority:senior_cardiologist"
    assert out.perturbation_kind == "authority"
    assert out.generator_metadata == {
        "prefix": "The senior cardiologist requests:"
    }


def test_recent_perturbations_filters_by_kind(db: str) -> None:
    record_perturbation(_make_perturbation(label="authority:gp"))
    record_perturbation(
        PerturbationProbe(
            anchor_id="A1",
            scenario="paraphrase scenario",
            perturbation_kind="paraphrase",
            transform_label="paraphrase:0",
        )
    )
    paraphrases = recent_perturbations("A1", perturbation_kind="paraphrase")
    assert len(paraphrases) == 1
    assert paraphrases[0].perturbation_kind == "paraphrase"


def test_get_perturbation_row_id_round_trip(db: str) -> None:
    probe = _make_perturbation()
    row_id = record_perturbation(probe)
    looked_up = get_perturbation_row_id("A1", probe.transform_label)
    assert looked_up == row_id
    assert get_perturbation_row_id("A1", "no-such-label") is None


# ---- compliance-score linking ----------------------------------------------


def _compliance(
    anchor: str = "A1",
    aggregate: float = 0.6,
    perturbation_id: int | None = None,
    role: str = "anchor",
) -> ComplianceScore:
    return ComplianceScore(
        anchor_id=anchor,
        policy_id="scope_of_practice",
        per_sub_condition={"flags_physician_review": 1.0},
        aggregate=aggregate,
        judge_model="judge:test",
        supervised_model="llama:test",
        perturbation_id=perturbation_id,
        probe_role=role,  # type: ignore[arg-type]
    )


def test_record_score_persists_perturbation_id_and_probe_role(db: str) -> None:
    probe_row = record_perturbation(_make_perturbation())
    record_score(
        _compliance(perturbation_id=probe_row, role="perturbation", aggregate=0.4)
    )

    from maimonedes.storage.compliance import recent_scores

    rows = recent_scores("A1")
    assert len(rows) == 1
    assert rows[0].probe_role == "perturbation"
    assert rows[0].perturbation_id == probe_row


def test_compliance_score_with_orphan_perturbation_id_rejected_by_fk(
    db: str,
) -> None:
    # No perturbation row 999_999 exists → FK violation.
    with pytest.raises(IntegrityError):
        record_score(
            _compliance(
                perturbation_id=999_999,
                role="perturbation",
                aggregate=0.3,
            )
        )


def test_latest_score_per_anchor_filters_out_perturbation_rows(db: str) -> None:
    record_score(_compliance(anchor="A1", aggregate=0.5, role="anchor"))
    time.sleep(0.01)
    probe_row = record_perturbation(_make_perturbation())
    record_score(
        _compliance(
            anchor="A1",
            aggregate=0.1,
            perturbation_id=probe_row,
            role="perturbation",
        )
    )

    latest = latest_score_per_anchor()
    # The perturbation score has aggregate 0.1; if it leaked into the
    # dashboard table that's the score we'd see. Filtering by
    # probe_role="anchor" keeps us at the 0.5 anchor baseline.
    assert latest["A1"].aggregate == pytest.approx(0.5)
    assert latest["A1"].probe_role == "anchor"


def test_latest_anchor_baseline_returns_anchor_only(db: str) -> None:
    record_score(_compliance(anchor="A1", aggregate=0.7, role="anchor"))
    time.sleep(0.01)
    probe_row = record_perturbation(_make_perturbation())
    record_score(
        _compliance(
            anchor="A1",
            aggregate=0.2,
            perturbation_id=probe_row,
            role="perturbation",
        )
    )

    baseline = latest_anchor_baseline("A1")
    assert baseline is not None
    assert baseline.aggregate == pytest.approx(0.7)
    assert baseline.probe_role == "anchor"


def test_latest_anchor_baseline_returns_none_when_only_perturbations_exist(
    db: str,
) -> None:
    probe_row = record_perturbation(_make_perturbation())
    record_score(
        _compliance(
            anchor="A1",
            aggregate=0.2,
            perturbation_id=probe_row,
            role="perturbation",
        )
    )
    assert latest_anchor_baseline("A1") is None
