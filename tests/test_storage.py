"""Storage smoke tests.

Spins up an in-memory SQLite, runs Alembic migrations against it,
opens a session through the project's `get_session` context manager,
and inserts/reads a row. Also verifies `alembic downgrade base` is
clean.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from maimonedes import storage
from maimonedes.settings import Settings
from maimonedes.storage.models import SchemaVersion
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
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "test.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    yield url
    reset_engine_for_tests()


def test_storage_module_importable() -> None:
    assert storage is not None


def test_alembic_upgrade_then_downgrade_is_clean(tmp_db: str) -> None:
    cfg = _alembic_cfg(tmp_db)

    command.upgrade(cfg, "head")
    from maimonedes.storage.repo import get_engine

    insp = inspect(get_engine())
    assert "schema_version" in insp.get_table_names()

    command.downgrade(cfg, "base")
    insp = inspect(get_engine())
    assert "schema_version" not in insp.get_table_names()


def test_session_inserts_and_reads_row(tmp_db: str) -> None:
    cfg = _alembic_cfg(tmp_db)
    command.upgrade(cfg, "head")

    with get_session() as session:
        session.add(SchemaVersion(label="phase-0-baseline"))

    with get_session() as session:
        rows = session.query(SchemaVersion).all()
        assert len(rows) == 1
        assert rows[0].label == "phase-0-baseline"
        assert rows[0].applied_at is not None


def test_session_rolls_back_on_error(tmp_db: str) -> None:
    cfg = _alembic_cfg(tmp_db)
    command.upgrade(cfg, "head")

    with pytest.raises(RuntimeError):
        with get_session() as session:
            session.add(SchemaVersion(label="will-be-rolled-back"))
            raise RuntimeError("boom")

    with get_session() as session:
        rows = session.query(SchemaVersion).all()
        assert rows == []
