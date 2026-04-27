"""Engine, session factory, and a transactional context manager."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from maimonedes.settings import Settings, get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _build_engine(url: str, **engine_kwargs: Any) -> Engine:
    # SQLite + multi-thread test runners need this; harmless elsewhere
    connect_args: dict[str, Any] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, future=True, connect_args=connect_args, **engine_kwargs)


def init_engine(settings: Settings | None = None, **engine_kwargs: Any) -> Engine:
    """Create (or recreate) the module-level engine + session factory."""
    global _engine, _SessionLocal
    settings = settings or get_settings()
    _engine = _build_engine(settings.database_url, **engine_kwargs)
    _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        init_engine()
    assert _engine is not None
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    return _SessionLocal


@contextmanager
def get_session() -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_for_tests() -> None:
    """Drop module-level state so tests can rebuild against a fresh URL."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
