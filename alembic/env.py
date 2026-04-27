"""Alembic environment.

Reads the connection URL from `maimonedes.settings.Settings` rather than
from alembic.ini, so application code and migrations share a single
source of truth (and so tests can override `DATABASE_URL` via env vars).
"""
from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from maimonedes.settings import get_settings
from maimonedes.storage.models import Base

config = context.config

if config.config_file_name is not None:
    # `disable_existing_loggers=False` keeps pytest's caplog handler and
    # any application-level loggers attached when migrations run inside
    # the test process.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Inject our application-level URL; never hard-coded in alembic.ini.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=url is not None and url.startswith("sqlite"),
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        future=True,
    )
    with connectable.connect() as connection:
        is_sqlite = connection.dialect.name == "sqlite"
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=is_sqlite,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
