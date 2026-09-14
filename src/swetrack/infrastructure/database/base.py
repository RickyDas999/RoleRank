"""SQLAlchemy engine/session setup for local persistence.

Defaults to a file-based SQLite database under ``var/`` (gitignored,
generated locally) so learning history survives across runs during
development, at $0 cost and with zero external services. Moving to
PostgreSQL later means setting ``SWETRACK_DATABASE_URL`` and adding a
driver dependency -- the ORM models and service layers built on top of
``Base``/``get_sessionmaker`` do not change.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

ROOT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_DATABASE_URL = f"sqlite:///{ROOT_DIR / 'var' / 'swetrack.db'}"


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model in the project."""


def get_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine, defaulting to the local SQLite file store.

    ``database_url`` overrides the ``SWETRACK_DATABASE_URL`` environment
    variable, which in turn overrides the local SQLite default. Pass
    ``"sqlite:///:memory:"`` for isolated, disk-free tests.
    """
    url = database_url or os.environ.get("SWETRACK_DATABASE_URL", DEFAULT_DATABASE_URL)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    if url.startswith("sqlite") and ":memory:" not in url:
        (ROOT_DIR / "var").mkdir(parents=True, exist_ok=True)
    return create_engine(url, connect_args=connect_args)


def get_sessionmaker(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create all tables from every imported model. Idempotent."""
    Base.metadata.create_all(engine)
