"""Database engine/session setup.

SQLite in WAL mode for safe concurrent reads during async writes. A small
schema_version migration hook lives here (see SCHEMA_VERSION).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from algotrading.db.models import Base, Meta

# Bump when you change models and need a migration. Simple, versioned migrations
# can be added to apply() as the schema evolves.
SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def get_engine(db_path: str = "data/algotrading.db"):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: async single-loop usage shares connections across tasks.
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


@lru_cache(maxsize=1)
def get_session_factory(db_path: str = "data/algotrading.db"):
    engine = get_engine(db_path)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(db_path: str = "data/algotrading.db") -> None:
    """Create tables if missing and set schema_version."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        row = session.get(Meta, "schema_version")
        if row is None:
            session.add(Meta(key="schema_version", value=str(SCHEMA_VERSION)))
            session.commit()


def get_schema_version(db_path: str = "data/algotrading.db") -> int:
    engine = get_engine(db_path)
    with Session(engine) as session:
        row = session.get(Meta, "schema_version")
        return int(row.value) if row else 0


def get_session(db_path: str = "data/algotrading.db") -> Session:
    return get_session_factory(db_path)()
