"""Database engine/session setup.

SQLite in WAL mode for safe concurrent reads during async writes. A small
schema_version migration hook lives here (see SCHEMA_VERSION).
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from algotrading.db.models import Base, Meta

log = logging.getLogger(__name__)

# Bump when you change models and need a migration. Simple, versioned migrations
# can be added to apply() as the schema evolves.
SCHEMA_VERSION = 3


@lru_cache(maxsize=1)
def get_engine(db_path: str = "data/algotrading.db"):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False: async single-loop usage shares connections across tasks.
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False, "timeout": 30},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout = 10000")  # 10s wait
        cur.close()

    return engine


@lru_cache(maxsize=1)
def get_session_factory(db_path: str = "data/algotrading.db"):
    engine = get_engine(db_path)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(db_path: str = "data/algotrading.db") -> None:
    """Create tables if missing, run migrations, and set schema_version."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        row = session.get(Meta, "schema_version")
        current_version = int(row.value) if row else 0
        
        # Run migrations if needed
        if current_version < SCHEMA_VERSION:
            _run_migrations(session, current_version, SCHEMA_VERSION)
        
        if row is None:
            session.add(Meta(key="schema_version", value=str(SCHEMA_VERSION)))
        else:
            row.value = str(SCHEMA_VERSION)
        session.commit()


def _run_migrations(session: Session, from_version: int, to_version: int) -> None:
    """Apply schema migrations incrementally."""
    from sqlalchemy import text
    
    if from_version < 2 <= to_version:
        # Migration v1 -> v2: Add trailing_stop_price and highest_price to positions table
        # Check if columns already exist (for fresh databases created by create_all)
        cols = session.execute(text("PRAGMA table_info(positions)")).fetchall()
        col_names = {c[1] for c in cols}  # c[1] is column name
        
        if "trailing_stop_price" not in col_names:
            session.execute(text("ALTER TABLE positions ADD COLUMN trailing_stop_price FLOAT"))
        if "highest_price" not in col_names:
            session.execute(text("ALTER TABLE positions ADD COLUMN highest_price FLOAT"))
        session.commit()
        log.info("Applied migration v1 -> v2: added trailing_stop_price and highest_price to positions table")

    if from_version < 3 <= to_version:
        # Migration v2 -> v3: advisory decision log (ai_decision_log).
        # create_all() above already adds a *missing* table to an existing DB,
        # so this is the safety net for the two cases create_all cannot cover:
        # a table that exists but predates a column, and a DB whose version row
        # was written by a build that had the table but no index.
        _ensure_decision_log(session)
        log.info("Applied migration v2 -> v3: ensured ai_decision_log table")


def _ensure_decision_log(session: Session) -> None:
    """Create/patch ``ai_decision_log`` without touching existing data.

    Additive only: either the table is created, or missing columns are appended
    with ``ALTER TABLE ... ADD COLUMN``. Nothing is ever dropped or rewritten —
    an existing database keeps every row it had.
    """
    from sqlalchemy import text

    from algotrading.db.models import AIDecisionLog

    bind = session.get_bind()
    cols = session.execute(text("PRAGMA table_info(ai_decision_log)")).fetchall()
    if not cols:
        # ``__table__`` is a real sqlalchemy Table at runtime (the declared type
        # on the declarative class is FromClause, hence the ignore).
        AIDecisionLog.__table__.create(bind, checkfirst=True)  # type: ignore[attr-defined]
        session.commit()
        return

    existing = {c[1] for c in cols}
    for column in AIDecisionLog.__table__.columns:
        if column.name in existing:
            continue
        # SQLite types come from the ORM column so the added column matches a
        # freshly created table exactly.
        session.execute(
            text(f"ALTER TABLE ai_decision_log ADD COLUMN {column.name} {column.type}")
        )
        log.info("ai_decision_log: added missing column %s", column.name)
    session.commit()


def get_schema_version(db_path: str = "data/algotrading.db") -> int:
    engine = get_engine(db_path)
    with Session(engine) as session:
        row = session.get(Meta, "schema_version")
        return int(row.value) if row else 0


def get_session(db_path: str = "data/algotrading.db") -> Session:
    return get_session_factory(db_path)()
