"""Strategy seeding: populate an empty database with starter strategies.

Shared by `scripts/init_db.py` (explicit init) and `main.py` (auto-seed on
first startup). Seeding is idempotent: it runs once, guarded by the
`strategies_seeded` meta key.
"""
from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from algotrading.config import load_strategy_definitions
from algotrading.db.models import Meta, Strategy

log = logging.getLogger(__name__)

SEEDED_KEY = "strategies_seeded"


def strategies_seeded(session: Session) -> bool:
    return session.query(Meta).filter(Meta.key == SEEDED_KEY).first() is not None


def seed_strategies(session: Session) -> int:
    """Seed starter strategies from config/strategies.yaml.

    The first strategy becomes the initial active version; the rest are
    `approved` (ready but not live). Returns the number of strategies seeded,
    or 0 if already seeded.
    """
    if strategies_seeded(session):
        log.info("Strategies already seeded; skipping.")
        return 0

    defs = load_strategy_definitions()
    for i, sd in enumerate(defs):
        params = {p.name: p.default for p in sd.params}
        session.add(
            Strategy(
                name=sd.name,
                version=1,
                params=json.dumps(params),
                status="active" if i == 0 else "approved",
                description=sd.description,
            )
        )
    session.add(Meta(key=SEEDED_KEY, value="1"))
    session.commit()
    log.info("Seeded %d strategies from config/strategies.yaml.", len(defs))
    return len(defs)


def ensure_seeded(session: Session) -> int:
    """Seed if the database is empty; safe to call at every startup."""
    if strategies_seeded(session):
        return 0
    return seed_strategies(session)