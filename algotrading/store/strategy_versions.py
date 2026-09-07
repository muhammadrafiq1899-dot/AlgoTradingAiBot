"""Controlled strategy-version release.

Exactly ONE strategy version is `active` at a time. Promoting a new version
retires the current active one and marks the newcomer active. This is the
"controlled release" seam the AI approval flow drives: an approved proposal is
shadow-backtested, then `create_new_version` + `promote_to_active` swap the
live strategy.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.db.models import Strategy


def latest_version(session: Session, name: str) -> Strategy | None:
    """Return the highest-version row for `name`, or None."""
    return session.execute(
        select(Strategy)
        .where(Strategy.name == name)
        .order_by(Strategy.version.desc())
    ).scalars().first()


def active_version(session: Session, name: str) -> Strategy | None:
    """Return the currently active version of `name`, or None."""
    return session.execute(
        select(Strategy).where(
            Strategy.name == name, Strategy.status == "active"
        )
    ).scalars().first()


def create_new_version(
    session: Session,
    name: str,
    params: dict[str, Any],
    description: str = "",
) -> Strategy:
    """Create the next version of `name` in `draft` status.

    Version numbers increment by 1 from the current latest. The new row is
    NOT auto-promoted — callers decide when (and whether) to release it.
    """
    latest = latest_version(session, name)
    next_version = (latest.version if latest else 0) + 1
    row = Strategy(
        name=name,
        version=next_version,
        params=json.dumps(params),
        status="draft",
        description=description,
    )
    session.add(row)
    session.commit()
    return row


def promote_to_active(session: Session, strategy: Strategy) -> Strategy:
    """Make `strategy` the single active version of its name.

    Retires the current active version (if any) of the same name, then marks
    `strategy` active and stamps `approved_at`. Returns the promoted row.
    """
    current = active_version(session, strategy.name)
    if current is not None and current.id != strategy.id:
        current.status = "retired"
    if strategy.status != "active":
        strategy.status = "active"
        strategy.approved_at = strategy.approved_at or datetime.now(timezone.utc)
    session.commit()
    return strategy
