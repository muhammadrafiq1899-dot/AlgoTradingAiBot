"""Controlled strategy-version release.

Exactly ONE strategy version is `active` at a time. Promoting a new version
retires the current active one and marks the newcomer active. This is the
"controlled release" seam the AI approval flow drives: an approved proposal is
shadow-backtested, then `create_new_version` + `promote_to_active` swap the
live strategy.

Strategy *code* lives on disk as a plugin file (see
:mod:`algotrading.strategy.plugins`); these rows own the version/status history
and the parameter snapshot. Only routes into this module are the human-approved
recommendation flow — the AI can propose, never apply.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.db.models import Strategy
from algotrading.strategy.plugins import StrategyPluginError, get_default_loader
from algotrading.strategy.validation import CodeValidationError

log = logging.getLogger(__name__)


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


def create_new_strategy(
    session: Session,
    name: str,
    template: str,
    params: dict[str, Any],
    param_schema: list[dict[str, Any]],
    indicator_deps: list[str],
    description: str = "",
    test_template: str = "",
) -> Strategy:
    """Register a brand-new AI/authored strategy: validate → write file → version.

    Refuses to shadow a built-in or an existing plugin. The plugin file is the
    durable home of the code, so the strategy survives restarts and can be
    hand-edited; the DB row carries version/status/params.
    """
    loader = get_default_loader()
    try:
        loader.write_strategy(
            name,
            template,
            description=description,
            params=param_schema or None,
            indicator_deps=indicator_deps or None,
            overwrite=False,
        )
    except (StrategyPluginError, CodeValidationError) as exc:
        # A write that fails to load must not leave the file behind: the name
        # would then be permanently unusable ("already exists") even after the
        # author fixes the code. Only clean up when nothing is registered — a
        # working plugin refuses the write before it reaches the filesystem.
        if loader.get_class(name) is None:
            try:
                loader.path_for(name).unlink(missing_ok=True)
            except OSError:  # pragma: no cover - cleanup is best effort
                log.warning("could not clean up failed strategy file %r", name)
        raise ValueError(f"cannot create strategy {name!r}: {exc}") from exc

    log.info("created strategy plugin %r", name)
    return create_new_version(session, name, params, description)


def update_strategy_code(
    session: Session,
    name: str,
    template: str,
    params: dict[str, Any] | None = None,
    description: str = "",
    param_schema: list[dict[str, Any]] | None = None,
    indicator_deps: list[str] | None = None,
) -> Strategy:
    """Edit the code of an existing plugin strategy, then cut a new version.

    Built-in strategies cannot be redefined here — use a parameter change (or a
    new strategy name) so shipped code isn't silently replaced.
    """
    loader = get_default_loader()
    if loader.get_class(name) is None:
        raise ValueError(
            f"{name!r} is not an editable plugin strategy "
            "(built-ins are changed via param_change)"
        )
    try:
        loader.write_strategy(
            name,
            template,
            description=description or (loader.info(name).description if loader.info(name) else ""),
            params=param_schema if param_schema is not None else (
                list(loader.info(name).params) if loader.info(name) else None
            ),
            indicator_deps=indicator_deps if indicator_deps is not None else (
                list(loader.info(name).indicator_deps) if loader.info(name) else None
            ),
            overwrite=True,
        )
    except (StrategyPluginError, CodeValidationError) as exc:
        raise ValueError(f"cannot update strategy {name!r}: {exc}") from exc

    latest = latest_version(session, name)
    effective_params = params if params is not None else (
        json.loads(latest.params or "{}") if latest else {}
    )
    log.info("updated strategy plugin %r", name)
    return create_new_version(session, name, effective_params, description or f"code update of {name}")


def promote_to_active(session: Session, strategy: Strategy) -> Strategy:
    """Make `strategy` THE active version — the only one.

    Retires every other active row, not just the same name's: the engine picks
    the newest active row without filtering by name (`StrategyEngine.
    get_active_strategy` orders by version), so leaving another strategy active
    would silently keep it in charge of a freshly approved one. Stamps
    `approved_at` and returns the promoted row.
    """
    others = session.execute(
        select(Strategy).where(
            Strategy.status == "active", Strategy.id != strategy.id
        )
    ).scalars().all()
    for row in others:
        row.status = "retired"
    if strategy.status != "active":
        strategy.status = "active"
        strategy.approved_at = strategy.approved_at or datetime.now(timezone.utc)
    session.commit()
    return strategy
