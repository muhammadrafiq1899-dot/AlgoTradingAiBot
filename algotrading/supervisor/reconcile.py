"""Reconcile local trade intents with exchange state.

Compares intents the bot believes are pending/sent against the exchange's
open orders (matched by idempotency key). Runs after a reconnect or restart
so the local ledger cannot silently diverge from the venue.

The gateway needs a `get_open_orders(symbol) -> list[dict]` method returning
dicts with a `client_order_id` key (implemented by the live gateway; the
paper gateway reports empty open orders).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.db.models import TradeIntent


def reconcile(session: Session, gateway, symbols: list[str]) -> list[str]:
    """Return a list of drift descriptions; empty list means reconciled."""
    drift: list[str] = []
    intents = session.execute(
        select(TradeIntent).where(TradeIntent.status.in_(["sent", "pending"]))
    ).scalars().all()

    by_symbol: dict[str, list[TradeIntent]] = {}
    for intent in intents:
        by_symbol.setdefault(intent.symbol, []).append(intent)

    for symbol, pending_intents in by_symbol.items():
        if symbol not in symbols:
            continue
        try:
            remote = gateway.get_open_orders(symbol)
        except Exception as exc:  # noqa: BLE001 - network/live errors are drift
            drift.append(f"{symbol}: cannot reach exchange ({exc})")
            continue
        remote_ids = {o.get("client_order_id") for o in remote}
        for intent in pending_intents:
            if intent.idempotency_key not in remote_ids:
                drift.append(
                    f"{symbol} intent {intent.idempotency_key} has no matching "
                    f"exchange order (status={intent.status})"
                )
    return drift


def reconcile_and_report(session: Session, gateway, symbols: list[str]) -> str:
    """Run a reconcile pass and return a human-readable one-liner."""
    drift = reconcile(session, gateway, symbols)
    return "\n".join(drift) if drift else "OK: no drift"
