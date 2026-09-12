"""Ledger: event-sourced trade lifecycle.

The single source of truth is the immutable `order_events` log. Positions and
closed trades are derived views, rebuilt from events on startup and maintained
incrementally as events arrive. This makes recovery after a crash or restart
deterministic: replay the events and the state is exactly as it was.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.db.models import OrderEvent, Position, Signal, Trade, TradeIntent, utcnow

log = logging.getLogger(__name__)


class Ledger:
    def __init__(self, session: Session) -> None:
        self._session = session

    # --- event recording ---

    def record_event(self, intent_id: int, event_type: str, payload: dict[str, Any] | None = None) -> None:
        ev = OrderEvent(
            intent_id=intent_id,
            event_type=event_type,
            payload_json=__import__("json").dumps(payload or {}),
        )
        self._session.add(ev)

    # --- intent transitions ---

    def mark_sent(self, intent: TradeIntent, exchange_order_id: str) -> None:
        intent.status = "sent"
        intent.exchange_order_id = exchange_order_id
        self.record_event(intent.id, "accepted", {"exchange_order_id": exchange_order_id})

    def mark_filled(self, intent: TradeIntent, fill_price: float, qty: float, fee: float = 0.0) -> None:
        intent.status = "filled"
        intent.avg_fill_price = fill_price
        intent.fee = fee
        self.record_event(
            intent.id,
            "fill",
            {"fill_price": fill_price, "qty": qty, "fee": fee},
        )
        self._apply_fill(intent, fill_price, qty, fee)

    def mark_canceled(self, intent: TradeIntent, reason: str = "") -> None:
        intent.status = "canceled"
        self.record_event(intent.id, "canceled", {"reason": reason})

    def mark_failed(self, intent: TradeIntent, error: str) -> None:
        intent.status = "failed"
        self.record_event(intent.id, "rejected", {"error": error})

    def mark_risk_skipped(self, intent: TradeIntent, reason: str) -> None:
        intent.status = "skipped"
        self.record_event(intent.id, "risk_skipped", {"reason": reason})

    def mark_trailing_stop_adjusted(self, symbol: str, new_stop_price: float) -> None:
        """Record a trailing stop adjustment event (for audit trail)."""
        # Find the latest fill intent for this symbol to attach the event
        from algotrading.db.models import TradeIntent
        intent = self._session.execute(
            select(TradeIntent)
            .where(TradeIntent.symbol == symbol, TradeIntent.side == "buy", TradeIntent.status == "filled")
            .order_by(TradeIntent.ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        if intent:
            self.record_event(
                intent.id,
                "trailing_stop_adjusted",
                {"symbol": symbol, "new_stop_price": new_stop_price},
            )

    # --- derived state application (idempotent by intent status) ---

    def _apply_fill(self, intent: TradeIntent, fill_price: float, qty: float, fee: float) -> None:
        symbol = intent.symbol
        pos = self._session.execute(
            select(Position).where(Position.symbol == symbol)
        ).scalar_one_or_none()

        if intent.side == "buy":
            if pos is None:
                pos = Position(symbol=symbol, qty=0.0, avg_price=0.0, opened_at=utcnow())
                self._session.add(pos)
            total_cost = pos.qty * pos.avg_price + qty * fill_price
            pos.qty += qty
            pos.avg_price = total_cost / pos.qty if pos.qty else 0.0
        else:  # sell closes the whole position in v1 spot
            if pos is None or pos.qty <= 0:
                log.warning("sell fill for %s with no open position", symbol)
                return
            realized = (fill_price - pos.avg_price) * pos.qty - fee
            strategy_id = None
            if intent.signal_id is not None:
                sig = self._session.get(Signal, intent.signal_id)
                strategy_id = sig.strategy_id if sig else None
            trade = Trade(
                symbol=symbol,
                entry_qty=pos.qty,
                entry_avg_price=pos.avg_price,
                exit_avg_price=fill_price,
                realized_pnl=realized,
                fees=fee,
                opened_at=pos.opened_at,
                closed_at=utcnow(),
                strategy_id=strategy_id,
            )
            self._session.add(trade)
            pos.qty = 0.0
            pos.avg_price = 0.0

    # --- recovery / rebuild ---

    def rebuild_positions(self) -> None:
        """Recompute all positions and trades from the event log.

        Deletes derived rows and replays `fill` events grouped by symbol. Called
        once at startup; safe because intents carry their side and fill price.
        """
        self._session.query(Position).delete()
        self._session.query(Trade).delete()

        fills = self._session.execute(
            select(TradeIntent, OrderEvent)
            .join(OrderEvent, OrderEvent.intent_id == TradeIntent.id)
            .where(OrderEvent.event_type == "fill")
        ).all()
        by_symbol: dict[str, list] = {}
        for intent, ev in fills:
            payload = __import__("json").loads(ev.payload_json or "{}")
            by_symbol.setdefault(intent.symbol, []).append(
                (intent, payload, ev.ts)
            )
        # Replay in ts order per symbol.
        for symbol, items in by_symbol.items():
            items.sort(key=lambda x: x[2] or 0)
            for intent, payload, _ts in items:
                self._apply_fill(
                    intent,
                    float(payload.get("fill_price", 0)),
                    float(payload.get("qty", intent.qty)),
                    float(payload.get("fee", 0)),
                )
        self._session.commit()
        log.info("rebuild: %d symbols with fills", len(by_symbol))
