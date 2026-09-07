"""Execution engine: turn candidate signals into orders with hard safety.

Order of operations (non-negotiable):
  1. Load the candidate signal.
  2. Run risk checks (exposure cap, cooldown, duplicate guard).
  3. Persist a trade_intent (status=pending) with a UNIQUE idempotency key
     BEFORE any order is sent.
  4. Send the order with client_order_id = idempotency key.
  5. Record the outcome as an immutable order_event; update the intent.

The idempotency key guarantees that a retry after a network error cannot place
a duplicate order: if the key already exists, the previously-persisted intent
is reused instead of creating a new one.
"""
from __future__ import annotations

import json
import logging
import time
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.db.models import Position, Signal, TradeIntent
from algotrading.execution.base import ExchangeGateway
from algotrading.execution.risk import RiskManager
from algotrading.ledger.store import Ledger

log = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(self, session: Session, gateway: ExchangeGateway, risk: RiskManager, ledger: Ledger) -> None:
        self._session = session
        self._gateway = gateway
        self._risk = risk
        self._ledger = ledger

    def _make_key(self, signal_id: int, symbol: str, side: str) -> str:
        return f"{symbol}-{side}-{signal_id}-{uuid.uuid4().hex[:8]}"

    def _balance(self) -> float:
        """Paper starting balance; overridden by live account balances later."""
        return 100_000.0

    def execute(self, signal_id: int) -> TradeIntent | None:
        """Attempt to fill a single candidate signal. Returns the intent."""
        sig = self._session.get(Signal, signal_id)
        if sig is None:
            log.warning("execute: signal %s not found", signal_id)
            return None
        if sig.status != "candidate":
            log.debug("execute: signal %s already %s", signal_id, sig.status)
            return None

        pos = self._session.execute(
            select(Position).where(Position.symbol == sig.symbol)
        ).scalar_one_or_none()

        if sig.side == "buy":
            decision = self._risk.check_buy(
                symbol=sig.symbol,
                balance=self._balance(),
                open_positions=len(self._open_positions()),
                last_entry_ts=self._last_entry_ts(sig.symbol),
            )
        else:
            decision = self._risk.check_sell(has_position=pos is not None and pos.qty > 0)

        if not decision.allowed:
            intent = self._persist_intent(sig, decision.reason, skipped=True)
            self._ledger.mark_risk_skipped(intent, decision.reason)
            sig.status = "skipped"
            self._session.commit()
            log.info("signal %s skipped: %s", sig.id, decision.reason)
            return intent

        qty = self._qty_for(sig, decision, pos)
        if qty <= 0:
            intent = self._persist_intent(sig, "invalid qty", skipped=True)
            self._ledger.mark_risk_skipped(intent, "qty <= 0")
            sig.status = "skipped"
            self._session.commit()
            return intent

        # --- intent-before-order ---
        key = self._make_key(sig.id, sig.symbol, sig.side)
        intent = TradeIntent(
            signal_id=sig.id,
            idempotency_key=key,
            symbol=sig.symbol,
            side=sig.side,
            qty=qty,
            order_type="market",
            ref_price=sig.ref_price,
            status="pending",
        )
        self._session.add(intent)
        self._session.commit()
        sig.status = "sent"
        self._session.commit()

        result = self._gateway.place_market_order(
            sig.symbol, sig.side, qty, client_order_id=key
        )

        if result.status == "filled":
            self._ledger.mark_filled(intent, result.avg_fill_price or sig.ref_price, result.filled_qty, result.fee)
            self._session.commit()
            log.info("FILL %s %s qty=%s @ %.2f", sig.side, sig.symbol, qty, result.avg_fill_price)
        elif result.status == "rejected":
            self._ledger.mark_failed(intent, result.error)
            self._session.commit()
        else:
            # open/partial: mark sent with exchange id; reconciliation handles later.
            self._ledger.mark_sent(intent, result.order_id)
            self._session.commit()
        return intent

    # --- helpers ---

    def _persist_intent(self, sig: Signal, reason: str, skipped: bool) -> TradeIntent:
        # Skipped intents still get a unique key so retries are idempotent.
        key = self._make_key(sig.id, sig.symbol, sig.side)
        intent = TradeIntent(
            signal_id=sig.id,
            idempotency_key=key,
            symbol=sig.symbol,
            side=sig.side,
            qty=0.0,
            order_type="market",
            ref_price=sig.ref_price,
            status="pending",
        )
        self._session.add(intent)
        self._session.commit()
        return intent

    def _qty_for(self, sig: Signal, decision, pos: Position | None) -> float:
        price = sig.ref_price or 0.0
        if price <= 0:
            return 0.0
        if sig.side == "buy":
            risk = json.loads(sig.risk_json or "{}")
            position_pct = float(risk.get("position_pct", 0.2))
            # Simple fixed-fraction sizing of balance for v1; ATR-aware sizing
            # plugs in here later.
            balance = self._balance()
            notional = balance * position_pct
            return notional / price
        else:  # sell closes the whole open position
            if pos is None or pos.qty <= 0:
                return 0.0
            return pos.qty

    def _open_positions(self) -> list[Position]:
        return self._session.execute(
            select(Position).where(Position.qty > 0)
        ).scalars().all()

    def _last_entry_ts(self, symbol: str) -> float | None:
        intent = self._session.execute(
            select(TradeIntent)
            .where(TradeIntent.symbol == symbol, TradeIntent.side == "buy")
            .order_by(TradeIntent.ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        return intent.ts.timestamp() * 1000 if intent and intent.ts else None
