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

Optional trailing stop support (gated by risk.trailing_stop_pct > 0):
  - On fill, initialize trailing stop at entry_price * (1 - trailing_stop_pct)
  - On each market tick, update trailing stop if price moved favorably
  - If price hits trailing stop, generate sell signal

Exchange-side protective stop (gated by risk.exchange_stop_enabled, live only):
  the local trailing stop above only exists while this process is running. To
  survive the bot being killed (Android SIGKILLs the whole Termux app), a BUY
  fill also rests a real stop order ON THE VENUE. The exchange then protects
  the position with no bot in the loop.

**Where the resting stop's identity lives — and why the event log.**
The stop order's identity (client order id, exchange order id, stop price) has
to be recoverable after a crash, so it must be persisted, not kept in memory.
It is stored as `order_events` rows — `stop_placed` on placement and
`stop_canceled` on removal, both carrying the ids in `payload_json` — and the
current resting stop is replayed from those events (`_resting_stop`).

Two reasons this is the store rather than a new column or table:
  * `order_events` is the ledger's source of truth and is append-only, so the
    history of "we rested a stop, we moved it, we cancelled it" is auditable
    and cannot be lost by a later code path forgetting to update a column;
  * the execution path must not grow the schema — adding columns here would
    couple the order path to a migration, and the events are exactly the facts
    we need (which order, which symbol, what price, when).

Order entry types: a signal may request a resting entry with
`sig.risk["order_type"] == "limit"` and `sig.risk["limit_offset_pct"]` (the
distance behind the reference price; positive = passive). Anything else, and
every exit, stays a market order.

Alerting is best-effort throughout (`algotrading.alerts.notify`): a webhook
that is slow or down must never break the ledger write or the tick.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.alerts import notify
from algotrading.db.models import OrderEvent, Position, Signal, Trade, TradeIntent
from algotrading.execution.base import ExchangeGateway, OrderResult
from algotrading.execution.risk import RiskManager
from algotrading.ledger.store import Ledger

log = logging.getLogger(__name__)

# Default fraction of balance per entry when a signal carries no position_pct.
# The hard ceiling is `risk.max_position_pct`, applied in RiskManager.
DEFAULT_POSITION_PCT = 0.2

# How many recent closed trades the daily-loss guard inspects (UTC day).
DAILY_LOSS_LOOKBACK_ROWS = 200

# Event types that describe the venue-held protective stop. Kept as plain
# strings (not an enum) because `order_events.event_type` is a free-form column
# shared with the ledger's own event vocabulary.
STOP_PLACED = "stop_placed"
STOP_CANCELED = "stop_canceled"
STOP_EVENT_TYPES = (STOP_PLACED, STOP_CANCELED)

# `order_events.event_type` is String(24) — keep these short enough to fit.
assert max(len(STOP_PLACED), len(STOP_CANCELED)) <= 24

# Signal `risk` keys a strategy may use to hand over its own stop price.
SIGNAL_STOP_KEYS = ("stop_price", "stop_loss", "stop")

# Statuses that mean "the venue accepted the order and it is resting there".
# A `filled` answer is handled separately: the stop executed on arrival.
_STOP_ACCEPTED_STATUSES = ("open", "partial", "unknown")


def _notify(kind: str, title: str, body: str = "", **fields: Any) -> None:
    """Emit an alert, swallowing every failure.

    Alerting is observability, not part of the trade: a misconfigured or dead
    webhook must never abort a fill, a cancel or a tick.
    """
    try:
        notify(kind, title, body, **fields)
    except Exception as exc:  # noqa: BLE001 - best-effort by contract
        log.debug("alert dropped (%s %s): %s", kind, title, exc)


def _risk_dict(sig: Signal) -> dict[str, Any]:
    """Parse `Signal.risk_json`, tolerating junk (a signal is user/AI data)."""
    try:
        data = json.loads(sig.risk_json or "{}")
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


class ExecutionEngine:
    def __init__(self, session: Session, gateway: ExchangeGateway, risk: RiskManager, ledger: Ledger) -> None:
        self._session = session
        self._gateway = gateway
        self._risk = risk
        self._ledger = ledger

    def _make_key(self, signal_id: int, symbol: str, side: str) -> str:
        return f"{symbol}-{side}-{signal_id}-{uuid.uuid4().hex[:8]}"

    def _balance(self) -> float:
        """Sizing basis: the venue's balance when it can report one, else the
        configured paper balance.

        This used to return a hardcoded ``100_000.0`` while
        ``risk.paper_initial_balance`` defaulted to ``10_000.0`` — a 10x gap
        between the configured basis and the one actually used.

        A gateway that reports ``None`` is saying "no opinion" (a bare
        PaperGateway), which is not an error and must stay quiet; only a real
        failure is worth a warning.
        """
        getter = getattr(self._gateway, "get_balance", None)
        if callable(getter):
            try:
                reported = getter()
            except Exception as exc:  # noqa: BLE001 - never fail the tick on this
                log.warning(
                    "gateway balance unavailable (%s); falling back to the configured balance",
                    exc,
                )
            else:
                if reported is not None:
                    balance = float(reported)
                    if balance > 0:
                        return balance
                    log.warning(
                        "gateway reported a non-positive balance (%s); using the configured balance",
                        balance,
                    )
        return float(self._risk._cfg.paper_initial_balance)

    def _daily_realized_pnl(self) -> float:
        """Realized PnL of trades closed since 00:00 UTC (the guard's window).

        Filtering happens in Python rather than SQL: SQLite hands back naive
        datetimes, so a SQL `>=` against a tz-aware boundary is a footgun.
        """
        day_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        rows = self._session.execute(
            select(Trade.closed_at, Trade.realized_pnl)
            .where(Trade.closed_at.is_not(None))
            .order_by(Trade.closed_at.desc())
            .limit(DAILY_LOSS_LOOKBACK_ROWS)
        ).all()
        total = 0.0
        for closed_at, pnl in rows:
            if closed_at is None or pnl is None:
                continue
            ts = closed_at if closed_at.tzinfo else closed_at.replace(tzinfo=timezone.utc)
            if ts >= day_start:
                total += float(pnl)
        return total

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
                daily_pnl=self._daily_realized_pnl(),
            )
        else:
            decision = self._risk.check_sell(has_position=pos is not None and pos.qty > 0)

        if not decision.allowed:
            intent = self._persist_intent(sig, decision.reason, skipped=True)
            self._ledger.mark_risk_skipped(intent, decision.reason)
            sig.status = "skipped"
            self._session.commit()
            log.info("signal %s skipped: %s", sig.id, decision.reason)
            _notify(
                "risk",
                f"skipped {sig.side} {sig.symbol}",
                decision.reason,
                symbol=sig.symbol,
                side=sig.side,
                reason=decision.reason,
            )
            return intent

        qty = self._qty_for(sig, decision, pos)
        if qty <= 0:
            intent = self._persist_intent(sig, "invalid qty", skipped=True)
            self._ledger.mark_risk_skipped(intent, "qty <= 0")
            sig.status = "skipped"
            self._session.commit()
            _notify(
                "risk",
                f"skipped {sig.side} {sig.symbol}",
                "qty <= 0",
                symbol=sig.symbol,
                side=sig.side,
                reason="qty <= 0",
            )
            return intent

        limit_price = self._entry_limit_price(sig)

        # --- intent-before-order ---
        key = self._make_key(sig.id, sig.symbol, sig.side)
        intent = TradeIntent(
            signal_id=sig.id,
            idempotency_key=key,
            symbol=sig.symbol,
            side=sig.side,
            qty=qty,
            order_type="limit" if limit_price is not None else "market",
            ref_price=sig.ref_price,
            status="pending",
        )
        self._session.add(intent)
        self._session.commit()
        sig.status = "sent"
        self._session.commit()

        # An exit must not race the protective stop: if both fill we would sell
        # the position twice (spot would reject the second one at best).
        if sig.side == "sell":
            self._cancel_resting_stop(sig.symbol, reason="entry exit")

        result = self._send_order(intent, sig, qty, limit_price, key)
        self._apply_result(intent, sig, result)
        return intent

    # --- order placement ---

    def _entry_limit_price(self, sig: Signal) -> float | None:
        """Limit price for a passive entry, or None for a market order.

        Only buys can rest: exits are protective and must fill, so a sell is
        always market no matter what the signal asks for.
        """
        if sig.side != "buy":
            return None
        risk = _risk_dict(sig)
        if str(risk.get("order_type", "market")).lower() != "limit":
            return None
        ref = float(sig.ref_price or 0.0)
        if ref <= 0:
            return None
        try:
            offset = float(risk.get("limit_offset_pct", 0.0) or 0.0)
        except (TypeError, ValueError):
            offset = 0.0
        # Positive offset = passive: the buy rests *below* the reference price,
        # buying the dip instead of chasing it. 0 means "at the reference",
        # which crosses immediately against a current-price quote.
        price = ref * (1 - offset / 100.0)
        return price if price > 0 else None

    def _send_order(
        self,
        intent: TradeIntent,
        sig: Signal,
        qty: float,
        limit_price: float | None,
        key: str,
    ) -> OrderResult:
        """Place the entry, degrading to a market order when the venue cannot
        rest limits (the optional capability is probed, never assumed)."""
        try:
            if limit_price is not None:
                placer = getattr(self._gateway, "place_limit_order", None)
                if callable(placer):
                    return placer(
                        sig.symbol, sig.side, qty, limit_price, client_order_id=key
                    )
                log.warning(
                    "%s cannot rest limit orders; falling back to a market order",
                    type(self._gateway).__name__,
                )
                intent.order_type = "market"
            return self._gateway.place_market_order(sig.symbol, sig.side, qty, client_order_id=key)
        except Exception as exc:  # noqa: BLE001 - a gateway failure is a rejected order
            log.error("order placement raised for %s %s: %s", sig.side, sig.symbol, exc)
            _notify(
                "error",
                f"order failed {sig.side} {sig.symbol}",
                str(exc),
                symbol=sig.symbol,
                side=sig.side,
            )
            return OrderResult(order_id=key, status="rejected", error=str(exc))

    def _apply_result(self, intent: TradeIntent, sig: Signal, result: OrderResult) -> None:
        if result.status == "filled":
            fill_price = result.avg_fill_price or sig.ref_price
            self._ledger.mark_filled(intent, fill_price, result.filled_qty, result.fee)
            if sig.side == "buy":
                self._init_trailing_stop(sig.symbol, fill_price)
                self._maybe_place_exchange_stop(intent, sig, fill_price)
            else:
                # Position closed: the venue stop has nothing left to protect.
                self._cancel_resting_stop(sig.symbol, reason="position closed")
            self._session.commit()
            log.info("FILL %s %s qty=%s @ %.2f", sig.side, sig.symbol, intent.qty, fill_price)
            _notify(
                "fill",
                f"{sig.side.upper()} {sig.symbol} @ {fill_price:.8g}",
                f"qty {intent.qty}",
                symbol=sig.symbol,
                side=sig.side,
                qty=intent.qty,
                price=fill_price,
                fee=result.fee,
            )
        elif result.status == "rejected":
            self._ledger.mark_failed(intent, result.error)
            self._session.commit()
        else:
            # open/partial/unknown: mark sent with exchange id; reconciliation
            # confirms later. A resting limit entry legitimately lands here —
            # it is NOT a fill, and the intent must not be recorded as one.
            self._ledger.mark_sent(intent, result.order_id)
            self._session.commit()
            if intent.order_type == "limit":
                log.info("limit entry resting for %s (order %s)", sig.symbol, result.order_id)

    # --- trailing stops ---

    def _init_trailing_stop(self, symbol: str, entry_price: float) -> None:
        """Initialize trailing stop for a new position if enabled."""
        trailing_pct = self._risk._cfg.trailing_stop_pct
        if not trailing_pct or trailing_pct <= 0:
            return
        stop_price = entry_price * (1 - trailing_pct / 100.0)
        pos = self._session.execute(
            select(Position).where(Position.symbol == symbol)
        ).scalar_one_or_none()
        if pos:
            pos.trailing_stop_price = stop_price
            self._session.commit()
            log.info("Trailing stop initialized for %s: %.2f", symbol, stop_price)

    def update_trailing_stops(self, current_prices: dict[str, float]) -> list[Signal]:
        """Check and update trailing stops for all open positions.

        Returns list of sell signals generated by trailing stop hits.
        """
        trailing_pct = self._risk._cfg.trailing_stop_pct
        if not trailing_pct or trailing_pct <= 0:
            return []

        generated_signals: list[Signal] = []
        positions = self._open_positions()

        for pos in positions:
            if pos.symbol not in current_prices:
                continue
            current_price = current_prices[pos.symbol]
            entry_price = pos.avg_price or current_price
            existing_stop = getattr(pos, 'trailing_stop_price', None)

            # Track highest price for trailing stop (only move up for longs).
            # `highest_price` is nullable and unset until the first tick after
            # entry, so fall back to the entry price instead of comparing to None.
            highest_price = pos.highest_price or entry_price
            if current_price > highest_price:
                highest_price = current_price
                pos.highest_price = highest_price
                self._session.commit()

            # Only check trailing stop if price moved favorably (new high)
            if current_price >= highest_price:
                should_update, new_stop = self._risk.check_trailing_stop(
                    entry_price, highest_price, existing_stop, is_long=True
                )
                if should_update and new_stop is not None:
                    pos.trailing_stop_price = new_stop
                    self._session.commit()
                    self._ledger.mark_trailing_stop_adjusted(pos.symbol, new_stop)
                    self._reprice_resting_stop(pos, new_stop)
                    log.debug("Trailing stop updated for %s: %.2f", pos.symbol, new_stop)

            # Check if price hit trailing stop
            if existing_stop and current_price <= existing_stop:
                log.info("Trailing stop hit for %s: price %.2f <= stop %.2f", pos.symbol, current_price, existing_stop)
                # The local exit is about to close the position; drop the venue
                # stop so the two cannot both fill.
                self._cancel_resting_stop(pos.symbol, reason="trailing stop hit")
                # Generate sell signal via risk manager
                sell_sig = Signal(
                    symbol=pos.symbol,
                    side="sell",
                    ref_price=current_price,
                    rationale=f"Trailing stop hit at {existing_stop:.2f}",
                    risk_json=json.dumps({"position_pct": 1.0}),
                    status="candidate",
                )
                self._session.add(sell_sig)
                self._session.commit()
                generated_signals.append(sell_sig)

        return generated_signals

    # --- exchange-side protective stop ---

    def _maybe_place_exchange_stop(self, intent: TradeIntent, sig: Signal, entry_price: float) -> None:
        """Rest a protective stop on the venue after a BUY fill.

        Gated by `risk.exchange_stop_enabled` and probed with getattr, so a
        gateway without the capability (or paper mode before it was enabled)
        simply keeps the local trailing stop. This runs *after* the fill event
        is recorded and never blocks it: if the venue refuses the stop we log
        and alert, but the position and its ledger entry stand.
        """
        cfg = self._risk._cfg
        if not cfg.exchange_stop_enabled:
            return
        placer = getattr(self._gateway, "place_stop_order", None)
        if not callable(placer):
            log.debug(
                "exchange stop enabled but %s has no place_stop_order; local stop only",
                type(self._gateway).__name__,
            )
            return

        stop_price = self._entry_stop_price(sig, entry_price, cfg)
        if stop_price is None:
            log.warning("exchange stop requested for %s but no stop price is available", sig.symbol)
            return
        if stop_price <= 0 or stop_price >= entry_price:
            # A long's protective stop must sit below the market, or the venue
            # either rejects it or fills it instantly at a worse price.
            log.warning(
                "refusing exchange stop for %s: stop %.8g is not below entry %.8g",
                sig.symbol, stop_price, entry_price,
            )
            return

        pos = self._session.execute(
            select(Position).where(Position.symbol == sig.symbol)
        ).scalar_one_or_none()
        qty = float(pos.qty) if pos and pos.qty > 0 else 0.0
        if qty <= 0:
            log.warning("no open position for %s; not resting a stop", sig.symbol)
            return

        resting = self._resting_stop(sig.symbol)
        if resting and self._same_stop(resting, stop_price, qty):
            # A retry (same intent, same stop, same size) must not double-place.
            log.debug("exchange stop already resting for %s at %.8g", sig.symbol, stop_price)
            return
        if resting:
            self._cancel_resting_stop(sig.symbol, reason="replaced")

        client_order_id = self._next_stop_client_id(intent.id)
        self._place_stop(
            symbol=sig.symbol,
            qty=qty,
            stop_price=stop_price,
            intent_id=intent.id,
            client_order_id=client_order_id,
            placer=placer,
            limit_price=self._stop_limit_price(sig, stop_price, cfg),
        )

    def _reprice_resting_stop(self, pos: Position, new_stop: float) -> None:
        """Move the venue stop to `new_stop` after the trailing stop advanced.

        Cancel then place, in that order: a brief unprotected window is better
        than two live stops that can each sell the whole position. When nothing
        is resting yet this still places one (the setting may have been turned
        on after entry). Best-effort throughout — a venue that refuses the new
        stop leaves the previous behaviour (local stop) intact.
        """
        cfg = self._risk._cfg
        if not cfg.exchange_stop_enabled or pos.qty <= 0:
            return
        placer = getattr(self._gateway, "place_stop_order", None)
        if not callable(placer):
            return
        intent_id = self._resting_intent_id(pos.symbol)
        self._cancel_resting_stop(pos.symbol, reason="trailing advance")
        if self._resting_stop(pos.symbol) is not None:
            # The cancel did not confirm; leaving the old stop in place is
            # safer than stacking a second one on the same position.
            return
        self._place_stop(
            symbol=pos.symbol,
            qty=float(pos.qty),
            stop_price=new_stop,
            intent_id=intent_id,
            client_order_id=self._next_stop_client_id(intent_id),
            placer=placer,
            limit_price=None,
        )

    def _place_stop(
        self,
        *,
        symbol: str,
        qty: float,
        stop_price: float,
        intent_id: int | None,
        client_order_id: str,
        placer: Any,
        limit_price: float | None,
    ) -> bool:
        """Place one resting stop and record it. Returns True when recorded."""
        try:
            result = placer(
                symbol,
                "sell",
                qty,
                stop_price,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        except Exception as exc:  # noqa: BLE001 - an optional capability must not break the tick
            log.warning("exchange stop placement failed for %s: %s", symbol, exc)
            _notify(
                "error",
                f"exchange stop failed for {symbol}",
                str(exc),
                symbol=symbol,
                stop_price=stop_price,
                client_order_id=client_order_id,
            )
            return False

        if result.status == "filled":
            # The stop was already through the market and executed on arrival:
            # there is nothing resting to track or cancel, and the position may
            # now be flat at the venue (the reconcile path confirms it).
            log.warning(
                "exchange stop for %s filled immediately on placement (stop %.8g was already through the market)",
                symbol, stop_price,
            )
            _notify(
                "risk",
                f"exchange stop filled on arrival {symbol}",
                f"stop {stop_price:.8g} qty {qty}",
                symbol=symbol,
                stop_price=stop_price,
                qty=qty,
            )
            return False

        if result.status not in _STOP_ACCEPTED_STATUSES:
            log.warning(
                "exchange stop rejected for %s (status=%s): %s",
                symbol, result.status, result.error,
            )
            _notify(
                "error",
                f"exchange stop rejected for {symbol}",
                result.error or result.status,
                symbol=symbol,
                stop_price=stop_price,
                status=result.status,
            )
            return False

        # `unknown` is recorded too: the POST may have reached the exchange
        # while the response was lost. The client order id is deterministic, so
        # recording it is what lets the next attempt cancel instead of stack.
        self._record_stop_event(
            STOP_PLACED,
            {
                "symbol": symbol,
                "side": "sell",
                "qty": qty,
                "stop_price": stop_price,
                "limit_price": limit_price,
                "client_order_id": client_order_id,
                "exchange_order_id": result.order_id,
                "intent_id": intent_id,
                "status": result.status,
            },
        )
        self._session.commit()
        log.info(
            "exchange stop resting for %s: qty=%s stop=%.8g (order %s, status %s)",
            symbol, qty, stop_price, client_order_id, result.status,
        )
        _notify(
            "risk",
            f"exchange stop resting {symbol}",
            f"stop {stop_price:.8g} qty {qty}",
            symbol=symbol,
            stop_price=stop_price,
            qty=qty,
            client_order_id=client_order_id,
        )
        return True

    def _cancel_resting_stop(self, symbol: str, reason: str = "") -> bool:
        """Cancel the symbol's resting stop, best-effort.

        A cancel failure is logged and alerted but never raised: the caller is
        usually in the middle of a ledger write (position closed, stop hit) and
        an unreachable venue must not roll that back.
        """
        resting = self._resting_stop(symbol)
        if resting is None:
            return False
        client_order_id = str(resting.get("client_order_id") or "")
        canceller = getattr(self._gateway, "cancel_order", None)
        if not callable(canceller):
            log.debug("gateway cannot cancel orders; forgetting the local stop record for %s", symbol)
        elif client_order_id:
            try:
                result = canceller(symbol, client_order_id)
            except Exception as exc:  # noqa: BLE001 - best-effort by contract
                log.warning("cancel of resting stop %s failed for %s: %s", client_order_id, symbol, exc)
                _notify(
                    "error",
                    f"stop cancel failed for {symbol}",
                    str(exc),
                    symbol=symbol,
                    client_order_id=client_order_id,
                )
                return False
            if result.status not in ("canceled", "filled", "partial"):
                log.warning(
                    "cancel of resting stop %s for %s was not confirmed (status=%s): %s",
                    client_order_id, symbol, result.status, result.error,
                )
                _notify(
                    "error",
                    f"stop cancel unconfirmed for {symbol}",
                    result.error or result.status,
                    symbol=symbol,
                    client_order_id=client_order_id,
                )
                return False

        self._record_stop_event(
            STOP_CANCELED,
            {
                "symbol": symbol,
                "client_order_id": client_order_id,
                "exchange_order_id": resting.get("exchange_order_id"),
                "intent_id": resting.get("intent_id"),
                "reason": reason,
            },
        )
        self._session.commit()
        log.info("exchange stop canceled for %s (%s)", symbol, reason or "no reason given")
        return True

    # --- resting-stop state, replayed from the event log ---

    def _stop_events(self) -> list[tuple[str, dict[str, Any]]]:
        """All stop-related events, oldest first (insertion order = id order)."""
        rows = self._session.execute(
            select(OrderEvent)
            .where(OrderEvent.event_type.in_(STOP_EVENT_TYPES))
            .order_by(OrderEvent.id)
        ).scalars().all()
        out: list[tuple[str, dict[str, Any]]] = []
        for ev in rows:
            try:
                payload = json.loads(ev.payload_json or "{}")
            except ValueError:
                continue
            if isinstance(payload, dict):
                out.append((ev.event_type, payload))
        return out

    def _resting_stop(self, symbol: str) -> dict[str, Any] | None:
        """The stop currently resting on the venue for `symbol`, or None.

        Replayed from the immutable log: a `stop_placed` starts it, and the
        matching `stop_canceled` (same client order id) ends it. This is the
        only source of truth for it — there is no column to drift out of sync,
        and a crash cannot lose it.
        """
        resting: dict[str, Any] | None = None
        for event_type, payload in self._stop_events():
            if payload.get("symbol") != symbol:
                continue
            if event_type == STOP_PLACED:
                resting = payload
            elif resting is not None and payload.get("client_order_id") == resting.get("client_order_id"):
                resting = None
        return resting

    def _resting_intent_id(self, symbol: str) -> int | None:
        """Entry intent that owns the resting stop (for a stable client id)."""
        resting = self._resting_stop(symbol)
        if resting and resting.get("intent_id") is not None:
            try:
                return int(resting["intent_id"])
            except (TypeError, ValueError):
                pass
        intent = self._session.execute(
            select(TradeIntent)
            .where(
                TradeIntent.symbol == symbol,
                TradeIntent.side == "buy",
                TradeIntent.status == "filled",
            )
            .order_by(TradeIntent.ts.desc())
            .limit(1)
        ).scalar_one_or_none()
        return intent.id if intent else None

    def _next_stop_client_id(self, intent_id: int | None) -> str:
        """Deterministic client order id for the next placement of one entry.

        Derived from the entry intent plus how many placements are already
        recorded for it, so a *retry of a failed placement* (no event recorded)
        reuses the same id — the venue then rejects the duplicate instead of
        letting a crash-and-retry stack two stops on one position. A genuine
        re-placement (trailing advance) gets the next id.

        Fits Binance's 36-char client order id limit and its allowed alphabet.
        """
        owner = int(intent_id) if intent_id is not None else 0
        placements = sum(
            1
            for event_type, payload in self._stop_events()
            if event_type == STOP_PLACED and payload.get("intent_id") == owner
        )
        return f"stop-{owner}-{placements + 1}"

    def _record_stop_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append one stop event. `order_events` is append-only: never update."""
        self._session.add(
            OrderEvent(
                intent_id=payload.get("intent_id"),
                event_type=event_type,
                payload_json=json.dumps(payload, default=str),
            )
        )

    @staticmethod
    def _same_stop(resting: dict[str, Any], stop_price: float, qty: float) -> bool:
        try:
            same_price = abs(float(resting.get("stop_price") or 0.0) - float(stop_price)) < 1e-12
            same_qty = abs(float(resting.get("qty") or 0.0) - float(qty)) < 1e-12
        except (TypeError, ValueError):
            return False
        return same_price and same_qty

    def _entry_stop_price(self, sig: Signal, entry_price: float, cfg: Any) -> float | None:
        """Stop price for the resting stop: the signal's own stop if it has one,
        else the configured trailing distance from the fill price."""
        risk = _risk_dict(sig)
        for key in SIGNAL_STOP_KEYS:
            value = risk.get(key)
            if value in (None, "", 0):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                log.warning("signal %s has a non-numeric risk['%s']=%r; ignoring", sig.id, key, value)
        pct = cfg.trailing_stop_pct
        if not pct or pct <= 0:
            return None
        return entry_price * (1 - pct / 100.0)

    def _stop_limit_price(self, sig: Signal, stop_price: float, cfg: Any) -> float | None:
        """Optional stop-limit price.

        Off unless the signal opts in (`risk['stop_limit']`): a market
        STOP_LOSS always fills once triggered, while a stop-limit can be left
        unfilled through a gap — the wrong trade-off for protection unless the
        caller explicitly asks for a price floor.
        """
        risk = _risk_dict(sig)
        if not risk.get("stop_limit"):
            return None
        offset = float(getattr(cfg, "exchange_stop_limit_offset_pct", 0.0) or 0.0)
        if offset <= 0:
            return None
        return stop_price * (1 - offset / 100.0)

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
            position_pct = float(risk.get("position_pct", DEFAULT_POSITION_PCT))
            # Routed through RiskManager so `max_position_pct` is enforced on
            # the order path — a signal's own risk dict is a *request*, not a cap.
            return self._risk.size_by_pct(self._balance(), price, position_pct)
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
