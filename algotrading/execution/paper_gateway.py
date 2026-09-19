"""Paper trading gateway: simulated fills against live or demo prices.

Fills at the current market price plus a configurable slippage. It never talks
to an exchange, so the whole pipeline (signal -> intent -> fill -> ledger) can
be exercised without API keys or network (demo mode) or against live public
prices (paper mode).

The optional gateway capabilities are simulated in memory:

* limit orders fill immediately when the limit crosses the current price,
  otherwise they rest and are listed by ``get_open_orders`` until cancelled;
* stop orders trigger when the current price is at/through the stop and then
  fill like a market order (same slippage model);
* resting orders are tracked by ``client_order_id`` — the same key the ledger
  uses, so paper and live reconcile identically.

Resting state is deliberately in-memory and process-local: with no venue to
hold the order, there is nothing to persist, and a restart legitimately
"cancels" it. The market-order behaviour and fee model are unchanged (a 0.1%
spot taker fee on the fill notional), so existing fill maths still holds.
"""
from __future__ import annotations

import logging
from typing import Any

from algotrading.execution.base import OrderResult
from algotrading.market.base import MarketDataProvider

log = logging.getLogger(__name__)


class PaperGateway:
    def __init__(
        self,
        provider: MarketDataProvider,
        slippage_pct: float = 0.05,
        fill_delay_ms: int = 0,
        initial_balance: float | None = None,
    ) -> None:
        self._provider = provider
        self._slippage = slippage_pct / 100.0
        self._fill_delay = fill_delay_ms
        # `None` = "no opinion": the engine then sizes from
        # `risk.paper_initial_balance`. The module passes the configured value,
        # so the venue balance and the sizing basis agree by construction.
        self._balance = initial_balance
        self._resting: dict[str, dict[str, Any]] = {}
        self._known: dict[str, OrderResult] = {}

    # --- mandatory capabilities ---

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        client_order_id: str,
    ) -> OrderResult:
        market = self._market_price(symbol, client_order_id)
        if market is None:
            return self._known[client_order_id]
        # Slippage: buy fills higher, sell fills lower.
        if side == "buy":
            fill = market * (1 + self._slippage)
        else:
            fill = market * (1 - self._slippage)
        return self._fill(symbol, side, qty, fill, client_order_id, market)

    def get_order(self, symbol: str, client_order_id: str) -> OrderResult:
        known = self._known.get(client_order_id)
        if known is not None:
            return known
        # Unknown paper orders are treated as synchronous fills (legacy
        # behaviour: nothing can be resting that we did not place).
        return OrderResult(order_id=client_order_id, status="filled")

    # --- optional capabilities (simulated) ---

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        client_order_id: str,
        time_in_force: str = "GTC",
    ) -> OrderResult:
        """Fill immediately if the limit crosses the market, else rest it.

        The fill uses the limit price, not the (better) market price: a crossing
        limit is filled at no worse than the limit, so the limit is the
        conservative and deterministic assumption for PnL.
        """
        market = self._market_price(symbol, client_order_id)
        if market is None:
            return self._known[client_order_id]
        crossed = market <= price if side == "buy" else market >= price
        if crossed:
            return self._fill(symbol, side, qty, price, client_order_id, market, kind="limit")
        order = {
            "symbol": symbol,
            "type": "limit",
            "side": side,
            "qty": qty,
            "price": price,
            "time_in_force": time_in_force,
        }
        result = OrderResult(order_id=client_order_id, status="open")
        self._rest(client_order_id, order, result)
        log.info("paper limit resting %s %s qty=%s @ %.2f (market %.2f)", side, symbol, qty, price, market)
        return result

    def place_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        limit_price: float | None = None,
        client_order_id: str = "",
    ) -> OrderResult:
        """Rest a stop, or trigger it now when price is already through it.

        A triggered stop becomes a market order, so it fills with the same
        slippage model as ``place_market_order``. An untriggered one rests
        until ``cancel_order`` (paper has no candle feed to re-check it, and
        the live venue has the same "rest until it triggers" semantics).
        """
        market = self._market_price(symbol, client_order_id)
        if market is None:
            return self._known[client_order_id]
        triggered = market >= stop_price if side == "buy" else market <= stop_price
        if triggered:
            fill = market * (1 + self._slippage) if side == "buy" else market * (1 - self._slippage)
            log.info("paper stop triggered %s %s @ stop %.2f (market %.2f)", side, symbol, stop_price, market)
            return self._fill(symbol, side, qty, fill, client_order_id, market, kind="stop")
        order = {
            "symbol": symbol,
            "type": "stop_limit" if limit_price is not None else "stop",
            "side": side,
            "qty": qty,
            "stop_price": stop_price,
            "price": limit_price,
        }
        result = OrderResult(order_id=client_order_id, status="open")
        self._rest(client_order_id, order, result)
        log.info("paper stop resting %s %s qty=%s @ stop %.2f", side, symbol, qty, stop_price)
        return result

    def cancel_order(self, symbol: str, client_order_id: str) -> OrderResult:
        """Cancel a resting order.

        Idempotent on purpose: an order that is not resting is already in the
        desired end state, and the engine's protective-stop cancel must never
        treat "nothing to cancel" as a failure that blocks the ledger write.
        """
        self._resting.pop(client_order_id, None)
        result = OrderResult(order_id=client_order_id, status="canceled")
        self._known[client_order_id] = result
        return result

    def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        return [
            {"client_order_id": cid, "order": dict(order)}
            for cid, order in self._resting.items()
            if order.get("symbol") == symbol
        ]

    def get_balance(self, asset: str = "USDT") -> float | None:
        """Configured paper balance, or ``None`` when none was configured.

        ``None`` means "no opinion" rather than 0.0: the engine then falls back
        to ``risk.paper_initial_balance``, which is what it did before this
        method existed. Returning a made-up number here would silently change
        the sizing basis of every existing paper deployment.
        """
        return self._balance

    # --- internals ---

    def _market_price(self, symbol: str, client_order_id: str) -> float | None:
        try:
            return self._provider.fetch_ticker_price(symbol)
        except Exception as exc:  # noqa: BLE001 - provider failure must not crash engine
            self._known[client_order_id] = OrderResult(
                order_id=client_order_id,
                status="rejected",
                error=f"price fetch failed: {exc}",
            )
            return None

    def _fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        fill_price: float,
        client_order_id: str,
        market: float,
        kind: str = "market",
    ) -> OrderResult:
        # Paper fee = 0.1% of notional (spot taker) for realism.
        fee = qty * fill_price * 0.001
        log.info(
            "paper %s fill %s %s qty=%s @ %.2f (market %.2f)",
            kind, side, symbol, qty, fill_price, market,
        )
        result = OrderResult(
            order_id=client_order_id,
            status="filled",
            filled_qty=qty,
            avg_fill_price=fill_price,
            fee=fee,
        )
        self._resting.pop(client_order_id, None)
        self._known[client_order_id] = result
        return result

    def _rest(self, client_order_id: str, order: dict[str, Any], result: OrderResult) -> None:
        self._resting[client_order_id] = order
        self._known[client_order_id] = result
