"""Live exchange gateway: real Binance Spot orders via the REST client.

Mirrors PaperGateway's interface (place_market_order / get_order plus the
optional capabilities) so the execution engine is identical in paper and live
modes. Signed endpoints require API keys in .env — mainnet keys
(BINANCE_API_KEY / BINANCE_API_SECRET) or, when the venue is the spot testnet,
BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET.

Safety notes:
- The caller persists intent-before-order, so a network error mid-request
  must not place a duplicate: create_order carries newClientOrderId = the
  idempotency key, and get_order can reconcile by that key.
- A failed POST that may or may not have reached the exchange returns status
  "unknown" so the engine stops retrying risk-sensitive actions until the
  reconcile job confirms.
- A limit or stop order is *not* a fill: the exchange accepts it and it rests
  (``status=NEW``), so those paths report ``open`` and the intent stays `sent`
  until reconciliation confirms.
- ``get_balance`` raises on failure instead of inventing a number: sizing a
  real order from a paper balance would be worse than not trading at all.
"""
from __future__ import annotations

import logging
from typing import Any

from algotrading.execution.base import OrderResult
from algotrading.market.binance_rest import BinanceError, BinanceRestClient

log = logging.getLogger(__name__)

# Binance market orders always fill entirely at the taker price.
_STATUS_FILLED = "filled"
_STATUS_UNKNOWN = "unknown"

# Error codes/messages Binance returns for "this order is not open". A cancel
# that finds nothing is the desired end state (the order is not resting), so it
# counts as canceled rather than as a failure — the engine's protective-stop
# cancel must not be retried forever because the stop already triggered.
_NO_ORDER_MARKERS = ("-2011", "-2013", "unknown order", "order does not exist")


class LiveGateway:
    def __init__(self, client: BinanceRestClient | None = None) -> None:
        self._client = client or BinanceRestClient()

    # --- mandatory capabilities ---

    def place_market_order(
        self,
        symbol: str,
        side: str,  # buy | sell
        qty: float,
        client_order_id: str,
    ) -> OrderResult:
        try:
            data = self._client.create_order(
                symbol=symbol,
                side=side.upper(),
                quantity=qty,
                order_type="MARKET",
                client_order_id=client_order_id,
            )
        except BinanceError as exc:
            log.error("live order failed for %s %s qty=%s: %s", side, symbol, qty, exc)
            return OrderResult(
                order_id=client_order_id,
                status=_STATUS_UNKNOWN,
                error=str(exc),
            )

        order_id = str(data.get("orderId", client_order_id))
        # MARKET orders are filled synchronously by Binance. If the payload has
        # no executedQty, treat as unknown rather than assuming a fill.
        filled_qty = float(data.get("executedQty", 0.0) or 0.0)
        avg, fee = self._avg_and_fee(data)
        if filled_qty > 0:
            log.info("live fill %s %s qty=%s @ %s", side, symbol, filled_qty, avg)
            return OrderResult(
                order_id=order_id,
                status=_STATUS_FILLED,
                filled_qty=filled_qty,
                avg_fill_price=avg,
                fee=fee,
            )
        return OrderResult(
            order_id=order_id,
            status=_STATUS_UNKNOWN,
            error="market order returned no executedQty",
        )

    def get_order(self, symbol: str, client_order_id: str) -> OrderResult:
        try:
            data = self._client.get_order(symbol, client_order_id=client_order_id)
        except BinanceError as exc:
            return OrderResult(
                order_id=client_order_id,
                status=_STATUS_UNKNOWN,
                error=str(exc),
            )
        status = str(data.get("status", "")).lower()  # NEW|PARTIALLY_FILLED|FILLED|...
        filled_qty = float(data.get("executedQty", 0.0))
        return OrderResult(
            order_id=str(data.get("orderId", client_order_id)),
            status="filled" if status == "filled" else "open",
            filled_qty=filled_qty,
            avg_fill_price=None,
            error="",
        )

    # --- optional capabilities ---

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        client_order_id: str,
        time_in_force: str = "GTC",
    ) -> OrderResult:
        try:
            data = self._client.place_limit_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                price=price,
                client_order_id=client_order_id,
                time_in_force=time_in_force,
            )
        except BinanceError as exc:
            log.error("live limit order failed for %s %s qty=%s: %s", side, symbol, qty, exc)
            return OrderResult(order_id=client_order_id, status="rejected", error=str(exc))
        return self._resting_result(data, client_order_id)

    def place_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        limit_price: float | None = None,
        client_order_id: str = "",
    ) -> OrderResult:
        try:
            data = self._client.place_stop_order(
                symbol=symbol,
                side=side,
                quantity=qty,
                stop_price=stop_price,
                limit_price=limit_price,
                client_order_id=client_order_id,
            )
        except BinanceError as exc:
            # A stop the venue refuses (e.g. stopPrice on the wrong side of the
            # market, or insufficient balance) is a *rejected* order, not an
            # ambiguous one: nothing is resting, so the caller may retry safely.
            log.error("live stop order failed for %s %s stop=%s: %s", side, symbol, stop_price, exc)
            return OrderResult(order_id=client_order_id, status="rejected", error=str(exc))
        log.info("live stop resting %s %s qty=%s @ stop %s", side, symbol, qty, stop_price)
        return self._resting_result(data, client_order_id)

    def cancel_order(self, symbol: str, client_order_id: str) -> OrderResult:
        try:
            data = self._client.cancel_order(symbol, client_order_id=client_order_id)
        except BinanceError as exc:
            message = str(exc).lower()
            if any(marker in message for marker in _NO_ORDER_MARKERS):
                # Already gone (filled, expired or cancelled): nothing to do.
                return OrderResult(order_id=client_order_id, status="canceled")
            return OrderResult(order_id=client_order_id, status=_STATUS_UNKNOWN, error=str(exc))
        return OrderResult(order_id=str(data.get("orderId", client_order_id)), status="canceled")

    def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch open orders for a symbol (used by the reconcile job).

        Returns list of dicts each with a `client_order_id` key. Network or
        auth errors surface as a BinanceError (treated as drift by reconcile).
        """
        raw = self._client.open_orders(symbol)
        return [
            {"client_order_id": str(o.get("clientOrderId", "")), "order": o}
            for o in raw
        ]

    def get_balance(self, asset: str = "USDT") -> float:
        """Free + locked balance at the venue.

        Propagates BinanceError on failure (the engine logs it and falls back to
        the configured paper balance) rather than returning 0.0 — a zero would
        silently refuse every entry, and a fabricated figure would size a real
        order from a number nobody measured.
        """
        return float(self._client.get_balance(asset))

    # --- helpers ---

    def _resting_result(self, data: dict[str, Any], fallback_id: str) -> OrderResult:
        """Map an accepted (NEW/resting) order payload, tolerating an instant fill."""
        status_raw = str(data.get("status", "")).lower()
        filled_qty = float(data.get("executedQty", 0.0) or 0.0)
        if status_raw == "filled" or (filled_qty > 0 and status_raw in ("", "new")):
            avg, fee = self._avg_and_fee(data)
            return OrderResult(
                order_id=str(data.get("orderId", fallback_id)),
                status=_STATUS_FILLED,
                filled_qty=filled_qty,
                avg_fill_price=avg,
                fee=fee,
            )
        if status_raw == "partially_filled":
            avg, fee = self._avg_and_fee(data)
            return OrderResult(
                order_id=str(data.get("orderId", fallback_id)),
                status="partial",
                filled_qty=filled_qty,
                avg_fill_price=avg,
                fee=fee,
            )
        if status_raw == "rejected":
            return OrderResult(
                order_id=str(data.get("orderId", fallback_id)),
                status="rejected",
                error=str(data.get("msg", "rejected by exchange")),
            )
        order_id = str(data.get("orderId", fallback_id))
        # Carry the exchange id in the error-free fields only; callers key on
        # the client_order_id they supplied for a later cancel.
        return OrderResult(order_id=order_id, status="open")

    @staticmethod
    def _avg_and_fee(data: dict[str, Any]) -> tuple[float | None, float]:
        fills = data.get("fills") or []
        if fills:
            qty_sum = sum(float(f["qty"]) for f in fills)
            avg = (
                sum(float(f["price"]) * float(f["qty"]) for f in fills) / qty_sum
                if qty_sum
                else None
            )
            fee = sum(float(f.get("commission", 0.0)) for f in fills)
            return avg, fee
        price = float(data.get("price", 0.0) or 0.0)
        return (price or None), 0.0
