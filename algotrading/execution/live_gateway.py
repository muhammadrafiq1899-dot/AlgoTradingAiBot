"""Live exchange gateway: real Binance Spot orders via the REST client.

Mirrors PaperGateway's interface (place_market_order / get_order) so the
execution engine is identical in paper and live modes. Signed endpoints
require API keys in .env (BINANCE_API_KEY / BINANCE_API_SECRET).

Safety notes:
- The caller persists intent-before-order, so a network error mid-request
  must not place a duplicate: create_order carries newClientOrderId = the
  idempotency key, and get_order can reconcile by that key.
- A failed POST that may or may not have reached the exchange returns status
  "unknown" so the engine stops retrying risk-sensitive actions until the
  reconcile job confirms.
"""
from __future__ import annotations

import logging

from algotrading.execution.base import OrderResult
from algotrading.market.binance_rest import BinanceError, BinanceRestClient

log = logging.getLogger(__name__)

# Binance market orders always fill entirely at the taker price.
_STATUS_FILLED = "filled"
_STATUS_UNKNOWN = "unknown"


class LiveGateway:
    def __init__(self, client: BinanceRestClient | None = None) -> None:
        self._client = client or BinanceRestClient()

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
        fills = data.get("fills") or []
        filled_qty = float(data.get("executedQty", 0.0))
        if fills:
            avg = sum(float(f["price"]) * float(f["qty"]) for f in fills) / sum(
                float(f["qty"]) for f in fills
            )
            fee = sum(float(f.get("commission", 0.0)) for f in fills)
        else:
            avg = float(data.get("price", 0.0)) or None
            fee = 0.0
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

    def get_open_orders(self, symbol: str) -> list[dict]:
        """Fetch open orders for a symbol (used by the reconcile job).

        Returns list of dicts each with a `client_order_id` key. Network or
        auth errors surface as a BinanceError (treated as drift by reconcile).
        """
        raw = self._client.open_orders(symbol)
        return [
            {"client_order_id": str(o.get("clientOrderId", "")), "order": o}
            for o in raw
        ]
