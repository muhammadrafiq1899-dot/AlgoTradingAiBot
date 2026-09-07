"""Paper trading gateway: simulated fills against live or demo prices.

Fills at the current market price plus a configurable slippage. It never talks
to an exchange, so the whole pipeline (signal -> intent -> fill -> ledger) can
be exercised without API keys or network (demo mode) or against live public
prices (paper mode).
"""
from __future__ import annotations

import logging

from algotrading.execution.base import OrderResult
from algotrading.market.base import MarketDataProvider

log = logging.getLogger(__name__)


class PaperGateway:
    def __init__(
        self,
        provider: MarketDataProvider,
        slippage_pct: float = 0.05,
        fill_delay_ms: int = 0,
    ) -> None:
        self._provider = provider
        self._slippage = slippage_pct / 100.0
        self._fill_delay = fill_delay_ms

    def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        client_order_id: str,
    ) -> OrderResult:
        try:
            market = self._provider.fetch_ticker_price(symbol)
        except Exception as exc:  # noqa: BLE001 - provider failure must not crash engine
            return OrderResult(
                order_id=client_order_id,
                status="rejected",
                error=f"price fetch failed: {exc}",
            )
        # Slippage: buy fills higher, sell fills lower.
        if side == "buy":
            fill = market * (1 + self._slippage)
        else:
            fill = market * (1 - self._slippage)
        # Paper fee = 0.1% of notional (spot taker) for realism.
        fee = qty * fill * 0.001
        log.info("paper fill %s %s qty=%s @ %.2f (market %.2f)", side, symbol, qty, fill, market)
        return OrderResult(
            order_id=client_order_id,
            status="filled",
            filled_qty=qty,
            avg_fill_price=fill,
            fee=fee,
        )

    def get_order(self, symbol: str, client_order_id: str) -> OrderResult:
        # Paper orders are synchronous: immediately filled (or rejected above).
        return OrderResult(order_id=client_order_id, status="filled")

    def get_open_orders(self, symbol: str) -> list[dict]:
        # Paper orders fill instantly; nothing stays open on the venue.
        return []
