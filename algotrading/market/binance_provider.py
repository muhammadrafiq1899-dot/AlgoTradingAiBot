"""Binance market data provider backed by the lightweight REST client."""
from __future__ import annotations

import time

from algotrading.market.base import Candle, MarketDataProvider
from algotrading.market.binance_rest import BinanceRestClient


class BinanceMarketProvider(MarketDataProvider):
    """Read-only market data from Binance Spot via REST (public endpoints).

    No API keys required for klines/ticker, so this works in paper and live.
    """

    def __init__(self, client: BinanceRestClient | None = None) -> None:
        self._client = client or BinanceRestClient()

    # --- MarketDataProvider ---

    def fetch_klines(self, symbol: str, interval: str, since_ms: int | None = None) -> list[Candle]:
        raw = self._client.klines(symbol, interval, start_ms=since_ms, limit=1000)
        return [self._to_candle(symbol, interval, row) for row in raw]

    def fetch_ticker_price(self, symbol: str) -> float:
        return self._client.ticker_price(symbol)

    # --- helpers ---

    @staticmethod
    def _to_candle(symbol: str, interval: str, row: list[float]) -> Candle:
        # Binance kline row: [open_time, o, h, l, c, v, close_time, ...]
        return Candle(
            symbol=symbol,
            interval=interval,
            ts=int(row[0]),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
        )

    def now_ms(self) -> int:
        return int(time.time() * 1000)
