"""Binance market data provider backed by the lightweight REST client."""
from __future__ import annotations

import time

from algotrading.market.base import Candle, MarketDataProvider
from algotrading.market.binance_rest import BinanceRestClient
from algotrading.market.circuit_breaker import CircuitBreaker, CircuitOpenError, get_circuit


class BinanceMarketProvider(MarketDataProvider):
    """Read-only market data from Binance Spot via REST (public endpoints).

    No API keys required for klines/ticker, so this works in paper and live.
    """

    def __init__(self, client: BinanceRestClient | None = None, circuit_breaker: CircuitBreaker | None = None) -> None:
        self._client = client or BinanceRestClient()
        self._circuit = circuit_breaker or get_circuit("binance_market", fail_threshold=5, reset_timeout=60.0)

    # --- MarketDataProvider ---

    def fetch_klines(self, symbol: str, interval: str, since_ms: int | None = None) -> list[Candle]:
        def _call():
            raw = self._client.klines(symbol, interval, start_ms=since_ms, limit=1000)
            return [self._to_candle(symbol, interval, row) for row in raw]

        try:
            return self._circuit.call(_call)
        except CircuitOpenError as exc:
            log.warning("Circuit breaker open for %s/%s: %s", symbol, interval, exc)
            raise

    def fetch_ticker_price(self, symbol: str) -> float:
        try:
            return self._circuit.call(self._client.ticker_price, symbol)
        except CircuitOpenError as exc:
            log.warning("Circuit breaker open for ticker %s: %s", symbol, exc)
            raise

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
