"""Synthetic candle provider for offline testing / demo mode.

Generates a deterministic-ish random walk of candles so the whole pipeline
(strategy -> signal -> intent -> fill -> ledger) can be exercised with no
network access or API keys.
"""
from __future__ import annotations

import math
import random
import time

from algotrading.market.base import Candle, MarketDataProvider

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
}


class DemoProvider(MarketDataProvider):
    """Synthetic price feed. Deterministic when seeded."""

    def __init__(
        self,
        seed: int = 42,
        start_price: float = 60_000.0,
        vol_per_step: float = 0.002,
    ) -> None:
        self._rng = random.Random(seed)
        self._price = start_price
        self._vol = vol_per_step
        self._history: dict[tuple[str, str], list[Candle]] = {}

    def _step(self) -> float:
        # Random walk with slight upward drift to occasionally trigger signals.
        drift = 0.0002
        ret = self._rng.gauss(drift, self._vol)
        self._price = max(1.0, self._price * (1 + ret))
        return self._price

    def _gen(self, symbol: str, interval: str, count: int) -> list[Candle]:
        interval_ms = INTERVAL_MS.get(interval, 60_000)
        now_ms = int(time.time() * 1000)
        aligned = now_ms - (now_ms % interval_ms)
        start_ts = aligned - (count - 1) * interval_ms

        candles: list[Candle] = []
        price = self._price
        for i in range(count):
            ts = start_ts + i * interval_ms
            o = price
            c = price * (1 + self._rng.gauss(0.0002, self._vol))
            h = max(o, c) * (1 + abs(self._rng.gauss(0, self._vol / 3)))
            l = min(o, c) * (1 - abs(self._rng.gauss(0, self._vol / 3)))
            v = self._rng.uniform(1, 10)
            candles.append(
                Candle(symbol, interval, ts, o, h, l, c, v)
            )
            price = c
        self._price = price
        self._history[(symbol, interval)] = candles
        return candles

    def fetch_klines(self, symbol: str, interval: str, since_ms: int | None = None) -> list[Candle]:
        count = 300
        candles = self._gen(symbol, interval, count)
        if since_ms is not None:
            candles = [c for c in candles if c.ts >= since_ms]
        return candles

    def fetch_ticker_price(self, symbol: str) -> float:
        return self._price

    def now_ms(self) -> int:
        return int(time.time() * 1000)
