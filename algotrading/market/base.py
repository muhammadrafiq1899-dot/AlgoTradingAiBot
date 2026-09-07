"""Market data provider interface and normalized candle model."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Candle:
    """Normalized OHLCV candle.

    ts is epoch milliseconds aligned to the start of the interval.
    All price fields are floats; volume in base asset units.
    """

    symbol: str
    interval: str  # e.g. "1m", "1h"
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "ts": self.ts,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class MarketDataProvider(ABC):
    """Fetches normalized candles from a venue.

    Implementations must not mutate exchange state — read-only market data.
    """

    @abstractmethod
    def fetch_klines(self, symbol: str, interval: str, since_ms: int | None = None) -> list[Candle]:
        """Fetch OHLCV candles for a symbol+interval, oldest first.

        If since_ms is given, return candles on/after that timestamp. Otherwise
        return the most recent candles (venue-defined count).
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_ticker_price(self, symbol: str) -> float:
        """Return the latest tradable price for a symbol."""
        raise NotImplementedError
