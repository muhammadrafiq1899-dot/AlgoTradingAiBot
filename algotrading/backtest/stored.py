"""Load stored candles for a backtest from the local database.

Both the chat `backtest` tool and the `/strategies` inline buttons need the same
thing: the configured symbol/interval, the candles on disk for it, and nothing
else. Centralising it here keeps the three formerly-duplicated
``CandleStore.get`` → ``Candle`` mappings in one place and guarantees the two
callers replay identical data.

The runner itself (:mod:`algotrading.backtest.runner`) stays pure and DB-free;
this module is only the DB adapter in front of it.
"""
from __future__ import annotations

from typing import Any, Sequence

from sqlalchemy.orm import Session

from algotrading.market.base import Candle
from algotrading.market.candles import CandleStore

DEFAULT_LIMIT = 500


def resolve_market(settings, symbol: str | None = None, interval: str | None = None) -> tuple[str, str]:
    """Pick the symbol/interval to backtest, defaulting to the configured set."""
    sym = symbol or (settings.market.symbols or ["BTC/USDT"])[0]
    iv = interval or ("1h" if "1h" in settings.market.intervals else "1m")
    return sym, iv


def load_candles(
    session: Session,
    settings,
    symbol: str | None = None,
    interval: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> tuple[str, str, list[Candle]]:
    """Return ``(symbol, interval, candles)`` oldest -> newest (empty if none)."""
    sym, iv = resolve_market(settings, symbol, interval)
    rows: Sequence[Any] = CandleStore(session).get(sym, iv, limit=limit)
    candles = [
        Candle(
            symbol=r.symbol,
            interval=r.interval,
            ts=r.ts,
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
        )
        for r in rows
    ]
    return sym, iv, candles
