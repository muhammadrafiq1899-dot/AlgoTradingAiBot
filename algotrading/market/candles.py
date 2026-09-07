"""Candle store: persist normalized candles and drive backfill."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from algotrading.db.models import Candle as CandleRow
from algotrading.market.base import Candle, MarketDataProvider

log = logging.getLogger(__name__)

# Per-interval millisecond durations
INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class CandleStore:
    """Persists normalized candles and manages backfill/cleanup."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, candles: list[Candle]) -> int:
        """Insert candles, replacing any that already exist (by PK). Returns count."""
        if not candles:
            return 0
        rows = [
            CandleRow(
                symbol=c.symbol,
                interval=c.interval,
                ts=c.ts,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
            )
            for c in candles
        ]
        # Upsert: delete-then-insert within a transaction for simplicity/correctness.
        for r in rows:
            self._session.execute(
                delete(CandleRow).where(
                    CandleRow.symbol == r.symbol,
                    CandleRow.interval == r.interval,
                    CandleRow.ts == r.ts,
                )
            )
            self._session.add(r)
        self._session.commit()
        return len(rows)

    def get(
        self, symbol: str, interval: str, limit: int = 500, ascending: bool = True
    ) -> list[CandleRow]:
        stmt = (
            select(CandleRow)
            .where(CandleRow.symbol == symbol, CandleRow.interval == interval)
            .order_by(CandleRow.ts.asc() if ascending else CandleRow.ts.desc())
            .limit(limit)
        )
        return list(self._session.execute(stmt).scalars())

    def latest_ts(self, symbol: str, interval: str) -> int | None:
        row = self.get(symbol, interval, limit=1, ascending=False)
        return row[0].ts if row else None

    def prune(self, symbol: str, interval: str, keep_hours: int = 48) -> int:
        """Delete candles older than keep_hours for a symbol+interval."""
        cutoff = int((datetime.now(timezone.utc) - timedelta(hours=keep_hours)).timestamp() * 1000)
        res = self._session.execute(
            delete(CandleRow).where(
                CandleRow.symbol == symbol,
                CandleRow.interval == interval,
                CandleRow.ts < cutoff,
            )
        )
        self._session.commit()
        return res.rowcount or 0


def backfill(
    provider: MarketDataProvider,
    store: CandleStore,
    symbol: str,
    interval: str,
    days: int,
) -> int:
    """Backfill `days` of historical candles for a symbol+interval.

    Paginates using ccxt's `since` (max ~1000 candles/request). Returns count
    of candles stored.
    """
    interval_ms = INTERVAL_MS[interval]
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    since_ms = int(since.timestamp() * 1000)
    # Align to interval boundary
    since_ms = since_ms - (since_ms % interval_ms)

    total = 0
    while since_ms < int(now.timestamp() * 1000):
        batch = provider.fetch_klines(symbol, interval, since_ms=since_ms)
        if not batch:
            break
        # Only store completed (closed) candles; skip the still-forming last one
        # unless it's the final batch boundary. Simpler: keep all, dedupe handles it.
        total += store.upsert(batch)
        last_ts = batch[-1].ts
        # Advance by one interval; if no progress, stop to avoid infinite loop.
        if last_ts + interval_ms <= since_ms:
            break
        since_ms = last_ts + interval_ms
        log.debug("backfill %s %s at %s (%s candles)", symbol, interval, since_ms, total)
    return total
