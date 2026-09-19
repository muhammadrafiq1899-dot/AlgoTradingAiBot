"""Candle store: persist normalized candles and drive backfill."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
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
        """Insert candles, replacing any that already exist (by PK).

        Returns the number of rows actually written (not the input length).

        Feeds hand back their whole recent window (up to 1000 candles) on every
        tick, and all but the last one or two of those are already stored. Two
        things keep this cheap — together they were costing ~10s per 1000 candles
        and pinning a full CPU core inside the tick on a phone:

        * Only rows at or after the newest stored candle for that symbol+interval
          are written. Older candles are immutable history; rewriting them every
          minute is pure DB/WAL churn. The newest incoming candle is still
          refreshed, because the still-forming candle changes every tick.
        * The write is a single batched ``INSERT ... ON CONFLICT DO UPDATE``
          instead of one DELETE per row. The old loop called
          ``session.execute(delete(...))`` once per row, and SQLAlchemy's default
          autoflush flushed every object added so far on each of those calls —
          making a 1000-row batch O(n^2).
        """
        if not candles:
            return 0

        latest_stored: dict[tuple[str, str], int | None] = {}
        rows: list[dict[str, Any]] = []
        for c in candles:
            key = (c.symbol, c.interval)
            if key not in latest_stored:
                latest_stored[key] = self.latest_ts(c.symbol, c.interval)
            stored = latest_stored[key]
            if stored is not None and c.ts < stored:
                continue  # immutable history we already have
            rows.append(
                {
                    "symbol": c.symbol,
                    "interval": c.interval,
                    "ts": c.ts,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
            )
        if not rows:
            return 0

        stmt = sqlite_insert(CandleRow)
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "interval", "ts"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "volume": stmt.excluded.volume,
            },
        )
        self._session.execute(stmt, rows)
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
