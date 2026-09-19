"""Candle store: persist normalized candles and drive backfill."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

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
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
    # Calendar months vary (28-31 days); 30 days is the stepping approximation.
    # It errs short for 31-day months, which only re-fetches overlapping candles
    # that the upsert dedupes.
    "1M": 2_592_000_000,
}

# Intervals whose length is an approximation rather than an exact duration.
APPROXIMATE_INTERVALS = frozenset({"1M"})

# How far a paging loop advances after a batch for an approximate interval.
# A calendar month is 28-31 days while INTERVAL_MS["1M"] is a flat 30, so
# stepping a flat 30 days can *overshoot* a short month (Feb 1 + 30d = Mar 3,
# skipping the March candle entirely). Stepping by the shortest possible month
# instead makes consecutive requests overlap: overlap is deduped by the upsert
# and a candle can never be skipped. Progress is still guaranteed because the
# step is strictly positive (28d > 0), so the loop cannot spin forever.
APPROXIMATE_STEP_MS: dict[str, int] = {"1M": 28 * 86_400_000}

# Hard ceiling on requests per backfill call. The progress guard below already
# stops a provider that ignores `since_ms`; this is the belt-and-braces cap for
# a provider that drifts forward by tiny amounts forever (a phone must not be
# held in a fetch loop).
MAX_PAGES = 10_000


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
    *,
    since_ms: int | None = None,
    max_rows: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> int:
    """Backfill up to `days` of historical candles for a symbol+interval.

    Pages through history with `provider.fetch_klines(..., since_ms=...)`
    (max ~1000 candles/request) and upserts each batch, so a re-run is cheap:
    `CandleStore.upsert` never rewrites immutable history.

    Args:
        provider: market data source.
        store: candle store to write into.
        symbol: e.g. "BTC/USDT".
        interval: one of INTERVAL_MS.
        days: history window ending now (ignored when `since_ms` is given).
        since_ms: explicit first candle timestamp to fetch. Callers that want a
            resumable download pass the newest stored candle here instead of
            re-fetching the whole window.
        max_rows: stop once this many rows have been *stored* (not fetched).
            This is the disk guard for a phone: the caller passes the remaining
            budget of its own row cap and can resume later. A batch is trimmed
            to the remaining budget before it is written, so the total is a hard
            ceiling rather than an approximate one.
        progress: called with the running stored-row count after every page.

    Returns:
        Number of rows actually written (post-dedupe), not the fetched count.

    Stepping rule (why '1M' is safe):
        Batches advance to `batch[-1].ts + step`, where `step` is the exact
        interval length for normal intervals and the *shortest possible month*
        (28 days, see APPROXIMATE_STEP_MS) for APPROXIMATE_INTERVALS. A flat
        30-day step would skip a candle whenever a calendar month is shorter
        than 30 days; a 28-day step only ever overlaps, and overlapping rows
        are deduped by the upsert, so no candle is skipped and no duplicate is
        stored. The loop terminates because the step is strictly positive, the
        `since_ms` guard stops a provider that ignores the cursor, and MAX_PAGES
        caps a provider that creeps forward forever.
    """
    if interval not in INTERVAL_MS:
        raise ValueError(
            f"unknown interval {interval!r}; known: {sorted(INTERVAL_MS)}"
        )
    if days < 0:
        raise ValueError("days must be >= 0")

    interval_ms = INTERVAL_MS[interval]
    step_ms = APPROXIMATE_STEP_MS.get(interval, interval_ms)

    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    if since_ms is None:
        since = now - timedelta(days=days)
        since_ms = int(since.timestamp() * 1000)
    since_ms = max(0, int(since_ms))
    # Align to interval boundary. `% interval_ms` always moves the start earlier
    # (or leaves it), so it can never skip a candle at the window edge.
    since_ms = since_ms - (since_ms % interval_ms)

    total = 0
    pages = 0
    while since_ms < now_ms:
        if pages >= MAX_PAGES:
            log.warning(
                "backfill %s %s: page cap reached (%s), stopping",
                symbol, interval, MAX_PAGES,
            )
            break
        batch = provider.fetch_klines(symbol, interval, since_ms=since_ms)
        if not batch:
            break
        if max_rows is not None:
            remaining = max_rows - total
            if remaining <= 0:
                break
            # Trim before writing: the row cap is a disk guard, so it has to be
            # a ceiling, not "whatever the last page happened to contain".
            if len(batch) > remaining:
                batch = batch[:remaining]
        total += store.upsert(batch)
        pages += 1
        if progress is not None:
            progress(total)
        if max_rows is not None and total >= max_rows:
            break
        last_ts = batch[-1].ts
        # Advance by one step; if that makes no progress the provider ignored
        # `since_ms`, so stop rather than refetch the same window forever.
        next_ms = last_ts + step_ms
        if next_ms <= since_ms:
            break
        since_ms = next_ms
        log.debug("backfill %s %s at %s (%s candles)", symbol, interval, since_ms, total)
    return total
