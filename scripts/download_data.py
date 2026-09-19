#!/usr/bin/env python3
"""Download market history into the local database (resumable, row-capped).

Usage:
    python scripts/download_data.py --symbols BTC/USDT --intervals 1h --days 365
    python scripts/download_data.py --days 90 --intervals 1h,4h --export data/candles.csv

The heavy lifting lives in `algotrading.market.download.download_history`; this
script only parses flags and prints progress. Safe to re-run: only missing
candles are fetched, and a run stopped by the row cap continues where it left
off on the next run. Stop the bot first if it is running (one SQLite writer).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root or from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algotrading.config import load_settings  # noqa: E402
from algotrading.db import get_session, init_db  # noqa: E402
from algotrading.market.binance_provider import BinanceMarketProvider  # noqa: E402
from algotrading.market.candles import CandleStore, INTERVAL_MS  # noqa: E402
from algotrading.market.download import (  # noqa: E402
    DEFAULT_MAX_ROWS,
    download_history,
    export_candles,
)

# Stored candles written by --export (per symbol/interval pair).
EXPORT_LIMIT = 100_000


def _split(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="Download candle history into the local database.",
    )
    parser.add_argument(
        "--symbols",
        default=",".join(settings.market.symbols),
        help="comma-separated pairs (default: market.symbols)",
    )
    parser.add_argument(
        "--intervals",
        default=",".join(settings.market.intervals),
        help=f"comma-separated intervals from {sorted(INTERVAL_MS)} "
             "(default: market.intervals)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=int(settings.market.backfill_days),
        help="history window in days (default: market.backfill_days)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help=f"row cap for this run (default: {DEFAULT_MAX_ROWS})",
    )
    parser.add_argument(
        "--export",
        default="",
        help="optional: also write the stored candles to a .csv or .jsonl file",
    )
    parser.add_argument("--db", default=settings.db_path,
                        help="database file to write into")
    args = parser.parse_args()

    symbols = _split(args.symbols)
    intervals = _split(args.intervals)
    if not symbols or not intervals:
        parser.error("--symbols and --intervals must not be empty")
    unknown = [i for i in intervals if i not in INTERVAL_MS]
    if unknown:
        parser.error(f"unknown interval(s) {unknown}; known: {sorted(INTERVAL_MS)}")

    init_db(args.db)
    print(
        f"Downloading {args.days} days for {', '.join(symbols)} "
        f"at {', '.join(intervals)} (row cap {args.max_rows})..."
    )

    def _progress(symbol: str, interval: str, rows: int) -> None:
        print(f"  {symbol} {interval}: {rows} candles stored")

    with get_session(args.db) as session:
        store = CandleStore(session)
        report = download_history(
            BinanceMarketProvider(),
            store,
            symbols,
            intervals,
            args.days,
            max_rows=args.max_rows,
            progress=_progress,
        )
        if args.export:
            rows = []
            for entry in report["results"]:
                rows.extend(
                    store.get(entry["symbol"], entry["interval"], limit=EXPORT_LIMIT)
                )
            written = export_candles(rows, args.export)
            print(f"Exported {written} candles to {args.export}")

    print(f"Stored {report['total_rows']} rows across {len(report['results'])} pair(s).")
    if report["truncated"]:
        print("Row cap reached — re-run to continue where this stopped.")
    for entry in report["results"]:
        if entry["error"]:
            print(f"  !! {entry['symbol']} {entry['interval']}: {entry['error']}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
