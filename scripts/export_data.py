#!/usr/bin/env python3
"""Export trades, the equity curve and a state summary from the database.

Usage:
    python scripts/export_data.py --dir data/exports
    python scripts/export_data.py --trades-csv data/trades.csv --summary-json data/summary.json

Read-only: the exporters never change bot state, and every file is written
atomically (temp file + rename), so an interrupted export never leaves a
half-written file. The same code backs `algobot export` and the API's
`/export/*` routes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root or from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algotrading.config import load_settings  # noqa: E402
from algotrading.db import get_session  # noqa: E402
from algotrading.store.export import (  # noqa: E402
    export_equity_csv,
    export_summary_json,
    export_trades_csv,
    export_trades_json,
)


def main() -> int:
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="Export trades / equity curve / summary from the bot database.",
    )
    parser.add_argument("--db", default=settings.db_path, help="database file to read")
    parser.add_argument("--dir", default="", help="write all four files into this directory")
    parser.add_argument("--trades-csv", default="", help="trades as CSV")
    parser.add_argument("--trades-json", default="", help="trades as a JSON array")
    parser.add_argument("--equity-csv", default="", help="equity curve as CSV")
    parser.add_argument("--summary-json", default="", help="state summary as JSON")
    args = parser.parse_args()

    out_dir = args.dir
    targets = {
        "trades_csv": args.trades_csv or (f"{out_dir}/trades.csv" if out_dir else ""),
        "trades_json": args.trades_json or (f"{out_dir}/trades.json" if out_dir else ""),
        "equity_csv": args.equity_csv or (f"{out_dir}/equity.csv" if out_dir else ""),
        "summary_json": args.summary_json or (f"{out_dir}/summary.json" if out_dir else ""),
    }
    if not any(targets.values()):
        parser.error("nothing to export: pass --dir, or one of --trades-csv/--trades-json/"
                     "--equity-csv/--summary-json")

    initial_balance = float(settings.risk.paper_initial_balance)
    with get_session(args.db) as session:
        if targets["trades_csv"]:
            n = export_trades_csv(session, targets["trades_csv"])
            print(f"trades      -> {targets['trades_csv']} ({n} rows)")
        if targets["trades_json"]:
            n = export_trades_json(session, targets["trades_json"])
            print(f"trades JSON -> {targets['trades_json']} ({n} rows)")
        if targets["equity_csv"]:
            n = export_equity_csv(
                session, targets["equity_csv"], initial_balance=initial_balance
            )
            print(f"equity      -> {targets['equity_csv']} ({n} points)")
        if targets["summary_json"]:
            summary = export_summary_json(
                session,
                targets["summary_json"],
                mode=settings.mode,
                initial_balance=initial_balance,
            )
            counts = summary["counts"]
            print(
                f"summary     -> {targets['summary_json']} "
                f"({counts['trades']} trades, {counts['open_positions']} open)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
