#!/usr/bin/env python3
"""Run a parameter search out-of-process, optionally leaving a PENDING proposal.

Usage:
    python scripts/optimize.py --strategy ema_crossover --symbol BTC/USDT \\
        --interval 1h --combinations 40
    python scripts/optimize.py --strategy ema_crossover --propose

The search runs in a SEPARATE process (nice'd, time-boxed, lockfile-guarded):
the bot's 60-second tick shares this interpreter with APScheduler and Telegram,
so a CPU-bound grid search must never run here — see
`algotrading/optimize/spawn.py` for the full reasoning.

`--propose` only ever creates a PENDING `AIRecommendation`. Nothing is applied:
approval is a human action in Telegram, by design (PROJECT_MAP.md §11).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from algotrading.config import load_settings  # noqa: E402
from algotrading.optimize.spawn import (  # noqa: E402
    SearchBusyError,
    start_search_subprocess,
    wait_search,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Out-of-process parameter search (proposal only, never applied)."
    )
    parser.add_argument("--strategy", default="ema_crossover", help="registered strategy name")
    parser.add_argument("--symbol", default=None, help="trading pair (default: configured)")
    parser.add_argument("--interval", default=None, help="candle interval (default: configured)")
    parser.add_argument(
        "--combinations", type=int, default=None,
        help="candidate cap (default settings.optimize.max_combinations)",
    )
    parser.add_argument(
        "--propose", action="store_true",
        help="leave a PENDING recommendation for the best candidate (human approval required)",
    )
    parser.add_argument("--timeout", type=int, default=None, help="wall-clock budget in seconds")
    parser.add_argument("--quiet", action="store_true", help="print only the artifact path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings()

    if not settings.optimize.enabled:
        print("optimize: disabled by settings.optimize.enabled=false")
        return 2

    try:
        handle = start_search_subprocess(
            strategy=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            combinations=args.combinations,
            timeout=args.timeout,
            settings=settings,
        )
    except SearchBusyError as exc:
        print(f"optimize: {exc}")
        return 1

    code = wait_search(handle, timeout=args.timeout, settings=settings)
    print(f"search finished (exit {code}); artifact: {handle.out_path}")
    if handle.log_path.exists() and code != 0:
        print(f"see the child log: {handle.log_path}")

    if args.propose:
        return _propose(settings, handle.out_path)
    return 0 if code == 0 else 1


def _propose(settings, artifact_path: Path) -> int:
    """Create a PENDING recommendation from the artifact. Never applies it."""
    from algotrading.db import get_session
    from algotrading.optimize.proposal import propose_from_result
    from algotrading.optimize.search import SearchResult

    if not artifact_path.exists():
        print(f"optimize: no artifact at {artifact_path}; nothing to propose")
        return 1
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    result = SearchResult.from_dict(payload.get("search") or {})

    with get_session(settings.db_path) as session:
        try:
            rec = propose_from_result(session, result)
        except ValueError as exc:
            print(f"optimize: not proposing: {exc}")
            return 1
    print(
        f"proposed recommendation #{rec.id} ({rec.kind}, {rec.strategy_name}) "
        f"status={rec.status} — approve or reject it in Telegram"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
