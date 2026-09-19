"""Child-process entry point for the parameter search.

Run it directly:

    python -m algotrading.optimize.runner --strategy ema_crossover \
        --symbol BTC/USDT --interval 1h --out result.json

It loads candles from the local database through the SAME loader the backtest
UI uses (`algotrading.backtest.stored`), searches parameter combinations under
the `settings.optimize` budgets, and writes a JSON artifact into
`settings.optimize.results_dir`. It prints exactly one summary line.

Deliberate limits, because this process runs unattended:

  * It imports no telegram, api, scheduler or execution module — there is no
    code path here that can send a message or place an order.
  * It never writes settings and never touches the active strategy: the artifact
    is data, and `algotrading.optimize.proposal` is the only thing that may turn
    it into a PENDING recommendation (for a human to approve).
  * `--timeout`/`optimize.timeout_seconds` bounds the search from the inside, so
    an artifact is still written when the parent would otherwise kill it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

log = logging.getLogger("algotrading.optimize.runner")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NO_DATA = 3
EXIT_FAILED = 4


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m algotrading.optimize.runner",
        description="Search strategy parameters out-of-process (proposal only).",
    )
    parser.add_argument("--strategy", default="ema_crossover", help="registered strategy name")
    parser.add_argument("--symbol", default=None, help="trading pair (default: configured)")
    parser.add_argument("--interval", default=None, help="candle interval (default: configured)")
    parser.add_argument(
        "--combinations", type=int, default=None,
        help="candidate cap (default settings.optimize.max_combinations)",
    )
    parser.add_argument(
        "--objective", default=None, choices=("sharpe", "pnl_drawdown", "total_pnl"),
        help="ranking objective (default settings.optimize.objective)",
    )
    parser.add_argument("--folds", type=int, default=None, help="walk-forward folds")
    parser.add_argument(
        "--grid", default=None,
        help='explicit parameter space as JSON, e.g. \'{"fast_period": [5, 9, 14]}\'',
    )
    parser.add_argument("--out", default=None, help="artifact path (default: results_dir)")
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="internal wall-clock budget in seconds (default settings.optimize.timeout_seconds)",
    )
    parser.add_argument("--quiet", action="store_true", help="print nothing on success")
    return parser.parse_args(argv)


def run(
    *,
    strategy: str = "ema_crossover",
    symbol: str | None = None,
    interval: str | None = None,
    combinations: int | None = None,
    objective: str | None = None,
    folds: int | None = None,
    grid: dict[str, Sequence[Any]] | None = None,
    out: str | Path | None = None,
    timeout: int | None = None,
    settings: Any = None,
    quiet: bool = False,
) -> tuple[int, dict[str, Any] | None, Path | None]:
    """Load candles, search, write the artifact. Returns (exit code, artifact, path).

    Separated from `main` so a test can exercise the whole pipeline without
    spawning a process (and without touching the real database file).
    """
    from algotrading.backtest.stored import load_candles
    from algotrading.config import load_settings
    from algotrading.db import get_session, init_db
    from algotrading.optimize.search import default_param_space, run_search

    settings = settings if settings is not None else load_settings()
    opt = settings.optimize
    if not opt.enabled:
        print("optimize: disabled by settings.optimize.enabled=false", file=sys.stderr)
        return EXIT_USAGE, None, None

    db_path = settings.db_path
    if not Path(db_path).exists():
        print(
            f"optimize: no database at {db_path}; run the bot once to backfill candles",
            file=sys.stderr,
        )
        return EXIT_NO_DATA, None, None
    # Idempotent: creating the schema is what makes the loader safe to call on a
    # fresh install, and it never mutates existing rows.
    init_db(db_path)

    max_candles = int(getattr(settings.backtest, "max_candles", 20_000))
    with get_session(db_path) as session:
        # The stored loader returns the OLDEST `limit` candles for the pair
        # (see CandleStore.get); that is the same window the Telegram backtest
        # replays, so a proposal cites the data the UI would show.
        sym, iv, candles = load_candles(
            session, settings, symbol=symbol, interval=interval, limit=max_candles
        )
    if not candles:
        print(
            f"optimize: no stored candles for {sym} {iv}; backfill data first",
            file=sys.stderr,
        )
        return EXIT_NO_DATA, None, None

    space = dict(grid) if grid else default_param_space(strategy, settings)
    budget = int(timeout if timeout is not None else opt.timeout_seconds)
    deadline = time.monotonic() + budget if budget > 0 else None

    result = run_search(
        candles,
        strategy,
        space,
        objective,
        folds=folds,
        max_combinations=combinations,
        settings=settings,
        deadline=deadline,
    )

    artifact = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "algotrading.optimize.runner",
        "db_path": db_path,
        "timeout_seconds": budget,
        "search": result.to_dict(),
    }

    out_path = Path(out) if out else _default_out_path(settings, strategy, sym, iv)
    if not out_path.is_absolute():
        out_path = Path(opt.results_dir) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=False), encoding="utf-8")

    if not quiet:
        print(result.summary_line() + f" -> {out_path}")
    return EXIT_OK, artifact, out_path


def _default_out_path(settings: Any, strategy: str, symbol: str, interval: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe = f"{strategy}_{symbol}_{interval}".replace("/", "-").replace(" ", "")
    return Path(settings.optimize.results_dir) / f"search_{safe}_{stamp}.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    grid = json.loads(args.grid) if args.grid else None
    try:
        code, _artifact, _path = run(
            strategy=args.strategy,
            symbol=args.symbol,
            interval=args.interval,
            combinations=args.combinations,
            objective=args.objective,
            folds=args.folds,
            grid=grid,
            out=args.out,
            timeout=args.timeout,
            quiet=args.quiet,
        )
    except Exception as exc:  # a child process must exit with a code, not a traceback alone
        log.error("optimize: search failed: %s", exc)
        print(f"optimize: failed: {exc}", file=sys.stderr)
        return EXIT_FAILED
    return code


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
