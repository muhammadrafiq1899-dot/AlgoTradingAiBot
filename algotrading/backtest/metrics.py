"""Backtest metrics: honest statistics on top of a replay result.

Pure Python on purpose. This runs on a phone (aarch64, no wheels for numpy/pandas
in the project's dependency set) and it must be importable by the optimizer child
without pulling in the bot's runtime. Everything here is stdlib `math`,
`statistics`, `datetime` and `random`.

Two families live in this module:

  * `compute_metrics` — descriptive statistics of ONE replay: Sharpe/Sortino from
    the per-bar equity curve, expectancy/profit factor/win-loss averages from the
    trades, max drawdown, exposure, trade durations and PnL breakdowns by
    calendar day/week/month/year.
  * `walk_forward`, `monte_carlo`, `bootstrap` — the "was that real?" layer. None
    of them is a substitute for out-of-sample data, and each says so in its own
    docstring; a number from a backtest is a hypothesis, not a result.

Every value returned is JSON-serialisable (no numpy scalars, no `inf`, no `nan`):
unbounded statistics come back as `None` and are documented where that happens.
"""
from __future__ import annotations

import logging
import math
import random
import statistics
from datetime import datetime, timezone
from typing import Any, Sequence

from algotrading.backtest.runner import BacktestResult

log = logging.getLogger(__name__)

# Bars per year per interval, used to annualise per-bar Sharpe/Sortino.
# A year of calendar time is the convention (crypto trades 24/7, so there is no
# "252 trading days" adjustment to make here): 365 days for sub-daily intervals,
# 365 for days, 52 for weeks, 12 for months. 1M/1w come from INTERVAL_MS'
# stepping convention, so they are approximations like the rest of that table.
BARS_PER_YEAR: dict[str, float] = {
    "1m": 365 * 24 * 60,
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "30m": 365 * 24 * 2,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
    "1w": 52,
    "1M": 12,
}
DEFAULT_BARS_PER_YEAR = 365 * 24  # assume 1h when the interval is unknown

# Percentiles reported by the resampling helpers.
PERCENTILES = (5, 50, 95)

DEFAULT_SEED = 42


def _interval_ms(interval: str) -> int:
    """Bar length in milliseconds; falls back to 1h for an unknown interval."""
    try:
        from algotrading.market.candles import INTERVAL_MS

        return int(INTERVAL_MS.get(interval, 3_600_000))
    except Exception:  # pragma: no cover - only without the package tree
        return 3_600_000


def annualisation_factor(interval: str) -> float:
    """sqrt(bars per year) — the multiplier that turns a per-bar ratio annual.

    Documented rather than magic: `sharpe = mean(bar_return)/stdev(bar_return) *
    sqrt(bars_per_year)`, with a zero risk-free rate and a sample (n-1) standard
    deviation. The absolute number is comparable across intervals only because
    every metric here divides by the same bars-per-year for its own interval.
    """
    return math.sqrt(BARS_PER_YEAR.get(interval, DEFAULT_BARS_PER_YEAR))


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (the classic `numpy.percentile` default).

    Implemented here (rather than imported) because numpy is not allowed; kept
    simple enough to check by hand in the tests.
    """
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * (q / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(sorted_values[int(pos)])
    return float(
        sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)
    )


def _safe_div(numerator: float, denominator: float) -> float | None:
    """Division that returns None instead of raising/inventing infinities."""
    if denominator == 0:
        return None
    return numerator / denominator


def per_bar_returns(equity_curve: Sequence[float]) -> list[float]:
    """Simple returns between consecutive equity points.

    A bar whose previous equity is <= 0 contributes nothing (the account is gone
    or the curve is degenerate); the alternative — log returns of a negative
    number — is a crash, not a metric.
    """
    returns: list[float] = []
    for prev, cur in zip(equity_curve, equity_curve[1:]):
        if prev <= 0:
            continue
        returns.append((cur - prev) / prev)
    return returns


def sharpe_ratio(
    equity_curve: Sequence[float], interval: str, *, risk_free: float = 0.0
) -> float | None:
    """Per-bar Sharpe, annualised. None when it is not defined (too few bars)."""
    returns = per_bar_returns(equity_curve)
    if len(returns) < 2:
        return None
    excess = [r - risk_free for r in returns]
    stdev = statistics.stdev(excess)
    if stdev == 0:
        return None
    return (statistics.fmean(excess) / stdev) * annualisation_factor(interval)


def sortino_ratio(
    equity_curve: Sequence[float], interval: str, *, target: float = 0.0
) -> float | None:
    """Annualised Sortino: mean excess return over downside deviation.

    Downside deviation is the root-mean-square of the negative excess returns
    (target 0 by default) over ALL bars, so a strategy that rarely loses but
    loses hard is penalised where Sharpe is not. None when there is no downside
    at all — that is "no measurable downside", not "infinitely good".
    """
    returns = per_bar_returns(equity_curve)
    if len(returns) < 2:
        return None
    excess = [r - target for r in returns]
    downside = [min(0.0, r) for r in excess]
    dd = math.sqrt(sum(d * d for d in downside) / len(downside))
    if dd == 0:
        return None
    return (statistics.fmean(excess) / dd) * annualisation_factor(interval)


def _bucket_key(ts_ms: int, period: str) -> str:
    """Calendar bucket for an epoch-ms timestamp, in UTC."""
    moment = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    if period == "day":
        return moment.strftime("%Y-%m-%d")
    if period == "week":
        year, week, _ = moment.isocalendar()
        return f"{year}-W{week:02d}"
    if period == "month":
        return moment.strftime("%Y-%m")
    if period == "year":
        return moment.strftime("%Y")
    raise ValueError(f"unknown period {period!r}")


def pnl_breakdown(result: BacktestResult, period: str) -> dict[str, dict[str, Any]]:
    """PnL/trade counts bucketed by calendar `period` ('day'|'week'|'month'|'year').

    Trades are bucketed by their EXIT timestamp: a trade is only a result once it
    is closed, and bucketing by entry would credit a Monday entry with a Friday
    loss. Buckets are complete (every UTC day/week/month/year that contains a
    closed trade appears, including empty-PnL ones), sorted oldest first.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for trade in result.trades:
        key = _bucket_key(trade.exit_ts, period)
        bucket = buckets.setdefault(
            key, {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0}
        )
        bucket["pnl"] += trade.pnl
        bucket["trades"] += 1
        if trade.pnl > 0:
            bucket["wins"] += 1
        elif trade.pnl < 0:
            bucket["losses"] += 1
    for bucket in buckets.values():
        bucket["pnl"] = round(bucket["pnl"], 8)
    return {key: buckets[key] for key in sorted(buckets)}


def max_drawdown_pct(equity_curve: Sequence[float]) -> float:
    """Largest peak-to-trough fall as a PERCENT of the running peak."""
    peak = None
    worst = 0.0
    for value in equity_curve:
        if peak is None or value > peak:
            peak = value
        if peak and peak > 0:
            worst = max(worst, (peak - value) / peak * 100.0)
    return worst


def compute_metrics(
    result: BacktestResult, candles: Sequence[Any] | None = None
) -> dict[str, Any]:
    """Descriptive statistics for one replay; JSON-serialisable, numpy-free.

    Args:
        result: a finished `BacktestResult` (fees and slippage already inside).
        candles: optional bar series the replay ran over. Used only to size the
            bar-return series when the result carries no equity curve and for
            nothing else — the trade breakdowns use the trade timestamps, so a
            metrics call on a persisted result does not need the data back.

    Returns:
        A flat dict. `sharpe`, `sortino` and `profit_factor` may be None when
        the sample cannot define them (see the individual docstrings); a caller
        that averages them must handle None rather than treating it as 0.
    """
    trades = list(result.trades)
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    n_bars = result.n_bars or (len(result.equity_curve) or len(candles or []))
    # The curve is optional (a result rebuilt from JSON has none), but exposure
    # is carried on the result either way: never fabricate a curve to derive it.
    equity = list(result.equity_curve)
    exposure = result.exposure

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    win_rate = result.win_rate if trades else 0.0
    avg_win = statistics.fmean(wins) if wins else None
    avg_loss = statistics.fmean(losses) if losses else None
    # Expectancy: average PnL per trade, decomposed as p*avg_win + (1-p)*avg_loss.
    # That identity holds for any trade list, so it is written the decomposed way
    # to make the win-rate/scale sensitivity visible.
    loss_rate = (len(losses) / len(trades)) if trades else 0.0
    expectancy = None
    if trades:
        expectancy = win_rate * (avg_win or 0.0) + loss_rate * (avg_loss or 0.0)

    durations_ms = [max(0, t.exit_ts - t.entry_ts) for t in trades]
    bar_ms = _interval_ms(result.interval)
    avg_duration_ms = statistics.fmean(durations_ms) if durations_ms else None

    return {
        # --- counts / headline ---
        "strategy_name": result.strategy_name,
        "symbol": result.symbol,
        "interval": result.interval,
        "n_trades": len(trades),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "n_flat": len(trades) - len(wins) - len(losses),
        "win_rate": round(win_rate, 6),
        "n_bars": n_bars,
        "initial_balance": round(result.initial_balance, 8),
        "final_balance": round(result.final_balance, 8),
        "total_pnl": round(sum(pnls), 8),
        "return_pct": round(
            (sum(pnls) / result.initial_balance * 100.0)
            if result.initial_balance
            else 0.0,
            6,
        ),
        "fees_paid": round(result.fees_paid, 8),
        # --- risk-adjusted ---
        "sharpe": _round_opt(sharpe_ratio(equity, result.interval)),
        "sortino": _round_opt(sortino_ratio(equity, result.interval)),
        "annualisation_factor": round(annualisation_factor(result.interval), 6),
        "max_drawdown": round(result.max_drawdown, 8),
        "max_drawdown_pct": round(max_drawdown_pct(equity), 6),
        # --- trade quality ---
        "expectancy": _round_opt(expectancy),
        "avg_win": _round_opt(avg_win),
        "avg_loss": _round_opt(avg_loss),
        "largest_win": round(max(wins), 8) if wins else None,
        "largest_loss": round(min(losses), 8) if losses else None,
        "gross_profit": round(gross_profit, 8),
        "gross_loss": round(gross_loss, 8),
        # None (not inf) when there are no losing trades: infinity is not JSON,
        # and "no losses yet" is not "infinitely good".
        "profit_factor": _round_opt(_safe_div(gross_profit, gross_loss)),
        # --- exposure / hold time ---
        "exposure": round(exposure, 6),
        "position_bars": result.position_bars,
        "avg_trade_duration_ms": _round_opt(avg_duration_ms),
        "avg_trade_duration_hours": _round_opt(
            avg_duration_ms / 3_600_000 if avg_duration_ms is not None else None
        ),
        "avg_trade_duration_bars": _round_opt(
            avg_duration_ms / bar_ms if avg_duration_ms is not None else None
        ),
        # --- time breakdowns ---
        "pnl_by_day": pnl_breakdown(result, "day"),
        "pnl_by_week": pnl_breakdown(result, "week"),
        "pnl_by_month": pnl_breakdown(result, "month"),
        "pnl_by_year": pnl_breakdown(result, "year"),
        # --- honesty pass-through ---
        "assumptions": list(result.assumptions),
    }


def _round_opt(value: float | None, digits: int = 8) -> float | None:
    return None if value is None else round(float(value), digits)


def _fold_bounds(n: int, folds: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) pairs, as equal as possible, earliest first."""
    size, remainder = divmod(n, folds)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(folds):
        stop = start + size + (1 if i < remainder else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def walk_forward(
    candles: Sequence[Any],
    strategy_name: str,
    params: dict[str, Any] | None = None,
    folds: int | None = None,
    *,
    settings: Any = None,
    run_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay the series in `folds` contiguous chunks and report per-fold metrics.

    The point is not a better backtest, it is a cheaper lie detector: a strategy
    whose edge only exists in one segment is not a strategy, and a single
    whole-series number hides that. `consistency` reports how many folds were
    profitable and how far apart their returns are.

    LIMITATIONS (state these whenever the output is quoted):

      - Folds are SEQUENTIAL, not purged or embargoed. Adjacent folds share
        market regime, and a strategy whose indicators look back N bars gets a
        truncated window at each fold start, so fold-1 results are not
        independent of fold-0 data. This is deliberately NOT a rolling
        re-fit walk-forward: nothing is trained per fold, so it measures
        parameter stability across time, not out-of-sample generalisation.
      - Each fold restarts from `initial_balance`, so no position carries across
        a boundary and no compounding happens between folds.
      - A fold whose candles cannot produce a trade is reported as such instead
        of being hidden; `consistent` is a coarse screen (see below), not a
        statistical test.

    Args:
        candles: full series, oldest -> newest.
        strategy_name: registered strategy name.
        params: strategy params (None -> {}).
        folds: fold count; None -> settings.backtest.walk_forward_folds, clamped
            down when there are too few candles to fill that many folds.
        settings: injected settings object (defaults to `load_settings()`).
        run_kwargs: extra overrides forwarded to `run_backtest`.

    Returns:
        JSON-serialisable dict: `per_fold` (index, bar range, metrics, return_pct)
        and `consistency` (profitable fold count, mean/median/spread of fold
        returns, an explicit `consistent` verdict) plus a `limitations` list.
    """
    candles = list(candles)
    n = len(candles)
    if n == 0:
        raise ValueError("walk_forward requires at least one candle")

    if folds is None:
        folds = int(getattr(getattr(settings, "backtest", None), "walk_forward_folds", 0) or 0)
        if folds <= 0:
            try:
                from algotrading.config import load_settings

                folds = int(load_settings().backtest.walk_forward_folds)
            except Exception:  # pragma: no cover - no config tree
                folds = 4
    # Fewer than 2 candles per fold is not a fold, it is noise; clamp instead of
    # raising so a caller with a short series gets a usable (labelled) answer.
    effective = max(1, min(int(folds), n // 2 or 1))
    bounds = _fold_bounds(n, effective)
    overrides = dict(run_kwargs or {})

    from algotrading.backtest.runner import run_backtest

    per_fold: list[dict[str, Any]] = []
    for index, (start, stop) in enumerate(bounds):
        chunk = candles[start:stop]
        if len(chunk) < 2:
            per_fold.append(
                {
                    "fold": index,
                    "start_bar": start,
                    "end_bar": stop,
                    "n_bars": len(chunk),
                    "skipped": True,
                    "reason": "fold too short to replay (<2 candles)",
                }
            )
            continue
        try:
            result = run_backtest(chunk, strategy_name, dict(params or {}), **overrides)
        except ValueError as exc:
            per_fold.append(
                {
                    "fold": index,
                    "start_bar": start,
                    "end_bar": stop,
                    "n_bars": len(chunk),
                    "skipped": True,
                    "reason": f"replay failed: {exc}",
                }
            )
            continue
        metrics = compute_metrics(result)
        per_fold.append(
            {
                "fold": index,
                "start_bar": start,
                "end_bar": stop,
                "n_bars": len(chunk),
                "start_ts": chunk[0].ts,
                "end_ts": chunk[-1].ts,
                "skipped": False,
                "return_pct": metrics["return_pct"],
                "metrics": metrics,
            }
        )

    replayed = [fold for fold in per_fold if not fold.get("skipped")]
    returns = [fold["return_pct"] for fold in replayed]
    profitable = sum(1 for value in returns if value > 0)
    dispersion = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    # "Consistent" is a coarse screen, deliberately not a p-value: a majority of
    # profit-making folds with at least one closed trade in the sample.
    consistent = bool(replayed) and profitable >= math.ceil(len(replayed) / 2)

    return {
        "strategy_name": strategy_name,
        "params": dict(params or {}),
        "requested_folds": int(folds),
        "folds": len(bounds),
        "n_bars": n,
        "replayed_folds": len(replayed),
        "total_trades": sum(fold["metrics"]["n_trades"] for fold in replayed),
        "per_fold": per_fold,
        "consistency": {
            "folds": len(bounds),
            "replayed_folds": len(replayed),
            "profitable_folds": profitable,
            "profitable_fraction": round(profitable / len(replayed), 6) if replayed else 0.0,
            "mean_return_pct": round(statistics.fmean(returns), 6) if returns else None,
            "median_return_pct": round(statistics.median(returns), 6) if returns else None,
            "stdev_return_pct": round(dispersion, 6),
            "best_return_pct": round(max(returns), 6) if returns else None,
            "worst_return_pct": round(min(returns), 6) if returns else None,
            "consistent": consistent,
            "verdict": (
                "most folds profitable; the edge is not one lucky segment"
                if consistent
                else "edge concentrated in a minority of folds; treat the "
                "whole-series number as unstable"
            ),
        },
        "limitations": [
            "Folds are sequential, not purged/embargoed: adjacent folds share "
            "regime and overlapping lookback windows.",
            "No re-fit per fold: this measures parameter stability over time, "
            "not out-of-sample generalisation.",
            "Each fold restarts from the initial balance; no position or "
            "compounding carries across a fold boundary.",
            "`consistent` is a coarse majority screen, not a significance test.",
        ],
    }


def _resample_stats(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        f"p{q}": round(_percentile(ordered, q), 8) for q in PERCENTILES
    } | {
        "mean": round(statistics.fmean(ordered), 8),
        "min": round(ordered[0], 8),
        "max": round(ordered[-1], 8),
    }


def _seed_default(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    try:
        from algotrading.config import load_settings

        return int(load_settings().backtest.random_seed)
    except Exception:  # pragma: no cover - no config tree
        return DEFAULT_SEED


def _drawdown_series(pnls: Sequence[float], initial_balance: float) -> tuple[float, float]:
    """(max drawdown in cash, max drawdown in percent) for a trade-ordered path."""
    equity = initial_balance
    peak = initial_balance
    worst = 0.0
    worst_pct = 0.0
    for pnl in pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        drop = peak - equity
        if drop > worst:
            worst = drop
        if peak > 0:
            worst_pct = max(worst_pct, drop / peak * 100.0)
    return worst, worst_pct


def monte_carlo(
    result: BacktestResult, runs: int | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Resample the realized trades WITH REPLACEMENT to see the spread of outcomes.

    What it answers: "if the same distribution of trades repeats, how bad can the
    sequence be?" — i.e. percentiles of final PnL and of the max drawdown, plus
    the fraction of runs that end down.

    WHAT IT DOES NOT MODEL (read before quoting a p-value from it):

      - Trades are drawn independently: win/loss streaks, regime clustering and
        volatility autocorrelation are all destroyed. Real drawdowns are usually
        WORSE than this suggests, because losses cluster.
      - The trade distribution itself is assumed correct — this resamples the
        same backtest that may be curve-fit, so it cannot detect overfitting.
      - No fee/slippage re-derivation (already inside each trade's PnL) and no
        position-size compounding: every resample sums the same trade sizes.
      - It is not a significance test and not out-of-sample validation.

    Deterministic for a given `seed` (default `settings.backtest.random_seed`).

    Returns:
        JSON-serialisable dict with `final_pnl`/`final_balance`/`max_drawdown`
        percentiles (p5/p50/p95), `prob_loss` and the run count.
    """
    trade_pnls = [float(t.pnl) for t in result.trades]
    n_runs = int(runs if runs is not None else 0) or _default_runs("monte_carlo_runs")
    resolved_seed = _seed_default(seed)
    if not trade_pnls or n_runs <= 0:
        return {
            "runs": max(0, n_runs),
            "trades": len(trade_pnls),
            "seed": resolved_seed,
            "final_pnl": {},
            "final_balance": {},
            "max_drawdown": {},
            "max_drawdown_pct": {},
            "prob_loss": None,
            "reason": "no closed trades to resample" if not trade_pnls else "runs=0",
        }

    rng = random.Random(resolved_seed)
    initial = result.initial_balance or result.final_balance or 1.0
    finals: list[float] = []
    drawdowns: list[float] = []
    drawdowns_pct: list[float] = []
    losses = 0
    for _ in range(n_runs):
        sample = [rng.choice(trade_pnls) for _ in range(len(trade_pnls))]
        finals.append(sum(sample))
        dd, dd_pct = _drawdown_series(sample, initial)
        drawdowns.append(dd)
        drawdowns_pct.append(dd_pct)
        if finals[-1] < 0:
            losses += 1

    return {
        "runs": n_runs,
        "trades": len(trade_pnls),
        "seed": resolved_seed,
        "percentiles": list(PERCENTILES),
        "final_pnl": _resample_stats(finals),
        "final_balance": {
            key: round(initial + value, 8) for key, value in _resample_stats(finals).items()
        },
        "max_drawdown": _resample_stats(drawdowns),
        "max_drawdown_pct": _resample_stats(drawdowns_pct),
        "prob_loss": round(losses / n_runs, 6),
        "observed_final_pnl": round(sum(trade_pnls), 8),
        "assumptions": [
            "Trades are resampled independently with replacement: streaks and "
            "regime clustering are destroyed, so real drawdowns are usually worse.",
            "The realized trade distribution is taken as given; overfitting in "
            "the underlying backtest is invisible to this method.",
            "Fixed trade sizes: no compounding, and no fee/slippage re-modeling "
            "beyond what is already inside each trade's PnL.",
        ],
    }


def bootstrap(
    result: BacktestResult, runs: int | None = None, seed: int | None = None
) -> dict[str, Any]:
    """Trade-level bootstrap of the MEAN trade return (the edge, not the path).

    Where `monte_carlo` asks about the distribution of outcomes, this asks the
    statistical question: "how precisely is the average trade return estimated,
    and could the edge be zero?". Each run draws `n_trades` trades with
    replacement and records the mean; the reported interval is that sampling
    distribution, which is the honest way to read a positive expectancy off a
    small trade count.

    WHAT IT DOES NOT MODEL: the same caveats as `monte_carlo` (independent
    draws, the sampled distribution taken as given, no compounding) plus one
    more — a confidence interval says nothing about whether the underlying
    strategy will keep working on future data. Deterministic for a given seed.

    Returns:
        JSON-serialisable dict: percentiles of the mean trade PnL and of the
        total return percent, plus the probability the resampled edge is <= 0.
    """
    trade_pnls = [float(t.pnl) for t in result.trades]
    n_runs = int(runs if runs is not None else 0) or _default_runs("bootstrap_runs")
    resolved_seed = _seed_default(seed)
    if not trade_pnls or n_runs <= 0:
        return {
            "runs": max(0, n_runs),
            "trades": len(trade_pnls),
            "seed": resolved_seed,
            "mean_trade_pnl": {},
            "total_return_pct": {},
            "prob_mean_le_zero": None,
            "reason": "no closed trades to resample" if not trade_pnls else "runs=0",
        }

    rng = random.Random(resolved_seed)
    initial = result.initial_balance or 1.0
    means: list[float] = []
    totals_pct: list[float] = []
    non_positive = 0
    for _ in range(n_runs):
        sample = [rng.choice(trade_pnls) for _ in range(len(trade_pnls))]
        mean = statistics.fmean(sample)
        means.append(mean)
        totals_pct.append(sum(sample) / initial * 100.0)
        if mean <= 0:
            non_positive += 1

    observed_mean = statistics.fmean(trade_pnls)
    return {
        "runs": n_runs,
        "trades": len(trade_pnls),
        "seed": resolved_seed,
        "percentiles": list(PERCENTILES),
        "mean_trade_pnl": _resample_stats(means),
        "total_return_pct": _resample_stats(totals_pct),
        "prob_mean_le_zero": round(non_positive / n_runs, 6),
        "observed_mean_trade_pnl": round(observed_mean, 8),
        "assumptions": [
            "Independent draws with replacement from the realized trades.",
            "The realized trade distribution is taken as given; this bounds "
            "sampling error, not the risk of overfitting or regime change.",
            "Fixed trade sizes; no compounding between trades.",
        ],
    }


def _default_runs(name: str) -> int:
    """Read a run budget from settings.backtest, falling back to 200."""
    try:
        from algotrading.config import load_settings

        return int(getattr(load_settings().backtest, name))
    except Exception:  # pragma: no cover - no config tree
        return 200


def metrics_report(result: BacktestResult) -> list[str]:
    """One-line-per-metric text block for a log line or a Telegram summary."""
    metrics = compute_metrics(result)
    lines = [
        f"trades {metrics['n_trades']}  win rate {metrics['win_rate']:.1%}"
        f"  pnl {metrics['total_pnl']:.2f} ({metrics['return_pct']:.2f}%)",
        f"sharpe {_fmt(metrics['sharpe'])}  sortino {_fmt(metrics['sortino'])}"
        f"  maxDD {metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)",
        f"expectancy {_fmt(metrics['expectancy'])}  profit factor "
        f"{_fmt(metrics['profit_factor'])}  exposure {metrics['exposure']:.1%}",
    ]
    for assumption in metrics["assumptions"]:
        lines.append(f"assumes: {assumption}")
    return lines


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


__all__ = [
    "BARS_PER_YEAR",
    "PERCENTILES",
    "annualisation_factor",
    "compute_metrics",
    "max_drawdown_pct",
    "metrics_report",
    "monte_carlo",
    "per_bar_returns",
    "pnl_breakdown",
    "sharpe_ratio",
    "sortino_ratio",
    "walk_forward",
    "bootstrap",
]
