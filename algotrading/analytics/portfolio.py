"""Portfolio exposure + a descriptive correlation view.

Two read-only answers for the Telegram surface and the advisory context:

* **Exposure** — per open position: quantity, average entry, last stored price,
  notional and what share of the sizing balance it occupies, plus total exposure
  and the open-position count.
* **Correlation** — a Pearson matrix over the last
  ``settings.analytics.portfolio_bars`` closes of every configured symbol.

Read both with the right expectations:

* Correlation here is **descriptive risk context, NOT a trading signal**. It says
  "these two positions have been moving together lately", which is useful to
  judge whether three "diversified" entries are really one bet. It is not a
  forecast, not a filter, and nothing in the execution path consumes it.
* It is computed on **stored candles**, so it **lags**: the last bar is the last
  one market_tick persisted, not the price on screen right now.
* It is a **linear** measure: it misses non-linear co-movement and is unstable on
  short windows, which is why symbols with too few bars or zero variance are
  skipped and reported rather than silently plotted.

Pure Python (``math``/``statistics`` only; no numpy/pandas) — see invariant #8.
"""
from __future__ import annotations

import logging
import math
from statistics import fmean
from typing import Any, Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.db.models import Candle, Position

log = logging.getLogger(__name__)

#: Bars needed before a correlation is meaningful. Below this the estimate is
#: noise (and with 2 points every pair is +/-1.0 by construction).
MIN_CORRELATION_BARS = 10

#: Reasons a symbol can be missing from the matrix.
SKIP_INSUFFICIENT = "insufficient bars"
SKIP_NO_VARIANCE = "zero variance"
SKIP_NO_CANDLES = "no stored candles"


def _symbols(settings: Any, session: Session) -> list[str]:
    """Configured universe, falling back to whatever is in the DB.

    A minimal settings stub (tests, partial configs) must not break the view, so
    every attribute is read defensively.
    """
    market = getattr(settings, "market", None)
    symbols = [str(s) for s in (getattr(market, "symbols", None) or []) if s]
    if symbols:
        return symbols
    rows = session.execute(select(Position.symbol).order_by(Position.symbol.asc())).scalars().all()
    return [str(s) for s in rows if s]


def _interval(settings: Any) -> str:
    """Evaluation interval, mirroring ``BotContext.primary_interval``."""
    market = getattr(settings, "market", None)
    explicit = getattr(market, "eval_interval", None)
    if explicit:
        return str(explicit)
    intervals = list(getattr(market, "intervals", None) or [])
    return "1h" if "1h" in intervals else (intervals[0] if intervals else "1h")


def _bars(settings: Any) -> int:
    analytics = getattr(settings, "analytics", None)
    try:
        return max(MIN_CORRELATION_BARS, int(getattr(analytics, "portfolio_bars", 200) or 200))
    except (TypeError, ValueError):
        return 200


def _balance(settings: Any) -> float:
    """Sizing basis for the exposure percentages.

    There is no balance table: the ledger is event-sourced from orders, and the
    configured paper balance is what sizing uses (``risk.paper_initial_balance``).
    Percentage-of-balance is therefore relative to that basis, not to live
    equity — which is exactly how the risk manager sizes.
    """
    risk = getattr(settings, "risk", None)
    try:
        value = float(getattr(risk, "paper_initial_balance", 10_000.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value


def _latest_prices(session: Session, symbols: Sequence[str]) -> dict[str, float]:
    """Newest stored close per symbol that has any candle."""
    prices: dict[str, float] = {}
    for symbol in symbols:
        row = session.execute(
            select(Candle)
            .where(Candle.symbol == symbol)
            .order_by(Candle.ts.desc())
            .limit(1)
        ).scalars().first()
        if row is not None:
            prices[symbol] = float(row.close)
    # Positions on symbols outside the configured universe still need a mark.
    held = session.execute(select(Position.symbol)).scalars().all()
    for symbol in held:
        if symbol in prices:
            continue
        row = session.execute(
            select(Candle)
            .where(Candle.symbol == symbol)
            .order_by(Candle.ts.desc())
            .limit(1)
        ).scalars().first()
        if row is not None:
            prices[symbol] = float(row.close)
    return prices


def _position_rows(
    session: Session, prices: dict[str, float], balance: float
) -> list[dict[str, Any]]:
    positions = session.execute(
        select(Position).where(Position.qty > 0).order_by(Position.symbol.asc())
    ).scalars().all()
    rows: list[dict[str, Any]] = []
    for pos in positions:
        last = prices.get(pos.symbol)
        notional = (last or 0.0) * float(pos.qty or 0.0)
        rows.append(
            {
                "symbol": pos.symbol,
                "qty": float(pos.qty or 0.0),
                "avg_price": float(pos.avg_price or 0.0),
                "last_price": last,
                "notional": notional,
                "pct_of_balance": (notional / balance * 100.0) if balance > 0 else 0.0,
            }
        )
    return rows


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson r of two equal-length series, or None if it is undefined."""
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = fmean(x), fmean(y)
    covariance = sum((a - mx) * (b - my) for a, b in zip(x, y))
    var_x = sum((a - mx) ** 2 for a in x)
    var_y = sum((b - my) ** 2 for b in y)
    if var_x <= 0 or var_y <= 0:
        return None
    return covariance / math.sqrt(var_x * var_y)


def _closes_by_ts(session: Session, symbol: str, interval: str, bars: int) -> dict[int, float]:
    rows = session.execute(
        select(Candle)
        .where(Candle.symbol == symbol, Candle.interval == interval)
        .order_by(Candle.ts.desc())
        .limit(bars)
    ).scalars().all()
    return {int(r.ts): float(r.close) for r in reversed(rows)}


def correlation_matrix(
    session: Session, symbols: Sequence[str], interval: str, bars: int
) -> dict[str, Any]:
    """Pearson matrix over the last ``bars`` closes per symbol.

    Series are aligned on the timestamps both symbols share (so a symbol with a
    gap is not shifted against the others), then trimmed to the newest ``bars``
    shared bars. Each symbol is reported as included or skipped-with-a-reason.
    """
    series: dict[str, dict[int, float]] = {}
    skipped: dict[str, str] = {}
    for symbol in symbols:
        closes = _closes_by_ts(session, symbol, interval, bars)
        if not closes:
            skipped[symbol] = SKIP_NO_CANDLES
            continue
        if len(closes) < MIN_CORRELATION_BARS:
            skipped[symbol] = SKIP_INSUFFICIENT
            continue
        if _variance(closes, closes.keys()) <= 0:
            # A flat series has no correlation with anything (0/0): excluded and
            # reported rather than plotted as a row of zeros.
            skipped[symbol] = SKIP_NO_VARIANCE
            continue
        series[symbol] = closes

    included = sorted(series)
    matrix: dict[str, dict[str, float | None]] = {a: {} for a in included}
    for a in included:
        for b in included:
            if a == b:
                matrix[a][b] = 1.0
                continue
            shared = sorted(set(series[a]) & set(series[b]))[-bars:]
            if len(shared) < MIN_CORRELATION_BARS:
                matrix[a][b] = None
                skipped.setdefault(f"{a} x {b}", SKIP_INSUFFICIENT)
                continue
            value = _pearson([series[a][ts] for ts in shared], [series[b][ts] for ts in shared])
            if value is None:
                # Both series vary overall but one is flat across the *shared*
                # bars — no correlation is defined for this pair.
                skipped.setdefault(f"{a} x {b}", SKIP_NO_VARIANCE)
                matrix[a][b] = None
                continue
            matrix[a][b] = round(value, 4)

    return {
        "symbols": included,
        "matrix": matrix,
        "skipped": skipped,
        "bars": bars,
        "interval": interval,
        "min_bars": MIN_CORRELATION_BARS,
    }


def _variance(series: dict[int, float], timestamps: Iterable[int]) -> float:
    values = [series[ts] for ts in timestamps if ts in series]
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def portfolio_snapshot(session: Session, settings: Any) -> dict[str, Any]:
    """Per-symbol exposure + open-position stats + a correlation matrix.

    Everything is derived from stored rows (positions, candles). Correlation is
    descriptive context computed on stored candles, so it lags the market and is
    never a trading signal.
    """
    symbols = _symbols(settings, session)
    interval = _interval(settings)
    bars = _bars(settings)
    balance = _balance(settings)
    prices = _latest_prices(session, symbols)
    rows = _position_rows(session, prices, balance)
    total_exposure = sum(r["notional"] for r in rows)
    correlation = correlation_matrix(session, symbols, interval, bars)

    return {
        "positions": rows,
        "open_positions": len(rows),
        "total_exposure": total_exposure,
        "total_pct_of_balance": (total_exposure / balance * 100.0) if balance > 0 else 0.0,
        "balance": balance,
        "prices": prices,
        "correlation": correlation,
        "interval": interval,
    }
