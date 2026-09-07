"""Pure-Python technical indicators.

No numpy/pandas/talib — keeps the Termux/Android footprint small and avoids
C-extension build failures. All functions operate on plain lists of floats
ordered oldest -> newest. Inputs may be list[float] or list[Candle] via the
candle-accessor helpers.

Conventions:
  - Input lists are ordered oldest -> newest.
  - Output lists have the same length; the leading (period-1) entries are NaN
    so downstream code can align them against the candle timestamps.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

Number = float
NaN = math.nan


def sma(values: Sequence[float], period: int) -> list[float]:
    """Simple moving average. Leading period-1 values are NaN."""
    if period <= 0:
        raise ValueError("period must be > 0")
    out: list[float] = [NaN] * len(values)
    if len(values) < period:
        return out
    window = sum(values[:period])
    out[period - 1] = window / period
    for i in range(period, len(values)):
        window += values[i] - values[i - period]
        out[i] = window / period
    return out


def ema(values: Sequence[float], period: int) -> list[float]:
    """Exponential moving average (seed = SMA of first `period` values)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    out: list[float] = [NaN] * len(values)
    if len(values) < period:
        return out
    alpha = 2.0 / (period + 1)
    # Seed with SMA of the first `period` values for stability.
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = alpha * values[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi(values: Sequence[float], period: int = 14) -> list[float]:
    """Relative Strength Index (Wilder's smoothing). Leading period values NaN."""
    if period <= 0:
        raise ValueError("period must be > 0")
    out: list[float] = [NaN] * len(values)
    if len(values) <= period:
        return out
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = _rs(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = _rs(avg_gain, avg_loss)
    return out


def _rs(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> list[float]:
    """Average True Range (Wilder's). Leading `period` values NaN."""
    if period <= 0:
        raise ValueError("period must be > 0")
    n = len(closes)
    out: list[float] = [NaN] * n
    if n <= period:
        return out
    trs: list[float] = []
    for i in range(n):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        trs.append(tr)
    prev = sum(trs[1 : period + 1]) / period
    out[period] = prev
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + trs[i]) / period
        out[i] = prev
    return out


# --- candle accessor helpers ----------------------------------------------

def closes(candles: Sequence[object]) -> list[float]:
    return [float(c.close) for c in candles]


def highs(candles: Sequence[object]) -> list[float]:
    return [float(c.high) for c in candles]


def lows(candles: Sequence[object]) -> list[float]:
    return [float(c.low) for c in candles]


def last_valid(values: Sequence[float]) -> float | None:
    """Return the most recent non-NaN value, or None if all NaN."""
    for v in reversed(values):
        if not math.isnan(v):
            return v
    return None
