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

import inspect
import math
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

from algotrading.strategy.validation import CodeValidationError, compile_indicator

Number = float
NaN = math.nan

# Registry for indicator functions with metadata
_INDICATOR_REGISTRY: Dict[str, Dict[str, Any]] = {}


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
    """Exponential moving average (seed = SMA of first `period` valid values).
    
    Skips leading NaN values. Returns NaN for all positions before the first
    valid seed can be computed.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    out: list[float] = [NaN] * len(values)
    
    # Find first `period` valid (non-NaN) values for seeding
    valid_indices = [i for i, v in enumerate(values) if not (v != v)]  # NaN check
    if len(valid_indices) < period:
        return out
    
    first_valid_idx = valid_indices[period - 1]
    seed_values = [values[valid_indices[i]] for i in range(period)]
    seed = sum(seed_values) / period
    alpha = 2.0 / (period + 1)
    
    out[first_valid_idx] = seed
    prev = seed
    
    for i in range(first_valid_idx + 1, len(values)):
        if values[i] == values[i]:  # not NaN
            prev = alpha * values[i] + (1 - alpha) * prev
            out[i] = prev
        else:
            out[i] = prev  # propagate last valid EMA
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


# --- Bollinger Bands ---------------------------------------------------------

def bollinger_bands(values: Sequence[float], period: int = 20, num_std: float = 2.0) -> tuple[list[float], list[float], list[float]]:
    """Bollinger Bands: (middle=SMA, upper, lower). Leading period-1 values NaN."""
    middle = sma(values, period)
    upper: list[float] = [NaN] * len(values)
    lower: list[float] = [NaN] * len(values)
    if len(values) < period:
        return middle, upper, lower
    
    # Compute rolling std dev
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        mean = middle[i]
        if math.isnan(mean):
            continue
        variance = sum((x - mean) ** 2 for x in window) / period
        std = math.sqrt(variance)
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std
    return middle, upper, lower


# --- MACD --------------------------------------------------------------------

def macd(values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[list[float], list[float], list[float]]:
    """MACD: (macd_line, signal_line, histogram). Leading slow-1 values NaN."""
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line: list[float] = [NaN] * len(values)
    for i in range(len(values)):
        if not math.isnan(ema_fast[i]) and not math.isnan(ema_slow[i]):
            macd_line[i] = ema_fast[i] - ema_slow[i]
    
    signal_line = ema(macd_line, signal)
    histogram: list[float] = [NaN] * len(values)
    for i in range(len(values)):
        if not math.isnan(macd_line[i]) and not math.isnan(signal_line[i]):
            histogram[i] = macd_line[i] - signal_line[i]
    return macd_line, signal_line, histogram


# --- SuperTrend --------------------------------------------------------------

def supertrend(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
               period: int = 10, multiplier: float = 3.0) -> tuple[list[float], list[bool]]:
    """
    SuperTrend indicator.
    Returns (supertrend_line, is_uptrend_list).
    is_uptrend=True means trend is bullish (buy signal when price > supertrend).
    """
    n = len(closes)
    if n < period + 1:
        return [NaN] * n, [False] * n
    
    # Calculate ATR
    atr_values = atr(highs, lows, closes, period)
    
    # Calculate basic upper/lower bands
    basic_upper: list[float] = [NaN] * n
    basic_lower: list[float] = [NaN] * n
    for i in range(n):
        basic_upper[i] = (highs[i] + lows[i]) / 2 + multiplier * atr_values[i]
        basic_lower[i] = (highs[i] + lows[i]) / 2 - multiplier * atr_values[i]
    
    # Calculate final upper/lower bands
    final_upper: list[float] = [NaN] * n
    final_lower: list[float] = [NaN] * n
    supertrend_line: list[float] = [NaN] * n
    is_uptrend: list[bool] = [False] * n
    
    for i in range(period, n):
        if math.isnan(atr_values[i]):
            continue
            
        if i == period:
            final_upper[i] = basic_upper[i]
            final_lower[i] = basic_lower[i]
        else:
            # Final upper band
            if basic_upper[i] < final_upper[i - 1] or closes[i - 1] > final_upper[i - 1]:
                final_upper[i] = basic_upper[i]
            else:
                final_upper[i] = final_upper[i - 1]
            
            # Final lower band
            if basic_lower[i] > final_lower[i - 1] or closes[i - 1] < final_lower[i - 1]:
                final_lower[i] = basic_lower[i]
            else:
                final_lower[i] = final_lower[i - 1]
        
        # Determine trend
        if i == period:
            if closes[i] <= final_upper[i]:
                is_uptrend[i] = False
                supertrend_line[i] = final_upper[i]
            else:
                is_uptrend[i] = True
                supertrend_line[i] = final_lower[i]
        else:
            if is_uptrend[i - 1]:
                if closes[i] <= final_lower[i]:
                    is_uptrend[i] = False
                    supertrend_line[i] = final_upper[i]
                else:
                    is_uptrend[i] = True
                    supertrend_line[i] = final_lower[i]
            else:
                if closes[i] >= final_upper[i]:
                    is_uptrend[i] = True
                    supertrend_line[i] = final_lower[i]
                else:
                    is_uptrend[i] = False
                    supertrend_line[i] = final_upper[i]
    
    return supertrend_line, is_uptrend


# --- VWAP --------------------------------------------------------------------

def vwap(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
         volumes: Sequence[float]) -> list[float]:
    """
    Volume Weighted Average Price (session VWAP).
    Resets daily - expects data from a single session.
    Leading values NaN.
    """
    n = len(closes)
    out: list[float] = [NaN] * n
    if n == 0:
        return out
    
    cum_pv = 0.0  # cumulative price * volume
    cum_vol = 0.0  # cumulative volume
    
    for i in range(n):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3
        pv = typical_price * volumes[i]
        cum_pv += pv
        cum_vol += volumes[i]
        if cum_vol > 0:
            out[i] = cum_pv / cum_vol
    return out


def vwap_anchored(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float],
                  volumes: Sequence[float], anchor_idx: int) -> list[float]:
    """
    Anchored VWAP from a specific index (e.g., session open).
    Returns NaN before anchor_idx.
    """
    n = len(closes)
    out: list[float] = [NaN] * n
    if n == 0 or anchor_idx >= n:
        return out
    
    cum_pv = 0.0
    cum_vol = 0.0
    
    for i in range(anchor_idx, n):
        typical_price = (highs[i] + lows[i] + closes[i]) / 3
        pv = typical_price * volumes[i]
        cum_pv += pv
        cum_vol += volumes[i]
        if cum_vol > 0:
            out[i] = cum_pv / cum_vol
    return out


# ----------------------------------------------------------------------------
# Indicator registry and registration functions
# ----------------------------------------------------------------------------

def register_indicator(func: Callable, metadata: Dict[str, Any] | None = None) -> None:
    """Register an indicator function in the global registry.
    
    Args:
        func: The indicator function to register
        metadata: Optional metadata about the indicator including:
            - description: Human-readable description
            - return_type: Type of return value (e.g., "list[float]", "tuple[list[float], list[float]]")
            - param_schema: List of parameter definitions
            - dependencies: List of other indicators this depends on
    """
    name = func.__name__
    if name in _INDICATOR_REGISTRY:
        raise ValueError(f"Indicator '{name}' is already registered")
    
    # Extract default metadata from function signature
    sig = inspect.signature(func)
    param_info = []
    for param_name, param in sig.parameters.items():
        param_info.append({
            "name": param_name,
            "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
            "default": param.default if param.default != inspect.Parameter.empty else None,
            "required": param.default == inspect.Parameter.empty,
        })
    
    # Build metadata
    meta = metadata or {}
    meta.update({
        "name": name,
        "module": func.__module__,
        "signature": str(sig),
        "params": param_info,
        "docstring": func.__doc__ or "",
    })
    
    _INDICATOR_REGISTRY[name] = meta


def get_indicator_metadata(name: str) -> Dict[str, Any]:
    """Get metadata for a registered indicator."""
    if name not in _INDICATOR_REGISTRY:
        raise ValueError(f"Indicator '{name}' not found in registry")
    return _INDICATOR_REGISTRY[name].copy()


def list_indicators() -> list[str]:
    """List all registered indicator names."""
    return sorted(_INDICATOR_REGISTRY.keys())


def create_indicator(
    name: str,
    code: str,
    metadata: Dict[str, Any] | None = None,
) -> Callable:
    """Create and register a new indicator from Python code.
    
    Args:
        name: Name for the new indicator
        code: Python function definition code
        metadata: Optional metadata about the indicator
        
    Returns:
        The created indicator function
        
    Raises:
        ValueError: If code is invalid or unsafe
    """
    # Shared safety gate: one validator for every piece of AI-authored code.
    try:
        indicator_func = compile_indicator(code, name)
    except CodeValidationError as exc:
        raise ValueError(str(exc)) from exc

    # Register the function
    register_indicator(indicator_func, metadata)
    
    # Also add to module namespace for backward compatibility
    globals()[name] = indicator_func
    
    return indicator_func


# Auto-register all built-in indicators on module load
def _register_builtin_indicators():
    """Register all built-in indicator functions."""
    builtins = [
        (sma, {
            "description": "Simple Moving Average",
            "return_type": "list[float]",
        }),
        (ema, {
            "description": "Exponential Moving Average",
            "return_type": "list[float]",
        }),
        (rsi, {
            "description": "Relative Strength Index",
            "return_type": "list[float]",
        }),
        (atr, {
            "description": "Average True Range",
            "return_type": "list[float]",
        }),
        (bollinger_bands, {
            "description": "Bollinger Bands",
            "return_type": "tuple[list[float], list[float], list[float]]",
        }),
        (macd, {
            "description": "Moving Average Convergence Divergence",
            "return_type": "tuple[list[float], list[float], list[float]]",
        }),
        (supertrend, {
            "description": "SuperTrend indicator",
            "return_type": "tuple[list[float], list[bool]]",
        }),
        (vwap, {
            "description": "Volume Weighted Average Price",
            "return_type": "list[float]",
        }),
        (vwap_anchored, {
            "description": "Anchored VWAP",
            "return_type": "list[float]",
        }),
        (closes, {
            "description": "Extract close prices from candles",
            "return_type": "list[float]",
        }),
        (highs, {
            "description": "Extract high prices from candles",
            "return_type": "list[float]",
        }),
        (lows, {
            "description": "Extract low prices from candles",
            "return_type": "list[float]",
        }),
        (last_valid, {
            "description": "Find last non-NaN value in sequence",
            "return_type": "float | None",
        }),
    ]
    
    for func, meta in builtins:
        try:
            register_indicator(func, meta)
        except ValueError:
            pass  # Already registered


# Register built-in indicators
_register_builtin_indicators()
