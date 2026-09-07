"""Starter strategies: EMA crossover and RSI mean-reversion.

Each strategy is a class holding its parameters; `evaluate` is pure. The
registry maps the YAML `name` to a class and validates params.
"""
from __future__ import annotations

from typing import Any, Sequence

from algotrading.market.base import Candle
from algotrading.strategy import indicators as ta
from algotrading.strategy.base import Signal


class EMACrossover:
    """Trend-following: buy when fast EMA crosses above slow EMA.

    A fresh cross on the *latest* candle only emits a signal (not every candle
    while one EMA leads the other) to avoid re-entry on every tick.
    """

    name = "ema_crossover"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.fast = int(params["fast_period"])
        self.slow = int(params["slow_period"])
        self.position_pct = float(params.get("position_pct", 0.2))
        if self.fast >= self.slow:
            raise ValueError("fast_period must be < slow_period")

    def evaluate(self, symbol: str, candles: Sequence[Candle]) -> Signal | None:
        if len(candles) < self.slow + 1:
            return None
        c = [x.close for x in candles]
        fast = ta.ema(c, self.fast)
        slow = ta.ema(c, self.slow)
        # last valid index (the latest candle); prev index for cross detection.
        i = len(c) - 1
        if i < 1:
            return None
        fast_now, slow_now = fast[i], slow[i]
        fast_prev, slow_prev = fast[i - 1], slow[i - 1]
        if fast_now is None or slow_now is None or fast_prev is None or slow_prev is None:
            return None

        price = c[i]
        # Golden cross: fast crossed above slow on the latest candle.
        if fast_prev <= slow_prev and fast_now > slow_now:
            return Signal(
                strategy_id=0,  # engine fills in the real id
                symbol=symbol,
                side="buy",
                ref_price=price,
                rationale=(
                    f"EMA({self.fast}) {fast_now:.2f} crossed above "
                    f"EMA({self.slow}) {slow_now:.2f}"
                ),
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        # Death cross: close position.
        if fast_prev >= slow_prev and fast_now < slow_now:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="sell",
                ref_price=price,
                rationale=(
                    f"EMA({self.fast}) {fast_now:.2f} crossed below "
                    f"EMA({self.slow}) {slow_now:.2f}"
                ),
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        return None


class RSIMeanReversion:
    """Mean-reversion: buy oversold bounce, exit overbought.

    Entry requires RSI to dip below `oversold` then recover above it (a hook,
    not a falling-knife catch). Exit when RSI exceeds `overbought`.
    """

    name = "rsi_mean_reversion"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.period = int(params["period"])
        self.oversold = float(params["oversold"])
        self.overbought = float(params["overbought"])
        self.position_pct = float(params.get("position_pct", 0.2))
        if self.oversold >= self.overbought:
            raise ValueError("oversold must be < overbought")

    def evaluate(self, symbol: str, candles: Sequence[Candle]) -> Signal | None:
        n = len(candles)
        if n < self.period + 2:
            return None
        c = [x.close for x in candles]
        rsi = ta.rsi(c, self.period)
        i = n - 1
        r_now = rsi[i]
        r_prev = rsi[i - 1]
        if r_now is None or r_prev is None:
            return None
        price = c[i]

        # Hook recovery from oversold.
        if r_prev <= self.oversold and r_now > self.oversold:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="buy",
                ref_price=price,
                rationale=f"RSI recovered {r_prev:.1f} -> {r_now:.1f} above {self.oversold}",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        # Exit at overbought.
        if r_now >= self.overbought:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="sell",
                ref_price=price,
                rationale=f"RSI {r_now:.1f} reached overbought {self.overbought}",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        return None


STRATEGIES: dict[str, type] = {
    EMACrossover.name: EMACrossover,
    RSIMeanReversion.name: RSIMeanReversion,
}
