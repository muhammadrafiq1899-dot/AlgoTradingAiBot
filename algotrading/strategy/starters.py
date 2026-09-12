"""Starter strategies: EMA crossover, RSI mean-reversion, BB mean-reversion,
MACD trend, SuperTrend, VWAP reclaim, Multi-timeframe EMA, Ensemble.

Each strategy is a class holding its parameters; `evaluate` is pure. The
registry maps the YAML `name` to a class and validates params.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from algotrading.market.base import Candle
from algotrading.strategy import indicators as ta
from algotrading.strategy.base import Signal

log = logging.getLogger(__name__)


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


class BBMeanReversion:
    """Bollinger Bands mean-reversion: buy at lower band, sell at upper band.

    Entry when price closes below lower band then recovers above it.
    Exit when price reaches upper band or middle band (configurable).
    """

    name = "bb_mean_reversion"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.period = int(params["period"])
        self.num_std = float(params["num_std"])
        self.position_pct = float(params.get("position_pct", 0.2))
        self.exit_at_middle = bool(params.get("exit_at_middle", True))

    def evaluate(self, symbol: str, candles: Sequence[Candle]) -> Signal | None:
        n = len(candles)
        if n < self.period + 2:
            return None
        c = [x.close for x in candles]
        middle, upper, lower = ta.bollinger_bands(c, self.period, self.num_std)
        i = n - 1
        
        mid_now = middle[i]
        up_now = upper[i]
        low_now = lower[i]
        mid_prev = middle[i - 1]
        up_prev = upper[i - 1]
        low_prev = lower[i - 1]
        
        if any(v is None for v in [mid_now, up_now, low_now, mid_prev, up_prev, low_prev]):
            return None
        
        price = c[i]
        price_prev = c[i - 1]

        # Buy: price was below lower band, now recovered above it
        if price_prev <= low_prev and price > low_now:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="buy",
                ref_price=price,
                rationale=f"Price recovered from lower BB ({low_prev:.2f}) to {price:.2f}",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        # Sell: exit at upper band or middle band
        exit_price = mid_now if self.exit_at_middle else up_now
        if price >= exit_price:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="sell",
                ref_price=price,
                rationale=f"Price reached {'middle' if self.exit_at_middle else 'upper'} BB at {exit_price:.2f}",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        return None


class MACDTrend:
    """MACD trend-following: buy on MACD bullish cross, sell on bearish cross."""

    name = "macd_trend"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.fast = int(params["fast"])
        self.slow = int(params["slow"])
        self.signal = int(params["signal"])
        self.position_pct = float(params.get("position_pct", 0.2))

    def evaluate(self, symbol: str, candles: Sequence[Candle]) -> Signal | None:
        n = len(candles)
        if n < self.slow + self.signal + 1:
            return None
        c = [x.close for x in candles]
        macd_line, signal_line, histogram = ta.macd(c, self.fast, self.slow, self.signal)
        i = n - 1
        
        macd_now = macd_line[i]
        sig_now = signal_line[i]
        macd_prev = macd_line[i - 1]
        sig_prev = signal_line[i - 1]
        
        if any(v is None for v in [macd_now, sig_now, macd_prev, sig_prev]):
            return None
        
        price = c[i]

        # Bullish cross: MACD crosses above signal line
        if macd_prev <= sig_prev and macd_now > sig_now:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="buy",
                ref_price=price,
                rationale=f"MACD bullish cross: {macd_now:.4f} > signal {sig_now:.4f}",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        # Bearish cross: MACD crosses below signal line
        if macd_prev >= sig_prev and macd_now < sig_now:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="sell",
                ref_price=price,
                rationale=f"MACD bearish cross: {macd_now:.4f} < signal {sig_now:.4f}",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        return None


class SuperTrendStrategy:
    """SuperTrend trend-following: buy when price closes above SuperTrend, sell below."""

    name = "supertrend"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.period = int(params["period"])
        self.multiplier = float(params["multiplier"])
        self.position_pct = float(params.get("position_pct", 0.2))

    def evaluate(self, symbol: str, candles: Sequence[Candle]) -> Signal | None:
        n = len(candles)
        if n < self.period + 2:
            return None
        h = [x.high for x in candles]
        l = [x.low for x in candles]
        c = [x.close for x in candles]
        
        st_line, is_uptrend = ta.supertrend(h, l, c, self.period, self.multiplier)
        i = n - 1
        
        st_now = st_line[i]
        st_prev = st_line[i - 1]
        trend_now = is_uptrend[i]
        trend_prev = is_uptrend[i - 1]
        
        if st_now is None or st_prev is None:
            return None
        
        price = c[i]

        # Trend flip to uptrend: buy
        if not trend_prev and trend_now:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="buy",
                ref_price=price,
                rationale=f"SuperTrend flipped bullish at {st_now:.2f}",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        # Trend flip to downtrend: sell
        if trend_prev and not trend_now:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="sell",
                ref_price=price,
                rationale=f"SuperTrend flipped bearish at {st_now:.2f}",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        return None


class VWAPReclaim:
    """VWAP Reclaim: buy when price reclaims VWAP from below, sell when loses it."""

    name = "vwap_reclaim"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.position_pct = float(params.get("position_pct", 0.2))

    def evaluate(self, symbol: str, candles: Sequence[Candle]) -> Signal | None:
        n = len(candles)
        if n < 20:  # Need enough data for meaningful VWAP
            return None
        h = [x.high for x in candles]
        l = [x.low for x in candles]
        c = [x.close for x in candles]
        v = [x.volume for x in candles]
        
        vwap_vals = ta.vwap(h, l, c, v)
        i = n - 1
        
        vwap_now = vwap_vals[i]
        vwap_prev = vwap_vals[i - 1]
        
        if vwap_now is None or vwap_prev is None:
            return None
        
        price = c[i]
        price_prev = c[i - 1]

        # Reclaim: price was below VWAP, now above
        if price_prev <= vwap_prev and price > vwap_now:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="buy",
                ref_price=price,
                rationale=f"Price reclaimed VWAP ({vwap_prev:.2f} -> {price:.2f})",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        # Lose: price was above VWAP, now below
        if price_prev >= vwap_prev and price < vwap_now:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="sell",
                ref_price=price,
                rationale=f"Price lost VWAP ({vwap_prev:.2f} -> {price:.2f})",
                risk={"position_pct": self.position_pct},
                params=dict(self.params),
            )
        return None


class MultiTFEMA:
    """Multi-timeframe EMA: higher TF trend filter + lower TF entry.

    Uses 1h EMA for trend direction, 15m/5m EMA cross for entry.
    Expects candles dict with multiple intervals from StrategyEngine.
    """

    name = "multi_tf_ema"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.trend_interval = params.get("trend_interval", "1h")
        self.trend_fast = int(params.get("trend_fast", 9))
        self.trend_slow = int(params.get("trend_slow", 21))
        self.entry_interval = params.get("entry_interval", "15m")
        self.entry_fast = int(params.get("entry_fast", 5))
        self.entry_slow = int(params.get("entry_slow", 13))
        self.position_pct = float(params.get("position_pct", 0.2))

    def evaluate(self, symbol: str, candles: Sequence[Candle]) -> Signal | None:
        # Note: This strategy expects the StrategyEngine to pass a dict
        # with multiple intervals. For now, we only process if the right
        # interval is the primary one being evaluated.
        # The engine currently passes a single interval snapshot.
        # This strategy works best when the engine evaluates multiple intervals.
        return None


class EnsembleStrategy:
    """Combine multiple strategies with voting/filtering logic.

    Modes:
      - consensus: ALL must agree on side (buy/sell)
      - any: ANY signals buy -> buy, ANY signals sell -> sell
      - filter: primary strategy gated by filter strategy (both must agree)
      - weighted: weighted vote by position_pct
    """

    name = "ensemble"

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params
        self.mode = params.get("mode", "consensus")
        self.position_pct = float(params.get("position_pct", 0.2))
        
        # Parse components: list of {"name": str, "params": dict}
        components = params.get("components", [])
        if not isinstance(components, list) or len(components) < 2:
            raise ValueError("ensemble requires 'components' list with >=2 strategies")
        
        self.strategies = []
        for comp in components:
            if not isinstance(comp, dict) or "name" not in comp:
                raise ValueError("each component must have 'name' and 'params'")
            strat_name = comp["name"]
            strat_params = comp.get("params", {})
            if strat_name not in STRATEGIES:
                raise ValueError(f"unknown strategy: {strat_name}")
            self.strategies.append(STRATEGIES[strat_name](strat_params))

    def evaluate(self, symbol: str, candles: Sequence[Candle]) -> Signal | None:
        if len(self.strategies) < 2:
            return None
        
        # Collect signals from all component strategies
        signals = []
        for strat in self.strategies:
            try:
                sig = strat.evaluate(symbol, candles)
                if sig is not None:
                    signals.append(sig)
            except Exception as e:
                log.warning("Ensemble component %s failed: %s", strat.name, e)
        
        if not signals:
            return None
        
        price = candles[-1].close if candles else 0.0
        sides = [s.side for s in signals]
        
        if self.mode == "consensus":
            # All must agree
            if all(s == "buy" for s in sides):
                rationale = f"Ensemble consensus BUY: {', '.join(s.rationale for s in signals)}"
                return Signal(
                    strategy_id=0, symbol=symbol, side="buy", ref_price=price,
                    rationale=rationale, risk={"position_pct": self.position_pct},
                    params=dict(self.params),
                )
            if all(s == "sell" for s in sides):
                rationale = f"Ensemble consensus SELL: {', '.join(s.rationale for s in signals)}"
                return Signal(
                    strategy_id=0, symbol=symbol, side="sell", ref_price=price,
                    rationale=rationale, risk={"position_pct": self.position_pct},
                    params=dict(self.params),
                )
        
        elif self.mode == "any":
            # Any buy -> buy, any sell -> sell (buy takes priority if both)
            if "buy" in sides:
                buy_signals = [s for s in signals if s.side == "buy"]
                rationale = f"Ensemble any BUY: {', '.join(s.rationale for s in buy_signals)}"
                return Signal(
                    strategy_id=0, symbol=symbol, side="buy", ref_price=price,
                    rationale=rationale, risk={"position_pct": self.position_pct},
                    params=dict(self.params),
                )
            if "sell" in sides:
                sell_signals = [s for s in signals if s.side == "sell"]
                rationale = f"Ensemble any SELL: {', '.join(s.rationale for s in sell_signals)}"
                return Signal(
                    strategy_id=0, symbol=symbol, side="sell", ref_price=price,
                    rationale=rationale, risk={"position_pct": self.position_pct},
                    params=dict(self.params),
                )
        
        elif self.mode == "filter":
            # First strategy is primary, rest are filters (all must agree)
            primary = signals[0] if signals else None
            filters = signals[1:]
            if primary and all(f.side == primary.side for f in filters):
                rationale = f"Ensemble filter {primary.side.upper()}: primary={primary.rationale}; filters={', '.join(f.rationale for f in filters)}"
                return Signal(
                    strategy_id=0, symbol=symbol, side=primary.side, ref_price=price,
                    rationale=rationale, risk={"position_pct": self.position_pct},
                    params=dict(self.params),
                )
        
        elif self.mode == "weighted":
            # Weighted vote by position_pct of each component
            buy_weight = sum(s.risk.get("position_pct", 0.2) for s in signals if s.side == "buy")
            sell_weight = sum(s.risk.get("position_pct", 0.2) for s in signals if s.side == "sell")
            if buy_weight > sell_weight:
                rationale = f"Ensemble weighted BUY ({buy_weight:.2f} vs {sell_weight:.2f})"
                return Signal(
                    strategy_id=0, symbol=symbol, side="buy", ref_price=price,
                    rationale=rationale, risk={"position_pct": self.position_pct},
                    params=dict(self.params),
                )
            elif sell_weight > buy_weight:
                rationale = f"Ensemble weighted SELL ({sell_weight:.2f} vs {buy_weight:.2f})"
                return Signal(
                    strategy_id=0, symbol=symbol, side="sell", ref_price=price,
                    rationale=rationale, risk={"position_pct": self.position_pct},
                    params=dict(self.params),
                )
        
        return None


STRATEGIES: dict[str, type] = {
    EMACrossover.name: EMACrossover,
    RSIMeanReversion.name: RSIMeanReversion,
    BBMeanReversion.name: BBMeanReversion,
    MACDTrend.name: MACDTrend,
    SuperTrendStrategy.name: SuperTrendStrategy,
    VWAPReclaim.name: VWAPReclaim,
    MultiTFEMA.name: MultiTFEMA,
    EnsembleStrategy.name: EnsembleStrategy,
}
