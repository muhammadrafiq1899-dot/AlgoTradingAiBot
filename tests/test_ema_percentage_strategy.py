"""Unit tests for EMAPercentageStrategy.

The crossover is engineered to land on the *last* candle (a long flat base
followed by a single small move), which is the only candle the strategy
inspects. EMA seeding over a flat series is exact, so these are deterministic.
"""
from typing import Sequence

from algotrading.market.base import Candle
from algotrading.strategy.starters import EMAPercentageStrategy

MINUTE_MS = 60_000


def create_test_candles(prices: Sequence[float]) -> list[Candle]:
    """Helper to create candles from a price sequence (oldest -> newest)."""
    base_ts = 1_609_459_200_000
    return [
        Candle(
            symbol="BTC/USDT",
            interval="1m",
            ts=base_ts + i * MINUTE_MS,
            open=price * 0.99,
            high=price * 1.01,
            low=price * 0.98,
            close=price,
            volume=100.0,
        )
        for i, price in enumerate(prices)
    ]


def _flat_then(last: float, n: int = 30, level: float = 100.0) -> list[Candle]:
    return create_test_candles([level] * n + [last])


def test_ema_percentage_strategy_bullish_crossover_within_threshold():
    """Buy when a bullish crossover lands within buy_pct% of the fast EMA."""
    candles = _flat_then(100.5)

    strategy = EMAPercentageStrategy({
        "fast_period": 9,
        "slow_period": 20,
        "buy_pct": 1.0,
        "exit_pct": 3.0,
        "position_pct": 0.2,
    })

    signal = strategy.evaluate("BTC/USDT", candles)
    assert signal is not None
    assert signal.symbol == "BTC/USDT"
    assert signal.side == "buy"
    assert signal.risk["position_pct"] == 0.2
    assert signal.ref_price == 100.5
    assert "bullish crossover" in signal.rationale.lower()


def test_ema_percentage_strategy_bullish_crossover_outside_threshold():
    """No buy when price is too far from the fast EMA at the crossover."""
    candles = _flat_then(105.0)  # ~4% above the fast EMA

    strategy = EMAPercentageStrategy({
        "fast_period": 9,
        "slow_period": 20,
        "buy_pct": 1.0,
        "exit_pct": 3.0,
        "position_pct": 0.2,
    })

    assert strategy.evaluate("BTC/USDT", candles) is None


def test_ema_percentage_strategy_bearish_crossover():
    """Sell on a bearish EMA crossover."""
    candles = _flat_then(99.5)

    strategy = EMAPercentageStrategy({
        "fast_period": 9,
        "slow_period": 20,
        "buy_pct": 2.0,
        "exit_pct": 3.0,
        "position_pct": 0.2,
    })

    signal = strategy.evaluate("BTC/USDT", candles)
    assert signal is not None
    assert signal.side == "sell"
    assert "bearish crossover" in signal.rationale.lower()


def test_ema_percentage_strategy_insufficient_data():
    """No signal until enough candles exist for the slow EMA."""
    candles = create_test_candles([100.0] * 10)

    strategy = EMAPercentageStrategy({
        "fast_period": 9,
        "slow_period": 20,
        "buy_pct": 2.0,
        "exit_pct": 3.0,
        "position_pct": 0.2,
    })

    assert strategy.evaluate("BTC/USDT", candles) is None


def test_ema_percentage_strategy_parameter_validation():
    """Constructor rejects incoherent / negative thresholds."""
    try:
        EMAPercentageStrategy({
            "fast_period": 20,
            "slow_period": 20,
            "buy_pct": 1.0,
            "exit_pct": 2.0,
            "position_pct": 0.2,
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "fast_period must be < slow_period" in str(e)

    try:
        EMAPercentageStrategy({
            "fast_period": 9,
            "slow_period": 20,
            "buy_pct": -1.0,
            "exit_pct": 2.0,
            "position_pct": 0.2,
        })
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "buy_pct and exit_pct must be positive" in str(e)


def test_ema_percentage_strategy_default_parameters():
    """Defaults match the documented values."""
    strategy = EMAPercentageStrategy({})
    assert strategy.fast == 9
    assert strategy.slow == 20
    assert strategy.buy_pct == 1.0
    assert strategy.exit_pct == 2.0
    assert strategy.position_pct == 0.2
