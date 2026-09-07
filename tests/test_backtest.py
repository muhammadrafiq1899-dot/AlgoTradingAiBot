"""Backtest runner: replay candles through a strategy, produce equity summary."""
import pytest

from algotrading.backtest.runner import run_backtest
from algotrading.market.base import Candle


def _candles(prices, interval="1h", symbol="BTC/USDT"):
    out = []
    for i, p in enumerate(prices):
        out.append(
            Candle(
                symbol=symbol,
                interval=interval,
                ts=1_600_000_000_000 + i * 3_600_000,
                open=p,
                high=p,
                low=p,
                close=p,
                volume=1.0,
            )
        )
    return out


def test_flat_prices_no_trades():
    res = run_backtest(
        _candles([100.0] * 40), "ema_crossover",
        {"fast_period": 3, "slow_period": 6, "position_pct": 0.2},
    )
    assert res.n_trades == 0
    assert res.win_rate == 0.0
    assert res.final_balance == pytest.approx(10_000.0)


def test_trend_round_trip():
    # Flat base, then a strong up move (golden cross -> buy), then a strong
    # down move (death cross -> sell). Should produce exactly one round trip.
    prices = (
        [100.0] * 12
        + [101, 102, 105, 110, 118, 127, 135, 142, 148, 153, 157, 160]
        + [155, 148, 139, 128, 115, 100, 88]
    )
    res = run_backtest(
        _candles(prices), "ema_crossover",
        {"fast_period": 3, "slow_period": 6, "position_pct": 0.2},
    )
    assert res.n_trades == 1
    assert res.n_wins == 1
    assert res.n_losses == 0
    assert res.total_pnl > 0
    assert res.final_balance > 10_000.0
    assert res.to_dict()["n_trades"] == 1


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        run_backtest(_candles([100.0] * 10), "no_such_strategy", {})


def test_invalid_params_raises():
    with pytest.raises(ValueError):
        run_backtest(
            _candles([100.0] * 10), "ema_crossover",
            {"fast_period": 6, "slow_period": 3},  # fast >= slow
        )


def test_empty_candles_raises():
    with pytest.raises(ValueError):
        run_backtest([], "ema_crossover", {})


def test_result_metadata():
    res = run_backtest(
        _candles([100.0] * 20), "ema_crossover",
        {"fast_period": 3, "slow_period": 6, "position_pct": 0.2},
    )
    assert res.symbol == "BTC/USDT"
    assert res.interval == "1h"
    assert res.strategy_name == "ema_crossover"
    assert res.params["fast_period"] == 3
