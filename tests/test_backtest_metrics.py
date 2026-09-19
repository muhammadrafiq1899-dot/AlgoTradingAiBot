"""Backtest metrics: hand-computed values, walk-forward folds, seeded resampling.

The point of these tests is that the arithmetic is checked against numbers worked
out on paper, not against a second call of the same function: a metric suite that
only agrees with itself proves nothing.
"""
import json
import math

import pytest

from algotrading.backtest.metrics import (
    annualisation_factor,
    bootstrap,
    compute_metrics,
    max_drawdown_pct,
    monte_carlo,
    per_bar_returns,
    pnl_breakdown,
    sharpe_ratio,
    sortino_ratio,
    walk_forward,
)
from algotrading.backtest.runner import BacktestResult, BacktestTrade, run_backtest
from algotrading.config import load_settings
from algotrading.market.base import Candle

HOUR_MS = 3_600_000
# 2024-01-01T00:00Z, 2024-01-02T00:00Z, 2024-02-01T00:00Z, 2025-01-01T00:00Z
T_JAN1 = 1_704_067_200_000
T_JAN2 = 1_704_153_600_000
T_FEB1 = 1_706_745_600_000
T_JAN1_2025 = 1_735_689_600_000
SETTINGS = load_settings()


def _trade(exit_ts, pnl, *, duration_ms=2 * HOUR_MS):
    return BacktestTrade(
        entry_ts=exit_ts - duration_ms,
        entry_price=100.0,
        exit_ts=exit_ts,
        exit_price=100.0 + pnl,
        pnl=pnl,
    )


def _result(pnls=(10.0, -5.0, 5.0, 2.0), **overrides):
    """A result with a known curve: +10%, -10%, +10% over four bars."""
    exits = (T_JAN1, T_JAN2, T_FEB1, T_JAN1_2025)
    trades = [_trade(ts, pnl) for ts, pnl in zip(exits, pnls)]
    defaults = dict(
        strategy_name="unit",
        params={},
        symbol="BTC/USDT",
        interval="1h",
        n_trades=len(trades),
        n_wins=sum(1 for p in pnls if p > 0),
        n_losses=sum(1 for p in pnls if p < 0),
        total_pnl=sum(pnls),
        max_drawdown=11.0,
        win_rate=3 / 4,
        final_balance=10_000.0 + sum(pnls),
        final_cash=10_000.0 + sum(pnls),
        trades=trades,
        initial_balance=10_000.0,
        n_bars=4,
        equity_curve=[10_000.0, 11_000.0, 9_900.0, 10_890.0],
        exposure=0.5,
        position_bars=2,
        fees_paid=1.0,
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def _candles(n, start=100.0, step=0.5, interval="1h"):
    return [
        Candle(
            symbol="BTC/USDT",
            interval=interval,
            ts=1_600_000_000_000 + i * HOUR_MS,
            open=start + i * step,
            high=start + i * step + 1.0,
            low=start + i * step - 1.0,
            close=start + i * step,
            volume=1.0,
        )
        for i in range(n)
    ]


# --- per-bar statistics -------------------------------------------------------


def test_per_bar_returns_are_simple_percentage_changes():
    assert per_bar_returns([100.0, 110.0, 99.0, 108.9]) == pytest.approx(
        [0.1, -0.1, 0.1]
    )


def test_per_bar_returns_skips_a_wiped_out_bar():
    """A previous equity of 0 cannot define a return; it is skipped, not inf."""
    assert per_bar_returns([0.0, 100.0]) == []


def test_annualisation_factor_is_sqrt_of_bars_per_year():
    assert annualisation_factor("1h") == pytest.approx(math.sqrt(365 * 24))
    assert annualisation_factor("1m") == pytest.approx(math.sqrt(365 * 24 * 60))
    # Unknown intervals fall back to 1h rather than dividing by nothing.
    assert annualisation_factor("7s") == pytest.approx(math.sqrt(365 * 24))


def _hand_sharpe():
    """Sharpe of returns [0.1, -0.1, 0.1] over 1h bars, worked out by hand."""
    returns = [0.1, -0.1, 0.1]
    mean = sum(returns) / 3                     # 1/30 = 0.0331333
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return mean / math.sqrt(variance) * math.sqrt(365 * 24)   # ~27.0185


def _hand_sortino():
    """Sortino of the same series: downside deviation = sqrt(0.1^2 / 3)."""
    returns = [0.1, -0.1, 0.1]
    mean = sum(returns) / 3
    downside = math.sqrt(sum(min(0.0, r) ** 2 for r in returns) / len(returns))
    return mean / downside * math.sqrt(365 * 24)              # ~54.0370


def test_sharpe_matches_the_hand_computed_value():
    value = sharpe_ratio([10_000.0, 11_000.0, 9_900.0, 10_890.0], "1h")
    assert value == pytest.approx(_hand_sharpe(), rel=1e-9)
    assert value == pytest.approx(27.0185, abs=1e-3)


def test_she_metrics_return_none_when_undefined():
    assert sharpe_ratio([10_000.0], "1h") is None            # one bar, no returns
    assert sharpe_ratio([10_000.0, 10_000.0, 10_000.0], "1h") is None  # zero variance
    # No downside at all -> Sortino is undefined, not infinite.
    assert sortino_ratio([10_000.0, 10_100.0, 10_200.0], "1h") is None


def test_sortino_matches_the_hand_computed_value():
    value = sortino_ratio([10_000.0, 11_000.0, 9_900.0, 10_890.0], "1h")
    assert value == pytest.approx(_hand_sortino(), rel=1e-9)
    assert value == pytest.approx(54.0370, abs=1e-3)


def test_max_drawdown_pct_is_peak_to_trough():
    # peak 11000 -> trough 9900 = 10%
    assert max_drawdown_pct([10_000.0, 11_000.0, 9_900.0, 10_890.0]) == pytest.approx(10.0)
    assert max_drawdown_pct([10_000.0, 9_000.0, 8_000.0]) == pytest.approx(20.0)
    assert max_drawdown_pct([]) == 0.0


# --- trade statistics ---------------------------------------------------------


def test_trade_statistics_match_by_hand():
    metrics = compute_metrics(_result())
    assert metrics["n_trades"] == 4
    assert metrics["n_wins"] == 3 and metrics["n_losses"] == 1
    assert metrics["win_rate"] == pytest.approx(0.75)
    assert metrics["total_pnl"] == pytest.approx(12.0)
    assert metrics["return_pct"] == pytest.approx(0.12)
    assert metrics["gross_profit"] == pytest.approx(17.0)
    assert metrics["gross_loss"] == pytest.approx(5.0)
    assert metrics["profit_factor"] == pytest.approx(3.4)
    assert metrics["avg_win"] == pytest.approx(17.0 / 3)
    assert metrics["avg_loss"] == pytest.approx(-5.0)
    assert metrics["largest_win"] == pytest.approx(10.0)
    assert metrics["largest_loss"] == pytest.approx(-5.0)
    # expectancy = 0.75 * (17/3) + 0.25 * (-5) = 4.25 - 1.25 = 3.0
    assert metrics["expectancy"] == pytest.approx(3.0)


def test_curve_statistics_flow_through_compute_metrics():
    metrics = compute_metrics(_result())
    assert metrics["sharpe"] == pytest.approx(_hand_sharpe(), rel=1e-9)
    assert metrics["sortino"] == pytest.approx(_hand_sortino(), rel=1e-9)
    assert metrics["max_drawdown"] == pytest.approx(11.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(10.0)
    assert metrics["exposure"] == pytest.approx(0.5)
    assert metrics["n_bars"] == 4
    assert metrics["fees_paid"] == pytest.approx(1.0)


def test_average_trade_duration_is_reported_in_three_units():
    metrics = compute_metrics(_result())
    assert metrics["avg_trade_duration_ms"] == pytest.approx(2 * HOUR_MS)
    assert metrics["avg_trade_duration_hours"] == pytest.approx(2.0)
    assert metrics["avg_trade_duration_bars"] == pytest.approx(2.0)


def test_profit_factor_is_none_without_losses():
    metrics = compute_metrics(_result(pnls=(1.0, 2.0)))
    assert metrics["profit_factor"] is None, "infinite profit factor is not JSON"
    assert metrics["avg_loss"] is None


def test_pnl_breakdowns_bucket_by_exit_timestamp():
    result = _result()
    assert list(pnl_breakdown(result, "day")) == [
        "2024-01-01", "2024-01-02", "2024-02-01", "2025-01-01",
    ]
    assert pnl_breakdown(result, "day")["2024-01-01"]["pnl"] == pytest.approx(10.0)
    assert pnl_breakdown(result, "day")["2024-01-02"]["losses"] == 1
    weeks = pnl_breakdown(result, "week")
    assert weeks["2024-W01"]["trades"] == 2
    assert weeks["2024-W01"]["pnl"] == pytest.approx(5.0)
    assert weeks["2024-W05"]["pnl"] == pytest.approx(5.0)
    months = pnl_breakdown(result, "month")
    assert months["2024-01"]["trades"] == 2
    assert months["2024-02"]["pnl"] == pytest.approx(5.0)
    years = pnl_breakdown(result, "year")
    assert years["2024"]["trades"] == 3
    assert years["2024"]["pnl"] == pytest.approx(10.0)
    assert years["2025"]["wins"] == 1
    with pytest.raises(ValueError):
        pnl_breakdown(result, "fortnight")


def test_compute_metrics_is_json_serialisable_and_numpy_free():
    metrics = compute_metrics(_result())
    payload = json.dumps(metrics)  # raises on non-JSON values
    assert '"sharpe"' in payload
    assert isinstance(metrics["pnl_by_month"], dict)
    assert all(isinstance(v, (int, float, str, type(None), dict, list))
               for v in metrics.values())


def test_compute_metrics_on_a_real_replay():
    prices = [100.0] * 12 + [101, 105, 118, 135, 148, 160] + [150, 128, 100]
    candles = [
        Candle(symbol="BTC/USDT", interval="1h", ts=1_600_000_000_000 + i * HOUR_MS,
               open=p, high=p, low=p, close=p, volume=1.0)
        for i, p in enumerate(prices)
    ]
    result = run_backtest(candles, "ema_crossover",
                          {"fast_period": 3, "slow_period": 6})
    metrics = compute_metrics(result, candles)
    assert metrics["n_trades"] == 1
    assert metrics["n_bars"] == len(candles)
    assert metrics["exposure"] == pytest.approx(result.exposure)
    assert metrics["assumptions"] == result.assumptions
    assert metrics["sharpe"] is not None


# --- walk-forward -------------------------------------------------------------


def test_walk_forward_splits_into_contiguous_folds():
    candles = _candles(400)
    report = walk_forward(candles, "ema_crossover", {"fast_period": 5, "slow_period": 12},
                          folds=4)
    assert report["folds"] == 4
    assert len(report["per_fold"]) == 4
    covered = [(f["start_bar"], f["end_bar"]) for f in report["per_fold"]]
    assert covered[0][0] == 0 and covered[-1][1] == len(candles)
    for (_, stop), (start, _) in zip(covered, covered[1:]):
        assert start == stop, "folds must be contiguous and non-overlapping"
    assert sum(f["n_bars"] for f in report["per_fold"]) == len(candles)


def test_walk_forward_clamps_the_fold_count_for_a_short_series():
    report = walk_forward(_candles(5), "ema_crossover",
                          {"fast_period": 2, "slow_period": 3}, folds=4)
    assert report["requested_folds"] == 4
    assert report["folds"] == 2, "5 candles cannot fill 4 folds"


def test_walk_forward_uses_the_configured_fold_count_by_default():
    report = walk_forward(_candles(400), "ema_crossover",
                          {"fast_period": 5, "slow_period": 12})
    assert report["folds"] == SETTINGS.backtest.walk_forward_folds


def test_walk_forward_consistency_summary_and_limitations():
    report = walk_forward(_candles(400), "ema_crossover",
                          {"fast_period": 5, "slow_period": 12}, folds=4)
    consistency = report["consistency"]
    for key in ("folds", "profitable_folds", "profitable_fraction", "mean_return_pct",
                "median_return_pct", "stdev_return_pct", "best_return_pct",
                "worst_return_pct", "consistent", "verdict"):
        assert key in consistency
    assert isinstance(consistency["consistent"], bool)
    assert 0 <= consistency["profitable_fraction"] <= 1
    assert report["limitations"] and any("purged" in line for line in report["limitations"])


def test_walk_forward_is_deterministic_and_rejects_no_data():
    args = (_candles(200), "ema_crossover", {"fast_period": 4, "slow_period": 9})
    first = walk_forward(*args, folds=3)["consistency"]
    second = walk_forward(*args, folds=3)["consistency"]
    assert first == second
    with pytest.raises(ValueError):
        walk_forward([], "ema_crossover", {})


def test_walk_forward_records_unbuildable_params_instead_of_crashing():
    # fast_period >= slow_period: the strategy refuses to build, so every fold is
    # reported as skipped and nothing is scored.
    report = walk_forward(_candles(60), "ema_crossover", {"fast_period": 9, "slow_period": 3},
                          folds=2)
    assert report["replayed_folds"] == 0
    assert all(f["skipped"] for f in report["per_fold"])
    assert report["consistency"]["consistent"] is False


# --- Monte Carlo / bootstrap --------------------------------------------------


def test_monte_carlo_is_deterministic_for_a_seed():
    result = _result(pnls=[10.0, -5.0, 7.0, -3.0, 2.0, -8.0] * 5)
    first = monte_carlo(result, runs=50, seed=7)
    second = monte_carlo(result, runs=50, seed=7)
    assert first == second
    different = monte_carlo(result, runs=50, seed=8)
    assert different["final_pnl"] != first["final_pnl"]
    assert first["seed"] == 7


def test_monte_carlo_percentiles_are_ordered_and_bounded():
    result = _result(pnls=[10.0, -5.0, 7.0, -3.0, 2.0, -8.0] * 5)
    report = monte_carlo(result, runs=100, seed=42)
    stats = report["final_pnl"]
    assert stats["p5"] <= stats["p50"] <= stats["p95"]
    assert stats["min"] <= stats["p5"] and stats["p95"] <= stats["max"]
    assert 0.0 <= report["prob_loss"] <= 1.0
    assert report["max_drawdown"]["p50"] >= 0.0
    assert report["observed_final_pnl"] == pytest.approx(sum(t.pnl for t in result.trades))
    assert report["assumptions"], "a resample must state what it does not model"


def test_monte_carlo_with_one_trade_is_a_point_mass():
    result = _result(pnls=(25.0,))
    report = monte_carlo(result, runs=20, seed=1)
    assert report["final_pnl"]["p5"] == report["final_pnl"]["p95"] == pytest.approx(25.0)
    assert report["prob_loss"] == 0.0
    assert report["max_drawdown"]["p95"] == 0.0


def test_monte_carlo_without_trades_reports_no_probability():
    report = monte_carlo(_result(pnls=()), runs=10, seed=1)
    assert report["prob_loss"] is None
    assert report["runs"] == 10


def test_monte_carlo_uses_the_configured_seed_and_run_budget_by_default():
    result = _result(pnls=[1.0, -1.0] * 20)
    report = monte_carlo(result)
    assert report["seed"] == SETTINGS.backtest.random_seed
    assert report["runs"] == SETTINGS.backtest.monte_carlo_runs


def test_bootstrap_bounds_the_mean_trade_return():
    result = _result(pnls=[10.0, -5.0, 7.0, -3.0, 2.0, -8.0] * 5)
    report = bootstrap(result, runs=100, seed=3)
    assert report["runs"] == 100
    assert report["mean_trade_pnl"]["p5"] <= report["mean_trade_pnl"]["p50"]
    assert report["mean_trade_pnl"]["p50"] == pytest.approx(
        report["observed_mean_trade_pnl"], rel=0.5
    )
    assert 0.0 <= report["prob_mean_le_zero"] <= 1.0
    assert report["total_return_pct"]["p95"] > report["total_return_pct"]["p5"]


def test_bootstrap_is_deterministic_and_handles_a_losing_edge():
    losing = _result(pnls=(-3.0, -1.0) * 10)
    report = bootstrap(losing, runs=25, seed=11)
    assert report["prob_mean_le_zero"] == 1.0
    assert report == bootstrap(losing, runs=25, seed=11)
    assert bootstrap(_result(pnls=()), runs=5, seed=1)["prob_mean_le_zero"] is None
