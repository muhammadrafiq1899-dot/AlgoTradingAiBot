"""Backtest runner: replay candles through a strategy, produce equity summary.

Covers both the original contract (which Telegram, the recommendation store and
this file consumed before parity) and the parity work: slippage, fees, protective
levels, risk-check simulation, the lazy window and the honesty fields.
"""
import pytest

from algotrading.backtest import runner as runner_mod
from algotrading.backtest.runner import (
    CandleWindow,
    GUARD_COOLDOWN,
    GUARD_DAILY_LOSS,
    EXIT_SIGNAL,
    EXIT_STOP,
    EXIT_TAKE_PROFIT,
    run_backtest,
)
from algotrading.config import load_settings
from algotrading.market.base import Candle
from algotrading.strategy.base import Signal

SETTINGS = load_settings()
HOUR_MS = 3_600_000


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


def _bars(rows, interval="1h", start_ts=1_600_000_000_000):
    """Candles from (close, high, low) rows."""
    out = []
    step = 60_000 if interval == "1m" else 3_600_000
    for i, row in enumerate(rows):
        close, high, low = row if isinstance(row, (tuple, list)) else (row, row, row)
        out.append(
            Candle(
                symbol="BTC/USDT",
                interval=interval,
                ts=start_ts + i * step,
                open=close,
                high=high,
                low=low,
                close=close,
                volume=1.0,
            )
        )
    return out


class _ScriptedStrategy:
    """Emits signals at scripted bar indices, ignoring the window contents.

    Gives a test exact control over WHEN a signal fires, which is the only way
    to assert stop/target/cooldown behaviour bar by bar.
    """

    name = "scripted"
    params: dict = {}

    def __init__(self, script, risk=None):
        self.script = dict(script)
        self.risk = dict(risk or {})

    def evaluate(self, symbol, candles):
        index = len(candles) - 1
        side = self.script.get(index)
        if side is None:
            return None
        return Signal(
            strategy_id=0,
            symbol=symbol,
            side=side,
            ref_price=candles[-1].close,
            rationale=f"scripted {side} at bar {index}",
            risk=dict(self.risk),
        )


@pytest.fixture()
def scripted(monkeypatch):
    """Install a scripted strategy in place of the registry lookup."""

    def _install(script, risk=None):
        strategy = _ScriptedStrategy(script, risk)
        monkeypatch.setattr(runner_mod, "build_strategy", lambda name, params: strategy)
        return strategy

    return _install


# --- original contract (unchanged behaviour) ----------------------------------


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


def test_positional_signature_still_works():
    """`run_backtest(candles, name, params, balance, fee)` is a public contract."""
    res = run_backtest(_candles([100.0] * 20), "ema_crossover", {"fast_period": 3, "slow_period": 6}, 5_000.0, 0.002)
    assert res.initial_balance == pytest.approx(5_000.0)
    assert res.fee_rate == pytest.approx(0.002)


def test_defaults_come_from_settings():
    """None overrides read settings.backtest / settings.risk, not constants."""
    res = run_backtest(_candles([100.0] * 20), "ema_crossover", {"fast_period": 3, "slow_period": 6})
    assert res.fee_rate == pytest.approx(SETTINGS.backtest.fee_rate)
    assert res.slippage_pct == pytest.approx(SETTINGS.backtest.slippage_pct)
    assert res.apply_risk_checks is SETTINGS.backtest.apply_risk_checks
    assert res.initial_balance == pytest.approx(SETTINGS.risk.paper_initial_balance)


# --- costs: slippage + fees ---------------------------------------------------


def test_slippage_moves_both_fills_against_the_position():
    prices = [100.0] * 12 + [101, 105, 118, 135, 148, 160] + [150, 128, 100]
    candles = _candles(prices)
    free = run_backtest(candles, "ema_crossover",
                        {"fast_period": 3, "slow_period": 6}, fee_rate=0.0,
                        slippage_pct=0.0)
    slipped = run_backtest(candles, "ema_crossover",
                           {"fast_period": 3, "slow_period": 6}, fee_rate=0.0,
                           slippage_pct=1.0)
    assert free.n_trades == slipped.n_trades == 1
    assert slipped.trades[0].entry_price > free.trades[0].entry_price
    assert slipped.trades[0].exit_price < free.trades[0].exit_price
    # 1% adverse on entry and exit, on top of an unchanged reference price.
    assert slipped.trades[0].entry_price == pytest.approx(
        slipped.trades[0].entry_ref_price * 1.01
    )
    assert slipped.trades[0].exit_price == pytest.approx(
        slipped.trades[0].exit_ref_price * 0.99
    )
    assert slipped.total_pnl < free.total_pnl


def test_fee_rate_is_charged_per_side_and_reported():
    prices = [100.0] * 12 + [101, 105, 118, 135, 148, 160] + [150, 128, 100]
    candles = _candles(prices)
    free = run_backtest(candles, "ema_crossover",
                        {"fast_period": 3, "slow_period": 6}, fee_rate=0.0)
    charged = run_backtest(candles, "ema_crossover",
                           {"fast_period": 3, "slow_period": 6}, fee_rate=0.002)
    assert charged.fees_paid > free.fees_paid == 0.0
    assert charged.total_pnl < free.total_pnl
    # Two sides on roughly one balance of notional.
    assert charged.fees_paid > 10_000.0 * 0.002 * 1.9
    assert charged.to_dict()["fee_rate"] == pytest.approx(0.002)


# --- protective levels from the signal's risk dict ----------------------------


def test_stop_pct_exits_at_the_stop_price(scripted):
    scripted({2: "buy"}, risk={"stop_pct": 2.0})
    candles = _bars([100.0, 100.0, 100.0, (100.0, 100.0, 97.0)])
    res = run_backtest(candles, "scripted", {}, fee_rate=0.0, slippage_pct=0.0)
    assert res.n_trades == 1
    trade = res.trades[0]
    assert trade.exit_reason == EXIT_STOP
    assert trade.exit_price == pytest.approx(98.0)  # 2% below the 100.0 fill
    assert trade.pnl < 0


def test_take_profit_pct_exits_at_the_target(scripted):
    scripted({2: "buy"}, risk={"take_profit_pct": 5.0})
    candles = _bars([100.0, 100.0, 100.0, (104.0, 106.0, 99.0)])
    res = run_backtest(candles, "scripted", {}, fee_rate=0.0, slippage_pct=0.0)
    assert res.n_trades == 1
    trade = res.trades[0]
    assert trade.exit_reason == EXIT_TAKE_PROFIT
    assert trade.exit_price == pytest.approx(105.0)
    assert trade.pnl > 0


def test_stop_wins_when_one_bar_touches_both_levels(scripted):
    """Documented pessimism: a bar that could have hit either is a stop."""
    scripted({2: "buy"}, risk={"stop_pct": 2.0, "take_profit_pct": 5.0})
    candles = _bars([100.0, 100.0, 100.0, (103.0, 106.0, 97.0)])
    res = run_backtest(candles, "scripted", {}, fee_rate=0.0, slippage_pct=0.0)
    assert res.trades[0].exit_reason == EXIT_STOP


def test_protective_levels_cannot_fire_on_the_fill_bar(scripted):
    """The fill is at the signal bar's close: that bar's range already happened."""
    scripted({2: "buy"}, risk={"stop_pct": 2.0})
    # Bar 2 itself has a low far below the stop; only bar 3 may exit.
    candles = _bars([100.0, 100.0, (100.0, 100.0, 90.0), (99.0, 99.0, 99.0)])
    res = run_backtest(candles, "scripted", {}, fee_rate=0.0, slippage_pct=0.0)
    assert res.n_trades == 0
    assert res.position_open_at_end is True


def test_absolute_stop_price_is_honoured(scripted):
    scripted({2: "buy"}, risk={"stop_price": 95.0})
    candles = _bars([100.0, 100.0, 100.0, (96.0, 97.0, 94.0)])
    res = run_backtest(candles, "scripted", {}, fee_rate=0.0, slippage_pct=0.0)
    assert res.trades[0].exit_reason == EXIT_STOP
    assert res.trades[0].exit_price == pytest.approx(95.0)


def test_slippage_is_applied_to_stop_fills_too(scripted):
    scripted({2: "buy"}, risk={"stop_pct": 2.0})
    candles = _bars([100.0, 100.0, 100.0, (100.0, 100.0, 97.0)])
    res = run_backtest(candles, "scripted", {}, fee_rate=0.0, slippage_pct=1.0)
    # The 2% stop is measured from the FILL (100 * 1.01), then slipped again.
    assert res.trades[0].exit_reason == EXIT_STOP
    assert res.trades[0].exit_price == pytest.approx(101.0 * 0.98 * 0.99)


def test_no_risk_keys_means_no_protective_exit(scripted):
    scripted({2: "buy"})
    candles = _bars([100.0, 100.0, 100.0, (95.0, 95.0, 90.0), (100.0, 100.0, 100.0)])
    res = run_backtest(candles, "scripted", {}, fee_rate=0.0, slippage_pct=0.0)
    assert res.n_trades == 0, "nothing closes the position without a sell signal"


def test_signal_sell_still_reports_the_signal_reason(scripted):
    scripted({2: "buy", 3: "sell"})
    res = run_backtest(_bars([100.0, 100.0, 100.0, 110.0]), "scripted",
                       {}, fee_rate=0.0, slippage_pct=0.0)
    assert res.trades[0].exit_reason == EXIT_SIGNAL


# --- risk-check simulation ----------------------------------------------------


def test_risk_checks_are_off_by_default_and_all_in():
    res = run_backtest(_candles([100.0] * 20 + [110.0] * 4), "ema_crossover", {"fast_period": 3, "slow_period": 6})
    assert res.apply_risk_checks is False
    assert res.skipped_by_guard == {GUARD_COOLDOWN: 0, GUARD_DAILY_LOSS: 0,
                                    "position_size": 0}


def test_cooldown_skips_an_entry_inside_the_window(scripted):
    """1m bars: a re-entry 2 minutes after the last one is inside the 300s gap."""
    candles = _bars([100.0] * 11, interval="1m")
    scripted({1: "buy", 2: "sell", 3: "buy", 4: "sell", 8: "buy", 9: "sell"})
    res = run_backtest(candles, "scripted", {}, apply_risk_checks=True, risk_pct=0.2,
                       fee_rate=0.0, slippage_pct=0.0)
    # Bar 3 is 120s after the bar-1 entry -> blocked; bar 8 is 7 minutes later -> allowed.
    assert res.skipped_by_guard[GUARD_COOLDOWN] == 1
    assert res.skipped_by_guard[GUARD_DAILY_LOSS] == 0
    assert res.n_trades == 2
    assert res.trades[0].entry_ts == candles[1].ts
    assert res.trades[1].entry_ts == candles[8].ts


def test_daily_loss_blocks_entries_for_the_rest_of_the_utc_day(scripted):
    day_start = 1_700_000_000_000 // 86_400_000 * 86_400_000
    # One losing round trip (-20% on a 20% position = -4% of balance) then a
    # later signal in the SAME UTC day (past the cooldown, so the daily-loss
    # guard is the only thing that can block it), then a fresh day.
    scripted({0: "buy", 6: "sell", 12: "buy", 21: "buy", 22: "sell"})
    rows = [(100.0, 100.0, 100.0)] * 6 + [(80.0, 80.0, 80.0)] * 6 + [(100.0, 100.0, 100.0)] * 9
    candles = _bars(rows, interval="1m", start_ts=day_start)
    for offset in (1, 2):
        candles.append(
            Candle(symbol="BTC/USDT", interval="1m",
                   ts=day_start + 86_400_000 + offset * 60_000,
                   open=100.0, high=100.0, low=100.0, close=100.0, volume=1.0)
        )
    res = run_backtest(candles, "scripted", {}, apply_risk_checks=True, risk_pct=0.2,
                       fee_rate=0.0, slippage_pct=0.0)
    assert res.skipped_by_guard[GUARD_DAILY_LOSS] == 1
    assert res.skipped_by_guard[GUARD_COOLDOWN] == 0
    assert res.n_trades == 2, "the losing trade, then the next day's re-entry"
    assert res.trades[0].pnl < 0
    assert res.trades[1].entry_ts >= day_start + 86_400_000


def test_max_position_pct_caps_an_explicit_risk_pct(scripted):
    scripted({1: "buy", 3: "sell"}, risk={"position_pct": 0.5})
    candles = _bars([100.0, 100.0, 100.0, 110.0])
    capped = run_backtest(candles, "scripted", {}, apply_risk_checks=True,
                          fee_rate=0.0, slippage_pct=0.0)
    uncapped = run_backtest(candles, "scripted", {}, fee_rate=0.0, slippage_pct=0.0,
                            risk_pct=0.5)
    # 50% of balance asked for, 20% (risk.max_position_pct) allowed.
    assert capped.capped_entries == 1
    assert capped.total_pnl == pytest.approx(10_000.0 * 0.2 * 0.10)
    assert uncapped.total_pnl == pytest.approx(10_000.0 * 0.5 * 0.10)


def test_spot_sizing_never_exceeds_the_balance(scripted):
    """A requested fraction above 1.0 means borrowing; spot cannot borrow."""
    scripted({1: "buy", 3: "sell"})
    candles = _bars([100.0, 100.0, 100.0, 110.0])
    leveraged = run_backtest(candles, "scripted", {}, risk_pct=3.0,
                             fee_rate=0.0, slippage_pct=0.0)
    all_in = run_backtest(candles, "scripted", {}, risk_pct=1.0,
                          fee_rate=0.0, slippage_pct=0.0)
    assert leveraged.total_pnl == pytest.approx(all_in.total_pnl)
    assert leveraged.final_balance == pytest.approx(11_000.0)


def test_zero_sizing_request_skips_the_entry(scripted):
    scripted({1: "buy"}, risk={"position_pct": 0.0})
    res = run_backtest(_bars([100.0, 100.0, 100.0, 110.0]), "scripted", {},
                       apply_risk_checks=True, fee_rate=0.0, slippage_pct=0.0)
    assert res.n_trades == 0
    assert res.skipped_by_guard["position_size"] == 1


def test_signal_position_pct_drives_sizing_when_risk_checks_are_on(scripted):
    scripted({1: "buy", 3: "sell"}, risk={"position_pct": 0.1})
    candles = _bars([100.0, 100.0, 100.0, 110.0])
    res = run_backtest(candles, "scripted", {}, apply_risk_checks=True,
                       fee_rate=0.0, slippage_pct=0.0)
    assert res.capped_entries == 0
    assert res.total_pnl == pytest.approx(10_000.0 * 0.1 * 0.10)


# --- the lazy window ----------------------------------------------------------


def test_candle_window_behaves_like_a_list_slice():
    candles = _candles([float(i) for i in range(10)])
    window = CandleWindow(candles, 4)
    assert len(window) == 5
    assert window[0] is candles[0]
    assert window[-1] is candles[4]
    assert [c.close for c in window] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert list(window[2:]) == list(candles[2:5])
    assert list(window[-2:]) == list(candles[3:5])
    assert list(window[::2]) == [candles[0], candles[2], candles[4]]
    with pytest.raises(IndexError):
        window[5]


def test_window_is_reused_without_copying_the_underlying_list():
    candles = _candles([float(i) for i in range(10)])
    window = CandleWindow(candles, -1)
    assert len(window) == 0
    window.advance_to(2)
    assert len(window) == 3 and window[-1] is candles[2]
    # `candles` is never copied: the window indexes the list it was built from.
    window.advance_to(9)
    assert window[9] is candles[9]


def test_lazy_window_gives_every_strategy_the_same_signals_as_a_list():
    prices = [100.0] * 20 + [100 + i for i in range(30)] + [130 - i for i in range(30)]
    candles = _candles(prices)
    for name, params in (
        ("ema_crossover", {"fast_period": 5, "slow_period": 12}),
        ("rsi_mean_reversion", {"period": 6, "oversold": 30.0, "overbought": 70.0}),
        ("macd_trend", {"fast": 5, "slow": 12, "signal": 4}),
        ("supertrend", {"period": 7, "multiplier": 2.0}),
    ):
        strategy = runner_mod.build_strategy(name, dict(params))
        window = CandleWindow(candles, -1)
        for i in range(len(candles)):
            window.advance_to(i)
            lazy = strategy.evaluate("BTC/USDT", window)
            eager = strategy.evaluate("BTC/USDT", list(candles[: i + 1]))
            if eager is None:
                assert lazy is None, f"{name} signalled only through the lazy window"
                continue
            assert lazy is not None, f"{name} lost its signal at bar {i}"
            assert (lazy.side, lazy.rationale, lazy.risk) == (
                eager.side, eager.rationale, eager.risk,
            )


def test_replay_matches_the_old_list_window_algorithm():
    """fee 0.001, no slippage, all-in: identical to the pre-parity runner."""
    prices = [100.0] * 12 + [101, 105, 118, 135, 148, 160] + [150, 128, 100]
    candles = _candles(prices)
    params = {"fast_period": 3, "slow_period": 6}
    res = run_backtest(candles, "ema_crossover", dict(params), fee_rate=0.001,
                       slippage_pct=0.0)
    reference_trades, reference_balance = _legacy_replay(candles, "ema_crossover", params)
    assert res.n_trades == reference_trades
    assert res.final_balance == pytest.approx(reference_balance, rel=1e-12)


def _legacy_replay(candles, strategy_name, params):
    """The pre-parity loop: list window, all-in, fill at close, fee per side."""
    strategy = runner_mod.build_strategy(strategy_name, dict(params))
    fee = 0.001
    balance = 10_000.0
    holding = False
    qty = entry = 0.0
    trades = 0
    for i in range(len(candles)):
        window = list(candles[: i + 1])
        close = window[-1].close
        sig = strategy.evaluate("BTC/USDT", window)
        if sig is not None and sig.side == "buy" and not holding:
            entry = close
            qty = balance / (entry + entry * fee)
            holding = True
        elif sig is not None and sig.side == "sell" and holding:
            proceeds = qty * close - qty * close * fee
            balance = proceeds
            holding = False
            qty = 0.0
            trades += 1
    return trades, balance


# --- result shape -------------------------------------------------------------


def test_to_dict_keeps_every_original_key():
    res = run_backtest(_candles([100.0] * 20), "ema_crossover", {"fast_period": 3, "slow_period": 6})
    data = res.to_dict()
    for key in (
        "strategy_name", "symbol", "interval", "n_trades", "n_wins", "n_losses",
        "win_rate", "total_pnl", "max_drawdown", "final_balance",
    ):
        assert key in data, f"{key} must stay in the persisted backtest shape"
    assert isinstance(data["n_trades"], int)
    assert isinstance(data["win_rate"], float)


def test_result_carries_curve_exposure_and_assumptions():
    prices = [100.0] * 12 + [101, 105, 118, 135, 148, 160] + [150, 128, 100]
    res = run_backtest(_candles(prices), "ema_crossover",
                       {"fast_period": 3, "slow_period": 6})
    assert len(res.equity_curve) == res.n_bars == len(prices)
    assert 0.0 < res.exposure < 1.0
    assert res.position_bars == sum(1 for _ in range(res.position_bars))
    assert res.assumptions and isinstance(res.assumptions, list)
    assert any("slippage" in line.lower() for line in res.assumptions)
    assert any("close" in line.lower() for line in res.assumptions)
    assert res.to_dict()["exposure"] == pytest.approx(res.exposure, abs=1e-4)


def test_equity_curve_starts_at_the_initial_balance():
    res = run_backtest(_candles([100.0] * 25), "ema_crossover", {"fast_period": 3, "slow_period": 6})
    assert res.equity_curve[0] == pytest.approx(10_000.0)
    assert res.max_drawdown == 0.0


def test_flat_result_reports_zero_exposure_and_no_costs():
    res = run_backtest(_candles([100.0] * 25), "ema_crossover", {"fast_period": 3, "slow_period": 6})
    assert res.exposure == 0.0
    assert res.fees_paid == 0.0
    assert res.final_cash == pytest.approx(10_000.0)


def test_open_position_is_marked_not_fabricated(scripted):
    scripted({2: "buy"})
    res = run_backtest(_bars([100.0, 100.0, 100.0, 120.0]), "scripted",
                       {}, fee_rate=0.0, slippage_pct=0.0)
    assert res.n_trades == 0
    assert res.position_open_at_end is True
    assert res.final_balance == pytest.approx(12_000.0)  # marked at the last close
    assert res.final_cash == pytest.approx(0.0)          # all-in: nothing realized
    assert any("still open" in line for line in res.assumptions)
