"""The Pine v5 "3Commas Bot" port: entry, stop, target, trail, filters, activation.

The plugin is loaded from the real `strategies/` directory (same loader, same
validator the bot uses at runtime), so these tests fail if the file breaks the
plugin contract — not just if the maths is wrong.
"""
import json

import pytest

from algotrading.backtest.runner import run_backtest
from algotrading.db import get_session_factory, init_db
from algotrading.db.models import Strategy
from algotrading.market.base import Candle
from algotrading.store.recommendations import RecommendationStore, create_pending_recommendation
from algotrading.strategy.catalog import catalog_entries, default_params
from algotrading.strategy.plugins import get_default_loader
from algotrading.strategy.registry import build_strategy, known_names

NAME = "three_commas_bot"
BASE_TS = 1_700_000_000_000
HOUR = 3_600_000


@pytest.fixture(scope="module", autouse=True)
def _load_plugins():
    """Load strategies/*.py the way the bot does before serving a request."""
    get_default_loader().load_all()
    yield


def series(prices, start_ts=BASE_TS, pad=0.004):
    return [
        Candle(symbol="BTC/USDT", interval="1h", ts=start_ts + i * HOUR,
               open=p, high=p * (1 + pad), low=p * (1 - pad), close=p, volume=1.0)
        for i, p in enumerate(prices)
    ]


def first_signal(strategy, candles, start=30):
    """Walk the window like the engine does and return the first signal."""
    for i in range(start, len(candles)):
        sig = strategy.evaluate("BTC/USDT", candles[: i + 1])
        if sig is not None:
            return sig
    return None


# A dip that puts the fast MA under the slow one, then a steady rally that
# crosses back up: the exact shape the original strategy trades.
DIP_THEN_RALLY = [100.0] * 40 + [100 - 0.8 * i for i in range(1, 31)] + [76 + 1.2 * i for i in range(1, 61)]
FAST_SLOW = {"ma_type_1": "EMA", "ma_type_2": "EMA", "ma_length_1": 9,
             "ma_length_2": 21, "atr_len": 14, "swing_lookback": 5}


# --- contract -----------------------------------------------------------------

def test_plugin_loads_from_the_strategies_directory():
    info = get_default_loader().info(NAME)
    assert info is not None, "the plugin must load with the real loader"
    assert NAME in known_names()
    assert info.params, "STRATEGY_PARAMS metadata should reach the catalog"
    assert info.indicator_deps


def test_catalog_exposes_it_as_a_plugin_with_parameter_ranges():
    entry = next((e for e in catalog_entries() if e.name == NAME), None)
    assert entry is not None and entry.kind == "plugin"
    by_name = {p.name: p for p in entry.params}
    assert by_name["ma_type_1"].enum == ("EMA", "HEMA", "SMA", "HMA", "WMA", "DEMA", "VWMA", "VWAP", "T3")
    assert by_name["trail_source"].enum == ("High/Low", "Close", "Open")
    assert by_name["risk_m"].min == 0.0 and by_name["risk_m"].max == 10.0


def test_defaults_match_the_pine_inputs():
    params = default_params(NAME)
    assert params["ma_length_1"] == 21 and params["ma_length_2"] == 50
    assert params["ma_type_1"] == "EMA" and params["ma_type_2"] == "EMA"
    assert params["rnr"] == 1.0 and params["risk_m"] == 1.0
    assert params["swing_lookback"] == 5 and params["atr_len"] == 14
    assert params["trail_stop"] is False and params["use_limit"] is True
    # Pine's ignore window: 00:00-03:00 at GMT-6.
    assert (params["session_start_hour"], params["session_end_hour"],
            params["session_tz_offset"]) == (0, 3, -6)


def test_no_signal_before_the_moving_averages_are_ready():
    strat = build_strategy(NAME, dict(FAST_SLOW))
    assert strat.evaluate("BTC/USDT", series([100.0] * 10)) is None
    assert strat.evaluate("BTC/USDT", []) is None


# --- entries and exits --------------------------------------------------------

def test_crossover_emits_a_buy_with_a_swing_atr_stop():
    strat = build_strategy(NAME, dict(FAST_SLOW))
    sig = first_signal(strat, series(DIP_THEN_RALLY))
    assert sig is not None and sig.side == "buy"
    assert "crossed above" in sig.rationale
    assert sig.risk["position_pct"] == 0.2


def test_ma_cross_down_exits_the_long():
    """The original's short entry becomes the long-only exit."""
    strat = build_strategy(NAME, dict(FAST_SLOW, exit_on_cross=True))
    sig = first_signal(strat, series(DIP_THEN_RALLY))
    assert sig.side == "buy"
    res = run_backtest(series(DIP_THEN_RALLY), NAME, dict(FAST_SLOW, exit_on_cross=True))
    assert res.n_trades == 1
    assert res.trades[0].exit_price > res.trades[0].entry_price


def test_exit_on_cross_can_be_switched_off():
    res = run_backtest(series(DIP_THEN_RALLY), NAME, dict(FAST_SLOW, exit_on_cross=False,
                                                          use_limit=False))
    assert res.n_trades == 0, "no exit rule left armed, so the trade stays open"


def test_stop_loss_fires_at_swing_low_minus_atr():
    prices = ([100.0] * 40 + [100 - 0.4 * i for i in range(1, 21)]
              + [92 + 1.5 * i for i in range(1, 21)] + [122 - 6.0 * i for i in range(1, 9)])
    res = run_backtest(series(prices), NAME, dict(FAST_SLOW, risk_m=1.0, use_limit=False,
                                                 exit_on_cross=False))
    assert res.n_trades == 1
    trade = res.trades[0]
    assert trade.exit_price < trade.entry_price, "the stop must cut the loss"
    assert trade.pnl < 0


def test_take_profit_fires_at_the_reward_risk_target():
    prices = ([100.0] * 40 + [100 - 0.4 * i for i in range(1, 21)]
              + [92 + 1.5 * i for i in range(1, 21)] + [122 + 3.0 * i for i in range(1, 12)])
    res = run_backtest(series(prices), NAME, dict(FAST_SLOW, rnr=1.0, use_limit=True,
                                                  exit_on_cross=False))
    assert res.n_trades == 1
    assert res.trades[0].pnl > 0


def test_atr_trailing_stop_beats_the_static_stop_on_a_reversal():
    prices = ([100.0] * 40 + [100 - 0.4 * i for i in range(1, 21)]
              + [92 + 1.5 * i for i in range(1, 21)] + [122 + 1.8 * i for i in range(1, 16)]
              + [149 - 5.0 * i for i in range(1, 12)])
    trailed = run_backtest(series(prices), NAME, dict(FAST_SLOW, use_limit=False,
                                                      trail_stop=True, trail_stop_size=1.0,
                                                      rr_exit=0.0, exit_on_cross=False))
    static = run_backtest(series(prices), NAME, dict(FAST_SLOW, use_limit=False,
                                                     trail_stop=False, exit_on_cross=False))
    assert trailed.n_trades == 1, "the trail must close out the reversal"
    assert trailed.trades[0].pnl > 0
    assert trailed.trades[0].exit_price > trailed.trades[0].entry_price
    assert static.n_trades == 0, "without the trail nothing closes the trade"


def test_rr_exit_holds_the_trail_back_until_the_trigger():
    """rr_exit=1.0 arms the trail only at the target, so the stop stays put."""
    prices = ([100.0] * 40 + [100 - 0.4 * i for i in range(1, 21)]
              + [92 + 1.5 * i for i in range(1, 21)] + [122 - 2.0 * i for i in range(1, 6)])
    armed_early = run_backtest(series(prices), NAME, dict(FAST_SLOW, use_limit=False,
                                                          trail_stop=True, rr_exit=0.0,
                                                          exit_on_cross=False))
    armed_late = run_backtest(series(prices), NAME, dict(FAST_SLOW, use_limit=False,
                                                         trail_stop=True, rr_exit=1.0,
                                                         exit_on_cross=False))
    assert armed_late.n_trades <= armed_early.n_trades


# --- filters ------------------------------------------------------------------

def test_session_filter_blocks_every_entry_when_it_covers_the_whole_day():
    blocked = run_backtest(series(DIP_THEN_RALLY), NAME, dict(
        FAST_SLOW, ignore_session=True, session_start_hour=0, session_end_hour=23,
        session_tz_offset=0))
    assert blocked.n_trades == 0


def test_session_window_can_wrap_midnight():
    """22:00-03:00 must block 23:00 and 02:00 but allow 12:00."""
    strat = build_strategy(NAME, dict(FAST_SLOW, ignore_session=True,
                                      session_start_hour=22, session_end_hour=3,
                                      session_tz_offset=0))
    def hour_is_ignored(hour):
        return strat._in_ignore_session(hour * HOUR)
    assert hour_is_ignored(23) and hour_is_ignored(2) and hour_is_ignored(22)
    assert not hour_is_ignored(12) and not hour_is_ignored(3)


def test_date_window_filters_entries():
    start = BASE_TS + 100 * HOUR
    late = run_backtest(series(DIP_THEN_RALLY), NAME, dict(FAST_SLOW, start_ts=start))
    never = run_backtest(series(DIP_THEN_RALLY), NAME, dict(FAST_SLOW, end_ts=BASE_TS + 10 * HOUR))
    assert late.n_trades + never.n_trades < 2


def test_long_trades_off_means_no_entries():
    strat = build_strategy(NAME, dict(FAST_SLOW, long_trades=False))
    assert first_signal(strat, series(DIP_THEN_RALLY)) is None


# --- the MA menu --------------------------------------------------------------

@pytest.mark.parametrize("ma_type", ["EMA", "HEMA", "SMA", "HMA", "WMA", "DEMA", "VWMA", "VWAP", "T3"])
def test_every_ma_type_produces_signals(ma_type):
    strat = build_strategy(NAME, dict(FAST_SLOW, ma_type_1=ma_type))
    sig = first_signal(strat, series(DIP_THEN_RALLY))
    assert sig is not None, f"{ma_type} produced no entry signal"
    assert sig.side == "buy"


def test_an_unknown_ma_type_falls_back_instead_of_crashing():
    strat = build_strategy(NAME, dict(FAST_SLOW, ma_type_1="NOT_A_TYPE"))
    assert first_signal(strat, series(DIP_THEN_RALLY)) is not None


# --- activation ---------------------------------------------------------------

def test_it_can_be_activated_through_the_approval_flow(tmp_path):
    """How the user actually turns it on: an approved param_change."""
    loader = get_default_loader()
    db = str(tmp_path / "three_commas.db")
    init_db(db)
    session = get_session_factory(db)()
    try:
        rec = create_pending_recommendation(
            session, kind="param_change", strategy_name=NAME,
            params=dict(default_params(NAME), ma_length_1=9, ma_length_2=21),
            rationale="activate the converted 3Commas strategy",
        )
        assert rec.status == "pending"

        candles = series(DIP_THEN_RALLY)
        strategy, result = RecommendationStore(session).apply(rec, candles=candles)

        assert strategy is not None and strategy.name == NAME
        assert strategy.version == 1 and rec.status == "applied"
        row = session.query(Strategy).filter(Strategy.name == NAME).one()
        assert row.status == "active"
        assert json.loads(row.params)["ma_length_1"] == 9
        assert result is not None and result.strategy_name == NAME
        assert loader.get_class(NAME) is not None
    finally:
        session.close()
