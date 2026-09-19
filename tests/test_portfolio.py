"""Portfolio view: exposure math, Pearson correlation, Telegram formatting.

The correlation expectation is hand-computed (see ``HAND_PEARSON``) so the test
does not re-implement the function it is checking. No network, no numpy.
"""
import pytest

from algotrading.analytics.portfolio import (
    MIN_CORRELATION_BARS,
    SKIP_INSUFFICIENT,
    SKIP_NO_CANDLES,
    SKIP_NO_VARIANCE,
    _pearson,
    correlation_matrix,
    portfolio_snapshot,
)
from algotrading.db import get_session_factory, init_db
from algotrading.db.models import Candle, Position
from algotrading.telegram.ui import format_portfolio

HOUR_MS = 3_600_000
BASE_TS = 1_700_000_000_000

# x = 1..10, y = 2,1,4,3,6,5,8,7,10,9
#   mean(x) = mean(y) = 5.5
#   cov = 77.5, var(x) = var(y) = 82.5  ->  r = 77.5 / 82.5
HAND_PEARSON = 77.5 / 82.5


def _settings(symbols=("BTC/USDT", "ETH/USDT"), bars=200, balance=100_000.0):
    market = type("M", (), {"symbols": list(symbols), "intervals": ["1h", "1m"],
                            "eval_interval": "1h"})()
    analytics = type("A", (), {"portfolio_bars": bars})()
    risk = type("R", (), {"paper_initial_balance": balance})()
    return type("S", (), {"market": market, "analytics": analytics, "risk": risk,
                          "mode": "paper"})()


@pytest.fixture()
def session(tmp_path):
    db = str(tmp_path / "portfolio.db")
    init_db(db)
    sess = get_session_factory(db)()
    yield sess
    sess.close()


def _closes(session, symbol, closes, start=BASE_TS):
    for i, close in enumerate(closes):
        session.add(Candle(symbol=symbol, interval="1h", ts=start + i * HOUR_MS,
                           open=close, high=close, low=close, close=close, volume=1.0))
    session.commit()


def test_pearson_matches_hand_computation():
    x = list(range(1, 11))
    y = [2, 1, 4, 3, 6, 5, 8, 7, 10, 9]
    assert _pearson(x, y) == pytest.approx(HAND_PEARSON)
    assert _pearson(x, x) == pytest.approx(1.0)
    assert _pearson(x, list(reversed(x))) == pytest.approx(-1.0)
    # Undefined, not zero: a flat series has no correlation.
    assert _pearson(x, [5.0] * 10) is None


def test_correlation_matrix_uses_the_hand_computed_value(session):
    _closes(session, "BTC/USDT", list(range(1, 11)))
    _closes(session, "ETH/USDT", [2, 1, 4, 3, 6, 5, 8, 7, 10, 9])

    matrix = correlation_matrix(session, ["BTC/USDT", "ETH/USDT"], "1h", 200)

    assert matrix["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert matrix["matrix"]["BTC/USDT"]["ETH/USDT"] == pytest.approx(HAND_PEARSON, abs=1e-4)
    assert matrix["matrix"]["BTC/USDT"]["BTC/USDT"] == 1.0
    # Symmetric.
    assert matrix["matrix"]["ETH/USDT"]["BTC/USDT"] == matrix["matrix"]["BTC/USDT"]["ETH/USDT"]
    assert matrix["skipped"] == {}


def test_short_history_symbol_is_skipped_and_reported(session):
    _closes(session, "BTC/USDT", list(range(1, 11)))
    _closes(session, "ETH/USDT", [1.0] * (MIN_CORRELATION_BARS - 1))

    matrix = correlation_matrix(session, ["BTC/USDT", "ETH/USDT"], "1h", 200)

    assert matrix["symbols"] == ["BTC/USDT"]
    assert matrix["skipped"]["ETH/USDT"] == SKIP_INSUFFICIENT


def test_zero_variance_symbol_is_skipped_and_reported(session):
    _closes(session, "BTC/USDT", list(range(1, 11)))
    _closes(session, "ETH/USDT", [100.0] * 12)   # flat

    matrix = correlation_matrix(session, ["BTC/USDT", "ETH/USDT"], "1h", 200)

    assert matrix["symbols"] == ["BTC/USDT"]
    assert matrix["skipped"]["ETH/USDT"] == SKIP_NO_VARIANCE


def test_symbol_without_candles_is_reported(session):
    _closes(session, "BTC/USDT", list(range(1, 11)))
    matrix = correlation_matrix(session, ["BTC/USDT", "SOL/USDT"], "1h", 200)
    assert matrix["skipped"]["SOL/USDT"] == SKIP_NO_CANDLES


def test_matrix_ignores_bars_beyond_the_configured_window(session):
    # 40 bars stored, but only the newest 10 are used: the co-movement lives in
    # the first 30 of each series, so a 10-bar window must NOT see it.
    btc = list(range(1, 31)) + [50.0] * 10
    eth = [2 * v for v in range(1, 31)] + [50.0] * 10
    _closes(session, "BTC/USDT", btc)
    _closes(session, "ETH/USDT", eth)

    matrix = correlation_matrix(session, ["BTC/USDT", "ETH/USDT"], "1h", 10)

    # Both are flat across the last 10 shared bars.
    assert matrix["symbols"] == []
    assert matrix["skipped"]["BTC/USDT"] == SKIP_NO_VARIANCE


# --- exposure -----------------------------------------------------------------

def test_portfolio_snapshot_exposure_math(session):
    _closes(session, "BTC/USDT", [50_000.0, 51_000.0])
    _closes(session, "ETH/USDT", [2_000.0, 2_100.0])
    session.add(Position(symbol="BTC/USDT", qty=0.5, avg_price=50_000.0))
    session.add(Position(symbol="ETH/USDT", qty=2.0, avg_price=2_000.0))
    session.commit()

    snapshot = portfolio_snapshot(session, _settings())

    assert snapshot["open_positions"] == 2
    rows = {row["symbol"]: row for row in snapshot["positions"]}
    # 0.5 * 51_000 = 25_500 on a 100_000 balance => 25.5%
    assert rows["BTC/USDT"]["last_price"] == pytest.approx(51_000.0)
    assert rows["BTC/USDT"]["notional"] == pytest.approx(25_500.0)
    assert rows["BTC/USDT"]["pct_of_balance"] == pytest.approx(25.5)
    # 2.0 * 2_100 = 4_200 => 4.2%
    assert rows["ETH/USDT"]["notional"] == pytest.approx(4_200.0)
    assert rows["ETH/USDT"]["pct_of_balance"] == pytest.approx(4.2)
    assert snapshot["total_exposure"] == pytest.approx(29_700.0)
    assert snapshot["total_pct_of_balance"] == pytest.approx(29.7)
    assert snapshot["balance"] == pytest.approx(100_000.0)


def test_portfolio_snapshot_without_positions_or_prices(session):
    _closes(session, "BTC/USDT", [50_000.0])
    session.add(Position(symbol="SOL/USDT", qty=3.0, avg_price=20.0))  # no candles
    session.commit()

    snapshot = portfolio_snapshot(session, _settings())

    row = snapshot["positions"][0]
    assert row["last_price"] is None
    assert row["notional"] == 0.0           # no mark => no notional claimed
    assert snapshot["total_exposure"] == 0.0
    assert snapshot["correlation"]["skipped"]["BTC/USDT"] == SKIP_INSUFFICIENT


def test_closed_positions_are_not_counted(session):
    _closes(session, "BTC/USDT", [50_000.0, 51_000.0])
    session.add(Position(symbol="BTC/USDT", qty=0.0, avg_price=50_000.0))
    session.commit()

    snapshot = portfolio_snapshot(session, _settings())
    assert snapshot["open_positions"] == 0
    assert snapshot["total_exposure"] == 0.0


# --- formatting ---------------------------------------------------------------

def test_format_portfolio_reports_exposure_and_labels_the_caveat(session):
    # Perfectly correlated series (ETH = 2 x BTC) with a known last close.
    btc = [50_000.0 + 100 * (i + 1) for i in range(10)]
    _closes(session, "BTC/USDT", btc)
    _closes(session, "ETH/USDT", [2 * value for value in btc])
    session.add(Position(symbol="BTC/USDT", qty=0.5, avg_price=50_000.0))
    session.commit()

    text = format_portfolio(portfolio_snapshot(session, _settings()))

    assert "<b>Portfolio</b>" in text
    assert "BTC/USDT" in text
    assert "25500.00" in text                    # 0.5 * 51_000 last close
    assert "25.5%" in text
    assert "not a trading signal" in text        # correlation is labelled context
    assert "lags" in text
    assert "1.00" in text                        # ETH = 2 x BTC exactly


def test_format_portfolio_renders_the_hand_computed_correlation(session):
    _closes(session, "BTC/USDT", list(range(1, 11)))
    _closes(session, "ETH/USDT", [2, 1, 4, 3, 6, 5, 8, 7, 10, 9])
    session.commit()

    text = format_portfolio(portfolio_snapshot(session, _settings()))

    assert "No open positions." in text
    assert "0.94" in text                        # the hand-computed r, rendered


def test_format_portfolio_escapes_and_truncates():
    snapshot = {
        "positions": [
            {"symbol": f"<b>FAKE{i}</b>/USDT", "qty": 1.0, "avg_price": 100.0,
             "last_price": 101.0, "notional": 101.0, "pct_of_balance": 1.0}
            for i in range(80)
        ],
        "open_positions": 80,
        "total_exposure": 8_080.0,
        "total_pct_of_balance": 80.8,
        "balance": 10_000.0,
        "correlation": {"symbols": [], "matrix": {}, "skipped": {}, "bars": 200,
                        "interval": "1h"},
    }
    text = format_portfolio(snapshot)

    assert "<b>FAKE0</b>" not in text           # dynamic values are escaped
    assert "&lt;b&gt;FAKE0&lt;/b&gt;" in text
    assert len(text) <= 4096
    assert "truncated" in text


def test_format_portfolio_handles_an_empty_view():
    text = format_portfolio({
        "positions": [], "open_positions": 0, "total_exposure": 0.0,
        "total_pct_of_balance": 0.0, "balance": 10_000.0,
        "correlation": {"symbols": [], "matrix": {}, "skipped": {"SOL/USDT": SKIP_NO_CANDLES},
                        "bars": 200, "interval": "1h"},
    })
    assert "No open positions." in text
    assert "Not enough stored candles" in text
    assert "SOL/USDT" in text and SKIP_NO_CANDLES in text
