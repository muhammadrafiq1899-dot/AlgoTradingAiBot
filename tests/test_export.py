"""P2-10: CSV/JSON exporters (trades, equity curve, summary).

The exports are the only way to get data out of the bot without copying the
SQLite file, so the tests pin the *content* (not just "a file appeared"): column
order, ISO timestamps, the equity math, and the empty-database case a fresh
install hits.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone

import pytest

from algotrading.db import get_session_factory, init_db
from algotrading.db.models import (
    AIRecommendation,
    AnalyticsSummary,
    Position,
    Strategy,
    Trade,
    TradeIntent,
)
from algotrading.store.export import (
    EQUITY_COLUMNS,
    TRADE_COLUMNS,
    export_equity_csv,
    export_summary_json,
    export_trades_csv,
    export_trades_json,
    equity_rows,
    summary_dict,
    trade_rows,
)

UTC = timezone.utc
T0 = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)


@pytest.fixture()
def session(tmp_path):
    db = str(tmp_path / "export.db")
    init_db(db)
    factory = get_session_factory(db)
    with factory() as s:
        yield s


def _add_trade(session, *, symbol="BTC/USDT", pnl=10.0, closed_at=T0, open_trade=False):
    session.add(
        Trade(
            symbol=symbol,
            entry_qty=1.0,
            entry_avg_price=100.0,
            exit_avg_price=None if open_trade else 110.0,
            realized_pnl=None if open_trade else pnl,
            fees=0.11,
            opened_at=T0 - timedelta(hours=1),
            closed_at=None if open_trade else closed_at,
            loss_reasons=json.dumps([] if pnl >= 0 else ["stop"]),
        )
    )
    session.commit()


def _rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


# ---------------------------------------------------------------------------
# trades
# ---------------------------------------------------------------------------

def test_trades_csv_content(session, tmp_path):
    _add_trade(session, pnl=10.0)
    _add_trade(session, symbol="ETH/USDT", pnl=-4.0, closed_at=T0 + timedelta(days=1))
    path = tmp_path / "trades.csv"

    assert export_trades_csv(session, path) == 2

    rows = _rows(path)
    assert rows[0] == list(TRADE_COLUMNS)
    assert len(rows) == 3
    first = dict(zip(rows[0], rows[1]))
    assert first["symbol"] == "BTC/USDT"
    assert first["realized_pnl"] == "10.0"
    assert first["opened_at"] == (T0 - timedelta(hours=1)).isoformat()
    assert first["closed_at"] == T0.isoformat()
    assert first["loss_reasons"] == "[]"
    # Oldest first: an equity curve and a diff of two exports line up.
    assert rows[1][1] == "BTC/USDT" and rows[2][1] == "ETH/USDT"


def test_trades_json_content(session, tmp_path):
    _add_trade(session, pnl=-4.0)
    path = tmp_path / "trades.json"

    assert export_trades_json(session, path) == 1

    payload = json.loads(path.read_text())
    assert isinstance(payload, list) and len(payload) == 1
    trade = payload[0]
    assert trade["symbol"] == "BTC/USDT"
    assert trade["realized_pnl"] == -4.0
    assert trade["closed_at"] == T0.isoformat()
    assert trade["loss_reasons"] == ["stop"]


def test_open_trade_sorts_last_and_has_no_close(session):
    _add_trade(session, symbol="OPEN/USDT", pnl=0.0, open_trade=True)
    _add_trade(session, symbol="CLOSED/USDT", pnl=5.0)
    rows = trade_rows(session)
    assert [r["symbol"] for r in rows] == ["CLOSED/USDT", "OPEN/USDT"]
    assert rows[1]["closed_at"] is None


# ---------------------------------------------------------------------------
# equity curve
# ---------------------------------------------------------------------------

def test_equity_curve_from_closed_trades(session, tmp_path):
    _add_trade(session, pnl=10.0, closed_at=T0)
    _add_trade(session, pnl=-4.0, closed_at=T0 + timedelta(hours=2))
    path = tmp_path / "equity.csv"

    assert export_equity_csv(session, path, initial_balance=1000.0) == 2

    rows = _rows(path)
    assert rows[0] == list(EQUITY_COLUMNS)
    assert [r[2] for r in rows[1:]] == ["1", "2"]          # trade_id
    assert [float(r[3]) for r in rows[1:]] == [10.0, -4.0]  # realized_pnl
    assert [float(r[4]) for r in rows[1:]] == [10.0, 6.0]   # cumulative_pnl
    assert [float(r[5]) for r in rows[1:]] == [1010.0, 1006.0]  # equity


def test_equity_curve_skips_unclosed_trades(session):
    _add_trade(session, symbol="OPEN/USDT", open_trade=True)
    _add_trade(session, symbol="CLOSED/USDT", pnl=7.0)
    points = equity_rows(session, initial_balance=100.0)
    assert len(points) == 1
    assert points[0]["equity"] == 107.0


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def test_summary_json_has_counts_metadata_and_analytics(session, tmp_path):
    session.add(
        Strategy(
            name="ema_crossover", version=3, status="active",
            params=json.dumps({"fast_period": 12, "slow_period": 26}),
        )
    )
    session.add(Position(symbol="BTC/USDT", qty=0.5, avg_price=100.0))
    session.add(
        TradeIntent(
            idempotency_key="k1", symbol="BTC/USDT", side="buy", qty=0.5,
            status="filled",
        )
    )
    session.add(
        AnalyticsSummary(
            period="30m", symbol="ALL", metrics_json=json.dumps({"n_trades": 2}),
            ts=T0,
        )
    )
    session.add(
        AnalyticsSummary(
            period="daily", symbol="ALL", metrics_json=json.dumps({"n_trades": 5}),
            ts=T0 - timedelta(days=1),
        )
    )
    session.add(AIRecommendation(kind="param_change", status="pending"))
    session.commit()
    _add_trade(session, pnl=10.0)

    path = tmp_path / "summary.json"
    summary = export_summary_json(session, path, mode="paper", initial_balance=1000.0)

    on_disk = json.loads(path.read_text())
    assert on_disk == summary
    assert summary["mode"] == "paper"
    assert summary["strategy"]["name"] == "ema_crossover"
    assert summary["strategy"]["version"] == 3
    assert summary["strategy"]["params"]["fast_period"] == 12
    assert summary["counts"]["trades"] == 1
    assert summary["counts"]["closed_trades"] == 1
    assert summary["counts"]["open_positions"] == 1
    assert summary["counts"]["trade_intents"] == 1
    assert summary["counts"]["pending_recommendations"] == 1
    assert summary["totals"]["realized_pnl"] == 10.0
    assert summary["totals"]["equity"] == 1010.0
    assert summary["analytics"]["30m"]["metrics"] == {"n_trades": 2}
    assert summary["analytics"]["daily"]["metrics"] == {"n_trades": 5}


def test_empty_database_exports_are_well_formed(session, tmp_path):
    trades_csv = tmp_path / "trades.csv"
    trades_json = tmp_path / "trades.json"
    equity_csv = tmp_path / "equity.csv"
    summary_path = tmp_path / "summary.json"

    assert export_trades_csv(session, trades_csv) == 0
    assert export_trades_json(session, trades_json) == 0
    assert export_equity_csv(session, equity_csv, initial_balance=500.0) == 0
    summary = export_summary_json(session, summary_path, mode="paper")

    assert _rows(trades_csv) == [list(TRADE_COLUMNS)]     # header only
    assert json.loads(trades_json.read_text()) == []
    assert _rows(equity_csv) == [list(EQUITY_COLUMNS)]
    assert summary["strategy"] is None
    assert summary["counts"]["trades"] == 0
    assert summary["counts"]["open_positions"] == 0
    assert summary["analytics"] == {"30m": None, "daily": None}
    assert summary["totals"]["realized_pnl"] == 0.0
    assert summary["totals"]["equity"] == 10000.0        # default initial balance


def test_summary_dict_without_active_strategy(session):
    summary = summary_dict(session, mode="paper")
    assert summary["strategy"] is None
    assert summary["counts"]["positions"] == 0


# ---------------------------------------------------------------------------
# file handling
# ---------------------------------------------------------------------------

def test_exports_are_atomic_and_stay_inside_the_given_path(session, tmp_path):
    _add_trade(session, pnl=1.0)
    out_dir = tmp_path / "nested" / "exports"
    path = out_dir / "trades.csv"

    export_trades_csv(session, path)
    export_trades_csv(session, path)  # rewrite must not leave a second file

    assert [p.name for p in out_dir.iterdir()] == ["trades.csv"]
    assert (tmp_path / "nested" / "exports" / "trades.csv").exists()


def test_export_refuses_when_the_path_is_a_directory(session, tmp_path):
    target = tmp_path / "adir"
    target.mkdir()
    with pytest.raises(OSError):
        export_trades_json(session, target)
