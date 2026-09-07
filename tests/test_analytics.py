"""Analytics metric math + summary persistence."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from algotrading.analytics.metrics import (
    SLIPPAGE,
    TREND_REVERSAL,
    compute_metrics,
)
from algotrading.analytics.service import AnalyticsService
from algotrading.db import init_db, get_session_factory
from algotrading.db.models import Trade


def _trade(pnl, entry=100.0, exit_=None, fees=0.0, opened=None, closed=None,
           loss_reasons="[]", symbol="BTC/USDT"):
    if exit_ is None:
        exit_ = entry + pnl
    return Trade(
        symbol=symbol,
        entry_qty=1.0,
        entry_avg_price=entry,
        exit_avg_price=exit_,
        realized_pnl=pnl,
        fees=fees,
        opened_at=opened or datetime.now(timezone.utc) - timedelta(hours=1),
        closed_at=closed or datetime.now(timezone.utc),
        loss_reasons=loss_reasons,
        strategy_id=1,
    )


def test_mixed_trades_metrics():
    # 2 wins, 1 loss
    trades = [_trade(10.0), _trade(5.0), _trade(-5.0)]
    m = compute_metrics(trades, period="30m", strategy_id=1)
    assert m.n_trades == 3
    assert m.n_wins == 2
    assert m.n_losses == 1
    assert m.win_rate == pytest.approx(2 / 3)
    assert m.total_pnl == pytest.approx(10.0)
    assert m.expectancy == pytest.approx(10.0 / 3)
    assert m.avg_win == pytest.approx(7.5)
    assert m.avg_loss == pytest.approx(5.0)
    # gross win 15 / gross loss 5
    assert m.profit_factor == pytest.approx(3.0)


def test_empty_trades_no_divide_by_zero():
    m = compute_metrics([], period="30m")
    assert m.n_trades == 0
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0
    assert m.expectancy == 0.0


def test_loss_tagging_trend_reversal():
    t = _trade(-5.0, entry=100.0, exit_=90.0)  # 10% adverse move
    m = compute_metrics([t])
    assert m.loss_tags.get(TREND_REVERSAL) == 1


def test_loss_tagging_slippage():
    t = _trade(-0.5, entry=100.0, exit_=99.5, fees=1.0)
    m = compute_metrics([t])
    assert m.loss_tags.get(SLIPPAGE) == 1


def test_no_tags_on_win():
    t = _trade(10.0)
    m = compute_metrics([t])
    assert m.loss_tags == {}


@pytest.fixture()
def session(tmp_path):
    from algotrading.db.models import Strategy

    db = str(tmp_path / "an.db")
    init_db(db)
    sess = get_session_factory(db)()
    sess.add(Strategy(name="ema_crossover", version=1, status="active"))
    sess.commit()
    return sess


def test_service_persists_summary(session):
    now = datetime.now(timezone.utc)
    session.add(_trade(10.0, closed=now - timedelta(minutes=5)))
    session.add(_trade(-4.0, closed=now - timedelta(minutes=2)))
    session.commit()

    svc = AnalyticsService(session, period="30m", lookback_minutes=30)
    summary = svc.run(strategy_id=1)
    assert summary.period == "30m"
    m = json.loads(summary.metrics_json)
    assert m["n_trades"] == 2
    assert m["win_rate"] == pytest.approx(0.5)
    assert m["total_pnl"] == pytest.approx(6.0)


def test_service_filters_old_trades(session):
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    session.add(_trade(10.0, closed=old))
    session.commit()

    svc = AnalyticsService(session, period="30m", lookback_minutes=30)
    summary = svc.run()
    m = json.loads(summary.metrics_json)
    assert m["n_trades"] == 0


def test_daily_summary(session):
    now = datetime.now(timezone.utc)
    # Clamp the earlier trade to just after today's UTC midnight so it always
    # falls inside the daily window regardless of the current hour of day.
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    session.add(_trade(3.0, closed=midnight + timedelta(hours=1)))
    session.add(_trade(-1.0, closed=now - timedelta(minutes=1)))
    session.commit()

    svc = AnalyticsService(session)
    summary = svc.run_daily()
    m = json.loads(summary.metrics_json)
    assert m["period"] == "daily"
    assert m["n_trades"] == 2
