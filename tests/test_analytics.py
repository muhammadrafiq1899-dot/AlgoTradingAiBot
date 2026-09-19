"""Analytics metric math + summary persistence."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from algotrading.analytics.metrics import (
    SLIPPAGE,
    TREND_REVERSAL,
    compute_metrics,
)
from algotrading.analytics.service import AnalyticsService
from algotrading.db import init_db, get_session_factory
from algotrading.db.models import Strategy, Trade


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


# --- P3: the daily job also maintains the advisory decision log ---------------

def _bot_context(db: str, *, decision_log_enabled: bool = True):
    from algotrading.config import AIConfig, MarketConfig, RiskConfig, Settings
    from algotrading.scheduler.jobs import BotContext
    from algotrading.supervisor.health import HealthMonitor

    settings = Settings(
        mode="paper",
        market=MarketConfig(symbols=["BTC/USDT"], intervals=["1h"], max_staleness_seconds=300),
        risk=RiskConfig(paper_initial_balance=100_000.0),
        ai=AIConfig(enabled=False, decision_log_enabled=decision_log_enabled),
    )
    return BotContext(
        settings=settings,
        session_factory=get_session_factory(db),
        provider=None,
        gateway=None,
        health=HealthMonitor(str(Path(db).with_name("heartbeat"))),
    )


def test_analytics_daily_maintains_the_decision_log(tmp_path):
    """The daily job observes approvals another module owns and scores them."""
    from algotrading.ai.decision_log import PENDING, WIN
    from algotrading.db.models import AIDecisionLog, AIRecommendation, Candle as CandleRow
    from algotrading.scheduler.jobs import analytics_daily

    db = str(tmp_path / "daily.db")
    init_db(db)
    ctx = _bot_context(db)

    applied_at = datetime.now(timezone.utc) - timedelta(days=10)
    sess = ctx.session_factory()
    sess.add(Strategy(name="ema_crossover", version=1, status="active"))
    rec = AIRecommendation(kind="param_change", strategy_name="ema_crossover",
                           content_json="{}", status="applied", reviewed_at=applied_at)
    sess.add(rec)
    sess.add(_trade(10.0, closed=applied_at + timedelta(days=3)))
    start_ms = int((applied_at - timedelta(hours=1)).timestamp() * 1000)
    for i, close in enumerate([100.0, 100.0, 99.0]):
        sess.add(CandleRow(symbol="BTC/USDT", interval="1h", ts=start_ms + i * 3_600_000,
                           open=close, high=close, low=close, close=close, volume=1.0))
    sess.commit()
    sess.close()
    assert rec.id is not None

    analytics_daily(ctx)

    sess = ctx.session_factory()
    try:
        entry = sess.query(AIDecisionLog).one()
        assert entry.recommendation_id == rec.id
        assert entry.outcome == WIN
        assert entry.pnl_pct == pytest.approx(10.0)
        assert entry.benchmark_pct == pytest.approx(-1.0)
        assert entry.reflection
    finally:
        sess.close()


def test_decision_log_tick_leaves_an_unelapsed_horizon_pending(tmp_path):
    from algotrading.ai.decision_log import PENDING
    from algotrading.db.models import AIDecisionLog, AIRecommendation
    from algotrading.scheduler.jobs import decision_log_tick

    db = str(tmp_path / "pending.db")
    init_db(db)
    ctx = _bot_context(db)

    sess = ctx.session_factory()
    sess.add(Strategy(name="ema_crossover", version=1, status="active"))
    sess.add(AIRecommendation(kind="param_change", strategy_name="ema_crossover",
                              content_json="{}", status="applied",
                              reviewed_at=datetime.now(timezone.utc) - timedelta(days=1)))
    sess.commit()
    sess.close()

    decision_log_tick(ctx)

    sess = ctx.session_factory()
    try:
        entry = sess.query(AIDecisionLog).one()
        assert entry.outcome == PENDING          # recorded, not yet judged
        assert entry.evaluated_at is None
    finally:
        sess.close()


def test_decision_log_tick_disabled_does_nothing(tmp_path):
    from algotrading.db.models import AIDecisionLog, AIRecommendation
    from algotrading.scheduler.jobs import decision_log_tick

    db = str(tmp_path / "disabled.db")
    init_db(db)
    ctx = _bot_context(db, decision_log_enabled=False)

    sess = ctx.session_factory()
    sess.add(Strategy(name="ema_crossover", version=1, status="active"))
    sess.add(AIRecommendation(kind="param_change", strategy_name="ema_crossover",
                              content_json="{}", status="applied",
                              reviewed_at=datetime.now(timezone.utc) - timedelta(days=30)))
    sess.commit()
    sess.close()

    decision_log_tick(ctx)

    sess = ctx.session_factory()
    try:
        assert sess.query(AIDecisionLog).count() == 0
    finally:
        sess.close()
