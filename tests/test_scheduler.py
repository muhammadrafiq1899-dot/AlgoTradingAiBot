"""M7: scheduler job wiring — market tick end-to-end, stale-data freeze,
reconcile, heartbeat, and the AI-review guard against stacked proposals.

Jobs run synchronously against a tmp database with a fake candle provider, so
nothing here touches the network or the exchange.
"""
import json
import time

import pytest
from sqlalchemy import select

from algotrading.config import AIConfig, ApiConfig, MarketConfig, RiskConfig, ScheduleConfig, Settings
from algotrading.db import init_db, get_session_factory
from algotrading.db.models import AIRecommendation, Position, Strategy, TradeIntent
from algotrading.execution import PaperGateway
from algotrading.market.base import Candle
from algotrading.scheduler.jobs import (
    BotContext,
    ai_review,
    analytics_tick,
    build_scheduler,
    heartbeat_tick,
    market_tick,
    reconcile_tick,
)
from algotrading.supervisor.health import HealthMonitor

HOUR_MS = 3_600_000


def _settings() -> Settings:
    return Settings(
        mode="paper",
        market=MarketConfig(
            symbols=["BTC/USDT"],
            intervals=["1m", "1h"],
            max_staleness_seconds=300,
        ),
        risk=RiskConfig(),
        schedule=ScheduleConfig(),
        ai=AIConfig(enabled=False),
        api=ApiConfig(enabled=False),
    )


@pytest.fixture()
def ctx(tmp_path):
    db = str(tmp_path / "sched.db")
    init_db(db)
    factory = get_session_factory(db)
    session = factory()
    session.add(
        Strategy(name="ema_crossover", version=1, status="active",
                 params=json.dumps({"fast_period": 12, "slow_period": 26, "position_pct": 0.2}))
    )
    session.commit()
    session.close()
    return BotContext(
        settings=_settings(),
        provider=None,  # replaced per test
        gateway=None,   # replaced per test
        session_factory=factory,
        health=HealthMonitor(str(tmp_path / "heartbeat")),
    )


class FakeProvider:
    """Returns a fixed candle series (and a ticker price) for every symbol/interval."""

    def __init__(self, candles):
        self._candles = candles
        self._price = candles[-1].close if candles else 0.0

    def fetch_klines(self, symbol, interval, since_ms=None):
        return list(self._candles)

    def fetch_ticker_price(self, symbol):
        return self._price


def _candles(prices, ts_start=None):
    """Candles at 1h alignment ending at `ts_start` (default: aligned now)."""
    if ts_start is None:
        now_ms = int(time.time() * 1000)
        ts_start = now_ms - (now_ms % HOUR_MS)
    return [
        Candle(symbol="BTC/USDT", interval="1h", ts=ts_start - (len(prices) - 1 - i) * HOUR_MS,
               open=p, high=p, low=p, close=p, volume=1.0)
        for i, p in enumerate(prices)
    ]


def _with_gateway(ctx, candles):
    provider = FakeProvider(candles)
    ctx.provider = provider
    ctx.gateway = PaperGateway(provider, slippage_pct=0.05)
    return ctx


def test_build_scheduler_registers_all_jobs(ctx):
    sched = build_scheduler(ctx)
    ids = {j.id for j in sched.get_jobs()}
    assert ids == {"market_tick", "analytics", "analytics_daily", "ai_review", "reconcile", "heartbeat"}
    for job in sched.get_jobs():
        assert job.max_instances == 1
        assert job.coalesce


def test_market_tick_stores_candles_and_fills_signal(ctx):
    # Flat base then a single jump: fast EMA crosses above slow on the LAST
    # candle -> deterministic buy signal at the end of the series.
    _with_gateway(ctx, _candles([100.0] * 35 + [200.0]))

    market_tick(ctx)

    session = ctx.session_factory()
    try:
        # Candles stored for both configured intervals.
        from algotrading.db.models import Candle as CandleRow

        assert session.execute(select(CandleRow)).scalars().all()

        intent = session.execute(select(TradeIntent)).scalars().one()
        assert intent.status == "filled"
        assert intent.side == "buy"
        assert intent.qty > 0

        pos = session.execute(select(Position).where(Position.symbol == "BTC/USDT")).scalar_one()
        assert pos.qty == pytest.approx(intent.qty)
    finally:
        session.close()


def test_market_tick_freezes_on_stale_data(ctx):
    # Candles end two hours ago — well beyond interval (1h) + max_staleness
    # (5m) = 65m, so market_tick must freeze and create no entries. Using a
    # fixed 2h offset (not "start of previous hour") keeps the test robust
    # regardless of where in the hour it runs.
    stale_end = int(time.time() * 1000) - 2 * HOUR_MS
    _with_gateway(ctx, _candles([100.0] * 35 + [200.0], ts_start=stale_end))

    market_tick(ctx)

    session = ctx.session_factory()
    try:
        assert session.execute(select(TradeIntent)).scalars().all() == []
        assert session.execute(select(Position)).scalars().all() == []
    finally:
        session.close()


def test_market_tick_empty_feed_is_noop(ctx):
    _with_gateway(ctx, [])
    market_tick(ctx)  # must not raise
    session = ctx.session_factory()
    try:
        assert session.execute(select(TradeIntent)).scalars().all() == []
    finally:
        session.close()


def test_ai_review_skips_when_pending_exists(ctx):
    ctx.settings.ai.enabled = True
    session = ctx.session_factory()
    session.add(AIRecommendation(kind="hypothesis", strategy_name="ema_crossover",
                                 content_json="{}", status="pending", rationale="open"))
    session.commit()
    session.close()

    ai_review(ctx)  # must return early without calling the LLM

    session = ctx.session_factory()
    try:
        recs = session.execute(select(AIRecommendation)).scalars().all()
        assert len(recs) == 1
    finally:
        session.close()


def test_ai_review_disabled_is_noop(ctx):
    ctx.settings.ai.enabled = False
    ai_review(ctx)  # no client, no network


def test_reconcile_tick_reports_ok_with_paper_gateway(ctx):
    _with_gateway(ctx, _candles([100.0] * 36))
    reconcile_tick(ctx)  # paper gateway reports no open orders -> no drift


def test_heartbeat_tick_writes_file(ctx):
    heartbeat_tick(ctx)
    assert ctx.health.last_beat() > 0


def test_analytics_tick_runs_without_trades(ctx):
    _with_gateway(ctx, _candles([100.0] * 36))
    analytics_tick(ctx)  # empty ledger -> zero-trade summary, no crash