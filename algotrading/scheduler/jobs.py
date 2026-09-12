"""APScheduler job registry: the periodic jobs that keep the bot alive.

Jobs are plain sync functions taking a `BotContext`. They run in the
scheduler's worker thread(s); each job opens its own SQLAlchemy session from
`ctx.session_factory` so concurrent jobs never share a Session.

Job roster (all max_instances=1 + coalesce, so slow ticks never stack):

  - market_tick:    fetch candles -> store -> evaluate active strategy -> execute
  - analytics_tick: 30m snapshot of closed-trade metrics
  - analytics_daily: midnight-UTC daily summary
  - ai_review:      daily AI proposal (PENDING only, never applied)
  - reconcile_tick: local intents vs exchange open orders
  - heartbeat_tick: touch heartbeat file for the external watchdog

Timings come from `ScheduleConfig` in config/settings.yaml.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from algotrading.analytics.service import AnalyticsService
from algotrading.ai.assistant import Assistant
from algotrading.ai.client import AIClient
from algotrading.config import Settings
from algotrading.db.models import AnalyticsSummary, AIRecommendation
from algotrading.execution import ExecutionEngine, RiskManager
from algotrading.execution.base import ExchangeGateway
from algotrading.ledger import Ledger
from algotrading.market.base import Candle, MarketDataProvider
from algotrading.market.candles import CandleStore
from algotrading.store.recommendations import RecommendationStore
from algotrading.strategy.engine import StrategyEngine
from algotrading.supervisor.health import HealthMonitor
from algotrading.supervisor.reconcile import reconcile_and_report

log = logging.getLogger(__name__)

# Candles kept per symbol+interval for strategy evaluation / AI feature windows.
SNAPSHOT_LIMIT = 500


@dataclass
class BotContext:
    """Everything a job needs. Assembled once in main.py; jobs are stateless."""

    settings: Settings
    provider: MarketDataProvider
    gateway: ExchangeGateway
    session_factory: Callable[[], Any]
    health: HealthMonitor
    # Optional callback fired after the AI review job saves a PENDING
    # recommendation (e.g. Telegram push with the approval keyboard).
    # Called from the scheduler thread; the app must make it thread-safe.
    on_recommendation: Callable[[AIRecommendation], None] | None = None
    _eval_interval: str = field(default="1h", init=False)

    def primary_interval(self) -> str:
        """Interval used for strategy evaluation: '1h' if configured, else first."""
        intervals = self.settings.market.intervals
        return "1h" if "1h" in intervals else (intervals[0] if intervals else "1m")


def _open_session(ctx: BotContext):
    """Context-managed session for one job run."""
    session = ctx.session_factory()
    try:
        yield session
    finally:
        session.close()


# --- market tick -------------------------------------------------------------

def market_tick(ctx: BotContext) -> None:
    """Fetch the universe, store candles, evaluate, and execute signals."""
    settings = ctx.settings
    eval_interval = ctx.primary_interval()
    session = ctx.session_factory()
    try:
        store = CandleStore(session)
        fetched = 0
        for symbol in settings.market.symbols:
            for interval in settings.market.intervals:
                try:
                    candles = ctx.provider.fetch_klines(symbol, interval)
                except Exception as exc:  # noqa: BLE001 - one bad symbol must not kill the tick
                    log.warning("market tick: fetch failed %s %s: %s", symbol, interval, exc)
                    continue
                fetched += store.upsert(candles)
        if fetched == 0:
            log.warning("market tick: no candles fetched; nothing to do")
            return

        # Freeze new entries on stale data (only protective exits allowed).
        newest_ts = _newest_ts(ctx, store, eval_interval)
        if newest_ts is None:
            log.warning("market tick: no %s candles stored yet; skipping evaluation", eval_interval)
            return
        staleness_ms = settings.market.max_staleness_seconds * 1000
        interval_ms = 3_600_000 if eval_interval == "1h" else 60_000
        if time.time() * 1000 - newest_ts > interval_ms + staleness_ms:
            log.warning(
                "market tick: data stale (%ds old); freezing new entries",
                int((time.time() * 1000 - newest_ts) / 1000),
            )
            return

        snapshot = _snapshot(ctx, store, eval_interval)
        if not snapshot:
            return

        engine = StrategyEngine(session, ctx.settings)
        candidates = engine.evaluate(snapshot)
        if not candidates:
            return
        execution = ExecutionEngine(
            session, ctx.gateway, RiskManager(settings.risk), Ledger(session)
        )
        for sig in candidates:
            try:
                execution.execute(sig.id)
            except Exception:  # noqa: BLE001 - a failed order must not stop the tick
                log.exception("market tick: execution failed for signal %s", sig.id)

        # Update trailing stops (if enabled) after all fills
        trailing_pct = settings.risk.trailing_stop_pct
        if trailing_pct and trailing_pct > 0:
            try:
                # Get current prices from latest candles
                current_prices = {}
                for symbol in settings.market.symbols:
                    latest = store.latest(symbol, eval_interval)
                    if latest:
                        current_prices[symbol] = latest.close
                if current_prices:
                    execution.update_trailing_stops(current_prices)
            except Exception:  # noqa: BLE001
                log.exception("market tick: trailing stop update failed")
    finally:
        session.close()


def _newest_ts(ctx: BotContext, store: CandleStore, interval: str) -> int | None:
    newest = None
    for symbol in ctx.settings.market.symbols:
        ts = store.latest_ts(symbol, interval)
        if ts is not None:
            newest = ts if newest is None else max(newest, ts)
    return newest


def _snapshot(ctx: BotContext, store: CandleStore, interval: str) -> dict[str, list[Candle]]:
    """Latest candles per symbol as normalized market Candle objects."""
    out: dict[str, list[Candle]] = {}
    for symbol in ctx.settings.market.symbols:
        rows = store.get(symbol, interval, limit=SNAPSHOT_LIMIT)
        if not rows:
            continue
        out[symbol] = [
            Candle(
                symbol=r.symbol,
                interval=r.interval,
                ts=r.ts,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
            )
            for r in rows
        ]
    return out


# --- analytics ---------------------------------------------------------------

def analytics_tick(ctx: BotContext) -> None:
    """Periodic (30m) metrics snapshot from the closed-trade ledger."""
    session = ctx.session_factory()
    try:
        strategy_id = None
        active = StrategyEngine(session, ctx.settings).get_active_strategy()
        if active is not None:
            strategy_id = active.id
        AnalyticsService(session, period="30m",
                         lookback_minutes=ctx.settings.schedule.analytics_minutes).run(strategy_id)
    finally:
        session.close()


def analytics_daily(ctx: BotContext) -> None:
    """Daily summary since UTC midnight."""
    session = ctx.session_factory()
    try:
        AnalyticsService(session).run_daily()
    finally:
        session.close()


# --- AI review ---------------------------------------------------------------

def ai_review(ctx: BotContext) -> None:
    """Daily advisory AI pass: propose ONE change as a PENDING recommendation.

    Skips when the AI is disabled or a PENDING recommendation already exists
    (one open question at a time — the user must approve/reject before the
    next proposal). The assistant never applies anything itself.
    """
    if not ctx.settings.ai.enabled:
        return
    session = ctx.session_factory()
    try:
        store = RecommendationStore(session)
        if store.pending(limit=1):
            log.info("ai_review: PENDING recommendation exists; skipping proposal")
            return

        engine = StrategyEngine(session, ctx.settings)
        active = engine.get_active_strategy()
        if active is None:
            log.warning("ai_review: no active strategy; skipping")
            return

        symbol = (ctx.settings.market.symbols or ["BTC/USDT"])[0]
        interval = ctx.primary_interval()
        candles = _snapshot(ctx, CandleStore(session), interval)[symbol]
        summaries: Sequence[AnalyticsSummary] = (
            session.query(AnalyticsSummary)
            .order_by(AnalyticsSummary.ts.desc())
            .limit(5)
            .all()
        )
        params = json.loads(active.params or "{}")

        assistant = Assistant(
            session,
            client=AIClient(config=ctx.settings.ai),
            strategy_name=active.name,
            params=params,
        )
        rec = assistant.propose(symbol, interval, summaries, candles)
        if rec is not None and ctx.on_recommendation is not None:
            try:
                ctx.on_recommendation(rec)
            except Exception:  # noqa: BLE001 - a failed alert must not crash the job
                log.exception("ai_review: recommendation alert failed")
    finally:
        session.close()


# --- reconcile + heartbeat ---------------------------------------------------

def reconcile_tick(ctx: BotContext) -> None:
    """Local trade intents vs exchange open orders (drift detection)."""
    session = ctx.session_factory()
    try:
        report = reconcile_and_report(session, ctx.gateway, ctx.settings.market.symbols)
        if report.startswith("OK"):
            log.info("reconcile: %s", report)
        else:
            log.warning("reconcile drift:\n%s", report)
    finally:
        session.close()


def heartbeat_tick(ctx: BotContext) -> None:
    """Touch the heartbeat file so the external watchdog knows we're alive."""
    ctx.health.beat()


# --- registry ----------------------------------------------------------------

def build_scheduler(ctx: BotContext) -> AsyncIOScheduler:
    """Build the AsyncIOScheduler with all jobs registered (UTC)."""
    sched = AsyncIOScheduler(timezone="UTC")
    sched_cfg = ctx.settings.schedule

    def _add(job_id: str, fn, trigger, **trigger_args) -> None:
        sched.add_job(
            fn,
            trigger,
            args=[ctx],
            id=job_id,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=30,
            **trigger_args,
        )

    _add("market_tick", market_tick, "interval", seconds=sched_cfg.market_tick_seconds)
    _add("analytics", analytics_tick, "interval", minutes=sched_cfg.analytics_minutes)
    _add("analytics_daily", analytics_daily, "cron", hour=sched_cfg.daily_analytics_hour)
    _add("ai_review", ai_review, "cron", hour=sched_cfg.ai_review_hour)
    _add("reconcile", reconcile_tick, "interval", minutes=sched_cfg.reconcile_minutes)
    _add("heartbeat", heartbeat_tick, "interval", seconds=60)
    return sched