"""Scheduler: APScheduler job registry that drives the trading pipeline.

One process, one event loop (see algotrading/main.py). Jobs are sync functions
run by APScheduler's AsyncIOScheduler in a worker thread; each job opens its
own DB session so no session is ever shared across threads.
"""
from algotrading.scheduler.jobs import (
    BotContext,
    ai_review,
    analytics_daily,
    analytics_tick,
    build_scheduler,
    heartbeat_tick,
    market_tick,
    reconcile_tick,
)

__all__ = [
    "BotContext",
    "ai_review",
    "analytics_daily",
    "analytics_tick",
    "build_scheduler",
    "heartbeat_tick",
    "market_tick",
    "reconcile_tick",
]