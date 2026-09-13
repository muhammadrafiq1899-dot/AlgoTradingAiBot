"""Builtin analytics module: metrics snapshots + the daily advisory AI pass.

The job *functions* live in :mod:`algotrading.scheduler.jobs` (they are plain
``ctx`` functions and stay unit-testable there); this module owns the decision
to schedule them. Disabling the module removes its jobs without touching the
scheduler core — the plug-and-play behaviour for non-execution parts.
"""
from __future__ import annotations

from typing import Any

from algotrading.modules.base import CAPABILITY_ANALYTICS, Module, ModuleSpec
from algotrading.modules.registry import register_module


@register_module
class AnalyticsModule(Module):
    spec = ModuleSpec(
        name="analytics.default",
        capability=CAPABILITY_ANALYTICS,
        description="Periodic metrics snapshots, daily summary, and AI review.",
        builtin=True,
    )

    def setup(self, ctx: Any) -> None:
        ctx.provide(CAPABILITY_ANALYTICS, self)

    def jobs(self, ctx: Any) -> list[Any]:
        from algotrading.scheduler.jobs import (
            JobSpec,
            ai_review,
            analytics_daily,
            analytics_tick,
        )

        sched_cfg = ctx.settings.schedule
        return [
            JobSpec("analytics", analytics_tick, "interval",
                    {"minutes": sched_cfg.analytics_minutes}),
            JobSpec("analytics_daily", analytics_daily, "cron",
                    {"hour": sched_cfg.daily_analytics_hour}),
            JobSpec("ai_review", ai_review, "cron",
                    {"hour": sched_cfg.ai_review_hour}),
        ]
