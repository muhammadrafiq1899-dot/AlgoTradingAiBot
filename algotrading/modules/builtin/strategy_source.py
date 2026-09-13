"""Builtin strategy module: load strategy plugins and keep them fresh.

Strategy *code* lives on disk (``strategies/*.py``) and is owned by the user or
the AI; this module points the process-wide plugin loader at the configured
directories and schedules a periodic reload so hand-edits take effect without a
restart. The strategy engine already rebuilds the active instance on every tick,
so a reloaded class is picked up immediately.

Nothing here can execute a trade: strategies are pure signal producers and the
locked execution gateway decides whether a signal becomes an order.
"""
from __future__ import annotations

import logging
from typing import Any

from algotrading.modules.base import CAPABILITY_STRATEGY, Module, ModuleSpec
from algotrading.modules.registry import register_module
from algotrading.strategy.plugins import (
    StrategyPluginLoader,
    configure_default_loader,
    get_default_loader,
)
from algotrading.strategy.registry import builtin_names

log = logging.getLogger(__name__)

RELOAD_JOB_ID = "strategy_plugins_reload"


def reload_strategy_plugins(ctx: Any) -> None:
    """Job body: reload any strategy plugin file whose mtime changed."""
    loader: StrategyPluginLoader = ctx.get(CAPABILITY_STRATEGY)
    changed = loader.reload_changed()
    if changed:
        log.info("reloaded strategy plugins: %s", ", ".join(sorted(changed)))


@register_module
class StrategyPluginModule(Module):
    spec = ModuleSpec(
        name="strategy.plugins",
        capability=CAPABILITY_STRATEGY,
        description="Load user/AI strategy files from disk and hot-reload edits.",
        builtin=True,
        params=("reload_seconds",),
    )

    def setup(self, ctx: Any) -> None:
        cfg = ctx.settings.modules
        if cfg.autoload_strategies:
            loader = configure_default_loader(cfg.strategy_plugin_paths)
        else:
            loader = get_default_loader()
        ctx.provide(CAPABILITY_STRATEGY, loader)
        log.info(
            "strategy module: %d built-in strategies, %d plugin strategies from %s",
            len(builtin_names()),
            len(loader.names()),
            ", ".join(str(p) for p in loader.paths),
        )

    def jobs(self, ctx: Any) -> list[Any]:
        seconds = int(self.params.get("reload_seconds", 300) or 0)
        if seconds <= 0:
            return []
        from algotrading.scheduler.jobs import JobSpec

        return [
            JobSpec(RELOAD_JOB_ID, reload_strategy_plugins, "interval", {"seconds": seconds})
        ]
