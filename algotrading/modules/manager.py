"""Module manager: resolves, sets up, starts, and stops the enabled modules.

``main.py`` no longer decides which provider/gateway/control surface to build —
it asks the manager for the configured set and then drives the lifecycle:

    manager = ModuleManager(settings, demo=demo)
    ctx = ...                       # session factory + health monitor
    manager.setup(ctx)              # synchronous: services land on ctx
    scheduler = build_scheduler(ctx)
    ctx.provide(CAPABILITY_SCHEDULER_CONTROL, controller)
    await manager.start(ctx)        # async: telegram polling, api server, ...
    ...
    await manager.stop(ctx)         # reverse order

A failing optional module is logged and skipped rather than taking down the
trading core: only a locked-capability (execution) failure is fatal.
"""
from __future__ import annotations

import logging
from typing import Any

from algotrading.modules.base import (
    LOCKED_CAPABILITIES,
    Module,
    ModuleError,
)
from algotrading.modules.registry import resolve_modules

log = logging.getLogger(__name__)


class ModuleManager:
    def __init__(self, settings, demo: bool = False, disabled: tuple[str, ...] = ()) -> None:
        self.settings = settings
        self.demo = demo
        self.disabled = tuple(disabled)
        self._modules: list[Module] = []
        self._started: list[Module] = []
        self._built = False

    # --- resolution ---

    @property
    def modules(self) -> list[Module]:
        return list(self._modules)

    def resolve(self) -> list[Module]:
        """Resolve (but do not set up) the enabled modules."""
        self._modules = resolve_modules(
            self.settings, demo=self.demo, disabled=self.disabled
        )
        return list(self._modules)

    def describe(self) -> list[tuple[str, str, bool]]:
        return [(m.name, m.spec.capability, m.spec.locked) for m in self._modules]

    # --- lifecycle ---

    def setup(self, ctx: Any) -> None:
        """Run each module's ``setup`` in dependency order.

        Providers go first so a consumer's ``setup`` can resolve what it needs.
        A module whose capability is locked must succeed; anything else is
        logged and dropped.
        """
        if not self._modules:
            self.resolve()

        healthy: list[Module] = []
        for module in list(self._modules):
            try:
                module.setup(ctx)
            except Exception as exc:  # noqa: BLE001
                if module.spec.capability in LOCKED_CAPABILITIES:
                    raise ModuleError(
                        f"locked module {module.name!r} failed to set up: {exc}"
                    ) from exc
                log.exception("module %s failed to set up; disabling it", module.name)
                continue
            healthy.append(module)

        missing = [
            f"{m.name} needs {req!r}"
            for m in healthy
            for req in m.spec.requires
            if not ctx.has(req)
        ]
        if missing:
            raise ModuleError("module dependencies unsatisfied: " + "; ".join(missing))

        self._modules = healthy
        self._built = True

    async def start(self, ctx: Any) -> None:
        """Start modules in order. Optional failures are non-fatal."""
        if not self._built:
            raise ModuleError("setup() must run before start()")
        for module in self._modules:
            try:
                await module.start(ctx)
            except Exception as exc:  # noqa: BLE001
                if module.spec.capability in LOCKED_CAPABILITIES:
                    raise ModuleError(
                        f"locked module {module.name!r} failed to start: {exc}"
                    ) from exc
                log.exception("module %s failed to start; continuing without it", module.name)
                continue
            self._started.append(module)

    async def stop(self, ctx: Any) -> None:
        """Stop started modules in reverse order (best-effort)."""
        for module in reversed(self._started):
            try:
                await module.stop(ctx)
            except Exception:  # noqa: BLE001 - shutdown must not raise
                log.exception("module %s failed to stop", module.name)
        self._started = []

    # --- contributions ---

    def collect_jobs(self, ctx: Any) -> list[Any]:
        """Gather scheduled jobs contributed by enabled modules."""
        jobs: list[Any] = []
        for module in self._modules:
            try:
                jobs.extend(module.jobs(ctx) or [])
            except Exception:  # noqa: BLE001
                log.exception("module %s failed to contribute jobs", module.name)
        return jobs


def build_manager(settings, demo: bool = False) -> ModuleManager:
    """Convenience factory used by ``main.py``."""
    return ModuleManager(settings, demo=demo)
