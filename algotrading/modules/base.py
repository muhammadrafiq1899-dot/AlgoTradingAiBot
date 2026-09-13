"""Plug-and-play module contract for every non-execution subsystem.

The bot is a core (deterministic execution + ledger) plus a set of swappable
modules. A module declares what *capability* it satisfies, optionally which
capabilities it depends on, and how it wires itself into the shared
:class:`algotrading.scheduler.jobs.BotContext`:

    market  → ctx.provider            (exchange/market data source)
    execution → ctx.gateway            (order gateway — LOCKED, see below)
    strategy  → strategy plugin loading
    analytics → scheduled metric snapshots
    control   → Telegram control surface
    api       → optional HTTP health/status API

Modules are selected by name from ``config/settings.yaml`` ``modules:`` block
(see :class:`algotrading.config.ModulesConfig`), so adding, removing, or
replacing a part is a configuration change — no edits to ``main.py``.

**Execution is deliberately special.** Modules in ``LOCKED_CAPABILITIES`` must
be built into the package (``ModuleSpec.builtin``) and cannot be disabled or
replaced by configuration. That keeps the order path deterministic and
auditable: the AI can propose strategy changes, but nothing at runtime can
swap out the code that talks to the exchange.
"""
from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar

log = logging.getLogger(__name__)

# Capability keys. A module provides exactly one; consumers ask the context for
# it by key.
CAPABILITY_MARKET = "market"
CAPABILITY_EXECUTION = "execution"
CAPABILITY_STRATEGY = "strategy"
CAPABILITY_ANALYTICS = "analytics"
CAPABILITY_CONTROL = "control"
CAPABILITY_API = "api"
# Not a module capability: the runtime attaches the scheduler controller so the
# control surface can pause/resume jobs without importing main's internals.
CAPABILITY_SCHEDULER_CONTROL = "scheduler.control"

# Setup order: dependencies first.
BUILD_ORDER: tuple[str, ...] = (
    CAPABILITY_MARKET,
    CAPABILITY_EXECUTION,
    CAPABILITY_STRATEGY,
    CAPABILITY_ANALYTICS,
    CAPABILITY_CONTROL,
    CAPABILITY_API,
)

# Capabilities the module system refuses to let config/AI change.
LOCKED_CAPABILITIES = frozenset({CAPABILITY_EXECUTION})


class ModuleError(RuntimeError):
    """Raised when a module cannot be resolved, built, or started."""


@dataclass(frozen=True)
class ModuleSpec:
    """Static metadata describing a module."""

    name: str
    capability: str
    description: str = ""
    requires: tuple[str, ...] = ()
    # True for modules shipped with the package. External modules cannot hold a
    # locked capability.
    builtin: bool = False
    enabled_by_default: bool = True
    # Extra config keys the module reads (documentation for the config block).
    params: tuple[str, ...] = ()

    @property
    def locked(self) -> bool:
        return self.capability in LOCKED_CAPABILITIES


class Module(ABC):
    """Base class for pluggable modules.

    Lifecycle: ``__init__(params)`` → ``setup(ctx)`` (synchronous, before the
    scheduler starts) → ``start(ctx)`` (async, after the scheduler is up) →
    ``stop(ctx)`` (async, on shutdown, reverse order).
    """

    spec: ClassVar[ModuleSpec]

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})

    # --- convenience ---
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def capability(self) -> str:
        return self.spec.capability

    def describe(self) -> str:
        return self.spec.description or self.spec.name

    # --- lifecycle hooks (override what you need) ---
    def setup(self, ctx: Any) -> None:
        """Wire services into ``ctx``. Synchronous; runs before the scheduler."""

    async def start(self, ctx: Any) -> None:
        """Start background work (polling loops, servers)."""

    async def stop(self, ctx: Any) -> None:
        """Tear down anything ``start`` created."""

    def jobs(self, ctx: Any) -> list[Any]:
        """Scheduled jobs this module contributes (list of JobSpec)."""
        return []

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<{type(self).__name__} {self.spec.name} ({self.spec.capability})>"


@dataclass
class ServiceBag:
    """Tiny capability registry attached to the app context.

    Jobs and modules resolve their collaborators from here instead of importing
    concrete implementations, which is what makes the parts swappable.
    """

    services: dict[str, Any] = field(default_factory=dict)

    def provide(self, capability: str, service: Any) -> None:
        if capability in self.services:
            log.debug("capability %r provided twice; replacing", capability)
        self.services[capability] = service

    def get(self, capability: str) -> Any:
        if capability not in self.services:
            raise ModuleError(
                f"capability {capability!r} is not available "
                f"(provided: {sorted(self.services)})"
            )
        return self.services[capability]

    def has(self, capability: str) -> bool:
        return capability in self.services


def sort_modules(modules: list[Module]) -> list[Module]:
    """Order modules so dependencies are set up before dependents.

    Ties are broken by the registry order (insertion), which keeps behaviour
    stable and predictable.
    """
    order = {cap: i for i, cap in enumerate(BUILD_ORDER)}
    return sorted(
        modules,
        key=lambda m: (order.get(m.spec.capability, len(BUILD_ORDER)), m.spec.name),
    )
