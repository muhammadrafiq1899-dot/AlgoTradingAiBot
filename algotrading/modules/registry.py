"""Module registry: name → class, plus config-driven resolution.

Selection rules (from ``config/settings.yaml`` ``modules:``):

* ``enabled: null`` (default) → a sensible default set derived from mode/demo.
* ``enabled: [names]``       → exactly those modules.
* ``disabled: [names]``      → removed from either set.
* ``external: [dotted.path]``→ imported and registered before resolution.
* ``params: {name: {...}}``  → passed to that module's ``__init__``.

Locked capabilities (execution) cannot be disabled, and only modules shipped in
the package (``ModuleSpec.builtin``) may provide them.
"""
from __future__ import annotations

import importlib
import logging
from typing import Sequence

from algotrading.modules.base import (
    LOCKED_CAPABILITIES,
    Module,
    ModuleError,
    ModuleSpec,
    sort_modules,
)

log = logging.getLogger(__name__)

__all__ = [
    "register_module",
    "get_module",
    "all_modules",
    "module_names",
    "load_external_modules",
    "default_enabled_names",
    "resolve_modules",
]

_MODULES: dict[str, type[Module]] = {}


def register_module(cls: type[Module]) -> type[Module]:
    """Register a module class under ``cls.spec.name``. Usable as a decorator."""
    spec = getattr(cls, "spec", None)
    if not isinstance(spec, ModuleSpec):
        raise ModuleError(f"{cls.__name__} must define a ModuleSpec in `spec`")
    existing = _MODULES.get(spec.name)
    if existing is not None and existing is not cls:
        raise ModuleError(f"module name {spec.name!r} is already registered by {existing.__name__}")
    _MODULES[spec.name] = cls
    return cls


def get_module(name: str) -> type[Module] | None:
    return _MODULES.get(name)


def all_modules() -> dict[str, type[Module]]:
    return dict(_MODULES)


def module_names() -> list[str]:
    return sorted(_MODULES)


def load_external_modules(paths: Sequence[str]) -> list[str]:
    """Import dotted paths and register any Module subclasses found.

    A path may be ``"my_pkg.my_module"`` (registers every Module subclass) or
    ``"my_pkg.my_module:MyModule"`` (registers just that class).
    """
    registered: list[str] = []
    for path in paths:
        module_path, _, class_name = path.partition(":")
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not be silent
            raise ModuleError(f"cannot import external module {module_path!r}: {exc}") from exc

        if class_name:
            obj = getattr(module, class_name, None)
            if not isinstance(obj, type) or not issubclass(obj, Module):
                raise ModuleError(f"{path!r} does not name a Module subclass")
            candidates = [obj]
        else:
            candidates = [
                obj
                for obj in vars(module).values()
                if isinstance(obj, type) and issubclass(obj, Module) and obj is not Module
            ]
            if not candidates:
                raise ModuleError(f"no Module subclasses found in {path!r}")

        for cls in candidates:
            spec = getattr(cls, "spec", None)
            if not isinstance(spec, ModuleSpec):
                continue
            # External code may never claim a locked capability.
            if spec.capability in LOCKED_CAPABILITIES:
                log.warning(
                    "ignoring external module %s: capability %r is locked",
                    spec.name, spec.capability,
                )
                continue
            register_module(cls)
            registered.append(spec.name)
    return registered


def default_enabled_names(settings, demo: bool = False) -> list[str]:
    """The module set used when ``modules.enabled`` is not set."""
    modules_cfg = settings.modules
    names: list[str] = []

    if demo:
        names.append("market.demo")
    else:
        names.append("market.binance")

    if demo or settings.mode == "paper":
        names.append("gateway.paper")
    else:
        names.append("gateway.live")

    names.append("strategy.plugins")
    names.append("analytics.default")
    names.append("control.telegram")
    if settings.api.enabled:
        names.append("api.http")
    return names


# Gateways that contradict --demo-data (which must never place real orders).
_DEMO_FORBIDDEN = frozenset({"gateway.live"})


def resolve_modules(
    settings, demo: bool = False, disabled: Sequence[str] = ()
) -> list[Module]:
    """Instantiate the enabled module set for these settings.

    ``disabled`` lets a caller (e.g. ``--no-telegram``) force a module off
    without mutating settings. Locked modules ignore it.

    Raises ModuleError on unknown names, duplicate capabilities, a locked
    capability claimed by external code, or an unsatisfied dependency.
    """
    cfg = settings.modules
    if cfg.external:
        load_external_modules(cfg.external)

    if cfg.enabled is None:
        selected = default_enabled_names(settings, demo)
    else:
        selected = list(cfg.enabled)

    if demo:
        offending = [n for n in selected if n in _DEMO_FORBIDDEN]
        if offending:
            raise ModuleError(
                f"--demo-data forces paper mode; cannot use {offending}"
            )

    disabled = set(cfg.disabled or ()) | set(disabled)
    final: list[str] = []
    for name in selected:
        cls = get_module(name)
        if cls is None:
            raise ModuleError(
                f"unknown module {name!r}; known: {module_names()}"
            )
        if name in disabled:
            if cls.spec.locked:
                log.warning("module %r is locked and cannot be disabled; keeping it", name)
            else:
                continue
        final.append(name)

    # Instances + capability uniqueness + builtin enforcement for locked caps.
    instances: list[Module] = []
    by_capability: dict[str, Module] = {}
    for name in final:
        cls = get_module(name)
        if cls.spec.capability in LOCKED_CAPABILITIES and not cls.spec.builtin:
            raise ModuleError(
                f"module {name!r} claims locked capability {cls.spec.capability!r} "
                "but is not a built-in module"
            )
        instance = cls(cfg.params.get(name, {}))
        if cls.spec.capability in by_capability:
            raise ModuleError(
                f"capability {cls.spec.capability!r} is provided by both "
                f"{by_capability[cls.spec.capability].name!r} and {name!r}"
            )
        by_capability[cls.spec.capability] = instance
        instances.append(instance)

    # Dependencies: required capabilities must be satisfied by the selection.
    for instance in instances:
        for requirement in instance.spec.requires:
            if requirement not in by_capability:
                raise ModuleError(
                    f"module {instance.name!r} requires capability {requirement!r}, "
                    "which no enabled module provides"
                )

    return sort_modules(instances)
