"""Strategy registry: map names to classes (built-in + plugin) and build them.

Built-in strategies ship in :mod:`algotrading.strategy.starters`. Strategies
authored by the user or the AI live on disk as plugin files (see
:mod:`algotrading.strategy.plugins`) and are consulted after the built-ins, so
a plugin can never silently shadow shipped code.
"""
from __future__ import annotations

import logging
from typing import Any

from algotrading.strategy.plugins import get_default_loader
from algotrading.strategy.starters import STRATEGIES

log = logging.getLogger(__name__)


class UnknownStrategyError(ValueError):
    pass


def _plugin_class(name: str) -> type | None:
    return get_default_loader().get_class(name)


def build_strategy(name: str, params: dict[str, Any]):
    """Instantiate a strategy class by name with the given params.

    Built-in registry first, then plugin files. Raises UnknownStrategyError if
    neither knows the name.
    """
    cls = STRATEGIES.get(name) or _plugin_class(name)
    if cls is None:
        raise UnknownStrategyError(
            f"Unknown strategy '{name}'. Known: {known_names()}"
        )
    return cls(params)


def builtin_names() -> list[str]:
    """Names of strategies compiled into the package."""
    return sorted(STRATEGIES)


def plugin_names() -> list[str]:
    """Names of strategies loaded from plugin files."""
    return get_default_loader().names()


def known_names() -> list[str]:
    """All buildable strategy names: built-ins plus loaded plugins."""
    return sorted(set(STRATEGIES) | set(get_default_loader().names()))


def is_builtin(name: str) -> bool:
    return name in STRATEGIES


def is_plugin(name: str) -> bool:
    return get_default_loader().get_class(name) is not None


def reload_plugins() -> list[str]:
    """Reload plugin files whose contents changed on disk."""
    return get_default_loader().reload_changed()
