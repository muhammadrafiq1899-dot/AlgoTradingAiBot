"""Strategy registry: map YAML strategy names to classes + validate params."""
from __future__ import annotations

from typing import Any

from algotrading.strategy.starters import STRATEGIES


class UnknownStrategyError(ValueError):
    pass


def build_strategy(name: str, params: dict[str, Any]):
    """Instantiate a strategy class by name with the given params."""
    cls = STRATEGIES.get(name)
    if cls is None:
        raise UnknownStrategyError(
            f"Unknown strategy '{name}'. Known: {sorted(STRATEGIES)}"
        )
    return cls(params)


def known_names() -> list[str]:
    return sorted(STRATEGIES)
