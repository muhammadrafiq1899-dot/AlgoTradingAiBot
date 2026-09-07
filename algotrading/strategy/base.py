"""Strategy interface and signal model.

A strategy is a pure, stateless function of a market snapshot: it reads the
candle series (and its own parameters) and returns an optional Signal. It must
NOT place orders, mutate DB state, or consult the AI — the engine and execution
layers own all side effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class Signal:
    """A candidate entry/exit produced by a strategy.

    `side` is 'buy' (open/scale long) or 'sell' (close/reduce). For v1 spot
    trading every 'sell' closes the whole position on that symbol.
    """
    strategy_id: int
    symbol: str
    side: str  # 'buy' | 'sell'
    ref_price: float
    rationale: str
    risk: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


class Strategy(Protocol):
    """Stateless strategy. Instances are parameterized (constructed with params)."""

    name: str
    params: dict[str, Any]

    def evaluate(self, symbol: str, candles: Sequence[Any]) -> Signal | None:
        """Return a Signal if the current candle crosses a threshold, else None.

        `candles` is a list of Candle ordered oldest -> newest.
        """
        ...
