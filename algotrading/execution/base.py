"""Exchange gateway interface + result types.

The execution engine depends only on this protocol, so paper and live modes
share the exact same orchestration path (intent-before-order, idempotency,
risk checks). Only the gateway that ultimately places the order differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class OrderResult:
    """Outcome of an order placement or status query."""
    order_id: str
    status: str            # filled | partial | open | rejected | canceled
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    fee: float = 0.0
    error: str = ""


class ExchangeGateway(Protocol):
    """Places orders against an exchange (paper or live)."""

    def place_market_order(
        self,
        symbol: str,
        side: str,          # buy | sell
        qty: float,
        client_order_id: str,
    ) -> OrderResult:
        ...

    def get_order(self, symbol: str, client_order_id: str) -> OrderResult:
        ...
