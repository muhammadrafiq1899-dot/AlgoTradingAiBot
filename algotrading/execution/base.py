"""Exchange gateway interface + result types.

The execution engine depends only on this protocol, so paper and live modes
share the exact same orchestration path (intent-before-order, idempotency,
risk checks). Only the gateway that ultimately places the order differs.

Capabilities are split into two tiers on purpose:

* **Mandatory** — ``place_market_order`` / ``get_order``. A gateway without
  these cannot trade at all, and paper/live must stay interchangeable through
  them.
* **Optional** — ``place_limit_order`` / ``place_stop_order`` / ``cancel_order``
  / ``get_open_orders`` / ``get_balance``. The engine probes these with
  ``getattr(gateway, name, None)`` and degrades gracefully when they are
  missing, so a thin test double (e.g. a stub that only fills market orders)
  is a valid gateway without implementing the whole surface. Both shipped
  gateways are full implementations.

Why the tiering: the exchange-side protective stop and the reconcile job are
add-ons to a working order path. Making them mandatory would force every test
double and future venue to stub methods it never uses, and a missing optional
capability must never break the tick — it just means "this venue cannot do it,
keep the local protection only".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class OrderResult:
    """Outcome of an order placement or status query."""
    order_id: str
    status: str            # filled | partial | open | rejected | canceled | unknown
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    fee: float = 0.0
    error: str = ""


class ExchangeGateway(Protocol):
    """Places orders against an exchange (paper or live)."""

    # --- mandatory ---

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

    # --- optional (probe with getattr; a partial gateway may omit them) ---

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        client_order_id: str,
        time_in_force: str = "GTC",
    ) -> OrderResult:
        """Rest a LIMIT order. May return status ``open`` — a limit order is
        not synchronously filled, so the engine marks the intent `sent` and
        leaves confirmation to the reconcile path."""
        ...

    def place_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        limit_price: float | None = None,
        client_order_id: str = "",
    ) -> OrderResult:
        """Rest a protective STOP on the venue.

        With ``limit_price=None`` this is a market stop, which is the safest
        protective default: it always fills once triggered. The returned
        ``order_id`` is what the engine records so a later cancel can find it.
        """
        ...

    def cancel_order(self, symbol: str, client_order_id: str) -> OrderResult:
        """Cancel one resting order. Best-effort: callers must tolerate failure."""
        ...

    def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Resting orders for ``symbol``, each with a ``client_order_id`` key."""
        ...

    def get_balance(self, asset: str = "USDT") -> float:
        """Free + locked balance of ``asset`` at the venue (sizing basis)."""
        ...
