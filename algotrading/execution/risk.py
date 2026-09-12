"""Risk module: position sizing, exposure caps, cooldowns, trailing stops.

Pure decision logic — no side effects. The execution engine calls these
functions before persisting an intent. Any guard that fails causes the signal
to be marked `skipped` with the reason recorded as a risk_skipped event.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from algotrading.config import RiskConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    qty: float = 0.0
    reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self._cfg = cfg

    # --- sizing ---

    def size_position(self, balance: float, price: float, atr_value: float | None = None) -> float:
        """Size a spot buy so that a stop-loss distance risks `risk_per_trade_pct`.

        If no ATR is available, fall back to a fixed fraction of balance.
        `balance` is total account balance (or paper starting balance).
        """
        risk_budget = balance * (self._cfg.risk_per_trade_pct / 100.0)
        if atr_value and atr_value > 0:
            stop_distance = max(atr_value, price * 0.001)
            qty = risk_budget / stop_distance
        else:
            qty = balance / price * (self._cfg.risk_per_trade_pct / 100.0)
        # Cap by max_position_pct of balance.
        max_notional = balance * (self._cfg.max_position_pct / 100.0)
        qty = min(qty, max_notional / price)
        return max(qty, 0.0)

    # --- trailing stops ---

    def compute_trailing_stop_price(self, entry_price: float, current_price: float, is_long: bool = True) -> float | None:
        """Calculate trailing stop price based on current price and trailing_stop_pct.

        Returns the trailing stop price, or None if trailing stops are disabled.
        """
        if not self._cfg.trailing_stop_pct or self._cfg.trailing_stop_pct <= 0:
            return None
        pct = self._cfg.trailing_stop_pct / 100.0
        if is_long:
            # Trail below current price
            return current_price * (1 - pct)
        else:
            # Trail above current price (for shorts)
            return current_price * (1 + pct)

    def check_trailing_stop(self, entry_price: float, current_price: float,
                            existing_stop: float | None, is_long: bool = True) -> tuple[bool, float | None]:
        """Check if trailing stop should be updated.

        Returns (should_update, new_stop_price).

        For long positions: track the highest price seen since entry.
        For short positions: track the lowest price seen since entry.

        Note: This stateless version computes the ideal stop based on current_price.
        The engine must track the highest/lowest price separately for proper
        trailing behavior. Here we return the stop based on current_price, but
        the engine should only call this when price moves favorably.
        """
        if not self._cfg.trailing_stop_pct or self._cfg.trailing_stop_pct <= 0:
            return False, None

        new_stop = self.compute_trailing_stop_price(entry_price, current_price, is_long)
        if new_stop is None:
            return False, None

        # Only update if the new stop is better (higher for long, lower for short)
        if is_long:
            if existing_stop is None or new_stop > existing_stop:
                return True, new_stop
        else:
            if existing_stop is None or new_stop < existing_stop:
                return True, new_stop

        return False, None

    # --- guards ---

    def check_buy(self, *, symbol: str, balance: float, open_positions: int,
                  last_entry_ts: float | None) -> RiskDecision:
        """Guards for opening a new long position on `symbol`."""
        if open_positions >= self._cfg.max_open_positions:
            return RiskDecision(False, reason="max_open_positions reached")
        if last_entry_ts is not None:
            since = (time.time() * 1000) - last_entry_ts
            if since < self._cfg.cooldown_seconds * 1000:
                return RiskDecision(False, reason="cooldown active")
        return RiskDecision(True)

    def check_sell(self, *, has_position: bool) -> RiskDecision:
        """Guards for closing. Selling without a position is an error/skip."""
        if not has_position:
            return RiskDecision(False, reason="no open position to close")
        return RiskDecision(True)
