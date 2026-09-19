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

    def clamp_position_pct(self, position_pct: float) -> float:
        """Clamp a requested fraction-of-balance to the configured ceiling.

        `max_position_pct` is the hard cap the order path must honour. Sizing
        requests arrive inside `Signal.risk` (strategy params, AI-authored
        params, or a future model), so the cap belongs here, not in the schema.
        """
        try:
            pct = float(position_pct)
        except (TypeError, ValueError):
            return 0.0
        if pct <= 0:
            return 0.0
        return min(pct, self._cfg.max_position_pct / 100.0)

    def size_by_pct(self, balance: float, price: float, position_pct: float) -> float:
        """Quantity for a buy sized as `position_pct` of balance, capped.

        Returns 0.0 when the inputs cannot produce a sane order (the caller
        records a `risk_skipped` event for `qty <= 0`).
        """
        if balance <= 0 or price <= 0:
            return 0.0
        return balance * self.clamp_position_pct(position_pct) / price

    def daily_loss_breached(self, daily_pnl: float, balance: float) -> tuple[bool, float]:
        """Is today's realized loss at/over `max_daily_loss_pct`?

        Returns `(breached, loss_pct)`. A disabled guard (pct 0, or
        `enforce_daily_loss: false`) never breaches, so the fail-open path is
        explicit and reviewable rather than accidental.
        """
        if not self._cfg.enforce_daily_loss:
            return False, 0.0
        limit = self._cfg.max_daily_loss_pct
        if limit <= 0 or balance <= 0 or daily_pnl >= 0:
            return False, 0.0
        loss_pct = (-daily_pnl / balance) * 100.0
        return loss_pct >= limit, loss_pct

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
                  last_entry_ts: float | None, daily_pnl: float = 0.0) -> RiskDecision:
        """Guards for opening a new long position on `symbol`.

        `daily_pnl` is today's realized PnL (UTC). A breach only blocks *new
        entries* — protective exits must always be able to run.
        """
        breached, loss_pct = self.daily_loss_breached(daily_pnl, balance)
        if breached:
            return RiskDecision(
                False,
                reason=(
                    f"daily loss limit reached ({loss_pct:.2f}% >= "
                    f"{self._cfg.max_daily_loss_pct:.2f}%)"
                ),
            )
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
