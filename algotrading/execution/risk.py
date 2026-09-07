"""Risk module: position sizing, exposure caps, cooldowns.

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
