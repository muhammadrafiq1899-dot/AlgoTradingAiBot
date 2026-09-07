"""Strategy engine: run the active strategy version over market snapshots.

Owns the signal lifecycle up to the point where a candidate signal is persisted
as `candidate` in the DB. The execution engine later consumes candidates, runs
risk checks, and either sends an order or marks the signal `skipped`.

Only ONE strategy version is active at a time (controlled release). A duplicate
signal guard prevents emitting the same entry repeatedly while a position is
already open on that symbol.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.db.models import Position, Signal, Strategy
from algotrading.strategy.base import Signal as CandidateSignal
from algotrading.strategy.registry import build_strategy, UnknownStrategyError

log = logging.getLogger(__name__)


class StrategyEngine:
    def __init__(self, session: Session) -> None:
        self._session = session

    # --- active version loading ---

    def get_active_strategy(self) -> Strategy | None:
        stmt = (
            select(Strategy)
            .where(Strategy.status == "active")
            .order_by(Strategy.version.desc())
            .limit(1)
        )
        return self._session.execute(stmt).scalar_one_or_none()

    # --- evaluation ---

    def evaluate(self, snapshot: dict[str, list]) -> list[Signal]:
        """Evaluate the active strategy against per-symbol candle lists.

        `snapshot` maps symbol -> list[Candle] (oldest -> newest). Returns the
        list of candidate Signals persisted this tick.
        """
        strat = self.get_active_strategy()
        if strat is None:
            log.info("No active strategy; skipping evaluation")
            return []
        try:
            params = json.loads(strat.params or "{}")
            inst = build_strategy(strat.name, params)
        except (UnknownStrategyError, ValueError) as exc:
            log.error("Cannot build active strategy %s: %s", strat.name, exc)
            return []

        candidates: list[Signal] = []
        for symbol, candles in snapshot.items():
            if len(candles) < 5:
                continue
            sig = inst.evaluate(symbol, candles)
            if sig is None:
                continue
            if self._skip_duplicate(strat, symbol, sig.side):
                self._persist(strat, symbol, sig, status="skipped", duplicate=True)
                continue
            rec = self._persist(strat, symbol, sig, status="candidate")
            candidates.append(rec)
        return candidates

    def _skip_duplicate(self, strat: Strategy, symbol: str, side: str) -> bool:
        """Avoid stacking entries on a symbol that already has an open position.

        For v1 spot: if a buy signal fires while we already hold that symbol, skip.
        A sell signal is always allowed (it closes)."""
        if side != "buy":
            return False
        pos = self._session.execute(
            select(Position).where(Position.symbol == symbol)
        ).scalar_one_or_none()
        return pos is not None and pos.qty > 0

    def _persist(self, strat: Strategy, symbol: str, sig: CandidateSignal, status: str, duplicate: bool = False) -> Signal:
        rec = Signal(
            strategy_id=strat.id,
            symbol=symbol,
            side=sig.side,
            ref_price=sig.ref_price,
            rationale=(f"[dup] {sig.rationale}" if duplicate else sig.rationale),
            risk_json=json.dumps(sig.risk),
            status=status,
        )
        self._session.add(rec)
        self._session.commit()
        log.debug("signal %s %s %s (status=%s)", sig.side, symbol, sig.ref_price, status)
        return rec


def now_ms() -> int:
    return int(time.time() * 1000)
