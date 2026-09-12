"""AI recommendation CRUD + controlled apply().

A recommendation starts PENDING (saved by the assistant). A human approves or
rejects it over Telegram. On approve, `apply()` runs an optional shadow
backtest, then (if it passes a threshold or unconditionally, per policy)
creates a new strategy version and promotes it to active.

The assistant never calls apply() — approval is always human-initiated.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.backtest.runner import BacktestResult, run_backtest
from algotrading.db.models import AIRecommendation, Strategy
from algotrading.market.base import Candle
from algotrading.store.strategy_versions import (
    create_new_version,
    promote_to_active,
)

log = logging.getLogger(__name__)

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
APPLIED = "applied"

# Recommendation kinds the LLM (daily assistant OR chat orchestrator) may emit.
ALLOWED_KINDS = {
    "param_change",
    "new_strategy",
    "hypothesis",
    "failure_analysis",
    "ensemble_strategy",
    "filter_strategy",
}
# Must match the registry names in algotrading/strategy/starters.py
ALLOWED_STRATEGY_NAMES = {
    "ema_crossover",
    "rsi_mean_reversion",
    "bb_mean_reversion",
    "macd_trend",
    "supertrend",
    "vwap_reclaim",
    "multi_tf_ema",
    "ensemble",
}


def create_pending_recommendation(
    session: Session,
    *,
    kind: str,
    strategy_name: str,
    params: dict[str, Any],
    rationale: str = "",
    position_pct: float | None = None,
) -> AIRecommendation:
    """Validate and persist a PENDING recommendation (human approval required).

    Shared by the daily AI review and the chat orchestrator. Raises ValueError
    on anything that must never reach the DB; never applies anything.
    """
    if kind not in ALLOWED_KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    
    # Validate ensemble/filter strategy params
    if kind in ("ensemble_strategy", "filter_strategy"):
        if strategy_name != "ensemble":
            raise ValueError(f"{kind} requires strategy_name='ensemble'")
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")
        components = params.get("components")
        if not isinstance(components, list) or len(components) < 2:
            raise ValueError(f"{kind} requires 'components' list with >=2 strategies")
        for comp in components:
            if not isinstance(comp, dict) or "name" not in comp:
                raise ValueError("each component must have 'name' and 'params'")
            if comp["name"] not in ALLOWED_STRATEGY_NAMES - {"ensemble"}:
                raise ValueError(f"unknown component strategy: {comp['name']}")
    else:
        if strategy_name not in ALLOWED_STRATEGY_NAMES:
            raise ValueError(f"unknown strategy_name {strategy_name!r}")
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")

    rec = AIRecommendation(
        kind=kind,
        strategy_name=strategy_name,
        content_json=json.dumps({"params": params, "position_pct": position_pct}),
        status=PENDING,
        rationale=rationale or "",
    )
    session.add(rec)
    session.commit()
    return rec


class RecommendationStore:
    def __init__(self, session: Session):
        self._session = session

    def save(self, rec: AIRecommendation) -> AIRecommendation:
        self._session.add(rec)
        self._session.commit()
        return rec

    def get(self, rec_id: int) -> AIRecommendation | None:
        return self._session.get(AIRecommendation, rec_id)

    def pending(self, limit: int = 20) -> Sequence[AIRecommendation]:
        return self._session.execute(
            select(AIRecommendation)
            .where(AIRecommendation.status == PENDING)
            .order_by(AIRecommendation.ts.desc())
            .limit(limit)
        ).scalars().all()

    def mark_rejected(self, rec: AIRecommendation) -> AIRecommendation:
        rec.status = REJECTED
        rec.reviewed_at = datetime.now(timezone.utc)
        self._session.commit()
        return rec

    def apply(
        self,
        rec: AIRecommendation,
        candles: Sequence[Candle],
        do_backtest: bool = True,
        min_profit_threshold: float | None = None,
    ) -> tuple[Strategy | None, BacktestResult | None]:
        """Apply an approved recommendation.

        Runs the shadow backtest (unless disabled), then creates + promotes a
        new strategy version. Returns (new_strategy, backtest_result).

        Raises:
            ValueError if the recommendation is not PENDING or lacks content.
        """
        if rec.status != PENDING:
            raise ValueError(f"cannot apply recommendation in status {rec.status!r}")
        content = json.loads(rec.content_json or "{}")
        params = content.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("recommendation content.params must be a dict")

        result = None
        if do_backtest and candles:
            try:
                result = run_backtest(
                    candles,
                    rec.strategy_name or "ema_crossover",
                    params,
                )
            except ValueError as exc:
                log.warning("backtest failed for recommendation %s: %s", rec.id, exc)
                raise ValueError(f"shadow backtest failed: {exc}") from exc

            if min_profit_threshold is not None and (
                result is None or result.total_pnl < min_profit_threshold
            ):
                rec.status = REJECTED
                rec.reviewed_at = datetime.now(timezone.utc)
                rec.backtest_json = json.dumps(result.to_dict()) if result else "{}"
                self._session.commit()
                log.info("rec %s auto-rejected: backtest pnl below threshold", rec.id)
                return None, result

        strategy = create_new_version(
            self._session,
            name=rec.strategy_name or "ema_crossover",
            params=params,
            description=f"from recommendation {rec.id}",
        )
        promote_to_active(self._session, strategy)

        rec.status = APPLIED
        rec.reviewed_at = datetime.now(timezone.utc)
        if result is not None:
            rec.backtest_json = json.dumps(result.to_dict())
        self._session.commit()
        return strategy, result
