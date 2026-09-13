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
from algotrading.strategy.registry import is_plugin, known_names
from algotrading.store.strategy_versions import (
    create_new_version,
    create_new_strategy,
    promote_to_active,
    update_strategy_code,
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
    "edit_strategy",   # rewrite the code of an existing plugin strategy
    "hypothesis",
    "failure_analysis",
    "ensemble_strategy",
    "filter_strategy",
    "new_indicator",  # New kind for AI-generated indicators
}


def allowed_strategy_names() -> set[str]:
    """Every buildable strategy name: built-ins plus loaded plugins.

    Computed per call so an AI-created plugin becomes a valid target for a
    follow-up param change without a restart.
    """
    return set(known_names())


# Kept for backwards compatibility only — prefer allowed_strategy_names().
ALLOWED_STRATEGY_NAMES = frozenset(known_names())


def create_pending_recommendation(
    session: Session,
    *,
    kind: str,
    strategy_name: str,
    params: dict[str, Any],
    rationale: str = "",
    position_pct: float | None = None,
    template: str = "",
    indicator_deps: list[str] | None = None,
    param_schema: list[dict[str, Any]] | None = None,
    test_template: str = "",
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
            if comp["name"] not in allowed_strategy_names() - {"ensemble"}:
                raise ValueError(f"unknown component strategy: {comp['name']}")
    elif kind == "new_strategy":
        # Validate new strategy fields
        if not template or not isinstance(template, str):
            raise ValueError("new_strategy requires 'template' string with Python code")
        
        # Basic validation - more comprehensive validation happens in apply()
        if "def evaluate" not in template and "class" not in template:
            raise ValueError("new_strategy template must define a strategy class with evaluate method")
        
        if indicator_deps is None:
            indicator_deps = []
        elif not isinstance(indicator_deps, list):
            raise ValueError("indicator_deps must be a list of indicator function names")
        
        if param_schema is None:
            param_schema = []
        elif not isinstance(param_schema, list):
            raise ValueError("param_schema must be a list of parameter definitions")
        
        if strategy_name in allowed_strategy_names():
            raise ValueError(
                f"strategy {strategy_name!r} already exists; use edit_strategy to change it"
            )
        
        # Store extended fields in params for backward compatibility
        params.update({
            "_template": template,
            "_indicator_deps": indicator_deps,
            "_param_schema": param_schema,
            "_test_template": test_template,
        })
    elif kind == "edit_strategy":
        # Rewrite the code of a strategy that already exists as a plugin file.
        # Built-ins are deliberately excluded: their code is shipped, so their
        # behaviour is changed through param_change, not by shadowing.
        if not is_plugin(strategy_name):
            raise ValueError(
                f"edit_strategy requires an existing plugin strategy; "
                f"{strategy_name!r} is not editable this way (built-ins use param_change)"
            )
        if not template or not isinstance(template, str):
            raise ValueError("edit_strategy requires 'template' string with Python code")
        if "def evaluate" not in template and "class" not in template:
            raise ValueError("edit_strategy template must define a strategy class with evaluate method")
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")
        
        if indicator_deps is None:
            indicator_deps = []
        elif not isinstance(indicator_deps, list):
            raise ValueError("indicator_deps must be a list of indicator function names")
        
        if param_schema is None:
            param_schema = []
        elif not isinstance(param_schema, list):
            raise ValueError("param_schema must be a list of parameter definitions")
        
        params.update({
            "_template": template,
            "_indicator_deps": indicator_deps,
            "_param_schema": param_schema,
            "_test_template": test_template,
        })
    elif kind == "new_indicator":
        # Validate new indicator fields
        if not template or not isinstance(template, str):
            raise ValueError("new_indicator requires 'template' string with Python code")
        
        # Basic validation - indicator template must define a function
        if "def " not in template:
            raise ValueError("new_indicator template must define a function")
        
        # Store indicator-specific fields
        params.update({
            "_indicator_template": template,
            "_indicator_name": strategy_name,  # Use strategy_name field for indicator name
            "_indicator_test": test_template,
        })
    else:
        if strategy_name not in allowed_strategy_names():
            raise ValueError(f"unknown strategy_name {strategy_name!r}")
        if not isinstance(params, dict):
            raise ValueError("params must be a dict")

    # Build content JSON with all fields
    content_data = {"params": params, "position_pct": position_pct}
    
    # Include extended fields for new strategies
    if kind in ("new_strategy", "edit_strategy"):
        content_data.update({
            "template": template,
            "indicator_deps": indicator_deps or [],
            "param_schema": param_schema or [],
            "test_template": test_template,
        })
    elif kind == "new_indicator":
        content_data.update({
            "template": template,
            "test_template": test_template,
            "indicator_name": strategy_name,  # Store indicator name separately
        })
    
    rec = AIRecommendation(
        kind=kind,
        strategy_name=strategy_name,
        content_json=json.dumps(content_data),
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

    def _shadow_backtest(
        self,
        rec: AIRecommendation,
        params: dict[str, Any],
        candles: Sequence[Candle],
        do_backtest: bool,
        min_profit_threshold: float | None,
    ) -> BacktestResult | None:
        """Replay a recommendation; reject it if it misses the threshold.

        Returns the result (None when skipped). On a threshold miss the row is
        marked REJECTED and the caller must not release the version. Raises
        ValueError if the strategy cannot be built at all.
        """
        if not do_backtest or not candles:
            return None
        try:
            result = run_backtest(
                candles,
                rec.strategy_name or "ema_crossover",
                params,
            )
        except ValueError as exc:
            log.warning("backtest failed for recommendation %s: %s", rec.id, exc)
            raise ValueError(f"shadow backtest failed: {exc}") from exc

        if min_profit_threshold is not None and result.total_pnl < min_profit_threshold:
            rec.status = REJECTED
            rec.reviewed_at = datetime.now(timezone.utc)
            rec.backtest_json = json.dumps(result.to_dict())
            self._session.commit()
            log.info("rec %s auto-rejected: backtest pnl below threshold", rec.id)
        return result

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

        # Shadow-backtest things that are already buildable. Code-authoring
        # kinds (new_strategy/edit_strategy) are replayed *after* their plugin
        # file is written — the name does not resolve before that, so
        # backtesting first would always fail with "Unknown strategy". An
        # indicator is a function, not a strategy: there is nothing to replay.
        result = None
        if rec.kind not in ("new_strategy", "edit_strategy", "new_indicator"):
            result = self._shadow_backtest(
                rec, params, candles, do_backtest, min_profit_threshold
            )
            if rec.status == REJECTED:
                return None, result

        # Handle new_strategy creation differently
        if rec.kind == "new_strategy":
            # Extract extended fields
            template = content.get("template", "")
            indicator_deps = content.get("indicator_deps", [])
            param_schema = content.get("param_schema", [])
            test_template = content.get("test_template", "")
            
            if not template:
                raise ValueError("new_strategy requires template field")
            
            # Create new strategy from template
            try:
                strategy = create_new_strategy(
                    self._session,
                    name=rec.strategy_name,
                    template=template,
                    params=params,
                    param_schema=param_schema,
                    indicator_deps=indicator_deps,
                    description=f"AI-generated from recommendation {rec.id}",
                    test_template=test_template,
                )
            except ValueError as exc:
                log.error("Failed to create new strategy %s: %s", rec.strategy_name, exc)
                raise ValueError(f"Failed to create new strategy: {exc}") from exc
        elif rec.kind == "edit_strategy":
            template = content.get("template", "")
            indicator_deps = content.get("indicator_deps", [])
            param_schema = content.get("param_schema", [])
            if not template:
                raise ValueError("edit_strategy requires template field")
            try:
                strategy = update_strategy_code(
                    self._session,
                    name=rec.strategy_name,
                    template=template,
                    params=params,
                    param_schema=param_schema,
                    indicator_deps=indicator_deps,
                    description=f"code update from recommendation {rec.id}",
                )
            except ValueError as exc:
                log.error("Failed to edit strategy %s: %s", rec.strategy_name, exc)
                raise ValueError(f"Failed to edit strategy: {exc}") from exc
        elif rec.kind == "new_indicator":
            # Handle new indicator creation
            template = content.get("template", "")
            indicator_name = content.get("indicator_name", rec.strategy_name)
            test_template = content.get("test_template", "")
            
            if not template:
                raise ValueError("new_indicator requires template field")
            
            # Import indicator module
            from algotrading.strategy import indicators
            
            try:
                # Create and register the new indicator
                indicator_func = indicators.create_indicator(
                    name=indicator_name,
                    code=template,
                    metadata={
                        "description": f"AI-generated indicator from recommendation {rec.id}",
                        "source": "ai_generated",
                        "test_template": test_template,
                    }
                )
                
                # For indicators, we don't create a DB strategy row
                # Instead, we log success and return None for strategy
                log.info("Successfully created new indicator '%s' from recommendation %s", 
                        indicator_name, rec.id)
                strategy = None  # Indicators don't have DB rows
            except ValueError as exc:
                log.error("Failed to create new indicator %s: %s", indicator_name, exc)
                raise ValueError(f"Failed to create new indicator: {exc}") from exc
        else:
            # Standard strategy update
            strategy = create_new_version(
                self._session,
                name=rec.strategy_name or "ema_crossover",
                params=params,
                description=f"from recommendation {rec.id}",
            )
        
        if rec.kind in ("new_strategy", "edit_strategy"):
            # Now that the plugin is registered, replay it like any other change.
            result = self._shadow_backtest(
                rec, params, candles, do_backtest, min_profit_threshold
            )
            if rec.status == REJECTED:
                # The authored draft plugin is kept for inspection but not released.
                return None, result

        # Indicators register a function, not a strategy version — there is
        # nothing to promote, and promote_to_active(None) would crash.
        if strategy is not None:
            promote_to_active(self._session, strategy)

        rec.status = APPLIED
        rec.reviewed_at = datetime.now(timezone.utc)
        if result is not None:
            rec.backtest_json = json.dumps(result.to_dict())
        self._session.commit()
        return strategy, result
