"""Daily AI assistant: propose strategy changes as PENDING recommendations.

The assistant is the ONLY place that talks to the LLM. It builds a
deterministic prompt from analytics summaries + recent candles, calls the
AIClient, validates the structured JSON, and persists an AIRecommendation in
PENDING status. It NEVER applies anything and never touches the exchange.

Failure of the LLM/network is non-fatal: the bot keeps running on the last
approved strategy. A failed proposal simply isn't persisted.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from algotrading.ai.client import AIClient, RecommendationError, extract_json
from algotrading.ai.prompt_builder import build_prompt
from algotrading.db.models import AIRecommendation
from algotrading.market.base import Candle
from algotrading.store.recommendations import (
    ALLOWED_KINDS,
    ALLOWED_STRATEGY_NAMES,
    PENDING,
)

log = logging.getLogger(__name__)


class Assistant:
    def __init__(self, session, client: AIClient | None = None,
                 strategy_name: str = "ema_crossover",
                 params: dict[str, Any] | None = None):
        self._session = session
        self._client = client or AIClient()
        self._strategy_name = strategy_name
        self._params = params or {}

    def _validate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate + normalize the LLM's JSON into a recommendation payload.

        Raises RecommendationError on any schema violation so bad model output
        never reaches the DB.
        """
        kind = data.get("kind")
        if kind not in ALLOWED_KINDS:
            raise RecommendationError(f"unknown kind {kind!r}")

        name = data.get("strategy_name") or self._strategy_name
        
        content = data.get("content") or {}
        if not isinstance(content, dict):
            raise RecommendationError("content must be a JSON object")

        params = content.get("params") or {}
        if not isinstance(params, dict):
            raise RecommendationError("content.params must be a JSON object")

        # Validate ensemble/filter strategy params
        if kind in ("ensemble_strategy", "filter_strategy"):
            if name != "ensemble":
                raise RecommendationError(f"{kind} requires strategy_name='ensemble'")
            components = params.get("components")
            if not isinstance(components, list) or len(components) < 2:
                raise RecommendationError(f"{kind} requires 'components' list with >=2 strategies")
            for comp in components:
                if not isinstance(comp, dict) or "name" not in comp:
                    raise RecommendationError("each component must have 'name' and 'params'")
                if comp["name"] not in ALLOWED_STRATEGY_NAMES - {"ensemble"}:
                    raise RecommendationError(f"unknown component strategy: {comp['name']}")
        else:
            if name not in ALLOWED_STRATEGY_NAMES:
                raise RecommendationError(f"unknown strategy_name {name!r}")

        # For a plain hypothesis there is no params change.
        return {
            "kind": kind,
            "strategy_name": name,
            "content_json": json.dumps({
                "params": params,
                "position_pct": content.get("position_pct"),
            }),
            "rationale": data.get("rationale", "") or "",
        }

    def propose(self, symbol: str, interval: str,
                summaries: Sequence[Any], candles: Sequence[Candle],
                max_feature_rows: int = 24) -> AIRecommendation | None:
        """Build the prompt, call the LLM, persist a PENDING recommendation.

        Returns the saved recommendation, or None if the LLM is disabled or
        fails (non-fatal — execution continues on the last approved strategy).
        """
        if not self._client.enabled:
            log.info("AI disabled; skipping proposal")
            return None

        messages = build_prompt(
            symbol, interval, summaries, candles,
            self._strategy_name, self._params, max_feature_rows=max_feature_rows,
        )
        try:
            data = self._client.complete_json(messages)
        except RecommendationError as exc:
            log.warning("AI proposal failed: %s", exc)
            return None

        try:
            payload = self._validate(data)
        except RecommendationError as exc:
            log.warning("AI proposal invalid: %s", exc)
            return None

        rec = AIRecommendation(
            kind=payload["kind"],
            strategy_name=payload["strategy_name"],
            content_json=payload["content_json"],
            status=PENDING,
            rationale=payload["rationale"],
        )
        self._session.add(rec)
        self._session.commit()
        log.info("AI proposed %s for %s (id=%s)", payload["kind"],
                 payload["strategy_name"], rec.id)
        return rec
