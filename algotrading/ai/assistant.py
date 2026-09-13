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
from algotrading.strategy.registry import is_plugin
from algotrading.strategy.validation import CodeValidationError, validate_strategy_code
from algotrading.store.recommendations import (
    ALLOWED_KINDS,
    PENDING,
    allowed_strategy_names,
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

    def _validate_ast(self, code: str) -> None:
        """Validate generated strategy code using the shared safety gate.

        Raises RecommendationError if the code is unsafe or malformed.
        """
        try:
            validate_strategy_code(code)
        except CodeValidationError as exc:
            raise RecommendationError(str(exc)) from exc

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
                if comp["name"] not in allowed_strategy_names() - {"ensemble"}:
                    raise RecommendationError(f"unknown component strategy: {comp['name']}")
        elif kind == "new_strategy":
            template = content.get("template", "")
            if not template or not isinstance(template, str):
                raise RecommendationError("new_strategy requires 'template' string with Python code")
            self._validate_ast(template)
            if name in allowed_strategy_names():
                raise RecommendationError(
                    f"strategy {name!r} already exists; use edit_strategy"
                )

            indicator_deps = content.get("indicator_deps", [])
            if indicator_deps and not isinstance(indicator_deps, list):
                raise RecommendationError("indicator_deps must be a list of indicator function names")

            param_names = [p["name"] for p in content.get("param_schema", [])] if content.get("param_schema") else []
            if param_names:
                for param in param_names:
                    placeholder = f"{{{{{param}}}}}"
                    if placeholder not in template:
                        log.warning("Parameter %s placeholder not found in template", param)
        elif kind == "edit_strategy":
            if not is_plugin(name):
                raise RecommendationError(
                    f"edit_strategy requires an existing plugin strategy; {name!r} is not editable this way"
                )
            template = content.get("template", "")
            if not template or not isinstance(template, str):
                raise RecommendationError("edit_strategy requires 'template' string with Python code")
            self._validate_ast(template)
        else:
            if name not in allowed_strategy_names():
                raise RecommendationError(f"unknown strategy_name {name!r}")

        # For a plain hypothesis there is no params change.
        content_json_data = {
            "params": params,
            "position_pct": content.get("position_pct"),
        }

        # Include extended fields for new/edited strategies
        if kind in ("new_strategy", "edit_strategy"):
            content_json_data.update({
                "template": content.get("template", ""),
                "indicator_deps": content.get("indicator_deps", []),
                "param_schema": content.get("param_schema", []),
                "test_template": content.get("test_template", ""),
            })

        return {
            "kind": kind,
            "strategy_name": name,
            "content_json": json.dumps(content_json_data),
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

        log.info("Created AI recommendation %s (%s)", rec.id, rec.kind)
        return rec
