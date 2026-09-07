"""OpenAI-compatible cloud LLM client.

A thin HTTP wrapper over the standard `/chat/completions` JSON API using
`requests`. It deliberately does NOT depend on the `openai` SDK so the Termux
footprint stays small and any OpenAI-compatible endpoint (base_url) can be
used. Output is validated as JSON so downstream schema validation can run.

No network calls are made in tests — callers inject a transport or we return
the parsed JSON from a provided `response` object.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests

from algotrading.config import AIConfig

log = logging.getLogger(__name__)


class RecommendationError(RuntimeError):
    """Raised when the LLM response cannot be parsed or is missing content."""


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from an LLM reply.

    LLMs often wrap JSON in ```json ... ``` fences or add trailing prose. We
    try a strict parse first, then strip fences and take the first balanced
    `{...}` block.
    """
    text = (text or "").strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass

    # Strip ```json / ``` fences.
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        data = json.loads(fenced)
        if isinstance(data, dict):
            return data
    except ValueError:
        pass

    # Fall back to the first balanced {...} block.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict):
                            return data
                    except ValueError:
                        continue
    raise RecommendationError("no JSON object found in LLM response")


class AIClient:
    """Client for an OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, config: AIConfig | None = None, timeout: float = 60.0):
        self._cfg = config or AIConfig()
        self._timeout = timeout
        self._session = requests.Session()

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled and bool(self._cfg.api_key)

    def complete_json(self, messages: list[dict[str, str]],
                      temperature: float | None = None) -> dict[str, Any]:
        """Send a chat completion and return the parsed JSON payload.

        Raises:
            RecommendationError if the call fails, the HTTP status is not 2xx,
            or the response body does not contain parseable JSON.
        """
        if not self.enabled:
            raise RecommendationError("AI is disabled (no api_key configured)")

        url = self._cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self._cfg.model,
            "messages": messages,
            "temperature": self._cfg.temperature if temperature is None else temperature,
        }

        try:
            resp = self._session.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self._cfg.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            log.warning("LLM request failed: %s", exc)
            raise RecommendationError(f"LLM request failed: {exc}") from exc

        if resp.status_code >= 400:
            log.warning("LLM HTTP %s: %s", resp.status_code, resp.text[:200])
            raise RecommendationError(f"LLM HTTP {resp.status_code}")

        try:
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RecommendationError(f"unexpected LLM response shape: {exc}") from exc

        return extract_json(content)
