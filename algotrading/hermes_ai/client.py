"""Hermes Agent LLM client for AlgoTradingAiBot's advisory components.

Drop-in replacement for :class:`algotrading.ai.client.AIClient` that talks to a
locally installed Hermes Agent CLI (``hermes chat -q``) instead of an external
OpenAI-compatible endpoint. Selected with ``USE_HERMES=true`` in ``.env``.

Why the invocation looks the way it does:

* ``--format stream-json`` — plain ``hermes chat -q`` output is human-facing: it
  echoes the query back and frames the answer in box drawing, so a naive
  "first JSON object in the output" parse returns the *prompt's* own JSON
  example. The final ``{"type": "result", ..., "text": ...}`` event carries the
  model's answer verbatim, so that is what we read.
* ``-t safe`` — the nested agent runs with Hermes' terminal-free toolset. The AI
  is advisory only (PROJECT_MAP invariant #3): it must never be able to run
  commands or edit files, which would bypass the human approval flow.
* ``--ignore-rules`` — no AGENTS.md/memory/skill injection into a JSON-only reply.
* ``--max-turns 1`` — the real tool loop (get_status/backtest/propose_change)
  lives in ``algotrading/telegram/chat.py``; this call must answer, not act.
* ``--source tool`` — keeps bot traffic out of the user's interactive session list.

Errors surface as :class:`algotrading.ai.client.RecommendationError` (the same
type the external client raises) so all callers handle one error type.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, List

from algotrading.ai.client import RecommendationError, extract_json

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120.0
# Hermes toolset with no terminal/file access — see module docstring.
SAFE_TOOLSET = "safe"


def _binary() -> str:
    """The hermes executable to run (override with HERMES_CLI)."""
    return os.getenv("HERMES_CLI", "").strip() or "hermes"


def _toolset() -> str:
    """Toolset for the advisory call (override with HERMES_TOOLSETS)."""
    return os.getenv("HERMES_TOOLSETS", "").strip() or SAFE_TOOLSET


def _default_timeout() -> float:
    """Per-call timeout in seconds (override with HERMES_TIMEOUT)."""
    raw = os.getenv("HERMES_TIMEOUT", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_TIMEOUT
    except ValueError:
        log.warning("Ignoring invalid HERMES_TIMEOUT=%r", raw)
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


def parse_stream_json(stdout: str) -> str | None:
    """Return the final answer text from ``--format stream-json`` output.

    Hermes emits newline-delimited JSON events: per-token ``{"type": "text"}``
    events and one final ``{"type": "result", "text": ...}`` event. We prefer
    the result event; if it is missing (older CLI, truncated stream) we fall
    back to concatenating the text events. Returns None when there is nothing
    usable, so the caller can report a clear error.
    """
    result_text: str | None = None
    streamed: list[str] = []
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue  # banner / spinner noise never reaches stdout with -Q, but be safe
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        text = event.get("text")
        if not isinstance(text, str):
            continue
        if event.get("type") == "result":
            if text.strip():
                result_text = text
        elif event.get("type") == "text":
            streamed.append(text)
    if result_text is not None:
        return result_text
    joined = "".join(streamed).strip()
    return joined or None


class HermesAgentClient:
    """AIClient-compatible client backed by the local ``hermes`` CLI."""

    def __init__(self, config: Any = None, timeout: float | None = None,
                 toolsets: str | None = None):
        """Initialize the client.

        Args:
            config: unused, kept for AIClient API compatibility.
            timeout: per-call timeout in seconds (default HERMES_TIMEOUT / 120).
            toolsets: Hermes toolset(s) for the call (default HERMES_TOOLSETS
                / "safe" — terminal-free).
        """
        self._timeout = timeout if timeout is not None else _default_timeout()
        self._toolsets = toolsets or _toolset()
        self._binary = shutil.which(_binary()) or _binary()
        self._available = self._check_hermes_available()
        self._enabled = self._available

    def _check_hermes_available(self) -> bool:
        """True when the hermes CLI is on PATH and responds to --version."""
        if shutil.which(self._binary) is None:
            log.warning("hermes CLI not found on PATH")
            return False
        try:
            result = subprocess.run(
                [self._binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    @property
    def enabled(self) -> bool:
        """True when the client is ready to use."""
        return self._enabled

    def _command(self, prompt: str) -> list[str]:
        """The full hermes invocation for one advisory call."""
        cmd = [
            self._binary,
            "chat",
            "-q", prompt,
            "--format", "stream-json",
            "-t", self._toolsets,
            "--source", "tool",
            "--ignore-rules",
            "--max-turns", "1",
        ]
        # Ask Hermes to wrap up before our own hard kill, so we get a reply
        # (and a parseable error) instead of a SIGKILL with no output.
        budget = int(self._timeout - 15)
        if budget >= 30:
            cmd += ["--run-budget", str(budget)]
        return cmd

    def complete_json(
        self, messages: List[Dict[str, str]], temperature: float | None = None
    ) -> Dict[str, Any]:
        """Run one query through Hermes and return the parsed JSON answer.

        Args:
            messages: OpenAI-style messages, flattened into a single prompt.
            temperature: unused, kept for AIClient API compatibility.

        Raises:
            RecommendationError: the CLI is missing, times out, exits without an
                answer, or the answer is not a JSON object.
        """
        if not self.enabled:
            raise RecommendationError("Hermes CLI is not available or not enabled")

        prompt = self._messages_to_prompt(messages)
        try:
            result = subprocess.run(
                self._command(prompt),
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as exc:
            log.warning("Hermes Agent request timed out: %s", exc)
            raise RecommendationError(
                f"Hermes Agent request timed out after {self._timeout:.0f}s"
            ) from exc
        except (FileNotFoundError, OSError) as exc:
            log.warning("Hermes Agent could not be started: %s", exc)
            raise RecommendationError(f"Could not run the hermes CLI: {exc}") from exc

        answer = parse_stream_json(result.stdout)
        if answer is None:
            detail = " / ".join(
                (result.stderr or result.stdout or "").strip().splitlines()[-3:]
            )
            raise RecommendationError(
                f"Hermes CLI returned no answer (exit {result.returncode})"
                + (f": {detail}" if detail else "")
            )
        try:
            return extract_json(answer)
        except RecommendationError as exc:
            log.warning("Hermes Agent did not return JSON: %s", answer[:200])
            raise RecommendationError(
                f"Hermes Agent did not return JSON: {answer[:200]}"
            ) from exc

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Flatten OpenAI-style messages into one prompt for ``hermes chat -q``."""
        prompt_parts = []
        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role.title()}: {content}")
        return "\n\n".join(prompt_parts)
