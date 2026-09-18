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
* ``--image PATH`` — an attached image (a Telegram screenshot of a chart/strategy)
  is handed to the nested agent as a real attachment. Hermes loads it natively
  for the model and exposes its ``vision_analyze`` tool for that turn; that tool
  is injected by the CLI for the attachment and is not a terminal/file tool, so
  the advisory-only guarantee above is unchanged.
* ``--ignore-rules`` — no AGENTS.md/memory/skill injection into a JSON-only reply.
* ``--max-turns 1`` (2 when an image is attached) — the real tool loop
  (get_status/backtest/propose_change) lives in ``algotrading/telegram/chat.py``;
  this call must answer, not act.
* ``--source tool`` — keeps bot traffic out of the user's interactive session list.

Errors surface as :class:`algotrading.ai.client.RecommendationError` (the same
type the external client raises) so all callers handle one error type.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from typing import Any, Dict, List

from algotrading.ai.client import RecommendationError, extract_json

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120.0
# Hermes toolset with no terminal/file access — see module docstring.
SAFE_TOOLSET = "safe"
# Turns allowed for one advisory call. An attached image costs one turn (the
# nested agent calls vision_analyze before it can answer), so image calls get a
# second one; without it the CLI exits non-zero on the exhausted budget.
MAX_TURNS = 1
IMAGE_MAX_TURNS = 2


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


def salvage_partial_reply(answer: str) -> dict[str, Any] | None:
    """Recover a reply object the model left mid-string.

    Long answers occasionally arrive truncated (provider stream cut): the JSON
    object is real but never closes. When the fragment is clearly a ``reply``
    action with a partial ``"text"``, return what was written instead of throwing
    the turn away. Anything else stays an error — guessing at tool arguments
    would be worse than failing.
    """
    text = (answer or "").strip()
    if not text.startswith("{"):
        return None
    marker = text.find('"text"')
    if marker == -1:
        return None
    if not re.search(r'"action"\s*:\s*"reply"', text[:marker]):
        return None

    rest = text[marker + len('"text"'):].lstrip()
    if not rest.startswith(":"):
        return None
    rest = rest[1:].lstrip()
    if not rest.startswith('"'):
        return None

    partial = rest[1:]
    # Drop a dangling escape or closing punctuation the model never finished.
    partial = partial.rstrip()
    while partial.endswith(("\\", '"', "}")):
        partial = partial[:-1].rstrip()
    partial = " ".join(partial.split())
    if not partial:
        return None
    return {"action": "reply", "text": f"{partial} […]"}


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

    def _command(self, prompt: str, image: str | None = None) -> list[str]:
        """The full hermes invocation for one advisory call.

        ``image`` is a local file path handed to the nested agent as an
        attachment (``--image``). It stays inside the same terminal-free
        invocation, so an image cannot escalate the agent's reach.
        """
        cmd = [
            self._binary,
            "chat",
            "-q", prompt,
            "--format", "stream-json",
            "-t", self._toolsets,
            "--source", "tool",
            "--ignore-rules",
            "--max-turns", str(IMAGE_MAX_TURNS if image else MAX_TURNS),
        ]
        if image:
            cmd += ["--image", str(image)]
        # Ask Hermes to wrap up before our own hard kill, so we get a reply
        # (and a parseable error) instead of a SIGKILL with no output.
        budget = int(self._timeout - 15)
        if budget >= 30:
            cmd += ["--run-budget", str(budget)]
        return cmd

    def complete_json(
        self, messages: List[Dict[str, str]], temperature: float | None = None,
        image: str | None = None,
    ) -> Dict[str, Any]:
        """Run one query through Hermes and return the parsed JSON answer.

        Args:
            messages: OpenAI-style messages, flattened into a single prompt.
            temperature: unused, kept for AIClient API compatibility.
            image: optional local image path attached to the query (``--image``).

        Raises:
            RecommendationError: the CLI is missing, times out, exits without an
                answer, the answer is not a JSON object, or the image path is not
                a readable file.
        """
        if not self.enabled:
            raise RecommendationError("Hermes CLI is not available or not enabled")
        if image is not None and not os.path.isfile(image):
            raise RecommendationError(f"image not found: {image}")

        answer = self._ask(messages, image=image)
        try:
            return extract_json(answer)
        except RecommendationError as exc:
            salvaged = salvage_partial_reply(answer)
            if salvaged is not None:
                # Long answers are occasionally cut mid-string by the provider.
                # A partial reply beats discarding the whole turn.
                log.warning("Hermes Agent answer was truncated; salvaging the partial reply")
                return salvaged
            log.warning("Hermes Agent did not return JSON: %s", answer[:200])
            raise RecommendationError(
                f"Hermes Agent did not return JSON: {answer[:200]}"
            ) from exc

    def complete_text(
        self, messages: List[Dict[str, str]], temperature: float | None = None,
        image: str | None = None,
    ) -> str:
        """Run one query and return the answer text verbatim (no JSON required).

        Use where an unstructured or partial answer is still useful — the
        per-image transcription pass, whose long JSON answers sometimes arrive
        truncated. Structured callers keep using :meth:`complete_json`.
        """
        if not self.enabled:
            raise RecommendationError("Hermes CLI is not available or not enabled")
        if image is not None and not os.path.isfile(image):
            raise RecommendationError(f"image not found: {image}")
        return self._ask(messages, image=image)

    def _ask(self, messages: List[Dict[str, str]], image: str | None = None) -> str:
        """Run the CLI once and return the text of its final answer event."""
        prompt = self._messages_to_prompt(messages)
        if image:
            prompt += (
                "\n\n(An image is attached to this message: "
                f"{os.path.basename(image)}. Look at it before answering.)"
            )
        try:
            result = subprocess.run(
                self._command(prompt, image=image),
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
        return answer

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
