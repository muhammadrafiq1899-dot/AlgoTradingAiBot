"""Hermes Agent provider tests: CLI invocation, stream-json parsing, gating.

No network and no real `hermes` process: subprocess.run is monkeypatched, and
the client's availability probe is stubbed. Covers the two failure modes that
matter — a text parse that would return the *prompt's* JSON example, and a
nested Hermes agent that is not locked down to a terminal-free toolset.
"""
import json
import subprocess
import types

import pytest

from algotrading.ai.client import RecommendationError
from algotrading.config import AIConfig, Settings, load_settings, validate_settings
from algotrading.db import get_session_factory, init_db
from algotrading.hermes_ai.client import HermesAgentClient, parse_stream_json
from algotrading.telegram import chat as chat_mod
from algotrading.telegram.chat import run_agent

# What `hermes chat -q` prints WITHOUT -Q / stream-json: the query is echoed
# back (including any JSON example the caller sent) and the answer is framed.
NOISY_STDOUT = """Query: Respond with ONLY {"action":"reply","text":"pong"}
Initializing agent...
------------------------------------------------------
{"action":"reply","text":"pong-from-echo"}
"""


def _result_event(text: str) -> str:
    return (
        '{"type": "system", "subtype": "init", "model": "m", "session_id": "s"}\n'
        '{"type": "text", "text": "{"}\n'
        '{"type": "text", "text": "}"}\n'
        + '{"type": "result", "session_id": "s", "exit_code": 0, "text": '
        + json.dumps(text)
        + "}\n"
    )


class RecordingRun:
    """subprocess.run stand-in that records the command and returns canned IO."""

    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        return types.SimpleNamespace(
            stdout=self.stdout, stderr=self.stderr, returncode=self.returncode
        )


@pytest.fixture()
def session_factory(tmp_path):
    db = str(tmp_path / "hermes.db")
    init_db(db)
    return get_session_factory(db)


@pytest.fixture()
def cli_ok(monkeypatch):
    """Pretend the hermes CLI is installed; return the recorder factory."""
    monkeypatch.setattr(HermesAgentClient, "_check_hermes_available", lambda self: True)
    return RecordingRun


# --- stream-json parsing ----------------------------------------------------

def test_parse_stream_json_prefers_result_event_over_echoed_prompt():
    # The prompt's own JSON example appears in the output; the answer must win.
    assert parse_stream_json(_result_event('{"action":"reply","text":"pong"}')) == (
        '{"action":"reply","text":"pong"}'
    )
    assert parse_stream_json(NOISY_STDOUT + _result_event('{"action":"reply"}')) == (
        '{"action":"reply"}'
    )


def test_parse_stream_json_falls_back_to_text_events():
    stream = '{"type": "text", "text": "{\\"action\\":"}\n{"type": "text", "text": "\\"reply\\"}"}\n'
    assert parse_stream_json(stream) == '{"action":"reply"}'


def test_parse_stream_json_ignores_banner_noise():
    assert parse_stream_json("Initializing agent...\n\nResume this session with:\n") is None
    assert parse_stream_json("") is None


# --- CLI invocation ---------------------------------------------------------

def test_complete_json_uses_stream_json_and_terminal_free_toolset(cli_ok, monkeypatch):
    run = cli_ok(_result_event('{"action":"reply","text":"hello"}'))
    monkeypatch.setattr("subprocess.run", run)
    client = HermesAgentClient()
    out = client.complete_json([{"role": "user", "content": "hi"}])

    assert out == {"action": "reply", "text": "hello"}
    cmd = run.calls[-1]
    assert cmd[1:3] == ["chat", "-q"]
    assert "--format" in cmd and cmd[cmd.index("--format") + 1] == "stream-json"
    # Advisory-only: no terminal/file tools in the nested agent.
    assert cmd[cmd.index("-t") + 1] == "safe"
    assert "--ignore-rules" in cmd
    assert "--source" in cmd and cmd[cmd.index("--source") + 1] == "tool"


def test_complete_json_propagates_timeout_as_recommendation_error(cli_ok, monkeypatch):
    def boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd="hermes", timeout=1)

    monkeypatch.setattr("subprocess.run", boom)
    with pytest.raises(RecommendationError):
        HermesAgentClient().complete_json([{"role": "user", "content": "hi"}])


def test_complete_json_raises_on_non_json_answer(cli_ok, monkeypatch):
    monkeypatch.setattr("subprocess.run", cli_ok(_result_event("I cannot help with that")))
    with pytest.raises(RecommendationError):
        HermesAgentClient().complete_json([{"role": "user", "content": "hi"}])


def test_complete_json_raises_on_empty_output(cli_ok, monkeypatch):
    monkeypatch.setattr("subprocess.run", cli_ok("", stderr="boom", returncode=1))
    with pytest.raises(RecommendationError):
        HermesAgentClient().complete_json([{"role": "user", "content": "hi"}])


def test_client_disabled_when_cli_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    client = HermesAgentClient()
    assert client.enabled is False
    with pytest.raises(RecommendationError):
        client.complete_json([{"role": "user", "content": "hi"}])


# --- provider gating in config ---------------------------------------------

def test_use_hermes_enables_assistant_without_api_key(monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    monkeypatch.setenv("USE_HERMES", "true")
    load_settings.cache_clear()
    try:
        settings = load_settings()
        assert settings.ai.use_hermes is True
        assert settings.ai.enabled is True          # <- the reported bug
        assert settings.ai.api_key == ""
        validate_settings(settings)                 # must not demand an api_key
    finally:
        load_settings.cache_clear()

    monkeypatch.setenv("USE_HERMES", "false")
    load_settings.cache_clear()
    try:
        settings = load_settings()
        assert settings.ai.use_hermes is False
        assert settings.ai.enabled is False
    finally:
        load_settings.cache_clear()


# --- chat orchestrator provider selection ----------------------------------

class StubHermes:
    """Stand-in for HermesAgentClient: records construction, replays an answer."""

    def __init__(self, response=None, enabled=True):
        StubHermes.last = self
        self._response = response
        self.enabled = enabled

    def complete_json(self, messages, temperature=None):
        if self._response is None:
            raise RecommendationError("Hermes CLI is not available or not enabled")
        return self._response


def _hermes_settings():
    return Settings(ai=AIConfig(enabled=True, use_hermes=True))


def test_run_agent_uses_hermes_when_use_hermes_is_set(session_factory, monkeypatch):
    monkeypatch.setattr(chat_mod, "HERMES_AI_AVAILABLE", True)
    monkeypatch.setattr(
        chat_mod, "HermesAgentClient",
        lambda *a, **k: StubHermes({"action": "reply", "text": "from hermes"}),
    )
    result = run_agent(session_factory, _hermes_settings(), "hi")
    assert result["text"] == "from hermes"


def test_run_agent_reports_missing_hermes_cli(session_factory, monkeypatch):
    monkeypatch.setattr(chat_mod, "HERMES_AI_AVAILABLE", True)
    monkeypatch.setattr(chat_mod, "HermesAgentClient", lambda *a, **k: StubHermes(enabled=False))
    result = run_agent(session_factory, _hermes_settings(), "hi")
    assert "PATH" in result["text"]


def test_run_agent_reports_unconfigured_when_no_provider(session_factory):
    settings = Settings(ai=AIConfig(enabled=False, use_hermes=False))
    result = run_agent(session_factory, settings, "hi")
    assert "not configured" in result["text"]
