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
from algotrading.hermes_ai.client import HermesAgentClient, parse_stream_json, salvage_partial_reply
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
        self.image_kwargs = []

    def complete_json(self, messages, temperature=None, **kwargs):
        self.image_kwargs.append(kwargs)
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


# --- image input ------------------------------------------------------------

def _chart(tmp_path):
    """A real file on disk: the client refuses anything that is not one."""
    path = tmp_path / "chart.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
    return path


def test_image_call_attaches_it_and_raises_the_turn_budget(cli_ok, monkeypatch, tmp_path):
    image = _chart(tmp_path)
    run = cli_ok(_result_event('{"action":"reply","text":"seen"}'))
    monkeypatch.setattr("subprocess.run", run)

    out = HermesAgentClient().complete_json(
        [{"role": "user", "content": "what is this?"}], image=str(image)
    )

    assert out == {"action": "reply", "text": "seen"}
    cmd = run.calls[-1]
    assert cmd[cmd.index("--image") + 1] == str(image)
    # Reading an image costs a turn (vision_analyze), so the budget grows by one.
    assert cmd[cmd.index("--max-turns") + 1] == "2"
    # ...and the advisory-only guarantee is untouched: same terminal-free toolset.
    assert cmd[cmd.index("-t") + 1] == "safe"
    assert "--ignore-rules" in cmd
    prompt = cmd[cmd.index("-q") + 1]
    assert "what is this?" in prompt
    assert "image is attached" in prompt.lower()


def test_text_only_call_keeps_the_old_invocation(cli_ok, monkeypatch):
    run = cli_ok(_result_event('{"action":"reply","text":"ok"}'))
    monkeypatch.setattr("subprocess.run", run)
    HermesAgentClient().complete_json([{"role": "user", "content": "hi"}])

    cmd = run.calls[-1]
    assert "--image" not in cmd
    assert cmd[cmd.index("--max-turns") + 1] == "1"


def test_missing_image_is_refused_before_the_cli_runs(cli_ok, monkeypatch, tmp_path):
    run = cli_ok(_result_event('{"action":"reply"}'))
    monkeypatch.setattr("subprocess.run", run)

    with pytest.raises(RecommendationError, match="image not found"):
        HermesAgentClient().complete_json(
            [{"role": "user", "content": "x"}], image=str(tmp_path / "nope.png")
        )
    assert run.calls == []


def test_run_agent_attaches_the_image_for_the_hermes_provider(
    session_factory, monkeypatch, tmp_path
):
    image = _chart(tmp_path)
    monkeypatch.setattr(chat_mod, "HERMES_AI_AVAILABLE", True)
    monkeypatch.setattr(
        chat_mod, "HermesAgentClient",
        lambda *a, **k: StubHermes({"action": "reply", "text": "seen"}),
    )
    result = run_agent(session_factory, _hermes_settings(), "what is this?",
                       image_path=str(image))

    assert result["text"] == "seen"
    assert StubHermes.last.image_kwargs == [{"image": str(image)}]


def test_run_agent_refuses_images_without_the_hermes_provider(session_factory, tmp_path):
    # API-key provider: no vision path, so the image is refused with a hint
    # rather than silently dropped.
    image = _chart(tmp_path)
    settings = Settings(ai=AIConfig(enabled=True, api_key="k"))
    result = run_agent(session_factory, settings, "what is this?", image_path=str(image))

    assert "USE_HERMES" in result["text"]
    assert "image_path" not in result


def test_run_agent_reports_an_unreadable_image_file(session_factory, monkeypatch, tmp_path):
    monkeypatch.setattr(chat_mod, "HERMES_AI_AVAILABLE", True)
    monkeypatch.setattr(
        chat_mod, "HermesAgentClient",
        lambda *a, **k: StubHermes({"action": "reply", "text": "seen"}),
    )
    StubHermes.last = None       # class-level recorder: clear the previous test's
    result = run_agent(session_factory, _hermes_settings(), "x",
                       image_path=str(tmp_path / "gone.png"))

    assert "again" in result["text"]
    assert StubHermes.last is None, "an unreadable image must not reach the provider"


# --- truncated answers -------------------------------------------------------
# The provider occasionally cuts a long answer mid-string. A partial reply is
# still worth showing; a partial tool call is not (guessing arguments is worse
# than failing), so only "reply" fragments are salvaged.

def test_truncated_reply_is_salvaged():
    cut = '{"action": "reply", "text": "Parts 1-3 describe one strategy: entry on EMA(9) cro'
    assert salvage_partial_reply(cut) == {
        "action": "reply",
        "text": "Parts 1-3 describe one strategy: entry on EMA(9) cro […]",
    }


def test_truncated_tool_call_is_not_salvaged():
    cut = '{"action": "tool", "name": "backtest", "args": {"strategy_name": "ema_crosso'
    assert salvage_partial_reply(cut) is None
    assert salvage_partial_reply("not json at all") is None
    assert salvage_partial_reply('{"action": "reply"}') is None


def test_complete_json_uses_the_salvaged_partial_reply(cli_ok, monkeypatch):
    cut = '{"action": "reply", "text": "the image shows an EMA crossover with an RSI fil'
    monkeypatch.setattr("subprocess.run", cli_ok(_result_event(cut)))
    out = HermesAgentClient().complete_json([{"role": "user", "content": "hi"}])
    assert out["action"] == "reply"
    assert "EMA crossover with an RSI fil" in out["text"]


def test_complete_text_returns_the_answer_verbatim(cli_ok, monkeypatch):
    partial = '{"description": "a chart with two EMAs and no close brace'
    monkeypatch.setattr("subprocess.run", cli_ok(_result_event(partial)))
    text = HermesAgentClient().complete_text(
        [{"role": "user", "content": "transcribe"}], image=None
    )
    assert text == partial      # no JSON requirement, nothing thrown away


def test_describe_pass_keeps_a_truncated_transcription(session_factory, tmp_path):
    """A cut-off transcription must not become "(unreadable)"."""
    images = [str(tmp_path / "a.png"), str(tmp_path / "b.png")]
    for path in images:
        (tmp_path / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 16)
    (tmp_path / "b.png").write_bytes((tmp_path / "a.png").read_bytes())
    partial = '{"description": "ENTRY: EMA(9) crosses above EMA(21) on the 1h chart. FILTER: RSI'
    seen = {}

    class CutOff:
        """complete_text hands back a JSON object the model never closed."""

        enabled = True

        def complete_text(self, messages, temperature=None, image=None):
            return partial

        def complete_json(self, messages, temperature=None, **kwargs):
            seen["user_turn"] = messages[1]["content"]
            return {"action": "reply", "text": "merged"}

    result = run_agent(session_factory, _hermes_settings(), "read these",
                       client=CutOff(), image_paths=images)

    assert result["text"] == "merged"
    assert "unreadable" not in seen["user_turn"]
    assert "ENTRY: EMA(9) crosses above EMA(21)" in seen["user_turn"]
