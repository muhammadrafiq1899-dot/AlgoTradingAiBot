"""Chat orchestrator tests: tool-using LLM agent with a fake client.

Covers the safety contract: the agent may only read state, backtest, and
create PENDING proposals — it never applies anything.
"""
import json
import random

import pytest

from algotrading.ai.client import RecommendationError
from algotrading.config import AIConfig, MarketConfig, Settings
from algotrading.db import get_session_factory, init_db
from algotrading.db.models import AIRecommendation, Strategy
from algotrading.market.base import Candle
from algotrading.market.candles import CandleStore
from algotrading.strategy.plugins import (
    DEFAULT_PLUGIN_DIR,
    configure_default_loader,
    get_default_loader,
)
from algotrading.telegram.chat import build_system_prompt, run_agent, strategy_catalog

# Minimal plugin used to prove the catalog is registry-driven, not hard-coded.
PLUGIN_CODE = '''
class MomentumNudge:
    """Buy when the last close ticks up."""

    def __init__(self, params=None):
        self.params = params or {}
        self.lookback = int(self.params.get("lookback", 3))

    def evaluate(self, symbol, candles):
        return None
'''


@pytest.fixture()
def plugin_dir(tmp_path):
    """Point the plugin loader at a temp dir, restoring it afterwards."""
    directory = tmp_path / "strategies"
    configure_default_loader([directory])
    yield directory
    configure_default_loader([DEFAULT_PLUGIN_DIR])


class FakeClient:
    """Scripted LLM: returns the queued JSON responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    @property
    def enabled(self):
        return True

    def complete_json(self, messages, temperature=None):
        self.calls.append(messages)
        if not self._responses:
            raise RecommendationError("no more scripted responses")
        return self._responses.pop(0)


@pytest.fixture()
def session_factory(tmp_path):
    db = str(tmp_path / "chat.db")
    init_db(db)
    sf = get_session_factory(db)
    with sf() as session:
        session.add(
            Strategy(name="ema_crossover", version=1, status="active",
                     params=json.dumps({"fast_period": 12, "slow_period": 26,
                                        "position_pct": 0.2}))
        )
        # ~100 candles so backtest has data to replay.
        rng = random.Random(7)
        price = 100.0
        candles = []
        for i in range(101):
            price *= 1 + (0.012 if i > 90 else -0.004) + rng.gauss(0, 0.002)
            candles.append(Candle("BTC/USDT", "1h", i * 3_600_000,
                                  price, price, price, price, 1.0))
        CandleStore(session).upsert(candles)
    return sf


def _settings():
    return Settings(
        ai=AIConfig(enabled=True, api_key="x", base_url="http://fake", model="m"),
        market=MarketConfig(symbols=["BTC/USDT"], intervals=["1h"]),
    )


def test_agent_plain_reply(session_factory):
    client = FakeClient([{"action": "reply", "text": "hello there"}])
    result = run_agent(session_factory, _settings(), "hi", client=client)
    assert result["text"] == "hello there"
    assert len(client.calls) == 1


def test_agent_backtest_tool_flow(session_factory):
    client = FakeClient([
        {"action": "tool", "name": "backtest",
         "args": {"strategy_name": "ema_crossover",
                  "params": {"fast_period": 5, "slow_period": 20, "position_pct": 0.2}}},
        {"action": "reply", "text": "done"},
    ])
    result = run_agent(session_factory, _settings(), "backtest ema 5/20", client=client)
    assert result["text"] == "done"
    # The tool result must have been fed back to the LLM.
    assert "Backtest ema_crossover" in client.calls[1][-1]["content"]


def test_agent_propose_change_creates_pending(session_factory):
    client = FakeClient([
        {"action": "tool", "name": "propose_change",
         "args": {"kind": "param_change", "strategy_name": "ema_crossover",
                  "params": {"fast_period": 8, "slow_period": 24}, "rationale": "test"}},
        {"action": "reply", "text": "proposal sent"},
    ])
    result = run_agent(session_factory, _settings(), "tune the strategy", client=client)
    assert result["text"] == "proposal sent"
    assert result.get("proposal_id") is not None
    with session_factory() as session:
        rec = session.get(AIRecommendation, result["proposal_id"])
        assert rec is not None
        assert rec.status == "pending"          # NEVER applied directly
        assert rec.kind == "param_change"
        assert json.loads(rec.content_json)["params"]["fast_period"] == 8


def test_agent_bad_proposal_is_feedback_not_crash(session_factory):
    client = FakeClient([
        {"action": "tool", "name": "propose_change",
         "args": {"kind": "bogus", "strategy_name": "ema_crossover",
                  "params": {"fast_period": 8}, "rationale": "x"}},
        {"action": "reply", "text": "sorry"},
    ])
    result = run_agent(session_factory, _settings(), "change something", client=client)
    assert result["text"] == "sorry"
    assert "tool propose_change failed" in client.calls[1][-1]["content"]
    with session_factory() as session:
        assert session.query(AIRecommendation).count() == 0  # nothing persisted


def test_agent_unknown_tool_feedback(session_factory):
    client = FakeClient([
        {"action": "tool", "name": "nuke_everything", "args": {}},
        {"action": "reply", "text": "ok"},
    ])
    result = run_agent(session_factory, _settings(), "do it", client=client)
    assert result["text"] == "ok"
    assert "unknown tool" in client.calls[1][-1]["content"]


def test_agent_ai_disabled(session_factory):
    settings = Settings(ai=AIConfig(enabled=False))
    result = run_agent(session_factory, settings, "hi", client=FakeClient([]))
    assert "not configured" in result["text"]


def test_agent_llm_failure_is_friendly(session_factory):
    client = FakeClient([])  # complete_json raises immediately
    result = run_agent(session_factory, _settings(), "hi", client=client)
    assert "LLM call failed" in result["text"]


# --- dynamic strategy catalog -------------------------------------------------

def test_strategy_catalog_lists_builtin_schemas():
    text = strategy_catalog()
    assert "- ema_crossover [built-in]" in text
    assert "fast_period (int, default 12, range 2-200)" in text
    assert "- rsi_mean_reversion [built-in]" in text
    # str/list params must render too (multi_tf_ema / ensemble).
    assert "trend_interval (str, default '1h')" in text
    assert "components (list)" in text


def test_strategy_catalog_lists_plugins(plugin_dir):
    get_default_loader().write_strategy(
        "momentum_nudge",
        PLUGIN_CODE,
        description="Test plugin",
        params=[{"name": "lookback", "type": "int", "default": 3, "min": 1, "max": 50}],
    )
    text = strategy_catalog()
    assert "- momentum_nudge [plugin]" in text
    assert "lookback (int, default 3, range 1-50)" in text
    assert "Test plugin" in text


def test_system_prompt_substitutes_catalog(plugin_dir):
    get_default_loader().write_strategy("momentum_nudge", PLUGIN_CODE, params=[])
    prompt = build_system_prompt()
    assert "momentum_nudge" in prompt
    assert "ema_crossover" in prompt
    assert "{{STRATEGY_CATALOG}}" not in prompt


def test_agent_sends_dynamic_catalog(session_factory, plugin_dir):
    get_default_loader().write_strategy("momentum_nudge", PLUGIN_CODE, params=[])
    client = FakeClient([{"action": "reply", "text": "ok"}])
    run_agent(session_factory, _settings(), "hi", client=client)

    system = client.calls[0][0]
    assert system["role"] == "system"
    assert "momentum_nudge" in system["content"]