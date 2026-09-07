"""AI client parsing + prompt builder determinism."""
import pytest

from algotrading.ai.client import AIClient, RecommendationError, extract_json
from algotrading.ai.prompt_builder import build_feature_window, build_prompt
from algotrading.config import AIConfig
from algotrading.market.base import Candle


def _candles(n=30, start=100.0):
    return [
        Candle(
            symbol="BTC/USDT", interval="1h",
            ts=1_600_000_000_000 + i * 3_600_000,
            open=start + i, high=start + i + 1, low=start + i - 1,
            close=start + i, volume=10.0 + i,
        )
        for i in range(n)
    ]


def test_extract_json_fenced():
    assert extract_json("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_extract_json_plain():
    assert extract_json('{"b": 2}') == {"b": 2}


def test_extract_json_with_prose():
    assert extract_json('Here you go: {"c": 3}\n\nHope that helps') == {"c": 3}


def test_extract_json_balanced_block():
    assert extract_json('prefix {"d": {"e": 4}} suffix') == {"d": {"e": 4}}


def test_extract_json_invalid_raises():
    with pytest.raises(RecommendationError):
        extract_json("no json here at all")


def test_client_disabled_raises():
    cfg = AIConfig(enabled=False, api_key="")
    with pytest.raises(RecommendationError):
        AIClient(cfg).complete_json([{"role": "user", "content": "hi"}])


def test_client_enabled_needs_key():
    # enabled but no key -> treated as disabled
    cfg = AIConfig(enabled=True, api_key="")
    assert not AIClient(cfg).enabled
    with pytest.raises(RecommendationError):
        AIClient(cfg).complete_json([{"role": "user", "content": "hi"}])


def test_prompt_builder_deterministic():
    p1 = build_prompt("BTC/USDT", "1h", [], _candles(), "ema_crossover",
                      {"fast_period": 12, "slow_period": 26})
    p2 = build_prompt("BTC/USDT", "1h", [], _candles(), "ema_crossover",
                      {"fast_period": 12, "slow_period": 26})
    assert p1 == p2


def test_prompt_builder_content():
    msgs = build_prompt("BTC/USDT", "1h", [], _candles(30), "ema_crossover",
                        {"fast_period": 12, "slow_period": 26})
    assert msgs[0]["role"] == "system"
    assert "ema_crossover" in msgs[1]["content"]
    assert "fast_period" in msgs[1]["content"]
    # 30 candles -> feature window excludes latest, caps at max_rows
    assert len(build_feature_window(_candles(30))) == 24
