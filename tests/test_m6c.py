"""M6c: recommendation store + apply(), assistant propose/validate, versioned release."""
import json

import pytest
from sqlalchemy import select

from algotrading.ai.assistant import Assistant
from algotrading.ai.client import AIClient, RecommendationError
from algotrading.backtest.runner import run_backtest
from algotrading.db import init_db, get_session_factory
from algotrading.db.models import AIRecommendation, Strategy
from algotrading.market.base import Candle
from algotrading.store.recommendations import RecommendationStore
from algotrading.store.strategy_versions import (
    active_version,
    create_new_version,
    latest_version,
    promote_to_active,
)


@pytest.fixture()
def session(tmp_path):
    init_db(str(tmp_path / "m6c.db"))
    sess = get_session_factory(str(tmp_path / "m6c.db"))()
    # Seed a baseline active strategy so controlled release has something to retire.
    sess.add(Strategy(name="ema_crossover", version=1, status="active",
                      params=json.dumps({"fast_period": 12, "slow_period": 26, "position_pct": 0.2})))
    sess.commit()
    yield sess
    sess.close()


def _candles(prices):
    return [
        Candle(symbol="BTC/USDT", interval="1h", ts=1_600_000_000_000 + i * 3_600_000,
               open=p, high=p, low=p, close=p, volume=1.0)
        for i, p in enumerate(prices)
    ]


def _rec(session, kind="param_change", name="ema_crossover", params=None,
         rationale="tune"):
    content = json.dumps({"params": params or {"fast_period": 10, "slow_period": 26},
                          "position_pct": 0.2})
    r = AIRecommendation(kind=kind, strategy_name=name,
                         content_json=content, status="pending", rationale=rationale)
    session.add(r)
    session.commit()
    return r


def test_create_and_promote_version(session):
    v2 = create_new_version(session, "ema_crossover", {"fast_period": 10, "slow_period": 26})
    assert v2.version == 2
    assert v2.status == "draft"

    promote_to_active(session, v2)
    v1 = session.execute(select(Strategy).where(Strategy.version == 1)).scalars().one()
    assert v1.status == "retired"
    assert active_version(session, "ema_crossover").id == v2.id


def test_latest_version(session):
    assert latest_version(session, "ema_crossover").version == 1
    create_new_version(session, "ema_crossover", {})
    assert latest_version(session, "ema_crossover").version == 2


def test_apply_runs_backtest_and_promotes(session):
    store = RecommendationStore(session)
    rec = _rec(session)
    # Slow period is 26, so we need enough candles for a clean golden cross.
    # Flat base, then a long monotonic climb (fast EMA crosses above slow),
    # then a strong down move (death cross) — one winning round trip.
    prices = (
        [100.0] * 30
        + list(range(100, 200, 2))
        + [198, 190, 175, 155, 135, 115, 95, 80]
    )
    strat, result = store.apply(rec, candles=_candles(prices))

    assert strat is not None
    assert strat.version == 2
    assert strat.status == "active"
    assert result is not None
    assert result.n_trades >= 1

    session.refresh(rec)
    assert rec.status == "applied"
    assert json.loads(rec.backtest_json)["n_trades"] == result.n_trades
    assert active_version(session, "ema_crossover").id == strat.id


def test_apply_non_pending_raises(session):
    store = RecommendationStore(session)
    rec = _rec(session)
    store.mark_rejected(rec)
    with pytest.raises(ValueError):
        store.apply(rec, candles=_candles([100.0] * 30))


def test_apply_threshold_auto_rejects(session):
    store = RecommendationStore(session)
    rec = _rec(session)
    strat, result = store.apply(
        rec, candles=_candles([100.0] * 40),
        min_profit_threshold=1_000_000.0,  # impossible to reach
    )
    assert strat is None
    session.refresh(rec)
    assert rec.status == "rejected"


def test_pending_and_reject(session):
    store = RecommendationStore(session)
    r1 = _rec(session, kind="hypothesis", params={})
    r2 = _rec(session, kind="param_change", params={"fast_period": 9})
    assert len(store.pending()) == 2

    store.mark_rejected(r1)
    assert len(store.pending()) == 1
    session.refresh(r1)
    assert r1.status == "rejected"


class _FakeClient:
    def __init__(self, data=None, enabled=True):
        self._data = data
        self.enabled = enabled

    def complete_json(self, messages):
        if not self.enabled:
            raise RecommendationError("disabled")
        return self._data


def test_assistant_proposes_pending(session):
    data = {"kind": "param_change", "strategy_name": "ema_crossover",
            "content": {"params": {"fast_period": 9, "slow_period": 26}},
            "rationale": "faster entry"}
    ast = Assistant(session, client=_FakeClient(data))
    rec = ast.propose("BTC/USDT", "1h", [], _candles([100.0] * 30))
    assert rec is not None
    assert rec.status == "pending"
    assert json.loads(rec.content_json)["params"]["fast_period"] == 9


def test_assistant_disabled_returns_none(session):
    ast = Assistant(session, client=_FakeClient(enabled=False))
    assert ast.propose("BTC/USDT", "1h", [], []) is None


def test_assistant_invalid_kind_returns_none(session):
    ast = Assistant(session, client=_FakeClient({"kind": "nonsense", "strategy_name": "ema_crossover", "content": {}}))
    assert ast.propose("BTC/USDT", "1h", [], []) is None
    assert len(session.execute(select(AIRecommendation)).scalars().all()) == 0


def test_assistant_unknown_strategy_returns_none(session):
    ast = Assistant(session, client=_FakeClient({"kind": "param_change", "strategy_name": "bogus", "content": {}}))
    assert ast.propose("BTC/USDT", "1h", [], []) is None


def test_assistant_accepts_rsi_mean_reversion(session):
    """The registered RSI strategy name must validate (registry name match)."""
    data = {"kind": "param_change", "strategy_name": "rsi_mean_reversion",
            "content": {"params": {"period": 14, "oversold": 30.0, "overbought": 70.0}},
            "rationale": "rsi tune"}
    ast = Assistant(session, client=_FakeClient(data))
    rec = ast.propose("BTC/USDT", "1h", [], [])
    assert rec is not None
    assert rec.status == "pending"
    assert json.loads(rec.content_json)["params"]["period"] == 14
