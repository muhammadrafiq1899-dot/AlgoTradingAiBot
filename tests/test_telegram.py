"""Telegram layer tests: auth allowlist, /status formatting, approval flow.

The approval tests wire real on_approve/on_reject callbacks (built the same way
`bot.build_application` does) into build_handlers and drive the inline
CallbackQueryHandler end-to-end against a stubbed query. No network is used.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from algotrading.config import RiskConfig
from algotrading.db import init_db, get_session_factory
from algotrading.db.models import (
    AIRecommendation,
    Candle,
    Position,
    Strategy,
    Trade,
    TradeIntent,
)
from algotrading.telegram.bot import _make_approve, _make_reject
from algotrading.telegram.commands import build_handlers
from algotrading.telegram.ui import (
    approval_keyboard,
    format_risk,
    format_status,
)


@pytest.fixture()
def session(tmp_path):
    db = str(tmp_path / "tg.db")
    init_db(db)
    sess = get_session_factory(db)()
    sess.add(
        Strategy(name="ema_crossover", version=1, status="active",
                 params=json.dumps({"fast_period": 12, "slow_period": 26, "position_pct": 0.2}))
    )
    sess.add(Position(symbol="BTC/USDT", qty=0.25, avg_price=50000.0))
    sess.commit()
    yield sess
    sess.close()


class FakeMessage:
    def __init__(self):
        self._sent = []

    async def reply_text(self, text, parse_mode=None):
        self._sent.append((text, parse_mode))

    async def edit_message_text(self, text, parse_mode=None):
        self._sent.append((text, parse_mode))


class FakeUser:
    def __init__(self, id):
        self.id = id


class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.answered = False
        self.message = FakeMessage()

    async def answer(self):
        self.answered = True

    async def edit_message_text(self, text, parse_mode=None):
        self.message._sent.append((text, parse_mode))


class FakeUpdate:
    def __init__(self, user_id, message=None, callback_query=None):
        self.effective_user = FakeUser(user_id)
        self.effective_message = message or FakeMessage()
        self.callback_query = callback_query


async def _run(handler, update, context=None):
    """Invoke the wrapped handler against a stub update."""
    # build_handlers returns PTB Handler objects; call .callback directly.
    await handler.callback(update, context or AsyncMock())


def _handlers(session, allowed=(1, 2)):
    settings = type("S", (), {"mode": "paper"})()
    risk = RiskConfig()
    return {
        next(iter(h.commands)): h
        for h in build_handlers(
            session=session,
            settings=settings,
            risk_cfg=risk,
            allowed_users=allowed,
        )
        if hasattr(h, "commands")
    }


def test_status_formatting(session):
    settings = type("S", (), {"mode": "paper"})()
    strategies = session.execute(select(Strategy)).scalars().all()
    positions = session.execute(select(Position)).scalars().all()
    intents = session.execute(select(TradeIntent)).scalars().all()
    trades = session.execute(select(Trade)).scalars().all()
    out = format_status(settings, session, strategies, positions, intents, trades,
                        prices={"BTC/USDT": 51000.0})
    assert "ema_crossover" in out
    assert "v1" in out
    assert "BTC/USDT" in out
    assert "0.25" in out
    assert "mode: paper" in out


def test_status_no_positions(session):
    session.query(Position).delete()
    session.commit()
    settings = type("S", (), {"mode": "paper"})()
    out = format_status(settings, session, [], [], [], [])
    assert "No open positions" in out


def test_risk_formatting(session):
    settings = type("S", (), {"mode": "paper"})()
    out = format_risk(settings, RiskConfig(risk_per_trade_pct=1.0, max_position_pct=20.0,
                                           max_open_positions=3, cooldown_seconds=300,
                                           slippage_pct=0.05), session)
    assert "1.0%" in out
    assert "20.0%" in out
    assert "3" in out
    assert "300s" in out
    assert "Open positions now: 1" in out


def test_approval_keyboard():
    kb = approval_keyboard(42)
    rows = kb.to_dict()["inline_keyboard"]
    assert [b["callback_data"] for b in rows[0]] == ["approve:42", "reject:42"]


def test_allowlisted_user_can_status(session):
    handlers = _handlers(session, allowed=(1,))
    update = FakeUpdate(user_id=1)
    asyncio.run(_run(handlers["status"], update))
    assert update.effective_message._sent, "allowed user should get a reply"


def test_outsider_gets_no_reply(session):
    handlers = _handlers(session, allowed=(1,))
    update = FakeUpdate(user_id=999)
    asyncio.run(_run(handlers["status"], update))
    assert not update.effective_message._sent, "outsider must be silently ignored"


# --- M6d: inline approve/reject drives the recommendation store ---


def _fake_settings():
    """Minimal settings stub mirroring the fields bot.py consumes."""
    market = type("M", (), {
        "symbols": ["BTC/USDT"],
        "intervals": ["1h", "1m"],
    })
    return type("S", (), {"market": market})()


def _seed_candles(session):
    """A clean golden-cross then death-cross series for a winning round trip."""
    prices = (
        [100.0] * 30
        + list(range(100, 200, 2))
        + [198, 190, 175, 155, 135, 115, 95, 80]
    )
    candles = [
        Candle(symbol="BTC/USDT", interval="1h", ts=1_600_000_000_000 + i * 3_600_000,
               open=p, high=p, low=p, close=p, volume=1.0)
        for i, p in enumerate(prices)
    ]
    for c in candles:
        session.add(c)
    session.commit()


def _pending_rec(session):
    rec = AIRecommendation(
        kind="param_change", strategy_name="ema_crossover",
        content_json=json.dumps({"params": {"fast_period": 10, "slow_period": 26},
                                 "position_pct": 0.2}),
        status="pending", rationale="tune",
    )
    session.add(rec)
    session.commit()
    return rec


def _approval_handlers(session):
    settings = _fake_settings()
    return build_handlers(
        session=session,
        settings=settings,
        risk_cfg=RiskConfig(),
        on_approve=_make_approve(settings, session),
        on_reject=_make_reject(session),
        allowed_users=(1,),
    )


def _find_callback(handlers):
    return next(h for h in handlers if hasattr(h, "pattern"))


async def _click(handler, data):
    update = FakeUpdate(user_id=1, callback_query=FakeQuery(data))
    await _run(handler, update)
    return update.callback_query


def test_approve_releases_strategy_version(session):
    _seed_candles(session)
    rec = _pending_rec(session)
    handlers = _approval_handlers(session)
    q = asyncio.run(_click(_find_callback(handlers), f"approve:{rec.id}"))

    session.refresh(rec)
    assert rec.status == "applied"
    assert session.execute(
        select(Strategy).where(Strategy.name == "ema_crossover", Strategy.status == "active")
    ).scalars().one().version == 2
    assert any("released" in text for text, _ in q.message._sent), q.message._sent


def test_reject_marks_recommendation_rejected(session):
    rec = _pending_rec(session)
    handlers = _approval_handlers(session)
    q = asyncio.run(_click(_find_callback(handlers), f"reject:{rec.id}"))

    session.refresh(rec)
    assert rec.status == "rejected"
    assert any("rejected" in text for text, _ in q.message._sent), q.message._sent


def test_approve_missing_recommendation(session):
    handlers = _approval_handlers(session)
    q = asyncio.run(_click(_find_callback(handlers), "approve:999"))
    assert any("not found" in text for text, _ in q.message._sent), q.message._sent
