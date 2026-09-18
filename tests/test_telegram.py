"""Telegram layer tests: auth allowlist, /status formatting, approval flow.

The approval tests wire real on_approve/on_reject callbacks (built the same way
`bot.build_application` does) into build_handlers and drive the inline
CallbackQueryHandler end-to-end against a stubbed query. No network is used.
"""
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select
from telegram import Chat, Message, PhotoSize, Update, User
from telegram.ext import MessageHandler

from algotrading.backtest.runner import BacktestResult
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
from algotrading.strategy.catalog import catalog_entries
from algotrading.strategy.plugins import (
    DEFAULT_PLUGIN_DIR,
    configure_default_loader,
    get_default_loader,
)
from algotrading.telegram.bot import (
    _make_approve,
    _make_backtest,
    _make_catalog_scores,
    _make_reject,
)
from algotrading.telegram import commands as commands_module
from algotrading.telegram.commands import (
    IMAGE_CAPTION_FALLBACK,
    UPLOAD_MAX_AGE_SECONDS,
    build_handlers,
    collect_image,
    reset_rate_limits,
)
from algotrading.telegram.ui import (
    approval_keyboard,
    backtest_keyboard,
    format_backtest_comparison,
    format_risk,
    format_status,
    format_strategy_catalog,
    rank_by_score,
)

PLUGIN_CODE = '''
class MomentumNudge:
    """Buy when the last close ticks up."""

    def __init__(self, params=None):
        self.params = params or {}

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


@pytest.fixture(autouse=True)
def _isolate_rate_limit():
    """Rate-limit state is process-global; keep tests independent of each other."""
    reset_rate_limits()
    yield
    reset_rate_limits()


@pytest.fixture(autouse=True)
def _isolate_photo_batches(monkeypatch):
    """Photo batches are process-global too — and photos answer instantly here."""
    monkeypatch.setattr(commands_module, "PHOTO_BATCH_DELAY_SECONDS", 0)
    commands_module.reset_photo_batches()
    yield
    commands_module.reset_photo_batches()


@pytest.fixture()
def session_factory(tmp_path):
    db = str(tmp_path / "tg.db")
    init_db(db)
    factory = get_session_factory(db)
    # Seed initial data
    sess = factory()
    sess.add(
        Strategy(name="ema_crossover", version=1, status="active",
                 params=json.dumps({"fast_period": 12, "slow_period": 26, "position_pct": 0.2}))
    )
    sess.add(Position(symbol="BTC/USDT", qty=0.25, avg_price=50000.0))
    sess.commit()
    sess.close()
    yield factory


@pytest.fixture()
def session(session_factory):
    """Backward compat: provide a session for direct DB access in tests."""
    sess = session_factory()
    yield sess
    sess.close()


class FakeMessage:
    def __init__(self):
        self._sent = []
        self._markups = []

    async def reply_text(self, text, parse_mode=None, reply_markup=None):
        self._sent.append((text, parse_mode))
        self._markups.append(reply_markup)

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


def _handlers(session_factory, allowed=(1, 2), **kwargs):
    settings = type("S", (), {"mode": "paper"})()
    risk = RiskConfig()
    return {
        next(iter(h.commands)): h
        for h in build_handlers(
            session_factory=session_factory,
            settings=settings,
            risk_cfg=risk,
            allowed_users=allowed,
            **kwargs,
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


def test_strategy_catalog_formatting():
    out = format_strategy_catalog(catalog_entries())
    assert "Available strategies" in out
    assert "ema_crossover" in out
    assert "[built-in]" in out
    assert "fast_period: int" in out
    assert "(2-200)" in out          # min-max range is shown


def test_strategy_catalog_includes_plugins(plugin_dir):
    get_default_loader().write_strategy(
        "momentum_nudge",
        PLUGIN_CODE,
        description="Test plugin",
        params=[{"name": "lookback", "type": "int", "default": 3, "min": 1, "max": 50}],
    )
    out = format_strategy_catalog(catalog_entries())
    assert "momentum_nudge" in out
    assert "[plugin]" in out
    assert "lookback: int" in out
    assert "(1-50)" in out


def test_strategy_catalog_truncates_for_telegram(plugin_dir):
    for i in range(60):
        get_default_loader().write_strategy(
            f"plugin_{i}", PLUGIN_CODE,
            params=[{"name": "x", "type": "int", "default": 1}],
        )
    out = format_strategy_catalog(catalog_entries())
    assert len(out) <= 4096          # Telegram's hard message cap
    assert "truncated" in out


def test_strategies_command_replies_to_allowed_user(session_factory):
    handlers = _handlers(session_factory, allowed=(1,))
    update = FakeUpdate(user_id=1)
    asyncio.run(_run(handlers["strategies"], update))
    sent = update.effective_message._sent
    assert sent, "allowed user should get the catalog"
    assert "ema_crossover" in sent[0][0]
    assert sent[0][1] == "HTML"


def test_strategies_command_ignores_outsider(session_factory):
    handlers = _handlers(session_factory, allowed=(1,))
    update = FakeUpdate(user_id=999)
    asyncio.run(_run(handlers["strategies"], update))
    assert not update.effective_message._sent


# --- /strategies backtest buttons ---------------------------------------------

def _backtest_handlers(session_factory, settings):
    return build_handlers(
        session_factory=session_factory,
        settings=settings,
        risk_cfg=RiskConfig(),
        on_backtest=_make_backtest(settings, session_factory),
        allowed_users=(1,),
    )


def test_backtest_keyboard_buttons_cover_catalog():
    kb = backtest_keyboard(catalog_entries())
    data = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert "bt:ema_crossover" in data
    assert all(d.startswith("bt:") for d in data)
    # Telegram's hard cap on callback_data.
    assert all(len(d.encode("utf-8")) <= 64 for d in data)


def test_backtest_keyboard_skips_overlong_names(plugin_dir):
    long_name = "s" * 70
    get_default_loader().write_strategy(long_name, PLUGIN_CODE, params=[])
    kb = backtest_keyboard(catalog_entries())
    data = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert f"bt:{long_name}" not in data
    assert all(len(d.encode("utf-8")) <= 64 for d in data)


def test_strategies_command_attaches_backtest_buttons(session_factory):
    handlers = _handlers(session_factory, allowed=(1,))
    update = FakeUpdate(user_id=1)
    asyncio.run(_run(handlers["strategies"], update))
    markup = update.effective_message._markups[0]
    data = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert "bt:ema_crossover" in data


def test_backtest_button_reports_result_when_params_match_defaults(session_factory, session):
    _seed_candles(session)
    settings = _fake_settings()
    handler = _find_callback(_backtest_handlers(session_factory, settings))
    update = FakeUpdate(user_id=1, callback_query=FakeQuery("bt:ema_crossover"))
    asyncio.run(_run(handler, update))

    sent = update.effective_message._sent
    assert sent, "tapping a backtest button should reply"
    assert "Backtest: ema_crossover" in sent[0][0]
    # Seeded params equal the shipped defaults, so there is nothing to compare.
    assert "match the catalog defaults" in sent[0][0]


def test_backtest_button_compares_current_vs_defaults(session_factory, session):
    _seed_candles(session)
    # A newer version with non-default params becomes the "current" set.
    session.add(Strategy(
        name="ema_crossover", version=2, status="draft",
        params=json.dumps({"fast_period": 5, "slow_period": 20, "position_pct": 0.2}),
    ))
    session.commit()

    settings = _fake_settings()
    handler = _find_callback(_backtest_handlers(session_factory, settings))
    update = FakeUpdate(user_id=1, callback_query=FakeQuery("bt:ema_crossover"))
    asyncio.run(_run(handler, update))

    text = update.effective_message._sent[0][0]
    assert "Backtest: ema_crossover" in text
    # Param diff + side-by-side results table, judged on risk-adjusted terms.
    assert "Params" in text and "fast_period" in text and "slow_period" in text
    assert "Results" in text
    assert "pnl/drawdown" in text
    assert "risk-adjusted" in text


# --- risk-adjusted verdict (pure, no DB) --------------------------------------

def _bt_result(pnl, drawdown, n_trades=1, balance=10_000.0):
    return BacktestResult(
        strategy_name="x", params={}, symbol="BTC/USDT", interval="1h",
        n_trades=n_trades, total_pnl=pnl, max_drawdown=drawdown, final_balance=balance,
    )


def _comparison(current, default):
    return format_backtest_comparison(
        "x", "BTC/USDT", "1h", 100, {"period": 5}, {"period": 20}, current, default,
    )


def test_verdict_prefers_risk_adjusted_winner_over_raw_pnl():
    # current earns more but risks 4x the drawdown -> worse per unit of risk.
    current = _bt_result(pnl=2000.0, drawdown=4000.0)   # 0.50
    default = _bt_result(pnl=1000.0, drawdown=1000.0)   # 1.00
    text = _comparison(current, default)
    assert "default is better risk-adjusted" in text
    assert "current has higher raw PnL but a worse drawdown" in text


def test_verdict_declares_current_when_ratio_is_higher():
    current = _bt_result(pnl=3000.0, drawdown=1000.0)   # 3.00
    default = _bt_result(pnl=1000.0, drawdown=1000.0)   # 1.00
    text = _comparison(current, default)
    assert "current is better risk-adjusted" in text
    assert "higher raw PnL" not in text


def test_verdict_handles_zero_drawdown():
    # A profitable run with no drawdown is the best possible ratio.
    current = _bt_result(pnl=500.0, drawdown=0.0)
    default = _bt_result(pnl=1000.0, drawdown=2000.0)
    text = _comparison(current, default)
    assert "current is better risk-adjusted" in text
    assert "\u221e" in text


def test_verdict_without_trades_is_neutral():
    text = _comparison(
        _bt_result(0.0, 0.0, n_trades=0),
        _bt_result(0.0, 0.0, n_trades=0),
    )
    assert "nothing to compare" in text


# --- /strategies risk-adjusted ranking ---------------------------------------

def test_rank_by_score_orders_best_first_and_unscored_last():
    entries = catalog_entries()
    scores = {
        "ema_crossover": _bt_result(pnl=1000.0, drawdown=1000.0),        # 1.00
        "rsi_mean_reversion": _bt_result(pnl=3000.0, drawdown=1000.0),  # 3.00
    }
    ranked = rank_by_score(entries, scores)

    assert ranked[0].name == "rsi_mean_reversion"
    assert ranked[1].name == "ema_crossover"
    assert ranked[-1].name not in scores          # unscored sorts last
    assert entries[0].name == "ema_crossover"     # input list is untouched


def test_catalog_renders_ranks_and_scores():
    entries = catalog_entries()
    scores = {"ema_crossover": _bt_result(pnl=1000.0, drawdown=1000.0)}
    text = format_strategy_catalog(rank_by_score(entries, scores), scores)

    assert "ranked by PnL" in text
    assert "#1 ema_crossover" in text
    assert "score 1.00" in text
    assert "score n/a" in text                    # unscored strategies
    assert len(text) <= 4096


def test_catalog_without_scores_is_unranked():
    text = format_strategy_catalog(catalog_entries())
    assert "ranked by" not in text
    assert "#1" not in text


def test_strategies_command_shows_scores(session_factory):
    async def fake_scores():
        return {"ema_crossover": _bt_result(pnl=3000.0, drawdown=1000.0)}

    handlers = _handlers(session_factory, on_catalog_scores=fake_scores)
    update = FakeUpdate(user_id=1)
    asyncio.run(_run(handlers["strategies"], update))

    text = update.effective_message._sent[0][0]
    assert "ranked by PnL" in text
    assert "#1 ema_crossover" in text
    assert "score 3.00" in text


def test_strategies_command_survives_scoring_failure(session_factory):
    async def boom():
        raise RuntimeError("scoring exploded")

    handlers = _handlers(session_factory, on_catalog_scores=boom)
    update = FakeUpdate(user_id=1)
    asyncio.run(_run(handlers["strategies"], update))

    text = update.effective_message._sent[0][0]
    assert "ema_crossover" in text
    assert "ranked by" not in text      # degrades to the unranked catalog


def test_catalog_scores_factory_backtests_every_strategy(session_factory, session):
    _seed_candles(session)
    settings = _fake_settings()
    scores = asyncio.run(_make_catalog_scores(settings, session_factory)())

    assert "ema_crossover" in scores
    assert scores["ema_crossover"] is not None
    assert scores["ema_crossover"].n_trades >= 0


def test_backtest_button_unknown_strategy_is_friendly(session_factory, session):
    _seed_candles(session)
    settings = _fake_settings()
    handler = _find_callback(_backtest_handlers(session_factory, settings))
    update = FakeUpdate(user_id=1, callback_query=FakeQuery("bt:not_a_strategy"))
    asyncio.run(_run(handler, update))

    assert "Unknown strategy" in update.effective_message._sent[0][0]


def test_backtest_button_ignores_outsider(session_factory):
    settings = _fake_settings()
    handlers = build_handlers(
        session_factory=session_factory,
        settings=settings,
        risk_cfg=RiskConfig(),
        on_backtest=_make_backtest(settings, session_factory),
        allowed_users=(1,),
    )
    update = FakeUpdate(user_id=999, callback_query=FakeQuery("bt:ema_crossover"))
    asyncio.run(_run(_find_callback(handlers), update))
    assert not update.effective_message._sent


def test_allowlisted_user_can_status(session_factory):
    handlers = _handlers(session_factory, allowed=(1,))
    update = FakeUpdate(user_id=1)
    asyncio.run(_run(handlers["status"], update))
    assert update.effective_message._sent, "allowed user should get a reply"


def test_outsider_gets_no_reply(session_factory):
    handlers = _handlers(session_factory, allowed=(1,))
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


def _approval_handlers(session_factory):
    settings = _fake_settings()
    return build_handlers(
        session_factory=session_factory,
        settings=settings,
        risk_cfg=RiskConfig(),
        on_approve=_make_approve(settings, session_factory),
        on_reject=_make_reject(session_factory),
        allowed_users=(1,),
    )


def _find_callback(handlers):
    return next(h for h in handlers if hasattr(h, "pattern"))


async def _click(handler, data):
    update = FakeUpdate(user_id=1, callback_query=FakeQuery(data))
    await _run(handler, update)
    return update.callback_query


def test_approve_releases_strategy_version(session_factory, session):
    _seed_candles(session)
    rec = _pending_rec(session)
    handlers = _approval_handlers(session_factory)
    q = asyncio.run(_click(_find_callback(handlers), f"approve:{rec.id}"))

    session.refresh(rec)
    assert rec.status == "applied"
    assert session.execute(
        select(Strategy).where(Strategy.name == "ema_crossover", Strategy.status == "active")
    ).scalars().one().version == 2
    assert any("released" in text for text, _ in q.message._sent), q.message._sent


def test_reject_marks_recommendation_rejected(session_factory, session):
    rec = _pending_rec(session)
    handlers = _approval_handlers(session_factory)
    q = asyncio.run(_click(_find_callback(handlers), f"reject:{rec.id}"))

    session.refresh(rec)
    assert rec.status == "rejected"
    assert any("rejected" in text for text, _ in q.message._sent), q.message._sent


def test_approve_missing_recommendation(session_factory):
    handlers = _approval_handlers(session_factory)
    q = asyncio.run(_click(_find_callback(handlers), "approve:999"))
    assert any("not found" in text for text, _ in q.message._sent), q.message._sent


# --- image input: photo / image document -> LLM with the picture attached ---

class FakeChat:
    def __init__(self, id=1):
        self.id = id
        self.actions = []
        self.sent = []          # (text, kwargs) from bot.send_message
        self.markups = []

    async def send_action(self, action):
        self.actions.append(action)

    async def send_chat_action(self, action, **kwargs):
        self.actions.append(action)

    async def send_message(self, chat_id, text=None, **kwargs):
        self.sent.append((text, kwargs))
        self.markups.append(kwargs.get("reply_markup"))
        return None


class FakeBot:
    def __init__(self, payload=None, chat=None):
        self.file = FakeTelegramFile(*(payload,) if payload else ())
        self.requested = []
        self.chat = chat or FakeChat()

    async def get_file(self, file_id):
        self.requested.append(file_id)
        return self.file

    async def send_message(self, chat_id, text=None, **kwargs):
        return await self.chat.send_message(chat_id, text=text, **kwargs)

    async def send_chat_action(self, chat_id, action, **kwargs):
        return await self.chat.send_chat_action(action)


class FakeContext:
    def __init__(self, payload=None, chat=None):
        self.chat = chat or FakeChat()
        self.bot = FakeBot(payload, chat=self.chat)


class FakeTelegramFile:
    """Stands in for telegram.File: writes bytes where the handler asked."""

    def __init__(self, payload=b"\x89PNG\r\n\x1a\n" + b"x" * 32):
        self.payload = payload
        self.saved_path = None

    async def download_to_drive(self, path):
        self.saved_path = path
        with open(path, "wb") as fh:
            fh.write(self.payload)


class FakeImageMessage(FakeMessage):
    """Message stub for photos/documents: a caption, never `text`."""

    def __init__(self, photo=None, document=None, caption=None):
        super().__init__()
        self.photo = photo or []
        self.document = document
        self.caption = caption


class FakePhotoSize:
    def __init__(self, file_id="photo-1", file_size=1024):
        self.file_id = file_id
        self.file_size = file_size


class FakeDocumentObject:
    def __init__(self, file_id="doc-1", file_size=1024,
                 mime_type="image/png", file_name="strategy.png"):
        self.file_id = file_id
        self.file_size = file_size
        self.mime_type = mime_type
        self.file_name = file_name


def _ai_settings(upload_dir, **overrides):
    """Settings stub with the ai.* fields the image path reads."""
    fields = {"enabled": True, "use_hermes": True, "images_enabled": True,
              "image_max_bytes": 5_000_000, "image_dir": str(upload_dir)}
    fields.update(overrides)
    return type("S", (), {"ai": type("AI", (), fields)(), "mode": "paper"})()


def _image_update(user_id=1, message=None, chat=None):
    update = FakeUpdate(user_id, message=message or FakeImageMessage(photo=[FakePhotoSize()]))
    update.effective_chat = chat or FakeChat()
    return update


def _real_update(**message_kwargs):
    """A genuine PTB Update, used only to probe handler filters."""
    return Update(update_id=1, message=Message(
        message_id=1, date=datetime.now(timezone.utc),
        chat=Chat(id=1, type="private"),
        from_user=User(id=1, first_name="x", is_bot=False),
        **message_kwargs,
    ))


_PHOTO_PROBE = dict(photo=[PhotoSize(file_id="f", file_unique_id="u", width=8, height=8)])
_TEXT_PROBE = dict(text="hello")


def test_photo_updates_route_to_the_image_handler_only(session_factory, tmp_path):
    handlers = build_handlers(
        session_factory=session_factory,
        settings=_ai_settings(tmp_path / "uploads"),
        risk_cfg=RiskConfig(),
        allowed_users=(1,),
    )
    message_handlers = [h for h in handlers if isinstance(h, MessageHandler)]
    assert len(message_handlers) == 2, "text + image handlers"

    photo_matches = [h for h in message_handlers if h.check_update(_real_update(**_PHOTO_PROBE))]
    text_matches = [h for h in message_handlers if h.check_update(_real_update(**_TEXT_PROBE))]

    assert len(photo_matches) == 1 and len(text_matches) == 1
    assert photo_matches[0] is not text_matches[0], "a photo must not hit the text handler"


def _image_handler(handlers):
    """The registered handler that accepts photos (probed with a real Update)."""
    probe = _real_update(**_PHOTO_PROBE)
    return next(h for h in handlers
                if isinstance(h, MessageHandler) and h.check_update(probe))


def _fake_run_agent(record, response=None):
    """Stand-in run_agent that records what the image flows hand it."""
    def fake(_sf, _settings, text, image_path=None, image_paths=None, **kwargs):
        record.append({
            "text": text,
            "image_paths": list(image_paths or []),
            "existed": all(Path(p).is_file() for p in (image_paths or [])),
            "bytes": Path(image_paths[0]).read_bytes()[:8] if image_paths else b"",
        })
        return response if response is not None else {"text": "ok"}
    return fake


def _image_handlers(session_factory, settings):
    """The photo handler and the callback handler, wired like the real app."""
    handlers = build_handlers(
        session_factory=session_factory, settings=settings,
        risk_cfg=RiskConfig(), allowed_users=(1,),
    )
    return _image_handler(handlers), _find_callback(handlers)


async def _send_photo(handler, update, context):
    """Run the photo handler, then the buffered flush it scheduled."""
    await _run(handler, update, context)
    await asyncio.sleep(0)                      # let the flush task start
    task = commands_module._photo_batch_tasks.get(update.effective_chat.id)
    if task is not None:
        await task


async def _send_batch(handler, updates, context):
    """Send several photos back-to-back, the way an album arrives."""
    for update in updates:
        await _run(handler, update, context)
    await asyncio.sleep(0)
    task = commands_module._photo_batch_tasks.get(updates[0].effective_chat.id)
    if task is not None:
        await task


async def _click_flow(handler, data, context):
    update = FakeUpdate(user_id=1, callback_query=FakeQuery(data))
    await _run(handler, update, context)
    return update.callback_query


def _flow_token(chat) -> str:
    """The token of the choice keyboard last sent to a chat."""
    markup = [m for m in chat.markups if m is not None][-1]
    data = markup.to_dict()["inline_keyboard"][0][0]["callback_data"]
    return data.split(":")[1]


def test_single_photo_is_answered_and_its_upload_deleted(session_factory, tmp_path, monkeypatch):
    settings = _ai_settings(tmp_path / "uploads")
    calls = []
    monkeypatch.setattr("algotrading.telegram.chat.run_agent",
                        _fake_run_agent(calls, {"text": "that is an EMA crossover chart"}))
    handler, _ = _image_handlers(session_factory, settings)
    context = FakeContext()
    update = _image_update(message=FakeImageMessage(
        photo=[FakePhotoSize(file_id="photo-1")], caption="can I backtest this?"))

    asyncio.run(_send_photo(handler, update, context))

    assert context.bot.requested == ["photo-1"]
    assert len(calls) == 1, "a lone photo is answered, not offered a choice"
    assert calls[0]["text"] == "can I backtest this?"     # the caption is the question
    assert calls[0]["existed"] is True                    # file outlives the AI call
    assert calls[0]["bytes"].startswith(b"\x89PNG")
    assert context.chat.sent[0][0] == "that is an EMA crossover chart"
    assert not Path(calls[0]["image_paths"][0]).exists(), "upload must be cleaned up"


def test_single_photo_without_caption_shows_the_proposal_card(session_factory, tmp_path, monkeypatch):
    settings = _ai_settings(tmp_path / "uploads")
    calls = []
    monkeypatch.setattr("algotrading.telegram.chat.run_agent", _fake_run_agent(calls, {
        "text": "proposed", "proposal_id": 7, "kind": "new_strategy",
        "strategy_name": "squeeze_breakout", "rationale": "from your screenshot"}))

    async def send():
        handler, _ = _image_handlers(session_factory, settings)
        context = FakeContext()
        await _send_photo(handler, _image_update(), context)
        return context

    context = asyncio.run(send())

    assert calls[0]["text"] == IMAGE_CAPTION_FALLBACK
    sent = [text for text, _ in context.chat.sent]
    assert sent[0] == "proposed"
    assert "AI proposal #7" in sent[1] and "squeeze_breakout" in sent[1]


def test_several_photos_are_batched_into_one_choice(session_factory, tmp_path, monkeypatch):
    settings = _ai_settings(tmp_path / "uploads")
    calls = []
    monkeypatch.setattr("algotrading.telegram.chat.run_agent", _fake_run_agent(calls))
    handler, _ = _image_handlers(session_factory, settings)
    chat = FakeChat()
    context = FakeContext(chat=chat)
    updates = [
        _image_update(chat=chat, message=FakeImageMessage(photo=[FakePhotoSize(file_id="p1")])),
        _image_update(chat=chat, message=FakeImageMessage(
            photo=[FakePhotoSize(file_id="p2")], caption="my exit rule")),
    ]

    asyncio.run(_send_batch(handler, updates, context))

    assert calls == [], "batched photos must not be read one by one"
    assert context.bot.requested == ["p1", "p2"]
    text, kwargs = chat.sent[0]
    assert "2 images received" in text
    assert kwargs["reply_markup"] is not None, "the user must get the choice buttons"
    assert _flow_token(chat) in commands_module._image_flows


def test_choice_combine_reads_all_images_in_one_call(session_factory, tmp_path, monkeypatch):
    settings = _ai_settings(tmp_path / "uploads")
    calls = []
    monkeypatch.setattr("algotrading.telegram.chat.run_agent",
                        _fake_run_agent(calls, {"text": "one strategy coming up"}))
    handler, callback = _image_handlers(session_factory, settings)
    chat = FakeChat()
    context = FakeContext(chat=chat)
    updates = [
        _image_update(chat=chat, message=FakeImageMessage(photo=[FakePhotoSize(file_id="p1")])),
        _image_update(chat=chat, message=FakeImageMessage(
            photo=[FakePhotoSize(file_id="p2")], caption="my exit rule")),
    ]

    async def run_flow():
        await _send_batch(handler, updates, context)
        return await _click_flow(callback, f"imgflow:{_flow_token(chat)}:combine", context)

    query = asyncio.run(run_flow())

    assert len(calls) == 1, "combine = one agent call over every image"
    assert len(calls[0]["image_paths"]) == 2
    assert calls[0]["existed"] is True
    assert calls[0]["text"] == "my exit rule"           # caption carries into the flow
    assert "Reading 2 images — one strategy from all" in query.message._sent[0][0]
    assert chat.sent[-1][0] == "one strategy coming up"
    assert all(not Path(p).exists() for p in calls[0]["image_paths"]), "uploads cleaned up"
    assert not commands_module._image_flows, "the token is consumed"


def test_choice_separate_reads_each_image_and_points_at_the_ensemble(
    session_factory, tmp_path, monkeypatch
):
    settings = _ai_settings(tmp_path / "uploads")
    calls = []
    monkeypatch.setattr("algotrading.telegram.chat.run_agent",
                        _fake_run_agent(calls, {"text": "one strategy per image"}))
    handler, callback = _image_handlers(session_factory, settings)
    chat = FakeChat()
    context = FakeContext(chat=chat)
    updates = [
        _image_update(chat=chat, message=FakeImageMessage(photo=[FakePhotoSize(file_id="p1")])),
        _image_update(chat=chat, message=FakeImageMessage(photo=[FakePhotoSize(file_id="p2")])),
    ]

    async def run_flow():
        await _send_batch(handler, updates, context)
        return await _click_flow(callback, f"imgflow:{_flow_token(chat)}:separate", context)

    asyncio.run(run_flow())

    assert len(calls) == 2, "separate = one agent call per image"
    assert [len(c["image_paths"]) for c in calls] == [1, 1]
    assert "ensemble" in chat.sent[-1][0], "the follow-up must point at the ensemble step"
    assert all(not Path(p).exists() for c in calls for p in c["image_paths"])


def test_unknown_or_expired_image_flow_is_friendly(session_factory, tmp_path):
    settings = _ai_settings(tmp_path / "uploads")
    handler, callback = _image_handlers(session_factory, settings)

    query = asyncio.run(_click_flow(callback, "imgflow:deadbeef:combine", FakeContext()))

    assert "expired" in query.message._sent[0][0]
    assert not commands_module._image_flows


def test_batches_bigger_than_the_limit_are_trimmed(session_factory, tmp_path, monkeypatch):
    settings = _ai_settings(tmp_path / "uploads")
    monkeypatch.setattr(commands_module, "IMAGE_BATCH_LIMIT", 2)
    handler, _ = _image_handlers(session_factory, settings)
    chat = FakeChat()
    context = FakeContext(chat=chat)
    updates = [
        _image_update(chat=chat, message=FakeImageMessage(photo=[FakePhotoSize(file_id=f"p{i}")]))
        for i in range(3)
    ]

    asyncio.run(_send_batch(handler, updates, context))

    text = chat.sent[0][0]
    assert "2 images received" in text
    assert "Only the first 2" in text
    assert len(commands_module._image_flows[_flow_token(chat)][1]) == 2


def test_collect_image_gates_before_downloading(tmp_path):
    cases = [
        ({"enabled": False}, "not configured"),
        ({"images_enabled": False}, "turned off"),
        ({"use_hermes": False}, "USE_HERMES"),
        ({"image_max_bytes": 10}, "MB"),          # 1 KB photo vs a 10-byte cap
    ]
    for overrides, expected in cases:
        context = FakeContext()
        path, error = asyncio.run(collect_image(
            _image_update(), context, _ai_settings(tmp_path / "uploads", **overrides)
        ))
        assert path is None, overrides
        assert expected in error, (overrides, error)
        assert context.bot.requested == [], "gating must happen before any download"


def test_collect_image_rejects_a_non_image_document(tmp_path):
    context = FakeContext()
    path, error = asyncio.run(collect_image(
        _image_update(message=FakeImageMessage(
            document=FakeDocumentObject(mime_type="application/pdf", file_name="plan.pdf")
        )),
        context,
        _ai_settings(tmp_path / "uploads"),
    ))
    assert path is None
    assert "only read images" in error
    assert context.bot.requested == []


def test_collect_image_accepts_a_document_and_prunes_stale_uploads(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    stale = uploads / "img-1-1.jpg"
    stale.write_bytes(b"old")
    old = time.time() - (UPLOAD_MAX_AGE_SECONDS + 60)
    os.utime(stale, (old, old))

    context = FakeContext()
    path, error = asyncio.run(collect_image(
        _image_update(message=FakeImageMessage(
            document=FakeDocumentObject(file_id="doc-9", mime_type="image/webp",
                                        file_name="setup.webp")
        )),
        context,
        _ai_settings(uploads),
    ))

    assert error == "" and path is not None
    assert path.suffix == ".webp" and path.exists()
    assert context.bot.requested == ["doc-9"]
    assert not stale.exists(), "a crash-leftover upload must be swept up"
