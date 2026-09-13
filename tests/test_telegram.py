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
from algotrading.telegram.commands import build_handlers, reset_rate_limits
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
