"""Full end-to-end scenario: an experienced trader with no coding background.

This is the "does the whole product actually work for a non-technical person"
test. It walks the same journey a real user takes, in order:

  1. First-run setup wizard (config .env without touching code)
  2. Start the bot in demo mode; watch the module graph compose
  3. The bot pulls candles, crosses EMAs, buys, arms a trailing stop, and the
     stop later closes the position by itself
  4. Every Telegram command, the ranked /strategies catalog, the inline
     backtest button, and the approve/reject buttons
  5. Asking the AI to create a strategy in plain English, then approving it
  6. Analytics snapshots, the HTTP API, reconciliation, and heartbeat
  7. Restart: state is rebuilt from the immutable event log

Everything runs offline against tmp_path databases and a scripted market feed;
nothing here talks to Binance, Telegram, or an LLM provider.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from algotrading import cli_setup
from algotrading.ai.client import RecommendationError
from algotrading.api import build_api
from algotrading.config import (
    AIConfig,
    ApiConfig,
    MarketConfig,
    ModulesConfig,
    RiskConfig,
    ScheduleConfig,
    Settings,
)
from algotrading.db import get_session_factory, init_db
from algotrading.db.models import (
    AIRecommendation,
    AnalyticsSummary,
    Position,
    Signal,
    Strategy,
    Trade,
    TradeIntent,
)
from algotrading.db.seed import ensure_seeded
from algotrading.ledger import Ledger
from algotrading.main import build_context
from algotrading.market.base import Candle as MarketCandle
from algotrading.market.base import MarketDataProvider
from algotrading.modules.base import CAPABILITY_EXECUTION, CAPABILITY_MARKET, Module, ModuleSpec
from algotrading.modules.registry import register_module
from algotrading.scheduler.jobs import (
    BotContext,
    analytics_daily,
    analytics_tick,
    heartbeat_tick,
    market_tick,
    reconcile_tick,
)
from algotrading.store.recommendations import (
    RecommendationStore,
    create_pending_recommendation,
)
from algotrading.strategy.catalog import catalog_entries
from algotrading.strategy.plugins import (
    DEFAULT_PLUGIN_DIR,
    configure_default_loader,
    get_default_loader,
)
from algotrading.strategy.registry import is_plugin, known_names
from algotrading.telegram.bot import (
    _make_approve,
    _make_backtest,
    _make_catalog_scores,
    _make_reject,
)
from algotrading.telegram.chat import run_agent
from algotrading.telegram.commands import build_handlers, reset_rate_limits

HOUR_MS = 3_600_000
OWNER = 7  # the trader's Telegram user id


# ---------------------------------------------------------------------------
# a feed the test controls (installed as a normal plug-and-play module)
# ---------------------------------------------------------------------------

class ScriptedFeed(MarketDataProvider):
    """Market data provider whose candles the test sets explicitly."""

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str], list[MarketCandle]] = {}

    def set(self, symbol: str, interval: str, candles: Sequence[MarketCandle]) -> None:
        self._by_key[(symbol, interval)] = list(candles)

    def fetch_klines(self, symbol: str, interval: str, since_ms: int | None = None):
        candles = self._by_key.get((symbol, interval), [])
        if since_ms is not None:
            candles = [c for c in candles if c.ts >= since_ms]
        return list(candles)

    def fetch_ticker_price(self, symbol: str) -> float:
        for key in ((symbol, "1h"), (symbol, "1m")):
            candles = self._by_key.get(key)
            if candles:
                return candles[-1].close
        return 0.0

    def now_ms(self) -> int:
        return int(time.time() * 1000)


class ScriptedMarketModule(Module):
    """A *third-party style* market module, proving the parts are swappable."""

    spec = ModuleSpec(
        name="market.scripted",
        capability=CAPABILITY_MARKET,
        description="Test-only scripted feed.",
        builtin=True,
    )
    feed: ScriptedFeed | None = None

    def setup(self, ctx: Any) -> None:
        assert self.feed is not None
        ctx.provider = self.feed
        ctx.provide(CAPABILITY_MARKET, self.feed)


register_module(ScriptedMarketModule)


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

@dataclass
class Bot:
    settings: Settings
    session_factory: Callable
    ctx: BotContext
    manager: Any
    feed: ScriptedFeed
    plugin_dir: Path

    def session(self):
        return self.session_factory()


@pytest.fixture()
def bot(tmp_path: Path):
    """A fully composed bot in demo/paper mode with a throwaway DB + plugins."""
    plugin_dir = tmp_path / "strategies"
    plugin_dir.mkdir()
    configure_default_loader([plugin_dir])
    reset_rate_limits()

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_path = str(tmp_path / "scenario.db")
    init_db(db_path)
    sf = get_session_factory(db_path)
    with sf() as session:
        ensure_seeded(session)  # first run seeds the starter strategies

    feed = ScriptedFeed()
    ScriptedMarketModule.feed = feed

    settings = Settings(
        mode="paper",
        market=MarketConfig(
            symbols=["BTC/USDT"],
            intervals=["1m", "1h"],
            max_staleness_seconds=10 ** 9,
        ),
        risk=RiskConfig(trailing_stop_pct=5.0, paper_initial_balance=10_000.0),
        schedule=ScheduleConfig(),
        ai=AIConfig(enabled=False),
        api=ApiConfig(enabled=False),
        modules=ModulesConfig(
            enabled=[
                "market.scripted",
                "gateway.paper",
                "strategy.plugins",
                "analytics.default",
            ],
            strategy_plugin_paths=[str(plugin_dir)],
        ),
        data_dir=str(data_dir),
        db_path=db_path,
    )
    ctx, manager = build_context(settings, demo=True, disabled=("control.telegram",))
    ctx.health.beat()

    yield Bot(
        settings=settings,
        session_factory=sf,
        ctx=ctx,
        manager=manager,
        feed=feed,
        plugin_dir=plugin_dir,
    )

    configure_default_loader([DEFAULT_PLUGIN_DIR])
    reset_rate_limits()


# ---------------------------------------------------------------------------
# deterministic candle series
# ---------------------------------------------------------------------------

def _anchored(closes: Sequence[float], symbol: str = "BTC/USDT",
              interval: str = "1h") -> list[MarketCandle]:
    """Index-stable candles; every tick uses a prefix of the same timeline."""
    now = int(time.time() * 1000)
    base = now - (now % HOUR_MS) - 100 * HOUR_MS
    return [
        MarketCandle(symbol, interval, base + i * HOUR_MS, c, c, c, c, 1.0)
        for i, c in enumerate(closes)
    ]


def _entry_closes() -> list[float]:
    """90 bars drifting down, then 11 up: a fresh golden cross on the last bar."""
    price = 100.0
    out: list[float] = []
    for i in range(90 + 11):
        price *= 1 + (-0.006 if i < 90 else 0.012)
        out.append(price)
    return out


def _crash(closes: list[float]) -> list[float]:
    """Three more bars that fall through any sane trailing stop."""
    out = list(closes)
    for factor in (0.80, 0.75, 0.70):
        out.append(out[-1] * factor)
    return out


def _prime_market(bot: Bot) -> list[float]:
    """Seed the crossover series and run one tick so a position is open."""
    closes = _entry_closes()
    bot.feed.set("BTC/USDT", "1h", _anchored(closes))
    market_tick(bot.ctx)
    return closes


# ---------------------------------------------------------------------------
# 1. first-run setup
# ---------------------------------------------------------------------------

def test_scenario_setup_wizard_configures_the_bot_without_code(tmp_path, monkeypatch, capsys):
    """The trader answers a few prompts; a .env appears and nothing else is edited."""
    env_path = tmp_path / ".env"
    monkeypatch.setattr(cli_setup, "ENV_PATH", env_path)
    # The wizard would normally verify the token over the network and shell out
    # to init the DB; both are outside the scenario's scope.
    monkeypatch.setattr(cli_setup, "validate_telegram", lambda token: token == "123:abc")
    monkeypatch.setattr(cli_setup, "_run", lambda cmd, what: 0)

    answers = iter(["123:abc", "7,8", "", "", "gpt-4o-mini"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    updates = cli_setup.prompt_for_fields({})
    assert updates["TELEGRAM_BOT_TOKEN"] == "123:abc"
    assert updates["TELEGRAM_ALLOWED_USERS"] == "7,8"
    # Blank AI key means "run without the assistant".
    assert updates["AI_API_KEY"] == ""
    assert updates["AI_BASE_URL"] == "https://api.openai.com/v1"

    cli_setup.apply_updates(updates)

    written = cli_setup.read_env()
    assert written["TELEGRAM_BOT_TOKEN"] == "123:abc"
    assert written["TELEGRAM_ALLOWED_USERS"] == "7,8"
    assert "AI_API_KEY=" in env_path.read_text(encoding="utf-8")

    # Re-running setup keeps existing values when the user just presses Enter.
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    kept = cli_setup.prompt_for_fields(cli_setup.read_env())
    assert kept["TELEGRAM_BOT_TOKEN"] == "123:abc"
    capsys.readouterr()  # swallow the wizard's chatter


# ---------------------------------------------------------------------------
# 2. startup composition
# ---------------------------------------------------------------------------

def test_scenario_startup_composes_swappable_modules(bot: Bot):
    """The enabled set is config, and execution is present no matter what."""
    names = {m.name for m in bot.manager.modules}
    assert {"market.scripted", "gateway.paper", "strategy.plugins",
            "analytics.default"} == names
    # Execution is installed and locked: it is the same object as ctx.gateway.
    assert bot.ctx.gateway is not None
    assert bot.ctx.get(CAPABILITY_EXECUTION) is bot.ctx.gateway
    # The default 9 starter strategies are seeded and exactly one is active.
    with bot.session() as session:
        rows = session.execute(select(Strategy)).scalars().all()
        assert len(rows) == len(catalog_entries())
        assert [r.name for r in rows if r.status == "active"] == ["ema_crossover"]

    # Modules contribute their scheduled jobs instead of main.py hard-coding them.
    job_ids = {job.job_id for job in bot.manager.collect_jobs(bot.ctx)}
    assert {"analytics", "analytics_daily", "ai_review", "strategy_plugins_reload"} <= job_ids


# ---------------------------------------------------------------------------
# 3. the bot trades and protects itself
# ---------------------------------------------------------------------------

def test_scenario_bot_opens_a_trade_then_trailing_stop_closes_it(bot: Bot):
    closes = _prime_market(bot)

    # --- tick 1: entry -----------------------------------------------------
    with bot.session() as session:
        pos = session.execute(select(Position)).scalar_one()
        assert pos.qty > 0
        assert pos.avg_price > 0
        armed_stop = pos.trailing_stop_price
        assert armed_stop is not None and armed_stop < pos.avg_price
        assert session.execute(
            select(TradeIntent).where(TradeIntent.side == "buy",
                                      TradeIntent.status == "filled")
        ).scalars().all()

    # --- tick 2: price rises; no new signal, but the stop must still trail --
    up = closes + [closes[-1] * 1.012]
    bot.feed.set("BTC/USDT", "1h", _anchored(up))
    market_tick(bot.ctx)

    with bot.session() as session:
        pos = session.execute(select(Position)).scalar_one()
        # The highest-price tracker was NULL until this tick; a naive comparison
        # against it would crash and silently skip trailing updates.
        assert pos.highest_price is not None
        assert pos.highest_price > pos.avg_price
        # The stop moved up even though the strategy emitted no candidate this
        # tick — protective exits must not depend on fresh entries.
        assert pos.trailing_stop_price > armed_stop

    # --- tick 3: crash; the stop closes the position with no strategy signal
    crashed = _crash(up)
    bot.feed.set("BTC/USDT", "1h", _anchored(crashed))
    market_tick(bot.ctx)

    with bot.session() as session:
        pos = session.execute(select(Position)).scalar_one()
        assert pos.qty == pytest.approx(0.0), "trailing stop should have flattened it"
        trades = session.execute(select(Trade)).scalars().all()
        assert len(trades) == 1
        assert trades[0].realized_pnl < 0  # sold into a crash
        # The exit came from the stop, not from the strategy's death cross.
        stops = session.execute(
            select(Signal).where(Signal.side == "sell")
        ).scalars().all()
        assert any("Trailing stop hit" in (s.rationale or "") for s in stops)
        assert session.execute(
            select(TradeIntent).where(TradeIntent.side == "sell",
                                      TradeIntent.status == "filled")
        ).scalars().all()


# ---------------------------------------------------------------------------
# 4. Telegram control surface
# ---------------------------------------------------------------------------

class FakeMessage:
    def __init__(self):
        self._sent: list[tuple[str, str | None]] = []
        self._markups: list[Any] = []

    async def reply_text(self, text, parse_mode=None, reply_markup=None):
        self._sent.append((text, parse_mode))
        self._markups.append(reply_markup)

    async def edit_message_text(self, text, parse_mode=None):
        self._sent.append((text, parse_mode))


class FakeChat:
    async def send_action(self, *args, **kwargs):
        return None


class FakeQuery:
    def __init__(self, data):
        self.data = data
        self.answered = False
        self.message = FakeMessage()

    async def answer(self):
        self.answered = True

    async def edit_message_text(self, text, parse_mode=None):
        self.message._sent.append((text, parse_mode))


class FakeUser:
    def __init__(self, id):
        self.id = id


class FakeUpdate:
    def __init__(self, user_id, message=None, callback_query=None):
        self.effective_user = FakeUser(user_id)
        self.effective_message = message or FakeMessage()
        self.effective_chat = FakeChat()
        self.callback_query = callback_query


async def _run(handler, update, context=None):
    await handler.callback(update, context or AsyncMock())


def _handlers(bot: Bot):
    return build_handlers(
        session_factory=bot.session_factory,
        settings=bot.settings,
        risk_cfg=bot.settings.risk,
        on_approve=_make_approve(bot.settings, bot.session_factory),
        on_reject=_make_reject(bot.session_factory),
        on_backtest=_make_backtest(bot.settings, bot.session_factory),
        on_catalog_scores=_make_catalog_scores(bot.settings, bot.session_factory),
        allowed_users=(OWNER,),
    )


def _by_command(handlers) -> dict[str, Any]:
    return {next(iter(h.commands)): h for h in handlers if hasattr(h, "commands")}


def _callback(handlers):
    return next(h for h in handlers if hasattr(h, "pattern"))


def test_scenario_telegram_commands_all_work(bot: Bot):
    _prime_market(bot)
    handlers = _by_command(_handlers(bot))

    async def invoke(command: str, monkeypatch_stdout=None):
        update = FakeUpdate(user_id=OWNER)
        await _run(handlers[command], update)
        return update.effective_message._sent

    # /help lists the commands the trader can use.
    sent = asyncio.run(invoke("help"))
    assert sent and "/strategies" in sent[0][0]

    # /status shows mode, the active strategy, and the open position.
    sent = asyncio.run(invoke("status"))
    assert "ema_crossover" in sent[0][0]
    assert "mode: paper" in sent[0][0]
    assert "BTC/USDT" in sent[0][0]

    # /risk shows the limits (including the trailing stop the user enabled).
    sent = asyncio.run(invoke("risk"))
    assert "5.0%" in sent[0][0] or "trailing" in sent[0][0].lower()

    # /strategy lists DB versions; /summary works before any analytics exist.
    sent = asyncio.run(invoke("strategy"))
    assert "ema_crossover" in sent[0][0]
    asyncio.run(invoke("summary"))

    # /strategies renders the live catalog, ranked, with backtest buttons.
    update = FakeUpdate(user_id=OWNER)
    asyncio.run(_run(handlers["strategies"], update))
    text, parse_mode = update.effective_message._sent[0]
    assert parse_mode == "HTML"
    assert "ranked by PnL" in text
    assert "#1" in text
    assert len(text) <= 4096  # Telegram's hard message cap
    buttons = [
        b.callback_data
        for row in update.effective_message._markups[0].inline_keyboard
        for b in row
    ]
    assert "bt:ema_crossover" in buttons

    # An outsider gets nothing at all.
    outsider = FakeUpdate(user_id=999)
    asyncio.run(_run(handlers["status"], outsider))
    assert not outsider.effective_message._sent


def test_scenario_telegram_backtest_button_compares_params(bot: Bot):
    _prime_market(bot)
    # A tweaked version becomes "current"; the shipped defaults are the baseline.
    with bot.session() as session:
        session.add(Strategy(
            name="ema_crossover", version=2, status="draft",
            params=json.dumps({"fast_period": 5, "slow_period": 20, "position_pct": 0.2}),
        ))
        session.commit()

    handler = _callback(_handlers(bot))
    update = FakeUpdate(user_id=OWNER, callback_query=FakeQuery("bt:ema_crossover"))
    asyncio.run(_run(handler, update))

    assert update.callback_query.answered
    text = update.effective_message._sent[0][0]
    assert "Backtest: ema_crossover" in text
    assert "fast_period" in text and "slow_period" in text   # param diff
    assert "pnl/drawdown" in text                            # risk-adjusted metric
    assert "risk-adjusted" in text                           # the verdict


def test_scenario_telegram_approve_releases_a_change(bot: Bot):
    _prime_market(bot)
    with bot.session() as session:
        rec = create_pending_recommendation(
            session,
            kind="param_change",
            strategy_name="ema_crossover",
            params={"fast_period": 10, "slow_period": 26, "position_pct": 0.2},
            rationale="faster entries",
        )
        rec_id = rec.id

    handler = _callback(_handlers(bot))
    update = FakeUpdate(user_id=OWNER, callback_query=FakeQuery(f"approve:{rec_id}"))
    asyncio.run(_run(handler, update))

    with bot.session() as session:
        assert session.get(AIRecommendation, rec_id).status == "applied"
        active = session.execute(
            select(Strategy).where(Strategy.name == "ema_crossover",
                                   Strategy.status == "active")
        ).scalars().one()
        assert active.version == 2
    assert any("released" in t for t, _ in update.callback_query.message._sent)


def test_scenario_telegram_reject_leaves_the_strategy_alone(bot: Bot):
    with bot.session() as session:
        rec = create_pending_recommendation(
            session,
            kind="param_change",
            strategy_name="ema_crossover",
            params={"fast_period": 99, "slow_period": 200},
            rationale="too extreme",
        )
        rec_id = rec.id

    handler = _callback(_handlers(bot))
    update = FakeUpdate(user_id=OWNER, callback_query=FakeQuery(f"reject:{rec_id}"))
    asyncio.run(_run(handler, update))

    with bot.session() as session:
        assert session.get(AIRecommendation, rec_id).status == "rejected"
        assert session.execute(
            select(Strategy).where(Strategy.name == "ema_crossover",
                                   Strategy.status == "active")
        ).scalars().one().version == 1


# ---------------------------------------------------------------------------
# 5. the AI assistant
# ---------------------------------------------------------------------------

NEW_STRATEGY_CODE = '''\
class DipBuyer:
    """Buy after two consecutive down closes, a simple dip-buying idea."""

    def __init__(self, params=None):
        self.params = params or {}
        self.lookback = int(self.params.get("lookback", 2))

    def evaluate(self, symbol, candles):
        if len(candles) < self.lookback + 1:
            return None
        window = candles[-(self.lookback + 1):]
        down = all(window[i].close < window[i - 1].close for i in range(1, len(window)))
        if down:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="buy",
                ref_price=candles[-1].close,
                rationale="dip buyer: consecutive down closes",
                risk={"position_pct": 0.1},
            )
        return None
'''


class FakeLLM:
    """Scripted LLM: replays queued JSON responses in order."""

    def __init__(self, responses: Sequence[dict]):
        self._responses = list(responses)
        self.calls: list[Any] = []

    @property
    def enabled(self):
        return True

    def complete_json(self, messages, temperature=None):
        self.calls.append(messages)
        if not self._responses:
            raise RecommendationError("no more scripted responses")
        return self._responses.pop(0)


def test_scenario_chat_answers_paths_and_plain_text_disabled(bot: Bot):
    _prime_market(bot)

    # Without an AI key, a plain-text message gets a friendly explanation.
    handlers = _handlers(bot)
    chat_handler = next(
        h for h in handlers if type(h).__name__ == "MessageHandler"
    )
    update = FakeUpdate(user_id=OWNER)
    update.effective_message.text = "how is my bot doing?"
    asyncio.run(_run(chat_handler, update))
    assert "not configured" in update.effective_message._sent[0][0]


def test_scenario_chat_creates_a_strategy_the_trader_approves(bot: Bot):
    _prime_market(bot)
    bot.settings.ai.enabled = True

    # The trader says: "make me a dip buying strategy". The agent backtests its
    # idea and files a PENDING proposal — it never applies anything itself.
    client = FakeLLM([
        {"action": "tool", "name": "backtest",
         "args": {"strategy_name": "ema_crossover",
                  "params": {"fast_period": 5, "slow_period": 20, "position_pct": 0.2}}},
        {"action": "tool", "name": "propose_change",
         "args": {"kind": "new_strategy", "strategy_name": "dip_buyer",
                  "params": {"lookback": 2},
                  "rationale": "buy after two down closes",
                  "template": NEW_STRATEGY_CODE,
                  "indicator_deps": [],
                  "param_schema": [{"name": "lookback", "type": "int",
                                    "default": 2, "min": 1, "max": 10}]}},
        {"action": "reply", "text": "I drafted a dip-buying strategy for your approval."},
    ])
    result = run_agent(bot.session_factory, bot.settings,
                       "create a dip buying strategy", client=client)

    assert result["proposal_id"]
    assert result["kind"] == "new_strategy"
    assert "approval" in result["text"]
    # Nothing is on disk until the human approves.
    assert not (bot.plugin_dir / "dip_buyer.py").exists()
    assert "dip_buyer" not in known_names()

    # The trader taps Approve on the proposal card.
    handler = _callback(_handlers(bot))
    update = FakeUpdate(user_id=OWNER,
                        callback_query=FakeQuery(f"approve:{result['proposal_id']}"))
    asyncio.run(_run(handler, update))

    # The approved code is now a real, loadable plugin with a version row.
    assert (bot.plugin_dir / "dip_buyer.py").exists()
    assert is_plugin("dip_buyer")
    assert "dip_buyer" in known_names()
    with bot.session() as session:
        row = session.execute(
            select(Strategy).where(Strategy.name == "dip_buyer")
        ).scalars().one()
        assert row.status == "active"
        assert json.loads(row.params)["lookback"] == 2

    # And the very next catalog render already knows about it.
    assert any(e.name == "dip_buyer" for e in catalog_entries())


def test_scenario_editing_agent_created_code_cuts_a_new_version(bot: Bot):
    # Seed an approved plugin, then edit it through the same approval path.
    with bot.session() as session:
        store = RecommendationStore(session)
        store.apply(
            create_pending_recommendation(
                session,
                kind="new_strategy", strategy_name="dip_buyer",
                params={"lookback": 2}, rationale="seed",
                template=NEW_STRATEGY_CODE,
                param_schema=[{"name": "lookback", "type": "int", "default": 2}],
            ),
            candles=[], do_backtest=False,
        )
        edit = create_pending_recommendation(
            session, kind="edit_strategy", strategy_name="dip_buyer",
            params={"lookback": 3}, rationale="widen the window",
            template=NEW_STRATEGY_CODE.replace("consecutive down closes",
                                               "three-bar dip"),
        )
        strategy, _ = RecommendationStore(session).apply(edit, candles=[], do_backtest=False)
        assert strategy is not None and strategy.version == 2
        assert "three-bar dip" in (bot.plugin_dir / "dip_buyer.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 6. analytics, API, reconciliation, heartbeat
# ---------------------------------------------------------------------------

def test_scenario_analytics_api_and_health(bot: Bot):
    # Trade once so there is something for analytics to summarise.
    closes = _prime_market(bot)
    bot.feed.set("BTC/USDT", "1h", _anchored(_crash(closes + [closes[-1] * 1.012])))
    market_tick(bot.ctx)

    analytics_tick(bot.ctx)
    analytics_daily(bot.ctx)
    with bot.session() as session:
        summaries = session.execute(select(AnalyticsSummary)).scalars().all()
        assert len(summaries) == 2
        assert json.loads(summaries[0].metrics_json)

    # Reconciliation is clean, and the heartbeat is fresh.
    reconcile_tick(bot.ctx)
    heartbeat_tick(bot.ctx)
    assert bot.ctx.health.last_beat() > 0

    # The internal API reports health openly and status behind the token.
    client = TestClient(build_api(bot.settings, bot.session_factory,
                                  bot.ctx.health, token="scenario-token"))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["mode"] == "paper"
    assert client.get("/status").status_code == 401
    status = client.get("/status", headers={"Authorization": "Bearer scenario-token"})
    assert status.status_code == 200
    assert status.json()["active_strategy"]["name"] == "ema_crossover"
    assert client.get("/metrics").status_code == 200


# ---------------------------------------------------------------------------
# 7. restart
# ---------------------------------------------------------------------------

def test_scenario_restart_rebuilds_state_from_the_event_log(bot: Bot):
    closes = _prime_market(bot)
    bot.feed.set("BTC/USDT", "1h", _anchored(_crash(closes + [closes[-1] * 1.012])))
    market_tick(bot.ctx)

    with bot.session() as session:
        trade_count = len(session.execute(select(Trade)).scalars().all())

        # Simulate a restart: derived state is discarded and replayed.
        Ledger(session).rebuild_positions()
        assert len(session.execute(select(Trade)).scalars().all()) == trade_count
        # Our round trip is closed, so no phantom position reappears.
        assert session.execute(select(Position)).scalar_one().qty == pytest.approx(0.0)
