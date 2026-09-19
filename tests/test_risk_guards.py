"""P0 regression tests: risk caps that are configured must actually apply.

These cover the four measured defects that made `config/settings.yaml` read as
documentation rather than enforcement:

  * `max_position_pct` was never applied on the buy path (only inside
    `RiskManager.size_position`, which the engine never called)
  * `max_daily_loss_pct` existed only in config validation
  * `_balance()` returned a hardcoded 100_000.0 while the configured paper
    balance defaulted to 10_000.0
  * the evaluation interval was silently 1h and the stale-data gate hardcoded
    `1h else 1m` arithmetic
"""
import json

import pytest

from algotrading.config import MarketConfig, RiskConfig, Settings
from algotrading.db import get_session_factory, init_db
from algotrading.db.models import OrderEvent, Position, Signal, Strategy, Trade, TradeIntent
from algotrading.execution import ExecutionEngine, PaperGateway, RiskManager
from algotrading.execution.risk import RiskDecision
from algotrading.ledger import Ledger
from algotrading.market.candles import INTERVAL_MS
from algotrading.scheduler.jobs import BotContext, stale_data
from sqlalchemy import select


@pytest.fixture()
def session(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    sess = get_session_factory(db)()
    sess.add(Strategy(name="ema_crossover", version=1, status="active", params="{}"))
    sess.commit()
    yield sess
    sess.close()


class FixedProvider:
    """Price source that never touches the network."""

    def __init__(self, price: float = 100.0) -> None:
        self.price = price

    def fetch_ticker_price(self, symbol: str) -> float:
        return self.price


def _buy_signal(session, *, symbol="BTC/USDT", price=100.0, position_pct=0.2) -> Signal:
    sig = Signal(
        symbol=symbol,
        side="buy",
        ref_price=price,
        risk_json=json.dumps({"position_pct": position_pct}),
        status="candidate",
    )
    session.add(sig)
    session.commit()
    return sig


def _engine(session, risk: RiskConfig, gateway=None) -> ExecutionEngine:
    gw = gateway or PaperGateway(FixedProvider(), slippage_pct=0.0)
    return ExecutionEngine(session, gw, RiskManager(risk), Ledger(session))


# --- RiskManager caps --------------------------------------------------------


def test_clamp_position_pct_caps_at_max_position_pct():
    risk = RiskManager(RiskConfig(max_position_pct=20.0))
    assert risk.clamp_position_pct(0.5) == pytest.approx(0.2)   # request above cap
    assert risk.clamp_position_pct(0.05) == pytest.approx(0.05)  # under cap passes
    assert risk.clamp_position_pct(0.0) == 0.0
    assert risk.clamp_position_pct(-0.3) == 0.0
    assert risk.clamp_position_pct("not-a-number") == 0.0


def test_size_by_pct_uses_balance_and_cap():
    risk = RiskManager(RiskConfig(max_position_pct=20.0))
    # 10_000 balance, 100 price, requested 50% -> capped to 20% = 20 units
    assert risk.size_by_pct(10_000.0, 100.0, 0.5) == pytest.approx(20.0)
    assert risk.size_by_pct(0.0, 100.0, 0.2) == 0.0
    assert risk.size_by_pct(10_000.0, 0.0, 0.2) == 0.0


def test_daily_loss_breached_boundaries_and_disable_switch():
    risk = RiskManager(RiskConfig(max_daily_loss_pct=3.0))
    breached, loss_pct = risk.daily_loss_breached(-200.0, 10_000.0)   # 2%
    assert breached is False and loss_pct == pytest.approx(2.0)

    breached, loss_pct = risk.daily_loss_breached(-300.0, 10_000.0)   # exactly 3%
    assert breached is True and loss_pct == pytest.approx(3.0)

    assert risk.daily_loss_breached(+50.0, 10_000.0)[0] is False      # a green day
    assert RiskManager(
        RiskConfig(max_daily_loss_pct=3.0, enforce_daily_loss=False)
    ).daily_loss_breached(-9_000.0, 10_000.0)[0] is False
    assert RiskManager(RiskConfig(max_daily_loss_pct=0.0)).daily_loss_breached(
        -9_000.0, 10_000.0
    )[0] is False


def test_check_buy_rejects_on_daily_loss_and_still_allows_sells():
    risk = RiskManager(RiskConfig(max_daily_loss_pct=3.0, max_open_positions=3))
    decision = risk.check_buy(
        symbol="BTC/USDT", balance=10_000.0, open_positions=0,
        last_entry_ts=None, daily_pnl=-400.0,
    )
    assert decision.allowed is False and "daily loss limit reached" in decision.reason

    # The guard is entry-only: closing a position is never blocked by it.
    assert risk.check_sell(has_position=True).allowed is True


# --- ExecutionEngine enforcement --------------------------------------------


def test_entry_size_capped_at_max_position_pct(session):
    """A signal asking for 50% of balance may only get max_position_pct (20%)."""
    risk = RiskConfig(paper_initial_balance=10_000.0, max_position_pct=20.0)
    engine = _engine(session, risk)
    sig = _buy_signal(session, position_pct=0.5)

    intent = engine.execute(sig.id)

    assert intent is not None and intent.status == "filled"
    # 10_000 * 0.20 / 100 = 20 units (not 50 units from the 50% request)
    assert intent.qty == pytest.approx(20.0)


def test_entry_size_uses_configured_balance_not_hardcoded(session):
    """The sizing basis is risk.paper_initial_balance, not 100_000."""
    risk = RiskConfig(paper_initial_balance=5_000.0, max_position_pct=20.0)
    engine = _engine(session, risk)
    sig = _buy_signal(session, position_pct=0.2)

    intent = engine.execute(sig.id)

    assert intent.qty == pytest.approx(10.0)  # 5_000 * 0.2 / 100


def test_gateway_balance_wins_when_available(session):
    """Live balances come from the venue when it can report one."""

    class BalanceGateway(PaperGateway):
        def get_balance(self) -> float:
            return 2_000.0

    risk = RiskConfig(paper_initial_balance=10_000.0, max_position_pct=20.0)
    engine = _engine(session, risk, gateway=BalanceGateway(FixedProvider(), 0.0))
    sig = _buy_signal(session, position_pct=0.2)

    intent = engine.execute(sig.id)

    assert intent.qty == pytest.approx(4.0)  # 2_000 * 0.2 / 100


def test_falling_back_to_configured_balance_when_gateway_errors(session):
    class BrokenGateway(PaperGateway):
        def get_balance(self) -> float:
            raise RuntimeError("venue unreachable")

    risk = RiskConfig(paper_initial_balance=10_000.0, max_position_pct=20.0)
    engine = _engine(session, risk, gateway=BrokenGateway(FixedProvider(), 0.0))
    sig = _buy_signal(session, position_pct=0.2)

    intent = engine.execute(sig.id)

    assert intent.qty == pytest.approx(20.0)  # 10_000 * 0.2 / 100


def test_daily_loss_blocks_new_entries_after_a_losing_day(session):
    """A 4% realized loss today blocks the buy and records why."""
    from datetime import datetime, timezone

    session.add(
        Trade(
            symbol="ETH/USDT",
            entry_qty=1.0,
            entry_avg_price=1_000.0,
            exit_avg_price=960.0,
            realized_pnl=-400.0,          # 4% of a 10_000 balance
            opened_at=datetime.now(timezone.utc),
            closed_at=datetime.now(timezone.utc),
        )
    )
    session.commit()

    engine = _engine(session, RiskConfig(paper_initial_balance=10_000.0, max_daily_loss_pct=3.0))
    sig = _buy_signal(session, position_pct=0.2)

    intent = engine.execute(sig.id)

    assert sig.status == "skipped"
    assert intent.status == "skipped" and intent.qty == 0.0
    events = session.execute(
        select(OrderEvent).where(OrderEvent.event_type == "risk_skipped")
    ).scalars().all()
    assert len(events) == 1
    assert "daily loss limit reached" in events[0].payload_json


def test_daily_loss_does_not_block_protective_sells(session):
    """With a loss limit breached, an exit still executes."""
    from datetime import datetime, timezone

    session.add(
        Trade(
            symbol="ETH/USDT", entry_qty=1.0, entry_avg_price=1_000.0,
            exit_avg_price=900.0, realized_pnl=-800.0,
            opened_at=datetime.now(timezone.utc), closed_at=datetime.now(timezone.utc),
        )
    )
    session.add(Position(symbol="BTC/USDT", qty=2.0, avg_price=100.0,
                         opened_at=datetime.now(timezone.utc)))
    session.commit()

    engine = _engine(session, RiskConfig(paper_initial_balance=10_000.0, max_daily_loss_pct=3.0))
    sig = Signal(
        symbol="BTC/USDT", side="sell",
        ref_price=100.0, risk_json="{}", status="candidate",
    )
    session.add(sig)
    session.commit()

    intent = engine.execute(sig.id)

    assert intent is not None
    assert intent.status == "filled"
    assert intent.qty == pytest.approx(2.0)


# --- evaluation interval / staleness ----------------------------------------


def test_primary_interval_prefers_explicit_eval_interval():
    settings = Settings(
        market=MarketConfig(intervals=["1m", "5m", "1h"], eval_interval="5m")
    )
    ctx = BotContext(
        settings=settings, session_factory=lambda: None, health=None, provider=None, gateway=None
    )
    assert ctx.primary_interval() == "5m"


def test_primary_interval_falls_back_to_legacy_rule():
    settings = Settings(market=MarketConfig(intervals=["1m", "5m", "1h"]))
    ctx = BotContext(
        settings=settings, session_factory=lambda: None, health=None, provider=None, gateway=None
    )
    assert ctx.primary_interval() == "1h"

    settings_5m_first = Settings(market=MarketConfig(intervals=["5m", "15m"]))
    ctx_5m = BotContext(
        settings=settings_5m_first, session_factory=lambda: None, health=None,
        provider=None, gateway=None,
    )
    assert ctx_5m.primary_interval() == "5m"


@pytest.mark.parametrize(
    "interval,age_ms,expected",
    [
        ("1m", 60_000, False),      # one 1m candle old + 300s grace -> fine
        ("1m", 400_000, True),      # over interval + grace
        ("15m", 900_000, False),    # the case the old hardcode would have frozen
        ("15m", 1_250_000, True),
        ("4h", 14_400_000, False),
        ("1d", 86_400_000, False),
        ("1d", 87_000_000, True),
    ],
)
def test_stale_data_uses_the_intervals_own_length(interval, age_ms, expected):
    settings = Settings(market=MarketConfig(max_staleness_seconds=300))
    now = 1_700_000_000_000
    assert stale_data(settings, now - age_ms, interval, now_ms=now) is expected


def test_interval_table_covers_every_configured_interval():
    """Every interval the shipped config stores must have a length in the table."""
    import yaml

    from algotrading.config import DEFAULT_SETTINGS

    with open(DEFAULT_SETTINGS, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for interval in raw["market"]["intervals"]:
        assert interval in INTERVAL_MS, f"{interval} missing from INTERVAL_MS"
    assert INTERVAL_MS["1w"] == 604_800_000
    assert INTERVAL_MS["30m"] == 1_800_000
