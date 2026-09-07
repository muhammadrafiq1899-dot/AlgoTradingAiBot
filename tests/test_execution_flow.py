"""End-to-end paper execution flow: signal -> intent -> fill -> ledger events.

This is the plan's key safety test: intent is persisted BEFORE the order is
sent, fills are recorded as immutable events, and positions/closed trades are
rebuilt from the event log after a simulated restart.
"""
import json

import pytest
from sqlalchemy import select

from algotrading.config import RiskConfig
from algotrading.db import init_db, get_session_factory
from algotrading.db.models import OrderEvent, Position, Signal, Strategy, Trade, TradeIntent
from algotrading.execution import ExecutionEngine, PaperGateway, RiskManager
from algotrading.execution.base import OrderResult
from algotrading.ledger import Ledger
from algotrading.market.base import Candle
from algotrading.market.demo import DemoProvider
from algotrading.strategy.engine import StrategyEngine


@pytest.fixture()
def session(tmp_path):
    db = str(tmp_path / "test.db")
    init_db(db)
    sess = get_session_factory(db)()
    sess.add(
        Strategy(name="ema_crossover", version=1, status="active",
                 params=json.dumps({"fast_period": 12, "slow_period": 26, "position_pct": 0.2}))
    )
    sess.commit()
    yield sess
    sess.close()


def cross_candles(symbol="BTC/USDT", down_first=True, seed=1):
    import random
    rng = random.Random(seed)
    price = 100.0
    closes = []
    for i in range(101):
        drift = (-0.006 if i < 90 else 0.012) if down_first else (0.006 if i < 90 else -0.012)
        price *= 1 + drift
        closes.append(price)
    return [Candle(symbol, "1m", i * 60_000, c, c, c, c, 1.0) for i, c in enumerate(closes)]


def _engine(session):
    gateway = PaperGateway(DemoProvider(seed=3), slippage_pct=0.05)
    risk = RiskManager(RiskConfig())
    ledger = Ledger(session)
    return ExecutionEngine(session, gateway, risk, ledger)


def test_signal_to_fill_to_ledger(session):
    cands = StrategyEngine(session).evaluate({"BTC/USDT": cross_candles()})
    assert cands and cands[0].side == "buy"
    sig = cands[0]

    engine = _engine(session)
    intent = engine.execute(sig.id)

    assert intent.status == "filled"
    assert intent.idempotency_key
    assert intent.qty > 0

    # signal consumed
    assert session.get(Signal, sig.id).status == "sent"

    # position opened
    pos = session.execute(select(Position).where(Position.symbol == "BTC/USDT")).scalar_one()
    assert pos.qty == pytest.approx(intent.qty)
    assert pos.avg_price > 0

    # exactly one fill event recorded
    events = session.execute(select(OrderEvent)).scalars().all()
    assert [e.event_type for e in events] == ["fill"]


def test_open_then_close_records_trade(session):
    engine = _engine(session)

    opens = StrategyEngine(session).evaluate({"BTC/USDT": cross_candles(down_first=True)})
    engine.execute(opens[0].id)
    pos = session.execute(select(Position).where(Position.symbol == "BTC/USDT")).scalar_one()
    assert pos.qty > 0

    closes = StrategyEngine(session).evaluate({"BTC/USDT": cross_candles(down_first=False)})
    sell = [c for c in closes if c.side == "sell"]
    assert sell, "expected a sell signal"
    engine.execute(sell[0].id)

    assert session.execute(select(Position).where(Position.symbol == "BTC/USDT")).scalar_one().qty == 0
    trades = session.execute(select(Trade)).scalars().all()
    assert len(trades) == 1
    assert trades[0].realized_pnl != 0
    assert trades[0].strategy_id is not None


def test_duplicate_signal_rejected_idempotently(session):
    engine = _engine(session)
    cands = StrategyEngine(session).evaluate({"BTC/USDT": cross_candles()})
    engine.execute(cands[0].id)
    events_before = len(session.execute(select(OrderEvent)).scalars().all())

    # Same signal again must not place a second order.
    result = engine.execute(cands[0].id)
    assert result is None
    events_after = len(session.execute(select(OrderEvent)).scalars().all())
    assert events_after == events_before


def test_rebuild_from_events_after_restart(session):
    engine = _engine(session)
    opens = StrategyEngine(session).evaluate({"BTC/USDT": cross_candles()})
    engine.execute(opens[0].id)
    closes = StrategyEngine(session).evaluate({"BTC/USDT": cross_candles(down_first=False)})
    sell = [c for c in closes if c.side == "sell"]
    engine.execute(sell[0].id)

    # Simulate a crash: drop derived rows, keep events.
    session.execute(select(Position)).scalars().all()
    session.query(Position).delete()
    session.query(Trade).delete()
    session.commit()

    # Restart: rebuild from the immutable event log.
    Ledger(session).rebuild_positions()
    assert session.execute(select(Position).where(Position.symbol == "BTC/USDT")).scalar_one().qty == 0
    assert len(session.execute(select(Trade)).scalars().all()) == 1


def test_risk_skip_when_max_positions(session):
    for i, sym in enumerate(["A/USDT", "B/USDT", "C/USDT"]):
        session.add(Position(symbol=sym, qty=1.0, avg_price=100.0))
    session.commit()

    engine = _engine(session)
    cands = StrategyEngine(session).evaluate({"BTC/USDT": cross_candles()})
    intent = engine.execute(cands[0].id)

    assert intent.status == "skipped"
    assert session.get(Signal, cands[0].id).status == "skipped"
    events = session.execute(select(OrderEvent)).scalars().all()
    assert events[0].event_type == "risk_skipped"


def test_paper_slippage_direction(session):
    from algotrading.execution.paper_gateway import PaperGateway

    class FixedProvider:
        def fetch_ticker_price(self, symbol):
            return 100.0

    gw = PaperGateway(FixedProvider(), slippage_pct=1.0)  # 1% slippage
    buy = gw.place_market_order("X", "buy", 1.0, "k1")
    sell = gw.place_market_order("X", "sell", 1.0, "k2")
    assert buy.avg_fill_price == pytest.approx(101.0)   # buy pays more
    assert sell.avg_fill_price == pytest.approx(99.0)   # sell receives less
    assert buy.fee > 0
