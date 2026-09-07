"""M7: live gateway + supervisor unit tests.

Covers:
- LiveGateway.place_market_order: filled path (executedQty + fills), unknown
  path (no executedQty), and the BinanceError -> "unknown" safety path.
- LiveGateway.get_order: filled / open status mapping.
- LiveGateway.get_open_orders: client_order_id extraction for reconcile.
- supervisor.reconcile: drift detection when a pending intent has no matching
  exchange order; "OK: no drift" when everything reconciles.
- supervisor.health: heartbeat file write/read, wake-lock availability.

All exchange interactions are faked via a stub client — nothing touches the
network.
"""
import json

import pytest

from algotrading.config import AIConfig, ApiConfig, MarketConfig, RiskConfig, ScheduleConfig, Settings
from algotrading.db import init_db, get_session_factory
from algotrading.db.models import Strategy, TradeIntent
from algotrading.execution.base import OrderResult
from algotrading.execution.live_gateway import LiveGateway
from algotrading.market.binance_rest import BinanceError
from algotrading.supervisor.health import HealthMonitor
from algotrading.supervisor.reconcile import reconcile, reconcile_and_report


class StubClient:
    """Records calls and returns canned payloads for the REST client."""

    def __init__(self):
        self.create_order_payload = {}
        self.create_order_error = None
        self.get_order_payload = {}
        self.get_order_error = None
        self.open_orders_payload = []
        self.calls = []

    def create_order(self, symbol, side, quantity, order_type="MARKET", client_order_id=None):
        self.calls.append(("create_order", symbol, side, quantity, client_order_id))
        if self.create_order_error:
            raise self.create_order_error
        return self.create_order_payload

    def get_order(self, symbol, client_order_id=None, order_id=None):
        self.calls.append(("get_order", symbol, client_order_id))
        if self.get_order_error:
            raise self.get_order_error
        return self.get_order_payload

    def open_orders(self, symbol=None):
        self.calls.append(("open_orders", symbol))
        return self.open_orders_payload


# ---------------------------------------------------------------------------
# LiveGateway: place_market_order
# ---------------------------------------------------------------------------

def test_live_place_market_order_filled_path():
    client = StubClient()
    client.create_order_payload = {
        "orderId": 12345,
        "executedQty": "0.5",
        "fills": [
            {"price": "61000.0", "qty": "0.3", "commission": "0.183"},
            {"price": "61100.0", "qty": "0.2", "commission": "0.1222"},
        ],
    }
    gw = LiveGateway(client)
    result = gw.place_market_order("BTCUSDT", "buy", 0.5, "idem-1")

    assert isinstance(result, OrderResult)
    assert result.order_id == "12345"
    assert result.status == "filled"
    assert result.filled_qty == 0.5
    # avg = (61000*0.3 + 61100*0.2) / 0.5 = (18300 + 12220) / 0.5 = 61040
    assert result.avg_fill_price == pytest.approx(61040.0)
    assert result.fee == pytest.approx(0.183 + 0.1222)
    assert client.calls[0] == ("create_order", "BTCUSDT", "BUY", 0.5, "idem-1")


def test_live_place_market_order_unknown_when_no_executed_qty():
    client = StubClient()
    client.create_order_payload = {"orderId": 12345}  # no executedQty
    gw = LiveGateway(client)
    result = gw.place_market_order("BTCUSDT", "buy", 0.5, "idem-2")

    assert result.status == "unknown"
    assert result.error  # explains why
    # fallback order_id when absent is the client order id; here orderId present
    assert result.order_id == "12345"


def test_live_place_market_order_error_returns_unknown():
    client = StubClient()
    client.create_order_error = BinanceError("insufficient balance")
    gw = LiveGateway(client)
    result = gw.place_market_order("BTCUSDT", "buy", 0.5, "idem-3")

    assert result.status == "unknown"
    assert "insufficient balance" in result.error
    # order_id falls back to the client idempotency key so reconcile can match
    assert result.order_id == "idem-3"


def test_live_place_market_order_uses_client_order_id_when_order_id_missing():
    client = StubClient()
    client.create_order_payload = {"executedQty": "1.0"}
    gw = LiveGateway(client)
    result = gw.place_market_order("BTCUSDT", "buy", 1.0, "idem-4")

    assert result.status == "filled"
    assert result.order_id == "idem-4"


# ---------------------------------------------------------------------------
# LiveGateway: get_order / get_open_orders
# ---------------------------------------------------------------------------

def test_live_get_order_filled_status():
    client = StubClient()
    client.get_order_payload = {"orderId": 99, "status": "FILLED", "executedQty": "0.5"}
    gw = LiveGateway(client)
    result = gw.get_order("BTCUSDT", "idem-1")

    assert result.status == "filled"
    assert result.filled_qty == 0.5
    assert client.calls[0][1] == "BTCUSDT"


def test_live_get_order_open_status():
    client = StubClient()
    client.get_order_payload = {"orderId": 99, "status": "NEW"}
    gw = LiveGateway(client)
    result = gw.get_order("BTCUSDT", "idem-1")

    assert result.status == "open"
    assert result.filled_qty == 0.0


def test_live_get_order_error_returns_unknown():
    client = StubClient()
    client.get_order_error = BinanceError("order not found")
    gw = LiveGateway(client)
    result = gw.get_order("BTCUSDT", "idem-1")

    assert result.status == "unknown"
    assert "order not found" in result.error


def test_live_get_open_orders_extracts_client_order_id():
    client = StubClient()
    client.open_orders_payload = [
        {"clientOrderId": "idem-1", "symbol": "BTCUSDT"},
        {"clientOrderId": "idem-2", "symbol": "BTCUSDT"},
    ]
    gw = LiveGateway(client)
    open_orders = gw.get_open_orders("BTCUSDT")

    ids = {o["client_order_id"] for o in open_orders}
    assert ids == {"idem-1", "idem-2"}
    assert client.calls[0] == ("open_orders", "BTCUSDT")


# ---------------------------------------------------------------------------
# supervisor.reconcile
# ---------------------------------------------------------------------------

def _settings():
    return Settings(
        mode="paper",
        market=MarketConfig(symbols=["BTC/USDT"], intervals=["1m"], max_staleness_seconds=300),
        risk=RiskConfig(),
        schedule=ScheduleConfig(),
        ai=AIConfig(enabled=False),
        api=ApiConfig(enabled=False),
    )


@pytest.fixture()
def session(tmp_path):
    db = str(tmp_path / "m7.db")
    init_db(db)
    factory = get_session_factory(db)
    s = factory()
    yield s
    s.close()


def _pending_intent(session, key, symbol="BTC/USDT"):
    intent = TradeIntent(
        symbol=symbol,
        side="buy",
        qty=0.1,
        idempotency_key=key,
        status="sent",
    )
    session.add(intent)
    session.commit()
    return intent


class FakeGateway:
    def __init__(self, open_orders=None):
        self._open_orders = open_orders or []

    def get_open_orders(self, symbol):
        return self._open_orders


def test_reconcile_detects_drift_for_orphaned_intent(session):
    _pending_intent(session, "idem-1")
    gw = FakeGateway(open_orders=[{"client_order_id": "idem-other"}])  # key not present
    drift = reconcile(session, gw, ["BTC/USDT"])

    assert len(drift) == 1
    assert "idem-1" in drift[0]


def test_reconcile_ok_when_intent_has_matching_exchange_order(session):
    _pending_intent(session, "idem-1")
    gw = FakeGateway(open_orders=[{"client_order_id": "idem-1"}])
    drift = reconcile(session, gw, ["BTC/USDT"])

    assert drift == []
    assert reconcile_and_report(session, gw, ["BTC/USDT"]) == "OK: no drift"


def test_reconcile_ignores_symbols_not_in_watchlist(session):
    _pending_intent(session, "idem-1", symbol="ETH/USDT")
    gw = FakeGateway(open_orders=[])
    drift = reconcile(session, gw, ["BTC/USDT"])  # ETH not watched

    assert drift == []


def test_reconcile_network_error_counts_as_drift(session):
    _pending_intent(session, "idem-1")

    class BrokenGateway:
        def get_open_orders(self, symbol):
            raise BinanceError("timeout")

    drift = reconcile(session, gw := BrokenGateway(), ["BTC/USDT"])
    assert len(drift) == 1
    assert "cannot reach exchange" in drift[0]


# ---------------------------------------------------------------------------
# supervisor.health
# ---------------------------------------------------------------------------

def test_health_beat_writes_and_reads_heartbeat(tmp_path):
    hm = HealthMonitor(str(tmp_path / "hb"))
    assert hm.last_beat() == 0.0  # nothing yet

    hm.beat()
    assert hm.last_beat() > 0.0
    assert (tmp_path / "hb").exists()


def test_wake_lock_noop_when_command_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("algotrading.supervisor.health.shutil.which", lambda _: None)
    from algotrading.supervisor.health import ensure_wake_lock
    # No subprocess should be attempted when termux-wake-lock is unavailable.
    assert ensure_wake_lock() is False


def test_wake_lock_succeeds_when_command_present(tmp_path, monkeypatch):
    monkeypatch.setattr("algotrading.supervisor.health.shutil.which", lambda _: "/usr/bin/termux-wake-lock")

    import subprocess
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("algotrading.supervisor.health.subprocess.run", fake_run)
    from algotrading.supervisor.health import ensure_wake_lock
    assert ensure_wake_lock() is True
    assert calls["cmd"] == ["termux-wake-lock"]
