"""P2-8: venue selection, real order types, and the exchange-side stop.

No network anywhere: the REST layer is exercised through a recording fake
session, and the engine runs against `PaperGateway` with a fixed price source.

Covered:
- paper limit orders: immediate fill when the limit crosses, otherwise resting,
  listed by `get_open_orders`, removable with `cancel_order`;
- paper stop orders: rest until triggered, then fill with the configured
  slippage (and the unchanged 0.1% fee model);
- paper `get_balance` (configured balance / "no opinion") and the untouched
  market-order maths;
- live REST request shapes for LIMIT, STOP_LOSS and STOP_LOSS_LIMIT
  (path, params, clientOrderId, timeInForce), `list_open_orders`, `get_balance`;
- venue selection: testnet base URL, explicit `base_url` winning, and a loud
  failure when the testnet keys are missing (never a silent mainnet fallback);
- the engine: a resting stop after a BUY fill when enabled, nothing when
  disabled or unsupported, cancel+re-place when the trailing stop advances,
  cancel when the position closes, a limit entry left `sent`, and alerting that
  can never break the tick.
"""
from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from algotrading.config import MarketConfig, RiskConfig, Settings
from algotrading.db import get_session_factory, init_db
from algotrading.db.models import OrderEvent, Position, Signal, Trade, TradeIntent
from algotrading.execution import ExecutionEngine, PaperGateway, RiskManager
from algotrading.execution import engine as engine_mod
from algotrading.execution.base import OrderResult
from algotrading.execution.live_gateway import LiveGateway
from algotrading.ledger import Ledger
from algotrading.market.binance_rest import (
    SPOT_BASE,
    TESTNET_BASE,
    BinanceError,
    BinanceRestClient,
    resolve_venue,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class FixedProvider:
    """Price source that never touches the network."""

    def __init__(self, price: float = 100.0) -> None:
        self.price = price

    def fetch_ticker_price(self, symbol: str) -> float:
        return self.price


@pytest.fixture()
def session(tmp_path):
    db = str(tmp_path / "p28.db")
    init_db(db)
    sess = get_session_factory(db)()
    yield sess
    sess.close()


def _buy_signal(session, *, symbol="BTC/USDT", price=100.0, risk=None) -> Signal:
    sig = Signal(
        symbol=symbol,
        side="buy",
        ref_price=price,
        risk_json=json.dumps(risk or {"position_pct": 0.2}),
        status="candidate",
    )
    session.add(sig)
    session.commit()
    return sig


def _sell_signal(session, *, symbol="BTC/USDT", price=100.0) -> Signal:
    sig = Signal(
        symbol=symbol, side="sell", ref_price=price, risk_json="{}", status="candidate"
    )
    session.add(sig)
    session.commit()
    return sig


def _engine(session, risk: RiskConfig, gateway) -> ExecutionEngine:
    return ExecutionEngine(session, gateway, RiskManager(risk), Ledger(session))


def _events(session, event_type: str) -> list[OrderEvent]:
    return session.execute(
        select(OrderEvent).where(OrderEvent.event_type == event_type).order_by(OrderEvent.id)
    ).scalars().all()


def _payload(event: OrderEvent) -> dict:
    return json.loads(event.payload_json or "{}")


@pytest.fixture()
def alerts(monkeypatch):
    """Record alerts instead of sending them (and prove they never raise)."""
    sent: list[tuple[str, str, dict]] = []

    def _record(kind, title, body="", **fields):
        sent.append((kind, title, fields))
        return True

    monkeypatch.setattr(engine_mod, "notify", _record)
    return sent


# ---------------------------------------------------------------------------
# paper gateway: limit orders
# ---------------------------------------------------------------------------

def test_paper_limit_rests_when_it_does_not_cross():
    gw = PaperGateway(FixedProvider(100.0), slippage_pct=0.0)
    result = gw.place_limit_order("BTC/USDT", "buy", 2.0, 90.0, "lim-1")

    assert result.status == "open"
    assert result.filled_qty == 0.0
    assert result.order_id == "lim-1"
    assert [o["client_order_id"] for o in gw.get_open_orders("BTC/USDT")] == ["lim-1"]
    assert gw.get_order("BTC/USDT", "lim-1").status == "open"


def test_paper_limit_fills_immediately_when_it_crosses():
    gw = PaperGateway(FixedProvider(100.0), slippage_pct=1.0)
    result = gw.place_limit_order("BTC/USDT", "buy", 2.0, 100.0, "lim-2")

    assert result.status == "filled"
    assert result.filled_qty == 2.0
    # Filled at the limit, not at the (better) market: conservative by design.
    assert result.avg_fill_price == pytest.approx(100.0)
    assert result.fee == pytest.approx(2.0 * 100.0 * 0.001)
    assert gw.get_open_orders("BTC/USDT") == []


def test_paper_sell_limit_fills_at_or_above_market_only():
    gw = PaperGateway(FixedProvider(100.0), slippage_pct=0.0)
    assert gw.place_limit_order("X", "sell", 1.0, 100.0, "s1").status == "filled"
    resting = gw.place_limit_order("X", "sell", 1.0, 110.0, "s2")
    assert resting.status == "open"
    assert [o["client_order_id"] for o in gw.get_open_orders("X")] == ["s2"]


# ---------------------------------------------------------------------------
# paper gateway: stop orders
# ---------------------------------------------------------------------------

def test_paper_stop_rests_until_triggered():
    gw = PaperGateway(FixedProvider(100.0), slippage_pct=0.0)
    result = gw.place_stop_order("BTC/USDT", "sell", 1.0, 95.0, client_order_id="stp-1")

    assert result.status == "open"
    open_orders = gw.get_open_orders("BTC/USDT")
    assert len(open_orders) == 1
    assert open_orders[0]["order"]["type"] == "stop"
    assert open_orders[0]["order"]["stop_price"] == 95.0


def test_paper_stop_triggers_with_slippage_and_fee():
    gw = PaperGateway(FixedProvider(100.0), slippage_pct=1.0)
    # market (100) <= stop (101) for a sell -> triggered now
    result = gw.place_stop_order("BTC/USDT", "sell", 1.0, 101.0, client_order_id="stp-2")

    assert result.status == "filled"
    assert result.avg_fill_price == pytest.approx(99.0)      # 1% adverse slippage
    assert result.fee == pytest.approx(1.0 * 99.0 * 0.001)
    assert gw.get_open_orders("BTC/USDT") == []


def test_paper_buy_stop_triggers_when_price_is_at_the_stop():
    gw = PaperGateway(FixedProvider(100.0), slippage_pct=1.0)
    assert gw.place_stop_order("X", "buy", 1.0, 99.0, client_order_id="stp-3").status == "filled"
    assert gw.place_stop_order("X", "buy", 1.0, 101.0, client_order_id="stp-4").status == "open"


def test_paper_stop_limit_keeps_the_limit_price():
    gw = PaperGateway(FixedProvider(100.0), slippage_pct=0.0)
    result = gw.place_stop_order("X", "sell", 1.0, 95.0, limit_price=94.9, client_order_id="stp-5")

    assert result.status == "open"
    order = gw.get_open_orders("X")[0]["order"]
    assert order["type"] == "stop_limit"
    assert order["stop_price"] == 95.0
    assert order["price"] == 94.9


# ---------------------------------------------------------------------------
# paper gateway: cancel / balance / unchanged market maths
# ---------------------------------------------------------------------------

def test_paper_cancel_removes_a_resting_order_and_is_idempotent():
    gw = PaperGateway(FixedProvider(100.0), slippage_pct=0.0)
    gw.place_limit_order("X", "buy", 1.0, 90.0, "cancel-1")

    first = gw.cancel_order("X", "cancel-1")
    assert first.status == "canceled"
    assert gw.get_open_orders("X") == []
    # Cancelling again is the desired end state, not an error.
    assert gw.cancel_order("X", "cancel-1").status == "canceled"


def test_paper_get_balance_reports_configured_value_or_nothing():
    assert PaperGateway(FixedProvider(100.0), initial_balance=5_000.0).get_balance() == 5_000.0
    # No configured balance = "no opinion": the engine keeps using the risk config.
    assert PaperGateway(FixedProvider(100.0)).get_balance() is None


def test_paper_market_order_maths_unchanged():
    gw = PaperGateway(FixedProvider(100.0), slippage_pct=1.0)
    buy = gw.place_market_order("X", "buy", 2.0, "k1")
    sell = gw.place_market_order("X", "sell", 2.0, "k2")

    assert buy.avg_fill_price == pytest.approx(101.0)
    assert sell.avg_fill_price == pytest.approx(99.0)
    assert buy.fee == pytest.approx(2.0 * 101.0 * 0.001)
    assert sell.fee == pytest.approx(2.0 * 99.0 * 0.001)
    assert buy.status == sell.status == "filled"


# ---------------------------------------------------------------------------
# live REST client: request shapes (fake session, no network)
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records requests and replays canned payloads."""

    def __init__(self, payloads=None):
        self.calls: list[tuple[str, str, dict, dict]] = []
        self.payloads = payloads or {}

    def _record(self, method, url, params, headers):
        self.calls.append((method, url, dict(params or {}), dict(headers or {})))
        return FakeResponse(self.payloads.get(method, self.payloads.get("default", {})))

    def post(self, url, params=None, headers=None, timeout=None):
        return self._record("POST", url, params, headers)

    def get(self, url, params=None, headers=None, timeout=None):
        return self._record("GET", url, params, headers)

    def delete(self, url, params=None, headers=None, timeout=None):
        return self._record("DELETE", url, params, headers)


def _client(payloads=None) -> tuple[BinanceRestClient, FakeSession]:
    client = BinanceRestClient(api_key="key", api_secret="secret")
    fake = FakeSession(payloads)
    client._session = fake
    return client, fake


def test_rest_limit_order_request_shape():
    client, fake = _client({"POST": {"orderId": 7, "status": "NEW"}})
    client.place_limit_order("BTC/USDT", "buy", 1.5, 61_000.0, "cid-1")

    method, url, params, headers = fake.calls[0]
    assert method == "POST"
    assert url == SPOT_BASE + "/api/v3/order"
    assert params["symbol"] == "BTCUSDT"
    assert params["side"] == "BUY"
    assert params["type"] == "LIMIT"
    assert params["timeInForce"] == "GTC"
    assert params["price"] == 61_000.0
    assert params["quantity"] == 1.5
    assert params["newClientOrderId"] == "cid-1"
    # signed request: timestamp + signature present, API key header set
    assert params["timestamp"] and params["signature"]
    assert headers["X-MBX-APIKEY"] == "key"


def test_rest_stop_loss_request_shape_has_no_limit_price():
    client, fake = _client({"POST": {"orderId": 9, "status": "NEW"}})
    client.place_stop_order("BTC/USDT", "sell", 1.5, 59_000.0, client_order_id="cid-2")

    _, _, params, _ = fake.calls[0]
    assert params["type"] == "STOP_LOSS"
    assert params["stopPrice"] == 59_000.0
    assert params["side"] == "SELL"
    # A market stop takes no price/timeInForce.
    assert "price" not in params
    assert "timeInForce" not in params
    assert params["newClientOrderId"] == "cid-2"


def test_rest_stop_loss_limit_request_shape():
    client, fake = _client({"POST": {"orderId": 10, "status": "NEW"}})
    client.place_stop_order(
        "BTC/USDT", "sell", 1.5, 59_000.0, 58_950.0, "cid-3", time_in_force="IOC"
    )

    _, _, params, _ = fake.calls[0]
    assert params["type"] == "STOP_LOSS_LIMIT"
    assert params["stopPrice"] == 59_000.0
    assert params["price"] == 58_950.0
    assert params["timeInForce"] == "IOC"
    assert params["newClientOrderId"] == "cid-3"


def test_rest_list_open_orders_and_cancel_use_the_order_paths():
    client, fake = _client({"GET": [{"orderId": 1, "clientOrderId": "a"}], "DELETE": {"status": "CANCELED"}})
    assert client.list_open_orders("BTC/USDT") == [{"orderId": 1, "clientOrderId": "a"}]
    assert fake.calls[-1][1] == SPOT_BASE + "/api/v3/openOrders"
    assert fake.calls[-1][2]["symbol"] == "BTCUSDT"

    client.cancel_order("BTC/USDT", client_order_id="a")
    assert fake.calls[-1][0] == "DELETE"
    assert fake.calls[-1][2]["origClientOrderId"] == "a"


def test_rest_get_balance_reuses_the_account_endpoint():
    client, fake = _client(
        {"GET": {"balances": [
            {"asset": "USDT", "free": "900.5", "locked": "99.5"},
            {"asset": "BTC", "free": "0.0", "locked": "0.0"},
        ]}}
    )
    assert client.get_balance("USDT") == pytest.approx(1_000.0)
    assert client.get_balance("btc") == 0.0        # zero balances are omitted
    assert client.get_balance("ETH") == 0.0
    assert fake.calls[0][1] == SPOT_BASE + "/api/v3/account"
    assert fake.calls[0][3]["X-MBX-APIKEY"] == "key"


# ---------------------------------------------------------------------------
# venue selection
# ---------------------------------------------------------------------------

def test_resolve_venue_picks_testnet_when_asked():
    assert resolve_venue(MarketConfig()) == (SPOT_BASE, False)
    assert resolve_venue(MarketConfig(use_testnet=True)) == (TESTNET_BASE, True)
    assert resolve_venue(Settings(market=MarketConfig(use_testnet=True))) == (TESTNET_BASE, True)


# --- startup validation follows the selected venue ---------------------------


def _control_secrets(monkeypatch, **values):
    from algotrading import config as config_module

    for name in (
        "binance_api_key",
        "binance_api_secret",
        "binance_testnet_key",
        "binance_testnet_secret",
        "api_token",
    ):
        monkeypatch.setitem(config_module._secrets, name, "")
    for name, value in values.items():
        monkeypatch.setitem(config_module._secrets, name, value)


def test_live_testnet_does_not_require_mainnet_keys(monkeypatch):
    from algotrading.config import validate_settings

    _control_secrets(monkeypatch, binance_testnet_key="tk", binance_testnet_secret="ts")
    validate_settings(Settings(mode="live", market=MarketConfig(use_testnet=True)))


def test_live_testnet_requires_testnet_keys(monkeypatch):
    from algotrading.config import validate_settings

    _control_secrets(monkeypatch)
    with pytest.raises(RuntimeError) as excinfo:
        validate_settings(Settings(mode="live", market=MarketConfig(use_testnet=True)))
    assert "BINANCE_TESTNET_API_KEY" in str(excinfo.value)


def test_live_mainnet_still_requires_mainnet_keys(monkeypatch):
    from algotrading.config import validate_settings

    # Testnet keys present, mainnet missing -> mainnet mode must still refuse.
    _control_secrets(monkeypatch, binance_testnet_key="tk", binance_testnet_secret="ts")
    with pytest.raises(RuntimeError) as excinfo:
        validate_settings(Settings(mode="live", market=MarketConfig(use_testnet=False)))
    assert "BINANCE_API_KEY" in str(excinfo.value)


def test_explicit_base_url_wins_over_testnet():
    cfg = MarketConfig(use_testnet=True, base_url="https://mirror.example/api/")
    assert resolve_venue(cfg) == ("https://mirror.example/api", True)

    client = BinanceRestClient.from_settings(Settings(market=cfg))
    assert client.base_url == "https://mirror.example/api"
    # testnet=False still means mainnet: an empty base_url must never become the URL
    assert BinanceRestClient().base_url == SPOT_BASE
    assert BinanceRestClient(testnet=True).base_url == TESTNET_BASE
    assert BinanceRestClient(testnet=True).testnet is True


def test_from_settings_uses_the_testnet_base_url():
    client = BinanceRestClient.from_settings(
        Settings(market=MarketConfig(use_testnet=True)), api_key="k", api_secret="s"
    )
    assert client.base_url == TESTNET_BASE


def test_missing_testnet_keys_fail_loudly_instead_of_falling_back_to_mainnet():
    settings = Settings(market=MarketConfig(use_testnet=True))
    with pytest.raises(BinanceError) as excinfo:
        BinanceRestClient.from_settings(settings, api_key="", api_secret="", require_keys=True)

    message = str(excinfo.value)
    assert "testnet" in message.lower()
    # The failure must not quietly produce a mainnet client.
    assert SPOT_BASE not in message

    # With keys present it builds, on the testnet base.
    ok = BinanceRestClient.from_settings(settings, api_key="k", api_secret="s", require_keys=True)
    assert ok.base_url == TESTNET_BASE


# ---------------------------------------------------------------------------
# live gateway module wiring (no network: the client is only constructed)
# ---------------------------------------------------------------------------

class _Ctx:
    def __init__(self, settings, market_provider=None):
        self.settings = settings
        self.gateway = None
        self.provider = None
        self.provided: dict[str, object] = {}
        self._market = market_provider

    def get(self, capability):
        return self._market

    def provide(self, capability, service):
        self.provided[capability] = service


def test_paper_module_gives_the_gateway_the_configured_balance():
    from algotrading.modules.builtin import execution_gateway as mod

    settings = Settings(risk=RiskConfig(paper_initial_balance=7_500.0, slippage_pct=0.05))
    ctx = _Ctx(settings, market_provider=FixedProvider())
    mod.PaperGatewayModule().setup(ctx)

    assert isinstance(ctx.gateway, PaperGateway)
    assert ctx.gateway.get_balance() == 7_500.0


def test_market_module_serves_the_same_venue_as_the_orders():
    """Testnet orders must be priced off testnet candles, and said so up front."""
    from algotrading.modules.builtin import market_provider as mod

    ctx = _Ctx(Settings(market=MarketConfig(use_testnet=True)))
    mod.BinanceMarketModule().setup(ctx)
    assert ctx.provider._client.base_url == TESTNET_BASE

    description = mod.BinanceMarketModule.spec.description.upper()
    assert "TESTNET" in description
    assert "THIN" in description          # the honest warning is part of the contract

    mainnet_ctx = _Ctx(Settings(market=MarketConfig()))
    mod.BinanceMarketModule().setup(mainnet_ctx)
    assert mainnet_ctx.provider._client.base_url == SPOT_BASE


def test_market_module_honours_an_explicit_base_url():
    from algotrading.modules.builtin import market_provider as mod

    ctx = _Ctx(Settings(market=MarketConfig(use_testnet=True, base_url="https://mirror.test")))
    mod.BinanceMarketModule().setup(ctx)
    assert ctx.provider._client.base_url == "https://mirror.test"


def _live_module_setup(settings, secrets, monkeypatch):
    from algotrading.modules.builtin import execution_gateway as mod

    monkeypatch.setattr(mod, "get_secret", lambda name: secrets.get(name, ""))
    ctx = _Ctx(settings)
    mod.LiveGatewayModule().setup(ctx)
    return ctx


def test_live_module_uses_testnet_keys_and_base_url(monkeypatch):
    settings = Settings(market=MarketConfig(use_testnet=True))
    ctx = _live_module_setup(
        settings,
        {"binance_testnet_key": "tk", "binance_testnet_secret": "ts"},
        monkeypatch,
    )
    assert isinstance(ctx.gateway, LiveGateway)
    assert ctx.gateway._client.base_url == TESTNET_BASE


def test_live_module_refuses_to_start_without_testnet_keys(monkeypatch):
    from algotrading.modules.base import ModuleError

    settings = Settings(market=MarketConfig(use_testnet=True))
    with pytest.raises(ModuleError) as excinfo:
        _live_module_setup(settings, {"binance_api_key": "mk", "binance_api_secret": "ms"}, monkeypatch)

    message = str(excinfo.value)
    assert "BINANCE_TESTNET_API_KEY" in message
    assert TESTNET_BASE in message


def test_live_module_mainnet_still_uses_mainnet_keys(monkeypatch):
    settings = Settings(market=MarketConfig())
    ctx = _live_module_setup(
        settings, {"binance_api_key": "mk", "binance_api_secret": "ms"}, monkeypatch
    )
    assert ctx.gateway._client.base_url == SPOT_BASE


# ---------------------------------------------------------------------------
# engine: exchange-side protective stop
# ---------------------------------------------------------------------------

def _stopped_engine(session, *, exchange_stop=True, trailing_pct=2.0, gateway=None):
    risk = RiskConfig(
        paper_initial_balance=10_000.0,
        max_position_pct=20.0,
        trailing_stop_pct=trailing_pct,
        exchange_stop_enabled=exchange_stop,
    )
    gw = gateway or PaperGateway(FixedProvider(100.0), slippage_pct=0.0)
    return _engine(session, risk, gw), gw


def test_engine_rests_a_stop_on_the_venue_after_a_buy_fill(session, alerts):
    engine, gw = _stopped_engine(session)
    sig = _buy_signal(session)

    intent = engine.execute(sig.id)

    assert intent.status == "filled"
    open_orders = gw.get_open_orders("BTC/USDT")
    assert len(open_orders) == 1
    order = open_orders[0]["order"]
    assert order["side"] == "sell"
    assert order["type"] == "stop"
    assert order["stop_price"] == pytest.approx(98.0)     # 100 * (1 - 2%)
    assert order["qty"] == pytest.approx(intent.qty)

    placed = _events(session, "stop_placed")
    assert len(placed) == 1
    payload = _payload(placed[0])
    assert payload["symbol"] == "BTC/USDT"
    assert payload["stop_price"] == pytest.approx(98.0)
    assert payload["client_order_id"] == f"stop-{intent.id}-1"
    assert payload["exchange_order_id"] == f"stop-{intent.id}-1"
    assert payload["intent_id"] == intent.id
    # and the fill alert went out best-effort
    assert any(kind == "fill" for kind, _title, _f in alerts)


def test_engine_places_nothing_when_exchange_stop_is_disabled(session, alerts):
    engine, gw = _stopped_engine(session, exchange_stop=False)
    intent = engine.execute(_buy_signal(session).id)

    assert intent.status == "filled"
    assert gw.get_open_orders("BTC/USDT") == []
    assert _events(session, "stop_placed") == []


def test_engine_places_nothing_when_the_gateway_lacks_the_capability(session, alerts):
    class ThinGateway:
        """A gateway with only the mandatory two methods (a thin test double)."""

        def __init__(self):
            self.calls = 0

        def place_market_order(self, symbol, side, qty, client_order_id):
            self.calls += 1
            return OrderResult(order_id=client_order_id, status="filled",
                               filled_qty=qty, avg_fill_price=100.0, fee=0.0)

        def get_order(self, symbol, client_order_id):
            return OrderResult(order_id=client_order_id, status="filled")

    thin = ThinGateway()
    engine, _ = _stopped_engine(session, gateway=thin)
    intent = engine.execute(_buy_signal(session).id)

    assert intent.status == "filled"          # the tick is intact
    assert thin.calls == 1
    assert _events(session, "stop_placed") == []
    assert session.execute(select(Position)).scalar_one().qty > 0


def test_engine_retry_does_not_double_place_the_resting_stop(session, alerts):
    engine, gw = _stopped_engine(session)
    sig = _buy_signal(session)
    intent = engine.execute(sig.id)

    # A retry of the same placement (crash between order and event commit, or a
    # reconcile re-run) must find the existing stop and leave it alone.
    engine._maybe_place_exchange_stop(intent, sig, 100.0)

    assert len(gw.get_open_orders("BTC/USDT")) == 1
    assert len(_events(session, "stop_placed")) == 1


def test_engine_uses_the_signals_own_stop_price_when_given(session, alerts):
    engine, gw = _stopped_engine(session)
    sig = _buy_signal(session, risk={"position_pct": 0.2, "stop_price": 90.0})

    engine.execute(sig.id)

    assert gw.get_open_orders("BTC/USDT")[0]["order"]["stop_price"] == pytest.approx(90.0)


def test_engine_skips_a_stop_that_is_not_below_the_entry(session, alerts):
    engine, gw = _stopped_engine(session)
    sig = _buy_signal(session, risk={"position_pct": 0.2, "stop_price": 120.0})

    intent = engine.execute(sig.id)

    assert intent.status == "filled"          # the fill still stands
    assert gw.get_open_orders("BTC/USDT") == []
    assert _events(session, "stop_placed") == []


def test_engine_cancels_the_resting_stop_when_the_position_closes(session, alerts):
    engine, gw = _stopped_engine(session)
    engine.execute(_buy_signal(session).id)
    assert len(gw.get_open_orders("BTC/USDT")) == 1

    sell = _sell_signal(session)
    intent = engine.execute(sell.id)

    assert intent.status == "filled"
    assert session.execute(select(Position)).scalar_one().qty == 0
    assert gw.get_open_orders("BTC/USDT") == []
    canceled = _events(session, "stop_canceled")
    assert len(canceled) == 1
    assert _payload(canceled[0])["symbol"] == "BTC/USDT"
    assert len(session.execute(select(Trade)).scalars().all()) == 1


def test_engine_moves_the_resting_stop_when_the_trailing_stop_advances(session, alerts):
    provider = FixedProvider(100.0)
    engine, gw = _stopped_engine(session, gateway=PaperGateway(provider, slippage_pct=0.0))
    intent = engine.execute(_buy_signal(session).id)

    # The market really moved: a 107.8 stop is only "resting" below 110.
    provider.price = 110.0
    engine.update_trailing_stops({"BTC/USDT": 110.0})

    open_orders = gw.get_open_orders("BTC/USDT")
    assert len(open_orders) == 1                       # moved, never stacked
    assert open_orders[0]["order"]["stop_price"] == pytest.approx(107.8)  # 110 * 0.98

    placed = _events(session, "stop_placed")
    assert len(placed) == 2
    assert _payload(placed[0])["client_order_id"] == f"stop-{intent.id}-1"
    assert _payload(placed[1])["client_order_id"] == f"stop-{intent.id}-2"
    assert len(_events(session, "stop_canceled")) == 1


def test_reprice_ignores_a_stop_that_is_already_through_the_market(session, alerts):
    """A stop the venue fills on arrival is not tracked as resting."""
    engine, gw = _stopped_engine(session)          # static price 100
    intent = engine.execute(_buy_signal(session).id)

    engine.update_trailing_stops({"BTC/USDT": 110.0})   # new stop 107.8 > market 100

    assert gw.get_open_orders("BTC/USDT") == []
    assert len(_events(session, "stop_placed")) == 1    # only the entry stop
    assert _events(session, "stop_canceled")[0].event_type == "stop_canceled"
    assert intent.status == "filled"


def test_engine_resting_stop_survives_a_restart_new_engine(session, alerts):
    engine, gw = _stopped_engine(session)
    intent = engine.execute(_buy_signal(session).id)

    # "Restart": a brand-new engine + gateway over the same DB (no memory shares).
    fresh_engine, fresh_gw = _stopped_engine(session)

    resting = fresh_engine._resting_stop("BTC/USDT")
    assert resting is not None
    assert resting["client_order_id"] == f"stop-{intent.id}-1"
    # The fresh gateway has no orders yet, so a re-placement would be a new one —
    # but the retry guard reads the event log first and refuses to double-place.
    fresh_engine._maybe_place_exchange_stop(intent, session.get(Signal, intent.signal_id), 100.0)
    assert len(_events(session, "stop_placed")) == 1
    assert fresh_gw.get_open_orders("BTC/USDT") == []


def test_engine_survives_a_rejected_stop_placement(session, alerts):
    class RejectingStopGateway(PaperGateway):
        def place_stop_order(self, symbol, side, qty, stop_price, limit_price=None, client_order_id=""):
            return OrderResult(order_id=client_order_id, status="rejected", error="insufficient balance")

    gw = RejectingStopGateway(FixedProvider(100.0), slippage_pct=0.0)
    engine, _ = _stopped_engine(session, gateway=gw)
    intent = engine.execute(_buy_signal(session).id)

    assert intent.status == "filled"
    assert _events(session, "stop_placed") == []
    assert any(kind == "error" for kind, _t, _f in alerts)


def test_engine_survives_a_raising_stop_placement(session, alerts):
    class ExplodingStopGateway(PaperGateway):
        def place_stop_order(self, symbol, side, qty, stop_price, limit_price=None, client_order_id=""):
            raise RuntimeError("venue unreachable")

    engine, _ = _stopped_engine(session, gateway=ExplodingStopGateway(FixedProvider(100.0), 0.0))
    intent = engine.execute(_buy_signal(session).id)

    assert intent.status == "filled"
    assert session.execute(select(Position)).scalar_one().qty > 0
    assert any(kind == "error" for kind, _t, _f in alerts)


def test_failed_stop_cancel_never_blocks_the_ledger_write(session, alerts):
    class BrokenCancelGateway(PaperGateway):
        def cancel_order(self, symbol, client_order_id):
            raise BinanceError("network timeout on DELETE")

    engine, gw = _stopped_engine(session, gateway=BrokenCancelGateway(FixedProvider(100.0), 0.0))
    engine.execute(_buy_signal(session).id)

    sell_intent = engine.execute(_sell_signal(session).id)

    assert sell_intent.status == "filled"                              # ledger write happened
    assert session.execute(select(Position)).scalar_one().qty == 0
    assert len(session.execute(select(Trade)).scalars().all()) == 1
    # the *place* event stays; the cancel was not recorded because it failed
    assert len(_events(session, "stop_canceled")) == 0
    assert any(kind == "error" for kind, _t, _f in alerts)


# ---------------------------------------------------------------------------
# engine: limit entries
# ---------------------------------------------------------------------------

def test_limit_entry_rests_and_marks_the_intent_sent(session, alerts):
    engine, gw = _stopped_engine(session, exchange_stop=False)
    sig = _buy_signal(session, risk={"position_pct": 0.2, "order_type": "limit", "limit_offset_pct": 5.0})

    intent = engine.execute(sig.id)

    assert intent.order_type == "limit"
    assert intent.status == "sent"                       # NOT filled
    assert session.get(Signal, sig.id).status == "sent"
    assert gw.get_open_orders("BTC/USDT")[0]["order"]["price"] == pytest.approx(95.0)
    # the ledger has an accepted (sent) event, never a fill
    assert [e.event_type for e in _events(session, "accepted")] == ["accepted"]
    assert _events(session, "fill") == []
    assert session.execute(select(Position)).scalars().all() == []


def test_limit_entry_fills_when_it_crosses(session, alerts):
    engine, gw = _stopped_engine(session, exchange_stop=False)
    # 0% offset = at the reference price, which crosses the current quote
    sig = _buy_signal(session, risk={"position_pct": 0.2, "order_type": "limit", "limit_offset_pct": 0.0})

    intent = engine.execute(sig.id)

    assert intent.order_type == "limit"
    assert intent.status == "filled"
    assert session.execute(select(Position)).scalar_one().qty == pytest.approx(intent.qty)


def test_limit_request_falls_back_to_market_on_a_thin_gateway(session, alerts):
    class ThinGateway:
        def place_market_order(self, symbol, side, qty, client_order_id):
            return OrderResult(order_id=client_order_id, status="filled",
                               filled_qty=qty, avg_fill_price=100.0, fee=0.0)

        def get_order(self, symbol, client_order_id):
            return OrderResult(order_id=client_order_id, status="filled")

    engine, _ = _stopped_engine(session, exchange_stop=False, gateway=ThinGateway())
    sig = _buy_signal(session, risk={"position_pct": 0.2, "order_type": "limit", "limit_offset_pct": 5.0})

    intent = engine.execute(sig.id)

    assert intent.status == "filled"     # degraded to market rather than stuck


def test_sell_never_rests_as_a_limit(session, alerts):
    engine, gw = _stopped_engine(session, exchange_stop=False)
    engine.execute(_buy_signal(session).id)
    sig = _sell_signal(session)
    sig.risk_json = json.dumps({"order_type": "limit", "limit_offset_pct": 5.0})
    session.commit()

    intent = engine.execute(sig.id)

    assert intent.order_type == "market"
    assert intent.status == "filled"


# ---------------------------------------------------------------------------
# engine: alerting is best-effort
# ---------------------------------------------------------------------------

def test_alerting_never_breaks_a_fill_or_a_skip(session, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("webhook down")

    monkeypatch.setattr(engine_mod, "notify", _boom)
    engine, _gw = _stopped_engine(session, exchange_stop=False)

    filled = engine.execute(_buy_signal(session).id)
    assert filled.status == "filled"

    # Risk skip path (max positions reached) must also survive a dead alerter.
    for sym in ("A/USDT", "B/USDT", "C/USDT"):
        session.add(Position(symbol=sym, qty=1.0, avg_price=1.0))
    session.commit()
    skipped = engine.execute(_buy_signal(session, symbol="ETH/USDT").id)
    assert skipped.status == "skipped"
    assert session.execute(select(TradeIntent)).scalars().all()


def test_risk_skip_emits_a_risk_alert(session, alerts):
    engine, _gw = _stopped_engine(session, exchange_stop=False)
    for sym in ("A/USDT", "B/USDT", "C/USDT"):
        session.add(Position(symbol=sym, qty=1.0, avg_price=1.0))
    session.commit()

    engine.execute(_buy_signal(session).id)

    assert any(kind == "risk" for kind, _t, _f in alerts)
