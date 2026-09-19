"""M7: internal API — /health is open, everything else needs the bearer token.

Also covers P2-10: the read-only HTML dashboard and the /export/* downloads.
"""
import json
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from algotrading.api import build_api
from algotrading.config import ApiConfig, Settings
from algotrading.db import init_db, get_session_factory
from algotrading.db.models import Candle, Position, Strategy, Trade
from algotrading.supervisor.health import HealthMonitor


@pytest.fixture()
def client(tmp_path):
    db = str(tmp_path / "api.db")
    init_db(db)
    factory = get_session_factory(db)
    session = factory()
    session.add(Strategy(name="ema_crossover", version=1, status="active",
                 params=json.dumps({"fast_period": 12, "slow_period": 26, "position_pct": 0.2})))
    # Add candles for both default symbols so market freshness check passes
    from algotrading.db.models import Candle
    now_ms = int(time.time() * 1000)
    for sym in ["BTC/USDT", "ETH/USDT"]:
        session.add(Candle(
            symbol=sym,
            interval="1m",
            ts=now_ms,
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1.0,
        ))
    session.commit()
    session.close()

    settings = Settings(mode="paper", db_path=db)
    # Create heartbeat file so scheduler check passes
    heartbeat_path = str(tmp_path / "heartbeat")
    import pathlib
    pathlib.Path(heartbeat_path).touch()
    app = build_api(
        settings,
        factory,
        HealthMonitor(heartbeat_path),
        token="sekrit",
    )
    return TestClient(app)


def test_health_is_open(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["mode"] == "paper"
    assert body["db"]["status"] == "ok"
    assert "latency_ms" in body["db"]
    assert "heartbeat_age_s" in body["scheduler"]
    assert "market" in body
    assert "scheduler" in body


def test_status_requires_token(client):
    assert client.get("/status").status_code == 401
    assert client.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_status_with_token(client):
    resp = client.get("/status", headers={"Authorization": "Bearer sekrit"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "paper"
    assert body["active_strategy"]["name"] == "ema_crossover"
    assert body["open_positions"] == []
    assert body["pending_recommendations"] == 0


def test_status_refused_when_no_token_configured(tmp_path):
    """Without a configured token, /status refuses everything (403)."""
    db = str(tmp_path / "api2.db")
    init_db(db)
    app = build_api(Settings(mode="paper", db_path=db), get_session_factory(db),
                    HealthMonitor(str(tmp_path / "hb")), token="")
    client = TestClient(app)
    assert client.get("/status", headers={"Authorization": "Bearer anything"}).status_code == 403


# ---------------------------------------------------------------------------
# P2-10: read-only dashboard + exports
# ---------------------------------------------------------------------------

TOKEN = {"Authorization": "Bearer sekrit"}
# A symbol is attacker-influenced data (config, exchange, AI-authored plugins):
# if it reaches the page unescaped, the page becomes script injection.
HOSTILE = "<script>alert(1)</script>"


def _rich_client(tmp_path, *, dashboard=True, name="api-rich.db"):
    """Client whose DB holds a position, a closed trade and a hostile symbol."""
    db = str(tmp_path / name)
    init_db(db)
    factory = get_session_factory(db)
    now_ms = int(time.time() * 1000)
    now = datetime.now(timezone.utc)
    with factory() as session:
        session.add(Strategy(name="ema_crossover", version=2, status="active",
                             params=json.dumps({"fast_period": 12, "slow_period": 26})))
        for sym in ["BTC/USDT", "ETH/USDT", HOSTILE]:
            session.add(Candle(symbol=sym, interval="1m", ts=now_ms, open=100.0,
                               high=111.0, low=99.0, close=110.0, volume=1.0))
        session.add(Position(symbol=HOSTILE, qty=1.5, avg_price=100.0))
        session.add(Trade(symbol=HOSTILE, entry_qty=1.0, entry_avg_price=100.0,
                          exit_avg_price=110.0, realized_pnl=10.0, fees=0.1,
                          opened_at=now - timedelta(hours=2), closed_at=now))
        session.commit()

    heartbeat = str(tmp_path / "heartbeat-rich")
    import pathlib
    pathlib.Path(heartbeat).touch()
    settings = Settings(mode="paper", db_path=db, api=ApiConfig(dashboard=dashboard))
    app = build_api(settings, factory, HealthMonitor(heartbeat), token="sekrit")
    return TestClient(app)


def test_dashboard_requires_token(tmp_path):
    client = _rich_client(tmp_path)
    assert client.get("/dashboard").status_code == 401
    assert client.get("/dashboard", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_dashboard_renders_and_escapes_hostile_values(tmp_path):
    client = _rich_client(tmp_path)
    resp = client.get("/dashboard", headers=TOKEN)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    page = resp.text

    # The hostile symbol is shown, but escaped — and never as live markup.
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
    assert "<script" not in page
    # The data the page is supposed to show.
    assert "ema_crossover" in page
    assert "v2" in page
    assert "paper" in page
    assert "Daily loss guard" in page
    assert "Open positions" in page
    assert "Market data freshness" in page
    assert "BTC/USDT" in page
    assert "Alerts" in page
    # Unrealized PnL of the hostile position: (110 - 100) * 1.5 = 15.00
    assert "15.00" in page
    # Today's realized PnL of the closed trade: 10.00
    assert "10.00" in page


def test_dashboard_is_read_only_no_forms_or_post(tmp_path):
    """Invariant: the browser surface can observe the bot, never drive it."""
    client = _rich_client(tmp_path)
    page = client.get("/dashboard", headers=TOKEN).text
    assert "<form" not in page.lower()
    assert "method=" not in page.lower()

    app = client.app
    methods = set()
    for route in app.routes:
        if getattr(route, "path", "") in ("/dashboard", "/export/trades.csv",
                                          "/export/trades.json", "/export/summary.json"):
            methods |= set(getattr(route, "methods", set()) or set())
    assert methods == {"GET"}


def test_dashboard_404_when_disabled(tmp_path):
    client = _rich_client(tmp_path, dashboard=False, name="api-nodash.db")
    assert client.get("/dashboard", headers=TOKEN).status_code == 404
    assert client.get("/dashboard").status_code == 404
    # Everything else still works.
    assert client.get("/health").status_code == 200
    assert client.get("/status", headers=TOKEN).status_code == 200


def test_export_trades_csv(tmp_path):
    client = _rich_client(tmp_path)
    resp = client.get("/export/trades.csv", headers=TOKEN)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert resp.headers["content-disposition"] == 'attachment; filename="trades.csv"'
    lines = resp.text.strip().splitlines()
    assert lines[0] == ("id,symbol,entry_qty,entry_avg_price,exit_avg_price,realized_pnl,"
                        "fees,opened_at,closed_at,strategy_id,loss_reasons")
    assert len(lines) == 2
    assert HOSTILE in lines[1]


def test_export_trades_json(tmp_path):
    client = _rich_client(tmp_path)
    resp = client.get("/export/trades.json", headers=TOKEN)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.headers["content-disposition"] == 'attachment; filename="trades.json"'
    payload = json.loads(resp.text)
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]["symbol"] == HOSTILE
    assert payload[0]["realized_pnl"] == 10.0


def test_export_summary_json(tmp_path):
    client = _rich_client(tmp_path)
    resp = client.get("/export/summary.json", headers=TOKEN)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.headers["content-disposition"] == 'attachment; filename="summary.json"'
    summary = json.loads(resp.text)
    assert summary["mode"] == "paper"
    assert summary["strategy"]["name"] == "ema_crossover"
    assert summary["counts"]["trades"] == 1
    assert summary["counts"]["open_positions"] == 1
    assert summary["totals"]["realized_pnl"] == 10.0


def test_export_endpoints_require_token(tmp_path):
    client = _rich_client(tmp_path)
    for path in ("/export/trades.csv", "/export/trades.json", "/export/summary.json"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer nope"}).status_code == 401


def test_health_still_open_after_new_routes(tmp_path):
    """The new routes must not have moved /health behind the token."""
    client = _rich_client(tmp_path)
    assert client.get("/health").status_code == 200
    body = client.get("/health").json()
    assert set(body) == {"status", "version", "mode", "uptime_s", "db", "market", "scheduler"}
    assert body["status"] == "ok"


def test_dashboard_renders_on_a_fresh_install(tmp_path):
    """No strategy, no positions, no heartbeat yet: the page must still render."""
    db = str(tmp_path / "fresh.db")
    init_db(db)
    app = build_api(
        Settings(mode="paper", db_path=db),
        get_session_factory(db),
        HealthMonitor(str(tmp_path / "never-beaten")),
        token="sekrit",
    )
    resp = TestClient(app).get("/dashboard", headers=TOKEN)

    assert resp.status_code == 200
    assert "none active" in resp.text
    assert "No open positions." in resp.text
    assert "No closed trades yet." in resp.text
    assert "never" in resp.text  # heartbeat age