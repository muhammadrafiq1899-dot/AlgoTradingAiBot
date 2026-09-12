"""M7: internal API — /health is open, /status requires the bearer token."""
import json
import time

import pytest
from fastapi.testclient import TestClient

from algotrading.api import build_api
from algotrading.config import Settings
from algotrading.db import init_db, get_session_factory
from algotrading.db.models import Strategy
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