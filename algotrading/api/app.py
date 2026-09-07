"""FastAPI app factory: internal health/status endpoints.

Served from algotrading/main.py as a task on the same event loop (uvicorn).
Reads the DB through a fresh session per request (from the shared session
factory) so it never races the scheduler's threads.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import FastAPI
from sqlalchemy import select, text

from algotrading.api.auth import require_token
from algotrading.config import Settings
from algotrading.db import get_schema_version
from algotrading.db.models import AIRecommendation, Position, Strategy, TradeIntent
from algotrading.supervisor.health import HealthMonitor
from algotrading import __version__

log = logging.getLogger(__name__)

_STARTED = time.time()


def build_api(
    settings: Settings,
    session_factory: Callable,
    health: HealthMonitor,
    token: str = "",
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: validated settings (mode, universe).
        session_factory: callable returning a fresh DB session.
        health: HealthMonitor for heartbeat age reporting.
        token: API_TOKEN from .env; empty disables /status entirely.
    """
    app = FastAPI(title="AlgoTrading internal API", version=__version__)

    def _session():
        return session_factory()

    @app.get("/health")
    def health_endpoint() -> dict:
        db_ok = True
        try:
            with _session() as s:
                s.execute(text("SELECT 1"))
        except Exception as exc:  # noqa: BLE001 - health must report, not raise
            log.error("health check: db unreachable: %s", exc)
            db_ok = False
        last_beat = health.last_beat()
        return {
            "status": "ok" if db_ok else "degraded",
            "version": __version__,
            "mode": settings.mode,
            "uptime_s": int(time.time() - _STARTED),
            "db": "ok" if db_ok else "error",
            "heartbeat_age_s": int(time.time() - last_beat) if last_beat else None,
        }

    @app.get("/status", dependencies=[require_token(token)])
    def status_endpoint() -> dict:
        with _session() as s:
            active = s.execute(
                select(Strategy).where(Strategy.status == "active").order_by(Strategy.version.desc())
            ).scalars().first()
            positions = s.execute(
                select(Position).where(Position.qty > 0)
            ).scalars().all()
            intents = s.execute(
                select(TradeIntent).order_by(TradeIntent.ts.desc()).limit(10)
            ).scalars().all()
            pending = s.execute(
                select(AIRecommendation).where(AIRecommendation.status == "pending")
            ).scalars().all()

        return {
            "mode": settings.mode,
            "schema_version": get_schema_version(settings.db_path),
            "active_strategy": {
                "name": active.name,
                "version": active.version,
                "params": active.params,
            } if active else None,
            "open_positions": [
                {"symbol": p.symbol, "qty": p.qty, "avg_price": p.avg_price}
                for p in positions
            ],
            "recent_intents": [
                {
                    "id": i.id,
                    "symbol": i.symbol,
                    "side": i.side,
                    "status": i.status,
                    "qty": i.qty,
                }
                for i in intents
            ],
            "pending_recommendations": len(pending),
        }

    return app