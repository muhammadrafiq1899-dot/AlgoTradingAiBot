"""FastAPI app factory: internal health/status endpoints.

Served from algotrading/main.py as a task on the same event loop (uvicorn).
Reads the DB through a fresh session per request (from the shared session
factory) so it never races the scheduler's threads.

Endpoints fall into two groups and the split is deliberate:

* open operational facts — ``/health`` (and ``/metrics`` when configured), no
  token, nothing sensitive;
* everything about trading state — ``/status``, ``/dashboard`` and the
  ``/export/*`` downloads — behind ``require_token``.

Every route here is GET and read-only: the API can observe the bot, never drive
it (there is no state-changing endpoint, by design).
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import FastAPI, Response
from sqlalchemy import select, text

from algotrading.api.auth import require_token
from algotrading.api.dashboard import collect_state, render_dashboard
from algotrading.api.metrics import (
    init_metrics,
    get_metrics,
    metrics_content_type,
)
from algotrading.config import Settings
from algotrading.db import get_schema_version
from algotrading.db.models import AIRecommendation, Candle, Position, Strategy, TradeIntent
from algotrading.store.export import (
    summary_dict,
    summary_json_text,
    trade_rows,
    trades_csv_text,
    trades_json_text,
)
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

    # Initialize Prometheus metrics (no-op if prometheus-client not installed)
    init_metrics(settings)

    def _session():
        return session_factory()

    def _check_db_latency(session_factory: Callable) -> tuple[bool, float | None]:
        """Check DB connectivity and measure latency."""
        start = time.perf_counter()
        try:
            with session_factory() as s:
                s.execute(text("SELECT 1"))
            return True, (time.perf_counter() - start) * 1000
        except Exception as exc:  # noqa: BLE001 - health must report, not raise
            log.error("health check: db unreachable: %s", exc)
            return False, None

    def _check_market_freshness(session_factory: Callable, symbols: list[str], max_staleness: int) -> tuple[bool, dict]:
        """Check if market data is fresh for all symbols."""
        try:
            with session_factory() as s:
                now_ms = int(time.time() * 1000)
                max_age_ms = max_staleness * 1000
                stale_symbols = []
                for symbol in symbols:
                    latest = s.execute(
                        select(Candle)
                        .where(Candle.symbol == symbol)
                        .order_by(Candle.ts.desc())
                        .limit(1)
                    ).scalar()
                    if latest is None:
                        stale_symbols.append(symbol)
                    elif (now_ms - latest.ts) > max_age_ms:
                        stale_symbols.append(symbol)
                return len(stale_symbols) == 0, {"stale_symbols": stale_symbols}
        except Exception as exc:
            log.error("health check: market freshness error: %s", exc)
            return False, {"error": str(exc)}

    @app.get("/health")
    def health_endpoint() -> dict:
        # DB check with latency
        db_ok, db_latency_ms = _check_db_latency(_session)

        # Heartbeat check
        last_beat = health.last_beat()
        heartbeat_age_s = int(time.time() - last_beat) if last_beat else None

        # Market data freshness (only in live/paper mode, not demo)
        market_ok, market_details = True, {}
        if settings.mode != "paper" or settings.market.symbols:
            market_ok, market_details = _check_market_freshness(
                _session, settings.market.symbols, settings.market.max_staleness_seconds
            )

        # Scheduler check (via heartbeat - if heartbeat is recent, scheduler is running)
        scheduler_ok = heartbeat_age_s is not None and heartbeat_age_s < settings.schedule.market_tick_seconds * 2

        # Overall status
        all_ok = db_ok and market_ok and scheduler_ok
        overall_status = "ok" if all_ok else "degraded"

        return {
            "status": overall_status,
            "version": __version__,
            "mode": settings.mode,
            "uptime_s": int(time.time() - _STARTED),
            "db": {
                "status": "ok" if db_ok else "error",
                "latency_ms": round(db_latency_ms, 1) if db_latency_ms else None,
            },
            "market": {
                "status": "ok" if market_ok else "stale",
                **market_details,
            },
            "scheduler": {
                "status": "ok" if scheduler_ok else "stale",
                "heartbeat_age_s": heartbeat_age_s,
            },
        }

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        """Prometheus metrics endpoint (open, no auth)."""
        return Response(content=get_metrics(), media_type=metrics_content_type())

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

    def _attachment(content: str, media_type: str, filename: str) -> Response:
        """Read-only download response with an attachment filename.

        The filename is a fixed constant per route (never derived from a query
        parameter): nothing user-supplied reaches a header.
        """
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # The dashboard is opt-out (`api.dashboard: false`). It is not registered at
    # all when disabled, so the route 404s for everyone — a disabled feature
    # should not advertise that it exists behind a token.
    if settings.api.dashboard:
        @app.get("/dashboard", dependencies=[require_token(token)])
        def dashboard_endpoint() -> Response:
            """Server-rendered read-only HTML page (no JS, no external assets)."""
            with _session() as s:
                state = collect_state(
                    s,
                    settings,
                    health,
                    uptime_seconds=int(time.time() - _STARTED),
                )
            return Response(
                content=render_dashboard(state),
                media_type="text/html",
            )

    @app.get("/export/trades.csv", dependencies=[require_token(token)])
    def export_trades_csv_endpoint() -> Response:
        """All trades as CSV."""
        with _session() as s:
            body = trades_csv_text(trade_rows(s))
        return _attachment(body, "text/csv", "trades.csv")

    @app.get("/export/trades.json", dependencies=[require_token(token)])
    def export_trades_json_endpoint() -> Response:
        """All trades as a JSON array."""
        with _session() as s:
            body = trades_json_text(trade_rows(s))
        return _attachment(body, "application/json", "trades.json")

    @app.get("/export/summary.json", dependencies=[require_token(token)])
    def export_summary_endpoint() -> Response:
        """Counts, totals, latest analytics and mode/strategy metadata."""
        with _session() as s:
            summary = summary_dict(
                s,
                mode=settings.mode,
                initial_balance=float(settings.risk.paper_initial_balance),
            )
        return _attachment(summary_json_text(summary), "application/json", "summary.json")

    return app