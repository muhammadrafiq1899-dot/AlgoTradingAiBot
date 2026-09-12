"""AlgoTrading entry point: async bootstrap wiring everything together.

One process, one event loop:

    scheduler (APScheduler) -> market tick -> strategy engine -> execution engine
    telegram (python-telegram-bot, polling) -> control surface + AI approvals
    api (FastAPI, optional) -> /health + /status for monitoring

Startup order (per plan): restore DB + rebuild derived state from events +
initial reconcile + safe resume. Live mode refuses to start without keys.

Run with:  python -m algotrading.main [--demo-data] [--no-telegram]
"""
from __future__ import annotations

import argparse
import asyncio
import html
import logging
import signal
import sys
from pathlib import Path

# Allow `python algotrading/main.py` from anywhere, not just `-m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algotrading.config import Settings, get_secret, load_settings, validate_settings  # noqa: E402
from algotrading.db import get_session, get_session_factory, init_db  # noqa: E402
from algotrading.db.seed import ensure_seeded  # noqa: E402
from algotrading.execution import LiveGateway, PaperGateway  # noqa: E402
from algotrading.ledger import Ledger  # noqa: E402
from algotrading.logging_config import setup_logging  # noqa: E402
from algotrading.market.binance_provider import BinanceMarketProvider  # noqa: E402
from algotrading.market.demo import DemoProvider  # noqa: E402
from algotrading.scheduler import BotContext, build_scheduler  # noqa: E402
from algotrading.supervisor.health import HealthMonitor, ensure_wake_lock  # noqa: E402
from algotrading.supervisor.reconcile import reconcile_and_report  # noqa: E402

log = logging.getLogger(__name__)


# --- CLI ---------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="algotrading", description=__doc__)
    parser.add_argument(
        "--demo-data",
        action="store_true",
        help="Use the synthetic demo candle feed (no network, no API keys). Forces paper mode.",
    )
    parser.add_argument(
        "--no-telegram",
        action="store_true",
        help="Run without the Telegram control surface (headless).",
    )
    return parser.parse_args(argv)


# --- components --------------------------------------------------------------

def build_context(settings: Settings, demo: bool) -> BotContext:
    """Assemble provider + gateway + session factory + health monitor.

    Raises SystemExit in live mode when Binance keys are missing — the bot
    refuses to start live without credentials (paper mode needs none).
    """
    session_factory = get_session_factory(settings.db_path)
    health = HealthMonitor(str(Path(settings.data_dir) / "heartbeat"))

    if demo:
        log.warning("--demo-data: using synthetic candles; forcing paper mode")
        provider = DemoProvider()
        gateway = PaperGateway(provider, slippage_pct=settings.risk.slippage_pct)
    elif settings.mode == "paper":
        provider = BinanceMarketProvider()
        gateway = PaperGateway(provider, slippage_pct=settings.risk.slippage_pct)
    else:  # live
        key = get_secret("binance_api_key")
        secret = get_secret("binance_api_secret")
        if not key or not secret:
            log.error(
                "LIVE mode requires BINANCE_API_KEY and BINANCE_API_SECRET in .env. "
                "Refusing to start."
            )
            raise SystemExit(1)
        log.warning(
            "LIVE MODE — the bot will place REAL orders on Binance Spot. "
            "This mode must only be enabled after explicit review."
        )
        from algotrading.market.binance_rest import BinanceRestClient

        provider = BinanceMarketProvider()
        gateway = LiveGateway(BinanceRestClient(api_key=key, api_secret=secret))

    return BotContext(
        settings=settings,
        provider=provider,
        gateway=gateway,
        session_factory=session_factory,
        health=health,
    )


class AppController:
    """Control surface for /start_bot and /stop_bot (pause/resume jobs)."""

    def __init__(self, scheduler) -> None:
        self._scheduler = scheduler

    async def start(self) -> None:
        for job in self._scheduler.get_jobs():
            if job.next_run_time is None:  # paused
                job.resume()
        log.info("scheduler resumed via Telegram")

    async def stop(self) -> None:
        for job in self._scheduler.get_jobs():
            job.pause()
        log.info("scheduler paused via Telegram")


# --- Telegram push for AI proposals -----------------------------------------

def _make_recommendation_pusher(app, allowed_users: list[int], loop: asyncio.AbstractEventLoop):
    """Return a thread-safe callback that pushes a PENDING rec to Telegram."""
    from algotrading.telegram.ui import approval_keyboard

    async def _push(rec) -> None:
        text = (
            f"🤖 <b>AI proposal #{rec.id}</b> [{rec.kind}] for {rec.strategy_name}:\n"
            f"{html.escape(rec.rationale or '(no rationale)')}"
        )
        for chat_id in allowed_users:
            try:
                await app.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode="HTML",
                    reply_markup=approval_keyboard(rec.id),
                )
            except Exception:  # noqa: BLE001 - alert failures must not crash the loop
                log.exception("failed to push recommendation %s to chat %s", rec.id, chat_id)

    def push(rec) -> None:
        asyncio.run_coroutine_threadsafe(_push(rec), loop)

    return push


# --- app lifecycle -----------------------------------------------------------

async def run(settings: Settings, demo: bool, no_telegram: bool) -> None:
    # 1. Restore DB, seed on first run, rebuild derived state, initial reconcile.
    init_db(settings.db_path)
    with get_session(settings.db_path) as session:
        ensure_seeded(session)
        Ledger(session).rebuild_positions()

    ctx = build_context(settings, demo)
    ctx.health.beat()

    # Startup reconcile: local intents vs exchange before any new orders.
    with get_session(settings.db_path) as session:
        log.info("startup reconcile: %s", reconcile_and_report(session, ctx.gateway, settings.market.symbols))

    scheduler = build_scheduler(ctx)
    scheduler.start()
    log.info(
        "scheduler started (%d jobs, mode=%s, %d symbols, tick=%ss)",
        len(scheduler.get_jobs()),
        "demo" if demo else settings.mode,
        len(settings.market.symbols),
        settings.schedule.market_tick_seconds,
    )

    tg_app = None
    controller = AppController(scheduler)
    loop = asyncio.get_running_loop()

    # 2. Telegram control surface (skip when disabled).
    if not no_telegram and get_secret("telegram_token"):
        from algotrading.telegram.bot import build_application

        tg_app = build_application(
            settings,
            get_session_factory(settings.db_path),
            controller=controller,
        )
        ctx.on_recommendation = _make_recommendation_pusher(
            tg_app, settings.telegram_allowed_users, loop
        )
        await tg_app.initialize()
        await tg_app.updater.start_polling()
        await tg_app.start()
        log.info("telegram bot started (polling)")
    elif no_telegram:
        log.info("running without Telegram (--no-telegram)")
    else:
        log.warning("TELEGRAM_BOT_TOKEN missing; running without control surface")

    # 3. Optional internal API.
    api_task: asyncio.Task | None = None
    if settings.api.enabled:
        token = get_secret("api_token")
        if not token:
            log.error("api.enabled is true but API_TOKEN is not set in .env; API not started")
        else:
            import uvicorn

            from algotrading.api import build_api

            api_app = build_api(settings, get_session_factory(settings.db_path), ctx.health, token=token)
            server = uvicorn.Server(
                uvicorn.Config(
                    api_app,
                    host=settings.api.host,
                    port=settings.api.port,
                    log_level="warning",
                )
            )
            api_task = asyncio.create_task(server.serve())
            log.info("internal API on http://%s:%s", settings.api.host, settings.api.port)

    # 4. Wait for shutdown signal, then tear down cleanly.
    stop_event = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # non-POSIX (e.g. Windows): rely on KeyboardInterrupt below

    try:
        await stop_event.wait()
    except KeyboardInterrupt:
        pass

    log.info("shutting down…")
    # wait=True: let an in-flight job (e.g. a slow market_tick) finish before
    # teardown, so the process actually exits after "bye" instead of lingering
    # on the scheduler's worker threads (which confused the watchdog/restarts).
    scheduler.shutdown(wait=True)

    if api_task is not None:
        api_task.cancel()
        try:
            await api_task
        except asyncio.CancelledError:
            pass

    if tg_app is not None:
        await tg_app.updater.stop()
        await tg_app.stop()
        await tg_app.shutdown()
    log.info("bye")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = load_settings()
    validate_settings(settings)
    setup_logging(
        log_level=settings.log_level,
        log_dir=settings.log_dir,
        log_format=settings.log_format,
        json_fields=settings.log_json_fields,
    )

    if not args.demo_data:
        ensure_wake_lock()

    try:
        asyncio.run(run(settings, demo=args.demo_data, no_telegram=args.no_telegram))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()