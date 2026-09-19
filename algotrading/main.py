"""AlgoTrading entry point: async bootstrap wiring everything together.

One process, one event loop. The trading core (scheduler + market tick +
strategy engine + deterministic execution + ledger) is fixed; everything around
it — market provider, order gateway, control surface, API, analytics, and the
strategy sources — is a **module** selected from ``config/settings.yaml``
``modules:`` and composed by :class:`algotrading.modules.ModuleManager`.

    modules.setup(ctx)          -> picks provider/gateway, loads strategy plugins
    scheduler (APScheduler)     -> market tick -> strategy -> execution
    modules.start(ctx)          -> telegram polling, optional API server

Startup order: restore DB + rebuild derived state from events + initial
reconcile + resume. Live mode refuses to start without credentials.

Run with:  python -m algotrading.main [--demo-data] [--no-telegram]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

# Allow `python algotrading/main.py` from anywhere, not just `-m`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algotrading.alerts import configure_alerter  # noqa: E402
from algotrading.config import Settings, load_settings, validate_settings  # noqa: E402
from algotrading.db import get_session, get_session_factory, init_db  # noqa: E402
from algotrading.db.seed import ensure_seeded  # noqa: E402
from algotrading.ledger import Ledger  # noqa: E402
from algotrading.logging_config import setup_logging  # noqa: E402
from algotrading.modules import (  # noqa: E402
    CAPABILITY_SCHEDULER_CONTROL,
    ModuleError,
    ModuleManager,
)
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

def build_context(
    settings: Settings,
    demo: bool,
    disabled: tuple[str, ...] = (),
) -> tuple[BotContext, ModuleManager]:
    """Compose the bot from the configured modules.

    Returns the scheduler context (with provider/gateway installed) and the
    manager, so the caller can drive start/stop. Raises ModuleError if a locked
    module (execution) fails or a required capability is missing.
    """
    session_factory = get_session_factory(settings.db_path)
    health = HealthMonitor(str(Path(settings.data_dir) / "heartbeat"))
    ctx = BotContext(settings=settings, session_factory=session_factory, health=health)

    manager = ModuleManager(settings, demo=demo, disabled=disabled)
    manager.setup(ctx)

    if ctx.provider is None or ctx.gateway is None:
        raise ModuleError(
            "market and execution modules must both be enabled "
            f"(got: {manager.describe()})"
        )
    # Modules contribute scheduled jobs; the scheduler merges them at build time.
    ctx.extra_jobs = manager.collect_jobs(ctx)
    if demo:
        log.warning("--demo-data: using synthetic candles; forcing paper mode")
    return ctx, manager


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


# --- app lifecycle -----------------------------------------------------------

async def run(settings: Settings, demo: bool, no_telegram: bool) -> None:
    # 1. Restore DB, seed on first run, rebuild derived state, initial reconcile.
    init_db(settings.db_path)
    with get_session(settings.db_path) as session:
        ensure_seeded(session)
        Ledger(session).rebuild_positions()

    # --no-telegram forces the control module off without editing config.
    disabled = ("control.telegram",) if no_telegram else ()
    ctx, manager = build_context(settings, demo, disabled)
    log.info(
        "modules: %s",
        ", ".join(f"{name}[{cap}]" for name, cap, _locked in manager.describe()),
    )
    ctx.health.beat()

    # Startup reconcile: local intents vs exchange before any new orders.
    with get_session(settings.db_path) as session:
        log.info(
            "startup reconcile: %s",
            reconcile_and_report(session, ctx.gateway, settings.market.symbols),
        )

    scheduler = build_scheduler(ctx)
    scheduler.start()
    log.info(
        "scheduler started (%d jobs, mode=%s, %d symbols, tick=%ss)",
        len(scheduler.get_jobs()),
        "demo" if demo else settings.mode,
        len(settings.market.symbols),
        settings.schedule.market_tick_seconds,
    )

    # The control surface reads the scheduler controller from the capability bag.
    ctx.provide(CAPABILITY_SCHEDULER_CONTROL, AppController(scheduler))

    # 2. Start modules (Telegram polling, optional API server, ...).
    await manager.start(ctx)

    # 3. Wait for shutdown signal, then tear down cleanly.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
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
    # teardown, so the process actually exits instead of lingering on the
    # scheduler's worker threads (which confused the watchdog/restarts).
    scheduler.shutdown(wait=True)
    await manager.stop(ctx)
    log.info("bye")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = load_settings()
    validate_settings(settings)
    # Wire the outbound alert channel before any job can fire: the default
    # alerter is a no-op, so an unconfigured bot silently drops alerts rather
    # than failing on them.
    configure_alerter(settings)
    setup_logging(
        log_level=settings.log_level,
        log_dir=settings.log_dir,
        log_format=settings.log_format,
        json_fields=settings.log_json_fields,
    )

    # Keep the CPU awake even in demo mode. This is a phone-first bot: without
    # the wake lock Android can freeze or kill the whole app as soon as the
    # screen goes off, and an unattended demo run is exactly that case.
    ensure_wake_lock()

    try:
        asyncio.run(run(settings, demo=args.demo_data, no_telegram=args.no_telegram))
    except KeyboardInterrupt:
        pass
    except ModuleError as exc:
        log.error("fatal module configuration error: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
