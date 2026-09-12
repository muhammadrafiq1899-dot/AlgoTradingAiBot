"""Telegram bot bootstrap: async Application + handler wiring.

The bot is a thin control surface. It reads config for the token + allowlist,
builds the PTB Application, registers the command/callback handlers from
`commands.py`, and (optionally) keeps a reference to an app controller so
/start_bot and /stop_bot can drive the scheduler.
"""
from __future__ import annotations

import logging

from telegram.ext import Application, ApplicationBuilder

from algotrading.config import Settings, get_secret
from algotrading.market.candles import CandleStore
from algotrading.store.recommendations import RecommendationStore
from algotrading.telegram.commands import build_handlers

log = logging.getLogger(__name__)


def build_application(
    settings: Settings,
    session_factory,
    controller=None,
) -> Application:
    """Build the Telegram Application with handlers bound to DB + settings.

    `controller` is an optional object exposing async `start()`/`stop()` used by
    /start_bot and /stop_bot. Pass None for pure status/read-only mode.

    Each handler gets its own session from session_factory() to avoid
    concurrent access issues and PendingRollbackError.
    """
    allowed_users = settings.telegram_allowed_users
    risk_cfg = settings.risk

    handlers = build_handlers(
        session_factory=session_factory,
        settings=settings,
        risk_cfg=risk_cfg,
        on_start=_make_start(controller),
        on_stop=_make_stop(controller),
        on_approve=_make_approve(settings, session_factory),
        on_reject=_make_reject(session_factory),
        allowed_users=allowed_users,
    )

    app = (
        ApplicationBuilder()
        .token(get_secret("telegram_token"))
        .concurrent_updates(True)
        .build()
    )
    for h in handlers:
        app.add_handler(h)
    return app


def _make_start(controller):
    async def start(update, context):
        if controller is None or not hasattr(controller, "start"):
            await update.effective_message.reply_text("Scheduler control not available.")
            return
        await controller.start()
        await update.effective_message.reply_text("Bot started.")

    return start


def _make_stop(controller):
    async def stop(update, context):
        if controller is None or not hasattr(controller, "stop"):
            await update.effective_message.reply_text("Scheduler control not available.")
            return
        await controller.stop()
        await update.effective_message.reply_text("Bot stopped.")

    return stop


def _stored_candles(session, settings) -> list:
    """Pull recent stored candles for the backtest, as market Candle objects."""
    from algotrading.market.base import Candle

    symbol = (settings.market.symbols or ["BTC/USDT"])[0]
    interval = "1h" if "1h" in settings.market.intervals else "1m"
    rows = CandleStore(session).get(symbol, interval, limit=500)
    return [
        Candle(
            symbol=r.symbol,
            interval=r.interval,
            ts=r.ts,
            open=r.open,
            high=r.high,
            low=r.low,
            close=r.close,
            volume=r.volume,
        )
        for r in rows
    ]


def _make_approve(settings, session_factory):
    """Factory for the inline /approve callback.

    Runs the shadow backtest + controlled release via RecommendationStore.apply.
    Missing candles make the backtest a no-op (the version is still released).

    Creates a new session per callback to avoid database locking issues.
    """
    async def approve(rec_id, update, context):
        session = session_factory()
        try:
            store = RecommendationStore(session)
            rec = store.get(int(rec_id))
            if rec is None:
                return "recommendation not found."
            candles = _stored_candles(session, settings)
            try:
                strategy, result = store.apply(rec, candles=candles)
            except ValueError as exc:
                return f"apply failed: {exc}"
            if strategy is None:
                return "backtest did not meet threshold; auto-rejected."
            if result is None:
                return f"released as v{strategy.version} (no backtest data)."
            return (
                f"released as v{strategy.version} — {result.n_trades} trades, "
                f"pnl {result.total_pnl:.2f}."
            )
        finally:
            session.close()

    return approve


def _make_reject(session_factory):
    """Factory for the inline /reject callback (marks PENDING -> rejected).

    Creates a new session per callback to avoid database locking issues.
    """
    async def reject(rec_id, update, context):
        session = session_factory()
        try:
            store = RecommendationStore(session)
            rec = store.get(int(rec_id))
            if rec is None:
                return "recommendation not found."
            store.mark_rejected(rec)
            return "rejected."
        finally:
            session.close()

    return reject


def run_bot(settings: Settings, session, controller=None) -> None:
    """Run the Telegram bot (blocking)."""
    app = build_application(settings, session, controller)
    log.info("Starting Telegram bot (polling)")
    app.run_polling()
