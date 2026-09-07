"""Telegram command handlers: status/start/stop/strategy/risk/summary + auth.

Auth is a strict allowlist: any user not in TELEGRAM_ALLOWED_USERS is ignored
(no reply — don't leak existence to outsiders). The application and handlers are
wired in `bot.py`; this module keeps the handler logic dependency-free enough to
unit test with a stub update object.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable

from telegram import Update
from telegram.ext import CallbackQueryHandler, CommandHandler, ContextTypes

from algotrading.db.models import Position, Signal, Strategy, Trade, TradeIntent
from algotrading.telegram.ui import (
    approval_keyboard,
    format_risk,
    format_status,
    format_strategies,
    format_summary,
)

log = logging.getLogger(__name__)

TEXT_MARKDOWN = "MarkdownV2"


def _auth_decorator(allowed: set[int]):
    """Wrap a handler so only allowlisted user IDs reach the real function."""

    def decorator(fn: Callable):
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
            user_id = update.effective_user.id if update.effective_user else None
            if user_id not in allowed:
                log.info("Ignoring non-allowlisted user %s", user_id)
                return
            return await fn(update, context)

        return wrapper

    return decorator


def build_handlers(
    session,
    settings,
    risk_cfg,
    on_start=None,
    on_stop=None,
    on_approve=None,
    on_reject=None,
    allowed_users: Iterable[int] = (),
) -> list:
    """Return PTB handlers bound to DB + settings + control callbacks.

    `on_start`/`on_stop` are async callables(update, context) that perform the
    actual scheduler start/stop; `on_approve`/`on_reject` handle the inline
    recommendation callbacks (M6). Passing None keeps the command as a stub.
    """
    allowed = set(allowed_users)
    auth = _auth_decorator(allowed)

    def session_for(_update):
        # SQLAlchemy Session is not thread-safe; handlers run in the bot's
        # event loop, so one shared session is fine here. (Scoped per request
        # is overkill for this app.)
        return session

    # --- /help ---
    @auth
    async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            "*AlgoTrading control*\n"
            "/help — this message\n"
            "/status — active strategy, positions, recent fills\n"
            "/start_bot — start the scheduler\n"
            "/stop_bot — stop the scheduler\n"
            "/strategy — list strategy versions\n"
            "/risk — current risk limits\n"
            "/summary — analytics summary (M5)\n"
            "/paper — switch to paper mode\n"
            "/live — switch to live mode (guarded)\n"
        )
        await update.effective_message.reply_text(text, parse_mode=TEXT_MARKDOWN)

    # --- /status ---
    @auth
    async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        sess = session_for(update)
        strategies = sess.query(Strategy).all()
        positions = sess.query(Position).all()
        intents = sess.query(TradeIntent).order_by(TradeIntent.ts.desc()).all()
        trades = sess.query(Trade).order_by(Trade.closed_at.desc()).all()
        text = format_status(
            settings,
            sess,
            strategies,
            positions,
            intents,
            trades,
            prices=_latest_prices(sess),
        )
        await update.effective_message.reply_text(text, parse_mode=TEXT_MARKDOWN)

    # --- /start_bot / /stop_bot ---
    @auth
    async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if on_start:
            await on_start(update, context)
        else:
            await update.effective_message.reply_text("Scheduler control not wired yet.")

    @auth
    async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if on_stop:
            await on_stop(update, context)
        else:
            await update.effective_message.reply_text("Scheduler control not wired yet.")

    # --- /strategy ---
    @auth
    async def strategy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        sess = session_for(update)
        strategies = sess.query(Strategy).order_by(Strategy.version.desc()).all()
        await update.effective_message.reply_text(
            format_strategies(strategies), parse_mode=TEXT_MARKDOWN
        )

    # --- /risk ---
    @auth
    async def risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        sess = session_for(update)
        await update.effective_message.reply_text(
            format_risk(settings, risk_cfg, sess), parse_mode=TEXT_MARKDOWN
        )

    # --- /summary ---
    @auth
    async def summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        sess = session_for(update)
        # M5 will populate AnalyticsSummary; query defensively.
        from algotrading.db.models import AnalyticsSummary

        summaries = sess.query(AnalyticsSummary).order_by(AnalyticsSummary.period.desc()).all()
        await update.effective_message.reply_text(
            format_summary(summaries, sess), parse_mode=TEXT_MARKDOWN
        )

    # --- inline approval callbacks (M6) ---
    @auth
    async def on_approval(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        data = query.data  # "approve:12" | "reject:12"
        action, _, rec_id = data.partition(":")
        rec_id = int(rec_id)
        handler = on_approve if action == "approve" else on_reject
        if handler is None:
            await query.edit_message_text(f"{action} not implemented yet.")
            return
        result = await handler(rec_id, update, context)
        await query.edit_message_text(
            f"Recommendation {rec_id} {action}d.\n{result or ''}", parse_mode=TEXT_MARKDOWN
        )

    return [
        CommandHandler("help", help_cmd),
        CommandHandler("status", status_cmd),
        CommandHandler("start_bot", start_cmd),
        CommandHandler("stop_bot", stop_cmd),
        CommandHandler("strategy", strategy_cmd),
        CommandHandler("risk", risk_cmd),
        CommandHandler("summary", summary_cmd),
        CallbackQueryHandler(on_approval, pattern=r"^(approve|reject):"),
    ]


def _latest_prices(sess) -> dict[str, float]:
    """Latest known close per symbol from the candles table (for mark-to-market)."""
    from sqlalchemy import func, select

    from algotrading.db.models import Candle

    # Correlated subquery: the latest candle for each symbol, then its close.
    latest = (
        select(Candle.symbol, Candle.close)
        .where(
            Candle.ts
            == select(func.max(Candle.ts))
            .where(Candle.symbol == Candle.symbol)
            .scalar_subquery()
        )
    )
    return {row[0]: row[1] for row in sess.execute(latest).all()}
