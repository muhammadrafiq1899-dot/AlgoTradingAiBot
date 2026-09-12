"""Telegram command handlers: status/start/stop/strategy/risk/summary + auth.

Auth is a strict allowlist: any user not in TELEGRAM_ALLOWED_USERS is ignored
(no reply — don't leak existence to outsiders). The application and handlers are
wired in `bot.py`; this module keeps the handler logic dependency-free enough to
unit test with a stub update object.
"""
from __future__ import annotations

import asyncio
import html
import logging
import time
from collections import defaultdict
from functools import wraps
from typing import Callable, Iterable

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from algotrading.db.models import Position, Signal, Strategy, Trade, TradeIntent
from algotrading.telegram.ui import (
    approval_keyboard,
    format_risk,
    format_status,
    format_strategies,
    format_summary,
)

log = logging.getLogger(__name__)

# HTML parse mode: MarkdownV2 rejects unescaped `_ ( ) . !` etc., and our
# texts are full of them (command names like /start_bot, "(M5)", prices).
# HTML only treats < > & specially, which these strings never contain.
TEXT_MARKDOWN = "HTML"

# Rate limiting: max commands per user per window
RATE_LIMIT_MAX_COMMANDS = 10
RATE_LIMIT_WINDOW_SEC = 60
_rate_limit_store: dict[int, list[float]] = defaultdict(list)
_rate_limit_lock = asyncio.Lock()


async def _check_rate_limit(user_id: int) -> bool:
    """Check if user is within rate limit. Returns True if allowed."""
    async with _rate_limit_lock:
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SEC
        # Prune old entries
        _rate_limit_store[user_id] = [ts for ts in _rate_limit_store[user_id] if ts > window_start]
        # Check limit
        if len(_rate_limit_store[user_id]) >= RATE_LIMIT_MAX_COMMANDS:
            return False
        _rate_limit_store[user_id].append(now)
        return True


def _rate_limited(fn: Callable):
    """Decorator to add rate limiting to a handler."""
    @wraps(fn)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.effective_user.id if update.effective_user else None
        if user_id is not None:
            allowed = await _check_rate_limit(user_id)
            if not allowed:
                log.warning("Rate limit exceeded for user %s", user_id)
                try:
                    await update.effective_message.reply_text(
                        f"⚠️ Rate limit exceeded. Max {RATE_LIMIT_MAX_COMMANDS} commands per {RATE_LIMIT_WINDOW_SEC}s."
                    )
                except Exception:
                    pass
                return
        return await fn(update, context)
    return wrapper


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
    session_factory,
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

    Each handler gets its own session from session_factory() to avoid
    concurrent access issues and PendingRollbackError when the database is
    locked by scheduler jobs running in worker threads.
    """
    allowed = set(allowed_users)
    auth = _auth_decorator(allowed)

    def session_for(_update):
        # Create a new session per request to avoid:
        # 1. Concurrent updates from multiple handlers (concurrent_updates=True)
        # 2. Database locks from scheduler jobs running in worker threads
        # 3. PendingRollbackError when a previous operation failed with "database is locked"
        return session_factory()

    # --- /help ---
    @auth
    @_rate_limited
    async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        text = (
            "<b>AlgoTrading control</b>\n"
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
    @_rate_limited
    async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        sess = session_for(update)
        try:
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
        finally:
            sess.close()

    # --- /start_bot / /stop_bot ---
    @auth
    @_rate_limited
    async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if on_start:
            await on_start(update, context)
        else:
            await update.effective_message.reply_text("Scheduler control not wired yet.")

    @auth
    @_rate_limited
    async def stop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if on_stop:
            await on_stop(update, context)
        else:
            await update.effective_message.reply_text("Scheduler control not wired yet.")

    # --- /strategy ---
    @auth
    @_rate_limited
    async def strategy_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        sess = session_for(update)
        try:
            strategies = sess.query(Strategy).order_by(Strategy.version.desc()).all()
            await update.effective_message.reply_text(
                format_strategies(strategies), parse_mode=TEXT_MARKDOWN
            )
        finally:
            sess.close()

    # --- /risk ---
    @auth
    @_rate_limited
    async def risk_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        sess = session_for(update)
        try:
            await update.effective_message.reply_text(
                format_risk(settings, risk_cfg, sess), parse_mode=TEXT_MARKDOWN
            )
        finally:
            sess.close()

    # --- /summary ---
    @auth
    @_rate_limited
    async def summary_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        sess = session_for(update)
        try:
            # M5 will populate AnalyticsSummary; query defensively.
            from algotrading.db.models import AnalyticsSummary

            summaries = sess.query(AnalyticsSummary).order_by(AnalyticsSummary.period.desc()).all()
            await update.effective_message.reply_text(
                format_summary(summaries, sess), parse_mode=TEXT_MARKDOWN
            )
        finally:
            sess.close()

    # --- natural-language chat (LLM orchestrator) ---
    @auth
    @_rate_limited
    async def chat_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Plain-text messages -> LLM orchestrator (runs in a worker thread).

        The agent can only read state, backtest, and create PENDING proposals
        (approval still required). It never trades or changes anything itself.
        """
        if not settings.ai.enabled:
            await update.effective_message.reply_text(
                "🤖 AI assistant is not configured. Set AI_API_KEY "
                "(+ AI_BASE_URL/AI_MODEL) in .env and restart."
            )
            return
        from algotrading.telegram.chat import run_agent

        text = update.effective_message.text or ""
        try:
            await update.effective_chat.send_action(ChatAction.TYPING)
        except Exception:  # noqa: BLE001 - typing indicator is best-effort
            pass
        result = await asyncio.to_thread(run_agent, session_factory, settings, text)
        await update.effective_message.reply_text(
            result.get("text") or "🤖 (no response)"
        )
        if result.get("proposal_id"):
            card = (
                f"🤖 <b>AI proposal #{result['proposal_id']}</b> "
                f"[{result.get('kind')}] for {result.get('strategy_name')}:\n"
                f"{html.escape(result.get('rationale') or '(no rationale)')}"
            )
            await update.effective_message.reply_text(
                card, parse_mode="HTML", reply_markup=approval_keyboard(result["proposal_id"])
            )

    # --- inline approval callbacks (M6) ---
    @auth
    @_rate_limited
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

    handlers = [
        CommandHandler("help", help_cmd),
        CommandHandler("status", status_cmd),
        CommandHandler("start_bot", start_cmd),
        CommandHandler("stop_bot", stop_cmd),
        CommandHandler("strategy", strategy_cmd),
        CommandHandler("risk", risk_cmd),
        CommandHandler("summary", summary_cmd),
        CallbackQueryHandler(on_approval, pattern=r"^(approve|reject):"),
    ]
    if session_factory is not None:
        handlers.append(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_cmd))
    return handlers


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
