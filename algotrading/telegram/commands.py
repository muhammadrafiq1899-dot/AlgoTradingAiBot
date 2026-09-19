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
import uuid
from collections import defaultdict
from functools import wraps
from pathlib import Path
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

from algotrading.ai.news import fetch_headlines
from algotrading.analytics.portfolio import portfolio_snapshot
from algotrading.db.models import Position, Signal, Strategy, Trade, TradeIntent
from algotrading.store.export import export_summary_json, export_trades_csv
from algotrading.strategy.catalog import catalog_entries
from algotrading.telegram.ui import (
    IMAGE_FLOW_MODES,
    approval_keyboard,
    backtest_keyboard,
    format_news,
    format_portfolio,
    format_risk,
    format_status,
    format_strategy_catalog,
    format_strategies,
    format_summary,
    image_flow_keyboard,
    rank_by_score,
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


def reset_rate_limits() -> None:
    """Clear all rate-limit windows (operational escape hatch + test isolation)."""
    _rate_limit_store.clear()


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


# --- image input (Telegram photo / image file -> advisory LLM) ---------------
# Fallback question when the picture arrives with no caption: without one the
# model tends to just describe the image instead of proposing something.
IMAGE_CAPTION_FALLBACK = (
    "What does this image show? Map it to the closest strategy I can backtest, "
    "or propose a new one if nothing fits."
)

# Uploaded images are deleted right after the AI reads them; this only sweeps up
# files an interrupted call left behind.
UPLOAD_MAX_AGE_SECONDS = 24 * 3600

# Fallback question when several pictures arrive together with no caption.
IMAGE_BATCH_CAPTION_FALLBACK = (
    "These images belong together. If they describe one strategy, write it as a "
    "single strategy; if they describe separate strategies, say which is which "
    "and what combining them would mean."
)

# Telegram delivers an album as one message per photo, and clients happily send
# a burst of separate photos. Both are buffered for this long, then answered once
# — a single photo waits the same (imperceptible) moment so behaviour is uniform.
PHOTO_BATCH_DELAY_SECONDS = 1.2

# A batch larger than this is trimmed: each image costs its own vision pass, and
# this runs on a phone.
IMAGE_BATCH_LIMIT = 6

# How long a buffered image set stays valid after the choice keyboard is sent.
IMAGE_FLOW_TTL_SECONDS = 15 * 60

IMAGE_FLOW_CHOICE_TEXT = (
    "🖼 {count} images received. How should I read them?\n\n"
    "🧩 One strategy from all — I read every image, then write a single strategy "
    "that combines the rules (entry from one, filter from another, …).\n"
    "🧱 Separate strategies → ensemble — each image becomes its own strategy; once "
    "you approve them I can draft an ensemble that votes on their signals."
)

IMAGE_FLOW_SEPARATE_FOLLOWUP = (
    "🧱 Read {count} image(s) separately. Approve the proposals you want, then "
    "send \"combine those strategies into an ensemble\" and I'll draft it "
    "(approving the ensemble makes it your active strategy)."
)

IMAGE_FLOW_EXPIRED = (
    "⌛ That image set expired — send the pictures again and I'll re-read them."
)

# Shown by /news when the feed module is switched off. Public RSS is opt-in
# because it is the only place the bot talks to a third-party website.
NEWS_DISABLED_TEXT = (
    "📰 News headlines are off. Set ai.news_enabled: true in "
    "config/settings.yaml (and pick news_feed_urls) and restart the bot."
)

# Buffered uploads, keyed by chat: [(path, caption), ...] awaiting a flush.
_photo_batches: dict[int, list[tuple[Path, str]]] = {}
# In-flight flush task per chat, so a newer photo can cancel the pending answer.
_photo_batch_tasks: dict[int, asyncio.Task] = {}
# Choice-keyboard state: token -> (chat_id, [path, ...], created_at). Callback
# data is capped at 64 bytes, so the paths never travel through Telegram.
_image_flows: dict[str, tuple[int, list[Path], float]] = {}


def reset_photo_batches() -> None:
    """Drop buffered uploads and pending image flows (tests + clean shutdown).

    Files already written to disk are left for :func:`_prune_uploads`; a restart
    simply forgets which pictures were waiting for a choice.
    """
    for task in list(_photo_batch_tasks.values()):
        task.cancel()
    _photo_batch_tasks.clear()
    _photo_batches.clear()
    _image_flows.clear()


def _delete_uploads(paths: Iterable[str | Path]) -> None:
    """Remove consumed uploads. Best effort: a leftover is pruned, not fatal."""
    for path in paths:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            log.debug("could not delete upload %s", path, exc_info=True)

# Telegram photos are JPEG; documents can be anything, so keep an allowlist.
_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
_MIME_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
}


def _uploads_dir(settings) -> Path:
    """Where incoming images are written (settings.ai.image_dir)."""
    return Path(getattr(settings.ai, "image_dir", "") or "data/uploads")


def _image_suffix(file_name: str | None, mime_type: str | None) -> str:
    """Extension for an upload: from its filename if sane, else from its MIME."""
    suffix = Path(file_name or "").suffix.lower()
    if suffix in _UPLOAD_SUFFIXES:
        return suffix
    return _MIME_SUFFIXES.get((mime_type or "").lower(), ".jpg")


def _image_path(directory: Path, user_id: int, suffix: str) -> Path:
    """Collision-free target path for one upload."""
    return directory / f"img-{int(time.time() * 1000)}-{user_id}{suffix}"


def _prune_uploads(directory: Path,
                   max_age_seconds: int = UPLOAD_MAX_AGE_SECONDS) -> None:
    """Delete stale uploads. Best effort: housekeeping never breaks the handler."""
    cutoff = time.time() - max_age_seconds
    try:
        for stale in directory.glob("img-*"):
            if stale.is_file() and stale.stat().st_mtime < cutoff:
                stale.unlink()
    except OSError:
        log.debug("could not prune uploads in %s", directory, exc_info=True)


async def collect_image(update: Update, context: ContextTypes.DEFAULT_TYPE,
                        settings) -> tuple[Path | None, str]:
    """Download the image in an incoming message to a local file.

    Returns ``(path, "")`` when the image is ready to hand to the AI, or
    ``(None, reason)`` with a user-facing explanation of the refusal. Gating
    (assistant enabled, images enabled, provider can actually see) happens before
    any bytes are fetched.
    """
    message = update.effective_message
    document = getattr(message, "document", None)
    if document is not None:
        mime = document.mime_type or ""
        if not mime.startswith("image/"):
            return None, ("🖼 I can only read images — that file is "
                          f"{mime or 'of an unknown type'}.")
        file_id, size = document.file_id, document.file_size
        suffix = _image_suffix(document.file_name, mime)
    else:
        photo = list(getattr(message, "photo", None) or [])
        if not photo:
            return None, "🖼 I couldn't find an image in that message."
        file_id, size, suffix = photo[-1].file_id, photo[-1].file_size, ".jpg"

    if not settings.ai.enabled:
        return None, ("🤖 AI assistant is not configured. Set AI_API_KEY or "
                      "USE_HERMES=true in .env and restart the bot.")
    if not getattr(settings.ai, "images_enabled", True):
        return None, ("🖼 Image input is turned off (ai.images_enabled: false in "
                      "config/settings.yaml).")
    if not getattr(settings.ai, "use_hermes", False):
        return None, ("🖼 Image input needs the local Hermes Agent: set "
                      "USE_HERMES=true in .env and restart the bot. "
                      "(The AI_API_KEY provider has no vision path.)")
    max_bytes = int(getattr(settings.ai, "image_max_bytes", 0) or 0)
    if max_bytes and size and size > max_bytes:
        return None, (f"🖼 That image is {size / 1e6:.1f} MB — I read up to "
                      f"{max_bytes / 1e6:.1f} MB. Send a smaller screenshot.")

    directory = _uploads_dir(settings)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _prune_uploads(directory)
        user_id = update.effective_user.id if update.effective_user else 0
        path = _image_path(directory, user_id, suffix)
        telegram_file = await context.bot.get_file(file_id)
        await telegram_file.download_to_drive(str(path))
    except Exception as exc:  # noqa: BLE001 - network/disk trouble must not crash
        log.warning("image download failed: %s", exc)
        return None, "🖼 I couldn't download that image — please send it again."

    # Telegram's declared size can be missing; verify what actually landed.
    actual = path.stat().st_size if path.exists() else 0
    if max_bytes and actual > max_bytes:
        path.unlink(missing_ok=True)
        return None, (f"🖼 That image is {actual / 1e6:.1f} MB — I read up to "
                      f"{max_bytes / 1e6:.1f} MB. Send a smaller screenshot.")
    if actual == 0:
        path.unlink(missing_ok=True)
        return None, "🖼 That image came through empty — please send it again."
    return path, ""


async def _deliver_agent_result(send, result: dict) -> None:
    """Send the agent's answer, plus the approval card when it proposed a change.

    ``send`` is any async callable taking ``(text, **kwargs)`` — a message's
    ``reply_text`` for chat replies, or a chat-bound ``context.bot.send_message``
    when the reply is not anchored to one incoming message (image batches).
    """
    await send(result.get("text") or "🤖 (no response)")
    if result.get("proposal_id"):
        card = (
            f"🤖 <b>AI proposal #{result['proposal_id']}</b> "
            f"[{result.get('kind')}] for {result.get('strategy_name')}:\n"
            f"{html.escape(result.get('rationale') or '(no rationale)')}"
        )
        await send(
            card, parse_mode=TEXT_MARKDOWN,
            reply_markup=approval_keyboard(result["proposal_id"]),
        )


def build_handlers(
    session_factory,
    settings,
    risk_cfg,
    on_start=None,
    on_stop=None,
    on_approve=None,
    on_reject=None,
    on_backtest=None,
    on_catalog_scores=None,
    allowed_users: Iterable[int] = (),
) -> list:
    """Return PTB handlers bound to DB + settings + control callbacks.

    `on_start`/`on_stop` are async callables(update, context) that perform the
    actual scheduler start/stop; `on_approve`/`on_reject` handle the inline
    recommendation callbacks (M6); `on_backtest(strategy_name, update, context)`
    runs the /strategies backtest buttons; `on_catalog_scores()` returns
    {strategy name: BacktestResult} used to rank the /strategies list. Passing
    None keeps the command as a stub.

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
            "/strategies — available strategies (tap one to backtest)\n"
            "/risk — current risk limits\n"
            "/summary — analytics summary (M5)\n"
            "/portfolio — exposure + correlation view (risk context, not a signal)\n"
            "/news — recent headlines (needs ai.news_enabled)\n"
            "/export — write trades (CSV) + summary (JSON) under data/exports/\n"
            "/paper — switch to paper mode\n"
            "/live — switch to live mode (guarded)\n"
            "…or just chat: ask about the bot, or send a photo/screenshot and "
            "the AI will read it (needs USE_HERMES=true). Send several at once "
            "and it asks whether to make one strategy or one per image.\n"
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

    # --- /strategies (live catalog: built-ins + plugins, with param ranges) ---
    @auth
    @_rate_limited
    async def strategies_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        entries = catalog_entries()
        # Scores are optional: if scoring fails (or there are no candles yet) the
        # catalog still renders, just unranked.
        scores: dict = {}
        if on_catalog_scores is not None:
            try:
                scores = await on_catalog_scores() or {}
            except Exception:  # noqa: BLE001 - never break the command over scoring
                log.exception("/strategies: scoring failed")
        ranked = rank_by_score(entries, scores)
        await update.effective_message.reply_text(
            format_strategy_catalog(ranked, scores),
            parse_mode=TEXT_MARKDOWN,
            reply_markup=backtest_keyboard(ranked),
        )

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

    # --- /portfolio (exposure + descriptive correlation; read-only) ---
    @auth
    @_rate_limited
    async def portfolio_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Exposure per symbol + a correlation matrix over stored closes.

        Read-only and advisory: nothing here feeds sizing or signals.
        """
        sess = session_for(update)
        try:
            text = format_portfolio(portfolio_snapshot(sess, settings))
        except Exception:  # noqa: BLE001 - a view must never break the bot
            log.exception("/portfolio failed")
            text = "📊 Could not build the portfolio view — see logs/algotrading.log."
        finally:
            sess.close()
        await update.effective_message.reply_text(text, parse_mode=TEXT_MARKDOWN)

    # --- /news (public RSS headlines; untrusted third-party data) ---
    @auth
    @_rate_limited
    async def news_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show the configured feed headlines, or say the feature is off.

        Fetched in a worker thread (network) with a TTL cache in ``ai/news.py``,
        so a repeated /news costs nothing. Headline text is shown as data only.
        """
        ai = getattr(settings, "ai", None)
        if ai is None or not getattr(ai, "news_enabled", False):
            await update.effective_message.reply_text(NEWS_DISABLED_TEXT)
            return
        try:
            headlines = await asyncio.to_thread(fetch_headlines, ai)
        except Exception:  # noqa: BLE001 - offline phone is normal, not an error
            log.warning("/news fetch failed", exc_info=True)
            headlines = []
        await update.effective_message.reply_text(
            format_news(headlines), parse_mode=TEXT_MARKDOWN
        )

    # --- /export (write the trade history to files on the device) ---
    @auth
    @_rate_limited
    async def export_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Dump trades + a summary to files under ``data/exports/``.

        Read-only with respect to trading state: it reads the DB and writes
        files, nothing else. Useful on the phone, where the practical way to get
        the history off the device is a path you can scp/cat — Telegram is just
        where that path is announced.
        """
        sess = session_for(update)
        try:
            stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            out_dir = Path(settings.data_dir) / "exports"
            out_dir.mkdir(parents=True, exist_ok=True)
            trades_path = out_dir / f"trades-{stamp}.csv"
            summary_path = out_dir / f"summary-{stamp}.json"
            n_trades = export_trades_csv(sess, trades_path)
            export_summary_json(sess, summary_path)
            text = (
                f"📤 <b>Exported</b> {n_trades} trade(s)\n"
                f"<code>{html.escape(str(trades_path))}</code>\n"
                f"<code>{html.escape(str(summary_path))}</code>"
            )
        except Exception:  # noqa: BLE001 - an export must never break the bot
            log.exception("/export failed")
            text = "📤 Export failed — see logs/algotrading.log."
        finally:
            sess.close()
        await update.effective_message.reply_text(text, parse_mode=TEXT_MARKDOWN)

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
                "(+ AI_BASE_URL/AI_MODEL) in .env — or set USE_HERMES=true to "
                "run it through the local Hermes Agent — and restart."
            )
            return
        from algotrading.telegram.chat import run_agent

        text = update.effective_message.text or ""
        try:
            await update.effective_chat.send_action(ChatAction.TYPING)
        except Exception:  # noqa: BLE001 - typing indicator is best-effort
            pass
        result = await asyncio.to_thread(run_agent, session_factory, settings, text)
        await _deliver_agent_result(update.effective_message.reply_text, result)

    # --- image chat: photo / image document -> same orchestrator, image attached ---

    async def _run_image_flow(context, chat_id: int, images: list, mode: str) -> None:
        """Read a batch of uploads — 'combine' (one strategy) or 'separate' (one each).

        Both flows keep the chat contract: read + propose only, human approval
        still required for every change.
        """
        from algotrading.telegram.chat import run_agent

        def send(text, **kwargs):
            return context.bot.send_message(chat_id, text=text, **kwargs)

        paths = [path for path, _ in images]
        caption = next((str(c).strip() for _, c in images if str(c or "").strip()), "")
        try:
            await context.bot.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:  # noqa: BLE001 - typing indicator is best-effort
            pass

        if mode == "separate":
            # One image per agent call: each picture becomes its own strategy, so
            # the ensemble flow can combine them afterwards.
            for path, image_caption in images:
                result = await asyncio.to_thread(
                    run_agent, session_factory, settings,
                    str(image_caption or "").strip() or IMAGE_CAPTION_FALLBACK,
                    image_paths=[str(path)],
                )
                await _deliver_agent_result(send, result)
            await send(IMAGE_FLOW_SEPARATE_FOLLOWUP.format(count=len(images)))
            return

        user_text = caption or (
            IMAGE_CAPTION_FALLBACK if len(paths) == 1 else IMAGE_BATCH_CAPTION_FALLBACK
        )
        result = await asyncio.to_thread(
            run_agent, session_factory, settings, user_text,
            image_paths=[str(p) for p in paths],
        )
        await _deliver_agent_result(send, result)

    async def _offer_image_flow(context, chat_id: int, images: list,
                                dropped: int = 0) -> None:
        """Ask how to read a multi-image batch, stashing the uploads by token.

        ``images`` is ``[(path, caption), ...]``: the caption has to survive the
        round trip through the keyboard, or the user's wording is lost.
        """
        # One live flow per chat: a fresh batch invalidates earlier tokens.
        for token in [t for t, (cid, _, _) in _image_flows.items() if cid == chat_id]:
            _image_flows.pop(token, None)
        token = uuid.uuid4().hex[:8]
        _image_flows[token] = (chat_id, list(images), time.time())
        text = IMAGE_FLOW_CHOICE_TEXT.format(count=len(images))
        if dropped:
            text += f"\n\n(Only the first {len(images)} are used.)"
        await context.bot.send_message(
            chat_id, text=text, reply_markup=image_flow_keyboard(token)
        )

    async def _flush_photo_batch(context, chat_id: int) -> None:
        """Answer a burst of photos once the user has stopped sending them."""
        try:
            await asyncio.sleep(PHOTO_BATCH_DELAY_SECONDS)
        except asyncio.CancelledError:
            return  # a newer photo re-scheduled this flush
        _photo_batch_tasks.pop(chat_id, None)
        images = _photo_batches.pop(chat_id, [])
        if not images:
            return
        dropped = max(0, len(images) - IMAGE_BATCH_LIMIT)
        if dropped:
            # Discard the extras now instead of waiting for the 24h sweep.
            _delete_uploads([path for path, _ in images[IMAGE_BATCH_LIMIT:]])
        images = images[:IMAGE_BATCH_LIMIT]
        if len(images) == 1:
            # A lone photo behaves exactly like it always did: read and answer.
            try:
                await _run_image_flow(context, chat_id, images, "combine")
            finally:
                _delete_uploads([path for path, _ in images])
            return
        await _offer_image_flow(context, chat_id, images, dropped)

    def _schedule_photo_flush(context, chat_id: int) -> None:
        previous = _photo_batch_tasks.get(chat_id)
        if previous is not None and not previous.done():
            previous.cancel()
        _photo_batch_tasks[chat_id] = asyncio.create_task(
            _flush_photo_batch(context, chat_id)
        )

    @auth
    @_rate_limited
    async def image_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
        """A photo or image file: buffered for a moment, then read in one go.

        One picture is answered immediately (with the image attached). Several
        pictures arriving together are answered once, after the user picks how to
        read them (one strategy, or one strategy each for the ensemble flow).
        The uploads are deleted as soon as they have been read.
        """
        message = update.effective_message
        path, error = await collect_image(update, context, settings)
        if path is None:
            await message.reply_text(error)
            return
        chat_id = update.effective_chat.id
        _photo_batches.setdefault(chat_id, []).append(
            (path, str(getattr(message, "caption", None) or ""))
        )
        _schedule_photo_flush(context, chat_id)

    # --- inline callbacks: approval (M6) + backtest buttons ---
    @auth
    @_rate_limited
    async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        action, _, payload = (query.data or "").partition(":")

        if action in ("approve", "reject"):
            rec_id = int(payload)
            handler = on_approve if action == "approve" else on_reject
            if handler is None:
                await query.edit_message_text(f"{action} not implemented yet.")
                return
            result = await handler(rec_id, update, context)
            await query.edit_message_text(
                f"Recommendation {rec_id} {action}d.\n{result or ''}", parse_mode=TEXT_MARKDOWN
            )
            return

        if action == "bt":
            if on_backtest is None:
                await query.edit_message_text("Backtest not wired yet.")
                return
            result = await on_backtest(payload, update, context)
            # Reply (don't edit) so the catalog + buttons stay usable for more taps.
            await update.effective_message.reply_text(result, parse_mode=TEXT_MARKDOWN)
            return

        if action == "imgflow":
            token, _, mode = payload.partition(":")
            entry = _image_flows.pop(token, None)
            if entry is None or mode not in IMAGE_FLOW_MODES:
                await query.edit_message_text(IMAGE_FLOW_EXPIRED)
                return
            chat_id, images, created = entry
            paths = [path for path, _ in images]
            if time.time() - created > IMAGE_FLOW_TTL_SECONDS:
                _delete_uploads(paths)
                await query.edit_message_text(IMAGE_FLOW_EXPIRED)
                return
            label = ("one strategy from all" if mode == "combine"
                     else "separate strategies → ensemble")
            await query.edit_message_text(
                f"🖼 Reading {len(paths)} images — {label}…"
            )
            try:
                await _run_image_flow(context, chat_id, images, mode)
            finally:
                _delete_uploads(paths)
            return

        log.warning("ignoring unknown callback action %r", action)

    handlers = [
        CommandHandler("help", help_cmd),
        CommandHandler("status", status_cmd),
        CommandHandler("start_bot", start_cmd),
        CommandHandler("stop_bot", stop_cmd),
        CommandHandler("strategy", strategy_cmd),
        CommandHandler("strategies", strategies_cmd),
        CommandHandler("risk", risk_cmd),
        CommandHandler("summary", summary_cmd),
        CommandHandler("portfolio", portfolio_cmd),
        CommandHandler("news", news_cmd),
        CommandHandler("export", export_cmd),
        CallbackQueryHandler(on_callback, pattern=r"^(approve|reject|bt|imgflow):"),
    ]
    if session_factory is not None:
        handlers.append(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_cmd))
        # Photos and image files: the same agent, with the picture attached.
        # Registered last so text routing is untouched.
        handlers.append(
            MessageHandler(filters.PHOTO | filters.Document.IMAGE, image_cmd)
        )
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
