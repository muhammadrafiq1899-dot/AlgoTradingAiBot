"""Telegram bot bootstrap: async Application + handler wiring.

The bot is a thin control surface. It reads config for the token + allowlist,
builds the PTB Application, registers the command/callback handlers from
`commands.py`, and (optionally) keeps a reference to an app controller so
/start_bot and /stop_bot can drive the scheduler.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

from telegram.ext import Application, ApplicationBuilder

from algotrading.config import Settings, get_secret
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
        on_backtest=_make_backtest(settings, session_factory),
        on_catalog_scores=_make_catalog_scores(settings, session_factory),
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


def _current_params(session, name: str) -> dict:
    """The params currently in effect for a strategy.

    Prefers the latest DB version (stripping internal `_`-prefixed keys left by
    AI proposals), falling back to the catalog's declared defaults so a freshly
    loaded plugin is still backtestable before it has a row.
    """
    from algotrading.store.strategy_versions import latest_version
    from algotrading.strategy.catalog import default_params

    row = latest_version(session, name)
    if row is not None:
        try:
            params = json.loads(row.params or "{}")
        except ValueError:
            params = {}
        if isinstance(params, dict):
            params = {k: v for k, v in params.items() if not k.startswith("_")}
            if params:
                return params
    return default_params(name)


def _make_backtest(settings, session_factory):
    """Factory for the /strategies inline backtest buttons.

    Replays the stored candles twice — once with the strategy's *current* params
    and once with the catalog *defaults* — then reports both side by side, so the
    user can see whether the live params actually beat the baseline. Deterministic
    and read-only: it never places an order.

    The session is created and closed inside the worker thread, mirroring the
    chat handler so SQLite connections stay single-threaded.
    """
    from algotrading.backtest.runner import run_backtest
    from algotrading.backtest.stored import load_candles
    from algotrading.strategy.catalog import default_params
    from algotrading.strategy.registry import known_names
    from algotrading.telegram.ui import format_backtest_comparison

    def _run(strategy_name: str) -> str:
        if strategy_name not in known_names():
            return f"Unknown strategy: {strategy_name}"
        session = session_factory()
        try:
            current = _current_params(session, strategy_name)
            defaults = default_params(strategy_name)
            symbol, interval, candles = load_candles(session, settings)
            if not candles:
                return f"no candles stored for {symbol} {interval} — cannot backtest yet"
            try:
                current_result = run_backtest(candles, strategy_name, dict(current))
                # Skip the second replay when the params are identical.
                default_result = (
                    current_result
                    if defaults == current
                    else run_backtest(candles, strategy_name, dict(defaults))
                )
            except (ValueError, KeyError, TypeError) as exc:
                return f"Backtest failed for {strategy_name}: {exc}"
            return format_backtest_comparison(
                strategy_name,
                symbol,
                interval,
                len(candles),
                current,
                defaults,
                current_result,
                default_result,
            )
        finally:
            session.close()

    async def backtest(strategy_name, update, context):
        try:
            return await asyncio.to_thread(_run, str(strategy_name))
        except Exception as exc:  # noqa: BLE001 - a button must never crash the bot
            log.warning("backtest button failed for %s: %s", strategy_name, exc)
            return f"Backtest failed for {strategy_name}: {exc}"

    return backtest


def _make_catalog_scores(
    settings,
    session_factory,
    max_strategies: int = 25,
    ttl_seconds: float = 60.0,
):
    """Factory for the /strategies risk-adjusted ranking.

    Backtests every catalog strategy against the stored candles using its
    *current* params (the same set the backtest buttons compare against) and
    returns ``{name: BacktestResult}``. A full pass is roughly
    O(strategies x candles^2), so it runs in a worker thread with its own
    session; the result is memoised for ``ttl_seconds`` keyed by the candle
    window, params, and strategy set, so repeated taps don't re-replay.

    Strategies beyond ``max_strategies`` are left unscored (they sort last) to
    bound the wait when many plugins are installed.
    """
    from algotrading.backtest.runner import run_backtest
    from algotrading.backtest.stored import load_candles
    from algotrading.strategy.catalog import catalog_entries

    cache: dict = {"key": None, "at": 0.0, "scores": {}}

    def _compute() -> dict:
        session = session_factory()
        try:
            symbol, interval, candles = load_candles(session, settings)
            if not candles:
                return {}

            entries = catalog_entries()
            params_by_name = {
                entry.name: _current_params(session, entry.name) for entry in entries
            }
            key = (
                symbol,
                interval,
                len(candles),
                candles[-1].ts,
                tuple(
                    (name, json.dumps(params, sort_keys=True, default=str))
                    for name, params in params_by_name.items()
                ),
            )
            now = time.monotonic()
            if cache["key"] == key and now - cache["at"] < ttl_seconds:
                return cache["scores"]

            if len(entries) > max_strategies:
                log.warning(
                    "/strategies: scoring %d of %d strategies (cap)",
                    max_strategies,
                    len(entries),
                )

            scores: dict = {}
            for name, params in list(params_by_name.items())[:max_strategies]:
                try:
                    scores[name] = run_backtest(candles, name, dict(params))
                except (ValueError, KeyError, TypeError) as exc:
                    log.warning("catalog scoring failed for %s: %s", name, exc)
                    scores[name] = None

            cache.update(key=key, at=now, scores=scores)
            return scores
        finally:
            session.close()

    async def scores() -> dict:
        return await asyncio.to_thread(_compute)

    return scores


def _stored_candles(session, settings) -> list:
    """Pull recent stored candles for the approval backtest, as Candle objects."""
    from algotrading.backtest.stored import load_candles

    return load_candles(session, settings)[2]


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
