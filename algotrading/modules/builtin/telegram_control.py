"""Builtin Telegram control surface module.

Starts the PTB polling application on ``start`` and tears it down on ``stop``.
The control surface can read state, research, backtest, and raise PENDING
proposals — it never trades. Approving a proposal goes through the existing
human approval flow (``RecommendationStore.apply``), which is unaffected by the
module system.

The scheduler controller (for ``/start_bot`` / ``/stop_bot``) is read from the
capability bag, so this module has no import-time dependency on ``main.py``.
"""
from __future__ import annotations

import asyncio
import html
import logging
from typing import Any

from algotrading.config import get_secret
from algotrading.modules.base import (
    CAPABILITY_CONTROL,
    CAPABILITY_SCHEDULER_CONTROL,
    Module,
    ModuleSpec,
)
from algotrading.modules.registry import register_module

log = logging.getLogger(__name__)


def make_recommendation_pusher(app, allowed_users: list[int], loop: asyncio.AbstractEventLoop):
    """Return a thread-safe callback that pushes a PENDING rec to Telegram."""
    from algotrading.telegram.ui import approval_keyboard

    async def _push(rec) -> None:
        text = (
            f"\U0001f916 <b>AI proposal #{rec.id}</b> [{rec.kind}] for {rec.strategy_name}:\n"
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


@register_module
class TelegramControlModule(Module):
    spec = ModuleSpec(
        name="control.telegram",
        capability=CAPABILITY_CONTROL,
        description="Telegram commands, chat orchestrator, and approval buttons.",
        builtin=True,
        params=("allow_chat",),
    )

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._app = None

    async def start(self, ctx: Any) -> None:
        if not get_secret("telegram_token"):
            log.warning("TELEGRAM_BOT_TOKEN missing; running without control surface")
            return

        from algotrading.telegram.bot import build_application

        controller = ctx.services.get(CAPABILITY_SCHEDULER_CONTROL)
        app = build_application(ctx.settings, ctx.session_factory, controller=controller)
        loop = asyncio.get_running_loop()
        ctx.on_recommendation = make_recommendation_pusher(
            app, ctx.settings.telegram_allowed_users, loop
        )
        await app.initialize()
        await app.updater.start_polling()
        await app.start()
        self._app = app
        log.info("telegram bot started (polling)")

    async def stop(self, ctx: Any) -> None:
        if self._app is None:
            return
        await self._app.updater.stop()
        await self._app.stop()
        await self._app.shutdown()
        self._app = None
