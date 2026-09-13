"""Builtin HTTP API module: optional FastAPI health/status endpoints.

Off by default; enabled by ``api.enabled: true`` + ``API_TOKEN`` in ``.env``
(which flips the default module set). Runs uvicorn as a task on the bot's own
event loop so it shares the process without touching the trading loop.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from algotrading.config import get_secret
from algotrading.modules.base import CAPABILITY_API, Module, ModuleSpec
from algotrading.modules.registry import register_module

log = logging.getLogger(__name__)


@register_module
class HttpApiModule(Module):
    spec = ModuleSpec(
        name="api.http",
        capability=CAPABILITY_API,
        description="FastAPI /health, /metrics, /status on the bot event loop.",
        builtin=True,
        enabled_by_default=False,
    )

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__(params)
        self._task: asyncio.Task | None = None
        self._server = None

    async def start(self, ctx: Any) -> None:
        token = get_secret("api_token")
        if not token:
            log.error("api.enabled is true but API_TOKEN is not set; API not started")
            return

        import uvicorn

        from algotrading.api import build_api

        api_app = build_api(
            ctx.settings, ctx.session_factory, ctx.health, token=token
        )
        self._server = uvicorn.Server(
            uvicorn.Config(
                api_app,
                host=ctx.settings.api.host,
                port=ctx.settings.api.port,
                log_level="warning",
            )
        )
        self._task = asyncio.create_task(self._server.serve())
        log.info(
            "internal API on http://%s:%s",
            ctx.settings.api.host,
            ctx.settings.api.port,
        )

    async def stop(self, ctx: Any) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - shutdown must never raise
                self._task.cancel()
            self._task = None
