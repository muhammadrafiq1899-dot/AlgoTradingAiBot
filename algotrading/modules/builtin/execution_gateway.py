"""Builtin execution gateways — the LOCKED part of the bot.

These are the only modules allowed to satisfy the ``execution`` capability.
They are ``builtin=True``, cannot be disabled via config, and external modules
claiming ``execution`` are rejected by the registry. The order path therefore
stays deterministic and auditable no matter what else is plugged in.

The gateway implementation itself (``PaperGateway`` / ``LiveGateway``) is
unchanged and remains the single place orders are created; the invariant
"persist the intent before sending the order" lives in ``execution/engine.py``.
"""
from __future__ import annotations

import logging
from typing import Any

from algotrading.config import get_secret
from algotrading.execution import LiveGateway, PaperGateway
from algotrading.modules.base import (
    CAPABILITY_EXECUTION,
    CAPABILITY_MARKET,
    Module,
    ModuleError,
    ModuleSpec,
)
from algotrading.modules.registry import register_module

log = logging.getLogger(__name__)


@register_module
class PaperGatewayModule(Module):
    spec = ModuleSpec(
        name="gateway.paper",
        capability=CAPABILITY_EXECUTION,
        description="Simulated fills against live prices with slippage + fees.",
        requires=(CAPABILITY_MARKET,),
        builtin=True,
    )

    def setup(self, ctx: Any) -> None:
        provider = ctx.get(CAPABILITY_MARKET)
        gateway = PaperGateway(
            provider, slippage_pct=ctx.settings.risk.slippage_pct
        )
        ctx.gateway = gateway
        ctx.provide(CAPABILITY_EXECUTION, gateway)


@register_module
class LiveGatewayModule(Module):
    spec = ModuleSpec(
        name="gateway.live",
        capability=CAPABILITY_EXECUTION,
        description="Real Binance Spot market orders (requires API keys).",
        requires=(CAPABILITY_MARKET,),
        builtin=True,
        enabled_by_default=False,
    )

    def setup(self, ctx: Any) -> None:
        key = get_secret("binance_api_key")
        secret = get_secret("binance_api_secret")
        if not key or not secret:
            raise ModuleError(
                "LIVE mode requires BINANCE_API_KEY and BINANCE_API_SECRET in .env"
            )
        from algotrading.market.binance_rest import BinanceRestClient

        log.warning(
            "LIVE MODE — real orders will be placed on Binance Spot. "
            "This must only be enabled after explicit review."
        )
        gateway = LiveGateway(BinanceRestClient(api_key=key, api_secret=secret))
        ctx.gateway = gateway
        ctx.provide(CAPABILITY_EXECUTION, gateway)
