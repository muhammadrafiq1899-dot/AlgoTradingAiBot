"""Builtin execution gateways — the LOCKED part of the bot.

These are the only modules allowed to satisfy the ``execution`` capability.
They are ``builtin=True``, cannot be disabled via config, and external modules
claiming ``execution`` are rejected by the registry. The order path therefore
stays deterministic and auditable no matter what else is plugged in.

The gateway implementation itself (``PaperGateway`` / ``LiveGateway``) is
unchanged and remains the single place orders are created; the invariant
"persist the intent before sending the order" lives in ``execution/engine.py``.

Venue selection for live mode comes from ``settings.market``: ``use_testnet``
routes orders to Binance's spot testnet and switches the credentials to
BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET, while an explicit
``base_url`` wins over both. There is no silent mainnet fallback — a missing
key for the selected venue raises here, at wiring time, instead of placing a
real order against the wrong market.
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
        description=(
            "Simulated fills against live prices with slippage + fees. Full "
            "gateway surface: market/limit/stop orders, cancel, open orders and "
            "a paper balance, all simulated in memory."
        ),
        requires=(CAPABILITY_MARKET,),
        builtin=True,
    )

    def setup(self, ctx: Any) -> None:
        provider = ctx.get(CAPABILITY_MARKET)
        gateway = PaperGateway(
            provider,
            slippage_pct=ctx.settings.risk.slippage_pct,
            # Same number the engine falls back to, so the simulated venue
            # balance and the sizing basis cannot disagree.
            initial_balance=ctx.settings.risk.paper_initial_balance,
        )
        ctx.gateway = gateway
        ctx.provide(CAPABILITY_EXECUTION, gateway)


@register_module
class LiveGatewayModule(Module):
    spec = ModuleSpec(
        name="gateway.live",
        capability=CAPABILITY_EXECUTION,
        description=(
            "Real Binance Spot orders (requires API keys). Venue from "
            "market.use_testnet / market.base_url: default is mainnet, testnet "
            "uses BINANCE_TESTNET_API_KEY/SECRET."
        ),
        requires=(CAPABILITY_MARKET,),
        builtin=True,
        enabled_by_default=False,
    )

    def setup(self, ctx: Any) -> None:
        from algotrading.market.binance_rest import BinanceRestClient, resolve_venue

        base_url, testnet = resolve_venue(ctx.settings)
        if testnet:
            key_name, secret_name = "binance_testnet_key", "binance_testnet_secret"
            hint = "BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET"
        else:
            key_name, secret_name = "binance_api_key", "binance_api_secret"
            hint = "BINANCE_API_KEY and BINANCE_API_SECRET"

        key = get_secret(key_name)
        secret = get_secret(secret_name)
        if not key or not secret:
            # Fail here, loudly: continuing without keys (or "just" using the
            # mainnet ones) would place orders on a venue the operator did not
            # choose.
            raise ModuleError(
                f"LIVE gateway on {base_url} requires {hint} in .env "
                f"(testnet={testnet})"
            )

        if testnet:
            log.warning(
                "LIVE MODE (TESTNET) — orders go to %s using the testnet keys. "
                "No real money moves, but testnet liquidity and history are thin; "
                "market data is served from the same venue.",
                base_url,
            )
        else:
            log.warning(
                "LIVE MODE — real orders will be placed on Binance Spot (%s). "
                "This must only be enabled after explicit review.",
                base_url,
            )

        client = BinanceRestClient.from_settings(
            ctx.settings, api_key=key, api_secret=secret, require_keys=True
        )
        gateway = LiveGateway(client)
        ctx.gateway = gateway
        ctx.provide(CAPABILITY_EXECUTION, gateway)
