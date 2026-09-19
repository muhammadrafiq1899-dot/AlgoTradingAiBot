"""Builtin market-data modules: real Binance prices or the offline demo feed.

Both satisfy the ``market`` capability by installing a
:class:`~algotrading.market.base.MarketDataProvider` on the context. Downstream
modules (execution, strategy) resolve the provider through the capability key,
so a third-party provider can be swapped in purely by configuration.

**Venue honesty.** Market data follows the configured venue, exactly like the
order path: ``market.use_testnet: true`` sends both orders *and* public data to
Binance's spot testnet, and ``market.base_url`` overrides both. This is
deliberate — pricing a testnet order off mainnet candles (or the reverse) would
make PnL, stops and backtests meaningless. The consequence is stated up front
rather than hidden: testnet has much thinner liquidity and a short history, so
its candles look nothing like the real market.
"""
from __future__ import annotations

import logging
from typing import Any

from algotrading.market.binance_provider import BinanceMarketProvider
from algotrading.market.binance_rest import BinanceRestClient, resolve_venue
from algotrading.market.demo import DemoProvider
from algotrading.modules.base import CAPABILITY_MARKET, Module, ModuleSpec
from algotrading.modules.registry import register_module

log = logging.getLogger(__name__)


@register_module
class BinanceMarketModule(Module):
    spec = ModuleSpec(
        name="market.binance",
        capability=CAPABILITY_MARKET,
        description=(
            "Public Binance REST klines/ticker (no API keys). Uses the same venue "
            "as the order path: with market.use_testnet=true this is the SPOT "
            "TESTNET feed, which has thin liquidity and a short history, not "
            "mainnet prices."
        ),
        builtin=True,
    )

    def setup(self, ctx: Any) -> None:
        base_url, testnet = resolve_venue(ctx.settings)
        if testnet:
            log.warning(
                "Market data comes from Binance spot TESTNET (%s): thin liquidity "
                "and a short history, so prices are NOT mainnet prices.",
                base_url,
            )
        provider = BinanceMarketProvider(client=BinanceRestClient.from_settings(ctx.settings))
        ctx.provider = provider
        ctx.provide(CAPABILITY_MARKET, provider)


@register_module
class DemoMarketModule(Module):
    spec = ModuleSpec(
        name="market.demo",
        capability=CAPABILITY_MARKET,
        description="Deterministic synthetic candle feed; no network, no keys.",
        builtin=True,
    )

    def setup(self, ctx: Any) -> None:
        provider = DemoProvider()
        ctx.provider = provider
        ctx.provide(CAPABILITY_MARKET, provider)
