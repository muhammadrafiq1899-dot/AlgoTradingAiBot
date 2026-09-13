"""Builtin market-data modules: real Binance prices or the offline demo feed.

Both satisfy the ``market`` capability by installing a
:class:`~algotrading.market.base.MarketDataProvider` on the context. Downstream
modules (execution, strategy) resolve the provider through the capability key,
so a third-party provider can be swapped in purely by configuration.
"""
from __future__ import annotations

from typing import Any

from algotrading.market.binance_provider import BinanceMarketProvider
from algotrading.market.demo import DemoProvider
from algotrading.modules.base import CAPABILITY_MARKET, Module, ModuleSpec
from algotrading.modules.registry import register_module


@register_module
class BinanceMarketModule(Module):
    spec = ModuleSpec(
        name="market.binance",
        capability=CAPABILITY_MARKET,
        description="Public Binance REST klines/ticker (paper or live prices).",
        builtin=True,
    )

    def setup(self, ctx: Any) -> None:
        provider = BinanceMarketProvider()
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
