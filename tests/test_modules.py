"""Module framework: config-driven selection, lifecycle, jobs, execution lock.

Nothing here touches the network or a real exchange: the paper gateway and the
Binance provider are only *constructed*, not called.
"""
import asyncio
import sys
import textwrap

import pytest

from algotrading.config import (
    AIConfig,
    ApiConfig,
    MarketConfig,
    ModulesConfig,
    RiskConfig,
    ScheduleConfig,
    Settings,
    validate_settings,
)
from algotrading.modules import ModuleError, ModuleManager, module_names
from algotrading.modules.base import CAPABILITY_EXECUTION, CAPABILITY_MARKET
from algotrading.modules.registry import load_external_modules
from algotrading.scheduler.jobs import BotContext
from algotrading.supervisor.health import HealthMonitor


def _settings(tmp_path, modules: ModulesConfig | None = None, mode: str = "paper",
              api_enabled: bool = False) -> Settings:
    return Settings(
        mode=mode,
        market=MarketConfig(symbols=["BTC/USDT"], intervals=["1m", "1h"]),
        risk=RiskConfig(),
        schedule=ScheduleConfig(),
        ai=AIConfig(enabled=False),
        api=ApiConfig(enabled=api_enabled),
        modules=modules or ModulesConfig(strategy_plugin_paths=[str(tmp_path / "strategies")]),
    )


def _ctx(settings, tmp_path) -> BotContext:
    return BotContext(
        settings=settings,
        session_factory=lambda: None,  # unused by setup
        health=HealthMonitor(str(tmp_path / "heartbeat")),
    )


def _names(manager: ModuleManager) -> set[str]:
    return {m.name for m in manager.resolve()}


# --- resolution --------------------------------------------------------------

def test_paper_defaults_include_builtin_modules(tmp_path):
    names = _names(ModuleManager(_settings(tmp_path)))
    assert {"market.binance", "gateway.paper", "strategy.plugins",
            "analytics.default", "control.telegram"} <= names
    assert "api.http" not in names  # api.enabled is false


def test_demo_selects_demo_feed_and_paper_gateway(tmp_path):
    names = _names(ModuleManager(_settings(tmp_path), demo=True))
    assert "market.demo" in names
    assert "gateway.paper" in names
    assert "market.binance" not in names


def test_live_mode_selects_live_gateway(tmp_path):
    names = _names(ModuleManager(_settings(tmp_path, mode="live")))
    assert "gateway.live" in names
    assert "gateway.paper" not in names


def test_api_module_selected_when_enabled(tmp_path):
    names = _names(ModuleManager(_settings(tmp_path, api_enabled=True)))
    assert "api.http" in names


def test_explicit_enabled_list_overrides_defaults(tmp_path):
    modules = ModulesConfig(
        enabled=["market.demo", "gateway.paper"],
        strategy_plugin_paths=[str(tmp_path / "strategies")],
    )
    names = _names(ModuleManager(_settings(tmp_path, modules=modules)))
    assert names == {"market.demo", "gateway.paper"}


def test_unknown_module_is_rejected(tmp_path):
    modules = ModulesConfig(enabled=["does.not.exist"],
                            strategy_plugin_paths=[str(tmp_path / "strategies")])
    with pytest.raises(ModuleError):
        ModuleManager(_settings(tmp_path, modules=modules)).resolve()


def test_execution_module_cannot_be_disabled(tmp_path):
    """The gateway is locked: config may not remove the order path."""
    names = _names(ModuleManager(_settings(tmp_path), disabled=("gateway.paper",)))
    assert "gateway.paper" in names


def test_demo_rejects_live_gateway(tmp_path):
    modules = ModulesConfig(
        enabled=["market.demo", "gateway.live"],
        strategy_plugin_paths=[str(tmp_path / "strategies")],
    )
    with pytest.raises(ModuleError):
        ModuleManager(_settings(tmp_path, modules=modules), demo=True).resolve()


def test_missing_dependency_is_rejected(tmp_path):
    # Execution requires market; selecting it alone must fail loudly.
    modules = ModulesConfig(
        enabled=["gateway.paper"],
        strategy_plugin_paths=[str(tmp_path / "strategies")],
    )
    with pytest.raises(ModuleError):
        ModuleManager(_settings(tmp_path, modules=modules)).resolve()


# --- external modules + the execution lock -----------------------------------

def test_external_module_cannot_claim_execution(tmp_path, monkeypatch):
    ext = tmp_path / "ext_mods"
    ext.mkdir()
    (ext / "rogue.py").write_text(textwrap.dedent('''
        from algotrading.modules.base import CAPABILITY_EXECUTION, Module, ModuleSpec

        class RogueGateway(Module):
            spec = ModuleSpec(
                name="gateway.rogue",
                capability=CAPABILITY_EXECUTION,
                builtin=False,
            )
    '''), encoding="utf-8")
    monkeypatch.syspath_prepend(str(ext))

    load_external_modules(["rogue"])

    # Refused: an external module may never hold a locked capability.
    assert "gateway.rogue" not in module_names()
    sys.modules.pop("rogue", None)


def test_external_module_registers_non_locked(tmp_path, monkeypatch):
    ext = tmp_path / "ext_mods2"
    ext.mkdir()
    (ext / "custom_market.py").write_text(textwrap.dedent('''
        from algotrading.modules.base import CAPABILITY_MARKET, Module, ModuleSpec

        class CustomMarket(Module):
            spec = ModuleSpec(
                name="market.custom",
                capability=CAPABILITY_MARKET,
                builtin=False,
            )
            def setup(self, ctx):
                ctx.provide(CAPABILITY_MARKET, "custom-provider")
    '''), encoding="utf-8")
    monkeypatch.syspath_prepend(str(ext))

    registered = load_external_modules(["custom_market:CustomMarket"])
    assert "market.custom" in registered
    assert "market.custom" in module_names()
    sys.modules.pop("custom_market", None)


# --- lifecycle ---------------------------------------------------------------

def test_setup_installs_services(tmp_path):
    settings = _settings(tmp_path)
    ctx = _ctx(settings, tmp_path)
    manager = ModuleManager(settings)
    manager.setup(ctx)

    assert ctx.provider is not None
    assert ctx.gateway is not None
    assert ctx.has(CAPABILITY_MARKET)
    assert ctx.has(CAPABILITY_EXECUTION)
    # The gateway is the same object exposed under both names.
    assert ctx.get(CAPABILITY_EXECUTION) is ctx.gateway


def test_setup_collects_module_jobs(tmp_path):
    settings = _settings(tmp_path)
    ctx = _ctx(settings, tmp_path)
    manager = ModuleManager(settings)
    manager.setup(ctx)

    ids = {job.job_id for job in manager.collect_jobs(ctx)}
    assert {"analytics", "analytics_daily", "ai_review"} <= ids


def test_start_and_stop_are_clean(tmp_path):
    settings = _settings(tmp_path)
    ctx = _ctx(settings, tmp_path)
    manager = ModuleManager(settings)
    manager.setup(ctx)

    async def cycle():
        await manager.start(ctx)
        await manager.stop(ctx)

    asyncio.run(cycle())  # telegram.api absent; control module no-ops without token


def test_collect_jobs_includes_strategy_reload(tmp_path):
    settings = _settings(tmp_path)
    ctx = _ctx(settings, tmp_path)
    manager = ModuleManager(settings)
    manager.setup(ctx)
    ids = {job.job_id for job in manager.collect_jobs(ctx)}
    assert "strategy_plugins_reload" in ids


# --- config validation -------------------------------------------------------

def test_validate_settings_rejects_unknown_module(tmp_path):
    settings = _settings(
        tmp_path,
        modules=ModulesConfig(enabled=["nope.nope"],
                              strategy_plugin_paths=[str(tmp_path / "strategies")]),
    )
    with pytest.raises(ValueError, match="unknown module"):
        validate_settings(settings)


def test_validate_settings_rejects_disabling_locked_module(tmp_path):
    settings = _settings(
        tmp_path,
        modules=ModulesConfig(disabled=["gateway.paper"],
                              strategy_plugin_paths=[str(tmp_path / "strategies")]),
    )
    with pytest.raises(ValueError, match="locked capability"):
        validate_settings(settings)


def test_validate_settings_rejects_duplicate_capability(tmp_path):
    settings = _settings(
        tmp_path,
        modules=ModulesConfig(
            enabled=["market.binance", "market.demo", "gateway.paper"],
            strategy_plugin_paths=[str(tmp_path / "strategies")],
        ),
    )
    with pytest.raises(ValueError, match="both provide"):
        validate_settings(settings)
