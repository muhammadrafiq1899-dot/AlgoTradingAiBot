"""Plug-and-play module system.

See :mod:`algotrading.modules.base` for the contract and
:mod:`algotrading.modules.registry` for how modules are selected from config.
Importing this package registers the builtin modules as a side effect.
"""
from algotrading.modules.base import (
    BUILD_ORDER,
    CAPABILITY_ANALYTICS,
    CAPABILITY_API,
    CAPABILITY_CONTROL,
    CAPABILITY_EXECUTION,
    CAPABILITY_MARKET,
    CAPABILITY_SCHEDULER_CONTROL,
    CAPABILITY_STRATEGY,
    LOCKED_CAPABILITIES,
    Module,
    ModuleError,
    ModuleSpec,
    ServiceBag,
)
from algotrading.modules.manager import ModuleManager, build_manager
from algotrading.modules.registry import (
    all_modules,
    get_module,
    module_names,
    register_module,
    resolve_modules,
)

# Register the builtin modules (side effect of the import).
from algotrading.modules import builtin  # noqa: E402,F401

__all__ = [
    "BUILD_ORDER",
    "CAPABILITY_ANALYTICS",
    "CAPABILITY_API",
    "CAPABILITY_CONTROL",
    "CAPABILITY_EXECUTION",
    "CAPABILITY_MARKET",
    "CAPABILITY_SCHEDULER_CONTROL",
    "CAPABILITY_STRATEGY",
    "LOCKED_CAPABILITIES",
    "Module",
    "ModuleError",
    "ModuleManager",
    "ModuleSpec",
    "ServiceBag",
    "all_modules",
    "build_manager",
    "get_module",
    "module_names",
    "register_module",
    "resolve_modules",
]
