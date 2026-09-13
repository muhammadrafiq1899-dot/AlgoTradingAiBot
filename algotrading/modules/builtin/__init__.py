"""Builtin modules shipped with the bot.

Importing this package registers every builtin module with the registry (each
module calls ``@register_module``). Add a new builtin by dropping a file here
that defines a ``Module`` subclass with a ``ModuleSpec`` and importing it below.
"""
from algotrading.modules.builtin import (  # noqa: F401
    analytics_jobs,
    execution_gateway,
    http_api,
    market_provider,
    strategy_source,
    telegram_control,
)
