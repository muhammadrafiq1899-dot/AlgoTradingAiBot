"""Live strategy catalog: what strategies exist and what parameters they take.

One source of truth for both consumers of this information — the chat prompt
(`algotrading/telegram/chat.py`) and the `/strategies` command
(`algotrading/telegram/ui.py`). Built-in schemas come from
``config/strategies.yaml``; plugin schemas come from the ``STRATEGY_PARAMS``
metadata of each ``strategies/*.py`` file.

Everything is recomputed per call, so a strategy the AI just created is visible
to the next chat turn or command without a restart.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from algotrading.config import load_strategy_definitions
from algotrading.strategy.plugins import get_default_loader
from algotrading.strategy.registry import is_builtin, known_names

log = logging.getLogger(__name__)

BUILTIN = "built-in"
PLUGIN = "plugin"


@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter of a strategy."""

    name: str
    type: str = "value"
    default: Any = None
    min: Any = None
    max: Any = None
    enum: tuple[Any, ...] = ()

    def range_text(self) -> str:
        """Human-readable bounds, or '' when unbounded."""
        if self.min is not None and self.max is not None:
            return f"{self.min}-{self.max}"
        if self.min is not None:
            return f">= {self.min}"
        if self.max is not None:
            return f"<= {self.max}"
        return ""

    def compact(self) -> str:
        """Single-line form for the LLM prompt, e.g. ``period (int, default 14)``."""
        bits = [self.type]
        # Collection defaults (ensemble components) would bloat the prompt.
        if self.default is not None and not isinstance(self.default, (list, dict)):
            bits.append(f"default {self.default!r}")
        bounds = self.range_text()
        if bounds:
            bits.append(f"range {bounds}")
        if self.enum:
            bits.append("one of " + "/".join(str(v) for v in self.enum))
        return f"{self.name} ({', '.join(bits)})"


@dataclass(frozen=True)
class StrategyEntry:
    """A strategy available to build, with its declared parameter schema."""

    name: str
    kind: str = BUILTIN
    description: str = ""
    params: tuple[ParamSpec, ...] = ()

    @property
    def is_builtin(self) -> bool:
        return self.kind == BUILTIN


def _param_from_mapping(raw: Any) -> ParamSpec | None:
    if isinstance(raw, ParamSpec):
        return raw
    if not isinstance(raw, dict) or not raw.get("name"):
        return None
    enum = raw.get("enum")
    return ParamSpec(
        name=str(raw["name"]),
        type=str(raw.get("type") or "value"),
        default=raw.get("default"),
        min=raw.get("min"),
        max=raw.get("max"),
        enum=tuple(enum) if isinstance(enum, (list, tuple)) else (),
    )


def catalog_entries() -> list[StrategyEntry]:
    """Every buildable strategy, built-ins first, then plugins."""
    entries: list[StrategyEntry] = []
    seen: set[str] = set()

    try:
        definitions = load_strategy_definitions()
    except Exception as exc:  # noqa: BLE001 - a bad schema file must not break callers
        log.warning("could not load strategy schemas: %s", exc)
        definitions = []

    for definition in definitions:
        seen.add(definition.name)
        params = tuple(
            spec for spec in (_param_from_mapping(p.dict()) for p in definition.params) if spec
        )
        entries.append(
            StrategyEntry(definition.name, BUILTIN, (definition.description or "").strip(), params)
        )

    loader = get_default_loader()
    for name, _cls in loader.items():
        seen.add(name)
        info = loader.info(name)
        params = tuple(
            spec
            for spec in (
                _param_from_mapping(p) for p in (info.params if info else ())
            )
            if spec
        )
        description = (info.description if info else "") or ""
        entries.append(StrategyEntry(name, PLUGIN, description.strip(), params))

    # Strategies registered without a schema still get listed by name so callers
    # know the name exists.
    for name in known_names():
        if name not in seen:
            entries.append(StrategyEntry(name, BUILTIN if is_builtin(name) else PLUGIN))

    return entries


def default_params(name: str, entries: list[StrategyEntry] | None = None) -> dict[str, Any]:
    """Declared default params for ``name``, from its catalog schema.

    Used as the fallback when a strategy has no version row yet (e.g. a freshly
    loaded plugin): backtesting it should use what it would run with today.
    """
    for entry in entries if entries is not None else catalog_entries():
        if entry.name == name:
            return {p.name: p.default for p in entry.params if p.default is not None}
    return {}
