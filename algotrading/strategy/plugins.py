"""Filesystem strategy plugins: user- and AI-authored strategies as real files.

A strategy plugin is a ``.py`` file under a plugin directory (default:
``strategies/`` at the repo root) defining a class with an
``evaluate(symbol, candles)`` method plus a module-level ``STRATEGY`` alias.

Keeping strategies on disk (instead of only inside a DB template column) makes
them inspectable, hand-editable, diffable, and version-controllable — which is
what "customisable by AI, editable by the user" needs. The DB still owns
*release* state (version / active / retired); the file owns the *code*.

The loader is the only writer of that directory, and every file it loads or
writes goes through :mod:`algotrading.strategy.validation`, so the AST-safety
rules are identical to the ones the AI proposals are held to.
"""
from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from algotrading.market.base import Candle
from algotrading.strategy.base import Signal
from algotrading.strategy.validation import CodeValidationError, compile_strategy, validate_strategy_code

log = logging.getLogger(__name__)

# Directory scanned by default, relative to the repo root.
DEFAULT_PLUGIN_DIR = "strategies"

_GENERATED_HEADER = "# strategy plugin {name!r} — edit by hand; validated on load.\n"


class StrategyPluginError(ValueError):
    """Raised when a plugin file cannot be loaded or written safely."""


@dataclass(frozen=True)
class StrategyPlugin:
    """Metadata for one loaded strategy plugin file."""

    name: str
    path: Path
    description: str = ""
    params: tuple[dict[str, Any], ...] = ()
    indicator_deps: tuple[str, ...] = ()
    mtime: float = 0.0


class StrategyPluginLoader:
    """Load strategy classes from one or more directories on disk."""

    def __init__(self, paths: Sequence[str | Path] = (DEFAULT_PLUGIN_DIR,)) -> None:
        self._paths = [Path(p) for p in paths] or [Path(DEFAULT_PLUGIN_DIR)]
        self._classes: dict[str, type] = {}
        self._info: dict[str, StrategyPlugin] = {}

    # --- directories ---

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    def _writable_dir(self) -> Path:
        """Directory new plugins are written to (first configured path)."""
        target = self._paths[0]
        target.mkdir(parents=True, exist_ok=True)
        return target

    def path_for(self, name: str) -> Path:
        return self._writable_dir() / f"{name}.py"

    # --- loading ---

    def load_all(self) -> int:
        """Load every ``*.py`` file in every configured directory. Returns count."""
        loaded = 0
        for directory in self._paths:
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.py")):
                if path.name.startswith("_"):
                    continue
                try:
                    if self.load_file(path) is not None:
                        loaded += 1
                except StrategyPluginError as exc:
                    log.warning("skipping strategy plugin %s: %s", path, exc)
        return loaded

    def load_file(self, path: str | Path) -> StrategyPlugin | None:
        """Load one plugin file, replacing any previous version of its name."""
        path = Path(path)
        if not path.exists():
            raise StrategyPluginError(f"plugin file not found: {path}")
        source = path.read_text(encoding="utf-8")
        name = _declared_name(source) or path.stem
        try:
            cls = compile_strategy(source, name)
        except CodeValidationError as exc:
            raise StrategyPluginError(str(exc)) from exc

        namespace = _exec_metadata(source, name)
        params = _normalise_params(namespace.get("STRATEGY_PARAMS"))
        deps = tuple(str(d) for d in (namespace.get("STRATEGY_INDICATORS") or ()))
        description = str(namespace.get("STRATEGY_DESCRIPTION") or (cls.__doc__ or "")).strip()

        self._classes[name] = cls
        self._info[name] = StrategyPlugin(
            name=name,
            path=path,
            description=description,
            params=params,
            indicator_deps=deps,
            mtime=path.stat().st_mtime,
        )
        log.info("loaded strategy plugin '%s' from %s", name, path)
        return self._info[name]

    def reload_changed(self) -> list[str]:
        """Reload plugins whose file mtime changed. Returns reloaded names."""
        reloaded: list[str] = []
        for name, info in list(self._info.items()):
            if not info.path.exists():
                self._forget(name)
                reloaded.append(name)
                continue
            if info.path.stat().st_mtime > info.mtime:
                if self.load_file(info.path) is not None:
                    reloaded.append(name)
        return reloaded

    def _forget(self, name: str) -> None:
        self._classes.pop(name, None)
        self._info.pop(name, None)

    # --- lookup ---

    def get_class(self, name: str) -> type | None:
        return self._classes.get(name)

    def info(self, name: str) -> StrategyPlugin | None:
        return self._info.get(name)

    def names(self) -> list[str]:
        return sorted(self._classes)

    def items(self) -> list[tuple[str, type]]:
        return sorted(self._classes.items())

    def as_dict(self) -> dict[str, type]:
        return dict(self._classes)

    # --- writing ---

    def write_strategy(
        self,
        name: str,
        code: str,
        *,
        description: str = "",
        params: Sequence[dict[str, Any]] | None = None,
        indicator_deps: Sequence[str] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Validate and persist a strategy plugin file, then load it.

        Refuses names that collide with a built-in strategy (those are edited
        through parameter changes, not by shadowing their code). Existing plugin
        files are only replaced when ``overwrite=True``.
        """
        if not name or not name.replace("_", "").isalnum():
            raise StrategyPluginError(f"invalid strategy name: {name!r}")
        if name in _builtin_names():
            raise StrategyPluginError(
                f"{name!r} is a built-in strategy; use a param change or a new name"
            )
        if name in self._classes and not overwrite:
            raise StrategyPluginError(
                f"strategy plugin {name!r} already exists; use edit_strategy to change it"
            )

        source = _compose_source(code, name, description, params, indicator_deps)
        # Validate before touching the filesystem so a bad write can't land.
        try:
            validate_strategy_code(source)
        except CodeValidationError as exc:
            raise StrategyPluginError(str(exc)) from exc

        path = self.path_for(name)
        tmp = path.with_suffix(".py.tmp")
        tmp.write_text(source, encoding="utf-8")
        os.replace(tmp, path)  # atomic: readers never see a half-written file
        info = self.load_file(path)
        if info is None:  # pragma: no cover - load_file raises on failure
            raise StrategyPluginError(f"failed to load written plugin {name!r}")
        return path

    def delete(self, name: str) -> bool:
        """Remove a plugin file (and unregister it). Returns True if it existed."""
        info = self._info.get(name)
        existed = info is not None
        if info is not None and info.path.exists():
            info.path.unlink()
            existed = True
        self._forget(name)
        return existed


# --- helpers -----------------------------------------------------------------

def _builtin_names() -> set[str]:
    from algotrading.strategy.starters import STRATEGIES

    return set(STRATEGIES)


def _declared_name(source: str) -> str | None:
    """Read a top-level ``NAME = "..."`` / ``STRATEGY_NAME = "..."`` if present."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("STRATEGY_NAME", "NAME"):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        return node.value.value
    return None


def _exec_metadata(source: str, name: str) -> dict[str, Any]:
    """Execute validated source to read plugin metadata (params/description)."""
    from algotrading.strategy import indicators as ta
    from algotrading.strategy.validation import safe_namespace

    namespace = safe_namespace({"Signal": Signal, "Candle": Candle, "ta": ta})
    try:
        exec(compile(source, f"<plugin-meta:{name}>", "exec"), namespace)
    except Exception:  # noqa: BLE001 - metadata is best-effort; compile already validated
        return {}
    return namespace


def _normalise_params(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, (list, tuple)):
        return ()
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict) and "name" in item:
            out.append(dict(item))
    return tuple(out)


def _compose_source(
    code: str,
    name: str,
    description: str,
    params: Sequence[dict[str, Any]] | None,
    indicator_deps: Sequence[str] | None,
) -> str:
    """Build the final file text: user code + metadata + a ``STRATEGY`` alias."""
    body = code.strip() + "\n"
    class_names = [
        n.name for n in ast.walk(ast.parse(body)) if isinstance(n, ast.ClassDef)
    ]
    if not class_names:
        raise StrategyPluginError("strategy code must define a class")

    lines = [_GENERATED_HEADER.format(name=name), body, "\n"]
    if "STRATEGY_DESCRIPTION" not in body:
        lines.append(f"STRATEGY_DESCRIPTION = {description!r}\n")
    if "STRATEGY_PARAMS" not in body and params is not None:
        lines.append(f"STRATEGY_PARAMS = {list(params)!r}\n")
    if "STRATEGY_INDICATORS" not in body and indicator_deps is not None:
        lines.append(f"STRATEGY_INDICATORS = {list(indicator_deps)!r}\n")
    if not any(f"STRATEGY = {cls}" in body for cls in class_names):
        lines.append(f"\nSTRATEGY = {class_names[-1]}\n")
    return "".join(lines)


# --- process-wide default loader (used by the strategy registry) -------------

_default_loader = StrategyPluginLoader([DEFAULT_PLUGIN_DIR])


def get_default_loader() -> StrategyPluginLoader:
    return _default_loader


def configure_default_loader(paths: Iterable[str | Path] | None = None) -> StrategyPluginLoader:
    """Point the default loader at ``paths`` and load everything found there."""
    global _default_loader
    _default_loader = StrategyPluginLoader(list(paths) if paths else [DEFAULT_PLUGIN_DIR])
    _default_loader.load_all()
    return _default_loader
