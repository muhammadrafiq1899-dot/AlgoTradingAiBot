"""Shared safety gate for AI/user-authored strategy and indicator code.

Everything that compiles code the bot did not ship with — the strategy plugin
loader, the recommendation applier, and the AI assistant — funnels through this
module. Keeping one implementation means the rules can't drift apart (they had
already been copy-pasted three times before this).

Two layers of defence:

1. ``validate_*_code`` parses the source and rejects dangerous AST nodes
   (``eval``/``exec``/``open``/``__import__``, imports of os/sys/socket/...).
2. ``compile_*`` executes the already-validated source in a namespace whose
   ``__builtins__`` is an explicit allowlist, so even a validator gap can't
   reach the dangerous builtins.

The compiled code never touches trading: strategies are pure functions of a
candle series, and the deterministic execution engine stays separate.
"""
from __future__ import annotations

import ast
import builtins
import math
from typing import Any, Callable

from algotrading.market.base import Candle
from algotrading.strategy.base import Signal

__all__ = [
    "CodeValidationError",
    "validate_strategy_code",
    "validate_indicator_code",
    "compile_strategy",
    "compile_indicator",
    "safe_namespace",
]


class CodeValidationError(ValueError):
    """Raised when generated code is syntactically invalid or unsafe."""


# Functions that execute arbitrary code or touch the world outside the sandbox.
_FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}
# Top-level packages that would let generated code escape the sandbox.
_FORBIDDEN_IMPORTS = {
    "os", "sys", "subprocess", "shutil", "socket", "pickle", "marshal",
    "ctypes", "multiprocessing", "importlib", "pathlib", "requests",
    "urllib", "http", "asyncio", "threading",
}
# Builtins exposed to generated code: pure computation only.
_SAFE_BUILTIN_NAMES = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "int", "isinstance", "len", "list", "map",
    "max", "min", "pow", "range", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "Exception", "ValueError",
    # Required by `class` statements executed via exec().
    "__build_class__",
}


def _safe_builtins() -> dict[str, Any]:
    """Explicit allowlist — never derive this from ``__builtins__`` (its type
    varies between a module and a dict depending on how the file was imported)."""
    return {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}


def safe_namespace(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """A restricted ``globals()`` mapping for executing generated code."""
    namespace: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        # Needed by `class` statements executed through exec().
        "__name__": "algotrading_strategy_plugin",
        "math": math,
    }
    if extra:
        namespace.update(extra)
    return namespace


def _parse(code: str) -> ast.Module:
    if not isinstance(code, str) or not code.strip():
        raise CodeValidationError("code must be a non-empty string")
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        raise CodeValidationError(f"invalid Python syntax: {exc}") from exc


def _check_ast(tree: ast.Module) -> None:
    """Walk the tree and reject dangerous calls/imports (depth-first)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                raise CodeValidationError(f"dangerous function call: {node.func.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise CodeValidationError(f"dunder attribute access is not allowed: {node.attr}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in _FORBIDDEN_IMPORTS:
                    raise CodeValidationError(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _FORBIDDEN_IMPORTS:
                raise CodeValidationError(f"forbidden import from: {node.module}")


def _has_method(tree: ast.Module, method: str) -> bool:
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        if any(
            isinstance(item, ast.FunctionDef) and item.name == method
            for item in cls.body
        ):
            return True
    return False


def validate_strategy_code(code: str) -> ast.Module:
    """Validate a strategy class definition. Returns the parsed tree."""
    tree = _parse(code)
    _check_ast(tree)
    if not any(isinstance(n, ast.ClassDef) for n in ast.walk(tree)):
        raise CodeValidationError("strategy code must define a class")
    if not _has_method(tree, "evaluate"):
        raise CodeValidationError("strategy class must define an 'evaluate' method")
    return tree


def validate_indicator_code(code: str, name: str | None = None) -> ast.Module:
    """Validate a standalone indicator function definition. Returns the tree."""
    tree = _parse(code)
    _check_ast(tree)
    funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not funcs:
        raise CodeValidationError("indicator code must define a function")
    if name is not None and not any(f.name == name for f in funcs):
        raise CodeValidationError(f"indicator code must define a function named {name!r}")
    return tree


def _indicator_module():
    # Imported lazily so this module can be imported by strategy.base consumers
    # without dragging the whole indicator package into a cycle.
    from algotrading.strategy import indicators as ta

    return ta


def compile_strategy(code: str, name: str) -> type:
    """Validate + compile a strategy class. Returns the class, named ``name``.

    The class must expose ``evaluate``; a module-level ``STRATEGY`` alias is
    honoured when present, otherwise the last class defined wins.
    """
    tree = validate_strategy_code(code)
    namespace = safe_namespace({"Signal": Signal, "Candle": Candle, "ta": _indicator_module()})
    try:
        exec(compile(tree, f"<strategy:{name}>", "exec"), namespace)
    except Exception as exc:  # noqa: BLE001 - surface as a validation error
        raise CodeValidationError(f"failed to execute strategy code: {exc}") from exc

    candidate = namespace.get("STRATEGY")
    if not isinstance(candidate, type) or not hasattr(candidate, "evaluate"):
        classes = [obj for obj in namespace.values() if isinstance(obj, type) and hasattr(obj, "evaluate")]
        if not classes:
            raise CodeValidationError("strategy code must define a class with an 'evaluate' method")
        candidate = classes[-1]

    candidate.name = name
    return candidate


def compile_indicator(code: str, name: str) -> Callable:
    """Validate + compile an indicator function. Returns the function."""
    tree = validate_indicator_code(code, name=name)
    namespace = safe_namespace({"NaN": math.nan, "Number": float})
    try:
        exec(compile(tree, f"<indicator:{name}>", "exec"), namespace)
    except Exception as exc:  # noqa: BLE001
        raise CodeValidationError(f"failed to execute indicator code: {exc}") from exc

    func = namespace.get(name)
    if not callable(func):
        raise CodeValidationError(f"indicator code must define a callable named {name!r}")
    func.__name__ = name
    return func
