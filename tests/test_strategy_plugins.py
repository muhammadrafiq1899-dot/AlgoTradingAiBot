"""Strategy plugins: filesystem loading, AI authoring, safety, hot-reload.

These cover the "strategies are customisable by AI and editable by the user"
requirement: the AI proposes code, a human approves, and the approved code
lands as a validated, loadable file on disk while the DB tracks its version.
"""
import json
import os

import pytest

from algotrading.db import get_session_factory, init_db
from algotrading.db.models import Strategy
from algotrading.store.recommendations import RecommendationStore, create_pending_recommendation
from algotrading.store.strategy_versions import latest_version
from algotrading.strategy.plugins import (
    DEFAULT_PLUGIN_DIR,
    StrategyPluginError,
    configure_default_loader,
    get_default_loader,
)
from algotrading.strategy.validation import CodeValidationError, compile_strategy
from algotrading.strategy.registry import (
    UnknownStrategyError,
    build_strategy,
    is_plugin,
    known_names,
)

# A minimal, pure strategy. `Signal` and `Candle` are injected by the loader.
PLUGIN_CODE = '''
class MomentumNudge:
    """Buy when the last close ticks up."""

    def __init__(self, params=None):
        self.params = params or {}
        self.lookback = int(self.params.get("lookback", 2))

    def evaluate(self, symbol, candles):
        if len(candles) < self.lookback + 1:
            return None
        if candles[-1].close > candles[-2].close:
            return Signal(
                strategy_id=0,
                symbol=symbol,
                side="buy",
                ref_price=candles[-1].close,
                rationale="plugin momentum up-tick",
                risk={"position_pct": 0.1},
            )
        return None
'''

EDITED_CODE = PLUGIN_CODE.replace("momentum up-tick", "plugin edited rebound")


@pytest.fixture()
def plugin_dir(tmp_path):
    """Point the process-wide plugin loader at a temp dir, restore afterwards."""
    directory = tmp_path / "strategies"
    configure_default_loader([directory])
    yield directory
    configure_default_loader([DEFAULT_PLUGIN_DIR])


def _candles(prices):
    from algotrading.market.base import Candle

    return [
        Candle(symbol="BTC/USDT", interval="1h", ts=i * 3_600_000,
               open=p, high=p, low=p, close=p, volume=1.0)
        for i, p in enumerate(prices)
    ]


# --- loading -----------------------------------------------------------------

def test_write_then_load_plugin(plugin_dir):
    loader = get_default_loader()
    path = loader.write_strategy(
        "momentum_nudge", PLUGIN_CODE,
        description="test plugin",
        params=[{"name": "lookback", "type": "int", "default": 2}],
    )
    assert path.exists()
    assert "momentum_nudge" in loader.names()
    assert is_plugin("momentum_nudge")
    assert "momentum_nudge" in known_names()

    strat = build_strategy("momentum_nudge", {"lookback": 2})
    assert strat.evaluate("BTC/USDT", _candles([100.0, 101.0, 102.0])) is not None
    assert strat.evaluate("BTC/USDT", _candles([100.0])) is None


def test_written_file_is_readable_python(plugin_dir):
    loader = get_default_loader()
    path = loader.write_strategy("momentum_nudge", PLUGIN_CODE, params=[])
    source = path.read_text(encoding="utf-8")
    # The loader appends metadata so the file is self-describing.
    assert "STRATEGY = " in source
    compile(source, str(path), "exec")  # must be valid Python


def test_plugin_cannot_shadow_builtin(plugin_dir):
    with pytest.raises(StrategyPluginError, match="built-in"):
        get_default_loader().write_strategy("ema_crossover", PLUGIN_CODE)


def test_unsafe_plugin_is_rejected(plugin_dir):
    loader = get_default_loader()
    with pytest.raises(StrategyPluginError):
        loader.write_strategy("evil", "import os\n\nclass Evil:\n    def evaluate(self, s, c):\n        return os.getcwd()\n")
    with pytest.raises(StrategyPluginError):
        loader.write_strategy("evil2", "class Evil2:\n    def evaluate(self, s, c):\n        return eval('1')\n")
    # Nothing was written for either attempt.
    assert "evil" not in loader.names()
    assert "evil2" not in loader.names()


def test_plugin_missing_evaluate_is_rejected(plugin_dir):
    with pytest.raises(StrategyPluginError):
        get_default_loader().write_strategy(
            "no_eval", "class NoEval:\n    pass\n"
        )


def test_unknown_plugin_strategy_raises(plugin_dir):
    with pytest.raises(UnknownStrategyError):
        build_strategy("not_a_real_strategy", {})


def test_reload_changed_picks_up_edits(plugin_dir):
    loader = get_default_loader()
    path = loader.write_strategy("momentum_nudge", PLUGIN_CODE, params=[])
    assert loader.reload_changed() == []

    # A later mtime simulates a hand-edit on disk.
    future = path.stat().st_mtime + 10
    os.utime(path, (future, future))
    assert "momentum_nudge" in loader.reload_changed()


# --- AI authoring flow -------------------------------------------------------

@pytest.fixture()
def session(tmp_path, plugin_dir):
    db = str(tmp_path / "plugins.db")
    init_db(db)
    factory = get_session_factory(db)
    sess = factory()
    yield sess
    sess.close()


def _new_strategy_rec(session, name, code, params=None):
    return create_pending_recommendation(
        session,
        kind="new_strategy",
        strategy_name=name,
        params=params or {"lookback": 3},
        rationale="test creation",
        template=code,
        indicator_deps=[],
        param_schema=[{"name": "lookback", "type": "int", "default": 3}],
    )


def test_approved_new_strategy_becomes_a_loaded_plugin(session, plugin_dir):
    rec = _new_strategy_rec(session, "momentum_nudge", PLUGIN_CODE)
    strategy, result = RecommendationStore(session).apply(rec, candles=[], do_backtest=False)

    assert strategy is not None
    assert strategy.status == "active"
    assert (plugin_dir / "momentum_nudge.py").exists()
    assert is_plugin("momentum_nudge")
    assert result is None


def test_approved_edit_rewrites_code_and_cuts_a_version(session, plugin_dir):
    # Create, approve, then edit through the same human-approval path.
    RecommendationStore(session).apply(
        _new_strategy_rec(session, "momentum_nudge", PLUGIN_CODE),
        candles=[], do_backtest=False,
    )
    edit = create_pending_recommendation(
        session,
        kind="edit_strategy",
        strategy_name="momentum_nudge",
        params={"lookback": 2},
        rationale="test edit",
        template=EDITED_CODE,
    )
    strategy, _ = RecommendationStore(session).apply(edit, candles=[], do_backtest=False)

    assert strategy is not None
    assert latest_version(session, "momentum_nudge").version == 2
    source = (plugin_dir / "momentum_nudge.py").read_text(encoding="utf-8")
    assert "plugin edited rebound" in source


def test_new_strategy_cannot_reuse_an_existing_name(session, plugin_dir):
    rec = _new_strategy_rec(session, "momentum_nudge", PLUGIN_CODE)
    RecommendationStore(session).apply(rec, candles=[], do_backtest=False)

    with pytest.raises(ValueError, match="already exists"):
        _new_strategy_rec(session, "momentum_nudge", PLUGIN_CODE)


def test_edit_strategy_rejects_builtins(session):
    with pytest.raises(ValueError, match="plugin strategy"):
        create_pending_recommendation(
            session,
            kind="edit_strategy",
            strategy_name="ema_crossover",
            params={},
            rationale="should not work",
            template=PLUGIN_CODE,
        )


def test_allowed_names_grow_with_plugins(session, plugin_dir):
    from algotrading.store.recommendations import allowed_strategy_names

    before = allowed_strategy_names()
    assert "momentum_nudge" not in before
    get_default_loader().write_strategy("momentum_nudge", PLUGIN_CODE, params=[])
    assert "momentum_nudge" in allowed_strategy_names()


def test_plugin_params_persisted_in_version_row(session, plugin_dir):
    RecommendationStore(session).apply(
        _new_strategy_rec(session, "momentum_nudge", PLUGIN_CODE, params={"lookback": 9}),
        candles=[], do_backtest=False,
    )
    row = session.query(Strategy).filter(Strategy.name == "momentum_nudge").one()
    assert json.loads(row.params)["lookback"] == 9


# --- ensemble components -----------------------------------------------------
# An AI-authored strategy is a plugin, so the ensemble must resolve components
# through the registry: before this, an ensemble built from image-derived
# strategies failed at approval time with "unknown strategy: <plugin>".

def test_ensemble_accepts_a_plugin_component(plugin_dir):
    get_default_loader().write_strategy("momentum_nudge", PLUGIN_CODE, params=[])
    ensemble = build_strategy("ensemble", {
        "mode": "consensus",
        "components": [
            {"name": "momentum_nudge", "params": {"lookback": 2}},
            {"name": "ema_crossover", "params": {"fast_period": 9, "slow_period": 21}},
        ],
    })

    # Steady climb (no fresh EMA cross) then one up-tick: only the plugin fires.
    candles = _candles([100.0 + i for i in range(30)] + [129.5, 130.0])
    signal = ensemble.evaluate("BTC/USDT", candles)

    assert signal is not None, "the plugin component's signal must reach the ensemble"
    assert signal.side == "buy"
    assert "plugin momentum up-tick" in signal.rationale


def test_ensemble_rejects_an_unknown_component(plugin_dir):
    with pytest.raises(UnknownStrategyError):
        build_strategy("ensemble", {
            "components": [
                {"name": "does_not_exist", "params": {}},
                {"name": "ema_crossover",
                 "params": {"fast_period": 9, "slow_period": 21}},
            ],
        })


# --- imports inside authoring code -------------------------------------------
# `strategies/README.md` documents `from algotrading.strategy import indicators
# as ta`, so that (and the bare `indicators` form models write) has to execute.
# Before this, the sandbox had no __import__ at all and approval died with
# "failed to execute strategy code: __import__ not found".

IMPORTING_CODE = '''
from algotrading.strategy import indicators as ta


class ImportingEma:
    """Buy when the close sits above its own SMA."""

    def __init__(self, params=None):
        self.params = params or {}
        self.period = int(self.params.get("period", 3))

    def evaluate(self, symbol, candles):
        closes = [c.close for c in candles]
        avg = ta.sma(closes, self.period)[-1]
        if avg is not None and closes[-1] > avg:
            return Signal(
                strategy_id=0, symbol=symbol, side="buy", ref_price=closes[-1],
                rationale="close above SMA", risk={"position_pct": 0.1},
            )
        return None
'''


def test_documented_indicator_import_compiles_and_runs():
    cls = compile_strategy(IMPORTING_CODE, "importing_ema")
    strategy = cls({"period": 3})
    signal = strategy.evaluate("BTC/USDT", _candles([100.0, 100.0, 101.0]))
    assert signal is not None and signal.side == "buy"


def test_bare_indicators_import_is_aliased():
    code = IMPORTING_CODE.replace(
        "from algotrading.strategy import indicators as ta", "from indicators import sma"
    ).replace("ta.sma(closes, self.period)", "sma(closes, self.period)")
    cls = compile_strategy(code, "importing_ema_bare")
    assert cls({"period": 3}).evaluate("BTC/USDT", _candles([100.0, 100.0, 101.0])) is not None


def test_other_imports_are_rejected_with_a_hint():
    code = IMPORTING_CODE.replace(
        "from algotrading.strategy import indicators as ta", "import numpy"
    )
    with pytest.raises(CodeValidationError, match="not available to strategy code"):
        compile_strategy(code, "importing_numpy")


def test_package_import_cannot_reach_siblings():
    code = IMPORTING_CODE.replace(
        "from algotrading.strategy import indicators as ta",
        "from algotrading.strategy import registry as ta",
    )
    with pytest.raises(CodeValidationError, match="not available to strategy code"):
        compile_strategy(code, "importing_sibling")


def test_forbidden_import_still_reports_forbidden():
    code = IMPORTING_CODE.replace(
        "from algotrading.strategy import indicators as ta", "import os"
    )
    with pytest.raises(CodeValidationError, match="forbidden import"):
        compile_strategy(code, "importing_os")


def test_runtime_import_hook_is_the_second_layer():
    """Even with the AST check bypassed, the hook refuses."""
    from algotrading.strategy.validation import safe_namespace

    builtins_map = safe_namespace()["__builtins__"]
    with pytest.raises(ImportError):
        builtins_map["__import__"]("os")
    assert builtins_map["__import__"]("math") is not None


def test_new_strategy_proposal_rejects_uncompilable_code(session, plugin_dir):
    """Bad code must fail at propose time, while the model can still fix it."""
    with pytest.raises(ValueError, match="template rejected"):
        _new_strategy_rec(session, "broken_import", "import numpy\n\nclass X:\n    def evaluate(self, s, c):\n        return None\n")

    with pytest.raises(ValueError, match="template rejected"):
        _new_strategy_rec(session, "no_evaluate", "class X:\n    pass\n")


# --- constructor shape -------------------------------------------------------
# The engine builds strategies as cls(params_dict). A keyword-style constructor
# used to load fine and then blow up on every build (backtest, engine, ensemble).

KEYWORD_CTOR_CODE = '''
class KeywordCtor:
    """Would explode as cls(params_dict)."""

    def __init__(self, period=14):
        self.period = int(period)

    def evaluate(self, symbol, candles):
        return None
'''


def test_keyword_constructor_is_rejected_with_a_fix_hint():
    with pytest.raises(CodeValidationError, match=r"def __init__\(self, params=None\)"):
        compile_strategy(KEYWORD_CTOR_CODE, "keyword_ctor")


def test_class_without_a_constructor_is_rejected():
    with pytest.raises(CodeValidationError, match="params_dict"):
        compile_strategy("class Bare:\n    def evaluate(self, s, c):\n        return None\n",
                         "bare")


def test_params_dict_constructor_passes_the_probe():
    """A constructor that reads dict keys (KeyError on {}) is a valid shape."""
    code = '''
class ReadsDict:
    """Reads params with a default."""

    def __init__(self, params=None):
        self.period = int((params or {}).get("period", 5))

    def evaluate(self, symbol, candles):
        return None
'''
    assert compile_strategy(code, "reads_dict") is not None


def test_propose_time_gate_rejects_a_keyword_constructor(session, plugin_dir):
    with pytest.raises(ValueError, match="template rejected"):
        _new_strategy_rec(session, "keyword_ctor", KEYWORD_CTOR_CODE)


def test_failed_write_does_not_poison_the_name(session, plugin_dir, monkeypatch):
    """A write that fails to load must leave no file behind.

    Without the cleanup, the name stays taken forever ("already exists") even
    after the author fixes the code — exactly what happened when a generated
    strategy failed at load time.
    """
    from algotrading.store.strategy_versions import create_new_strategy

    loader = get_default_loader()

    def failing_load(path):
        raise StrategyPluginError("simulated load failure")

    monkeypatch.setattr(loader, "load_file", failing_load)
    with pytest.raises(ValueError, match="cannot create strategy"):
        create_new_strategy(session, "broken_load", PLUGIN_CODE, params={},
                            param_schema=[], indicator_deps=[])

    assert not (plugin_dir / "broken_load.py").exists(), "the half-made file must go"

    # The name is free again once the code is fixable.
    monkeypatch.undo()
    create_new_strategy(session, "broken_load", PLUGIN_CODE, params={},
                        param_schema=[], indicator_deps=[])
    assert loader.get_class("broken_load") is not None
