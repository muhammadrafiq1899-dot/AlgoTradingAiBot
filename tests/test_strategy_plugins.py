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
