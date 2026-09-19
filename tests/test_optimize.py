"""Parameter search: grid caps, ranking, artifacts, proposals and spawning.

The safety tests matter as much as the maths ones: this package must be able to
RANK parameters and MUST NOT be able to apply them.
"""
import json
import pathlib
import subprocess

import pytest
from sqlalchemy import select

from algotrading.backtest.runner import run_backtest
from algotrading.config import load_settings
from algotrading.db import get_session_factory, init_db
from algotrading.db.models import AIRecommendation, Strategy
from algotrading.market.base import Candle
from algotrading.market.candles import CandleStore
from algotrading.optimize import proposal as proposal_mod
from algotrading.optimize import runner as runner_mod
from algotrading.optimize import search as search_mod
from algotrading.optimize import spawn as spawn_mod
from algotrading.optimize.proposal import propose_from_result
from algotrading.optimize.search import (
    Candidate,
    SearchResult,
    default_param_space,
    generate_grid,
    objective_value,
    run_search,
)
from algotrading.optimize.spawn import (
    SearchBusyError,
    acquire_lock,
    build_command,
    release_lock,
    start_search_subprocess,
    wait_search,
)

HOUR_MS = 3_600_000


# --- fixtures / helpers -------------------------------------------------------


def _settings(tmp_path, **optimize_overrides):
    """A deep copy of the real settings pointed at a tmp dir and tmp DB."""
    base = load_settings()
    settings = base.model_copy(deep=True) if hasattr(base, "model_copy") else base.copy(deep=True)
    settings.optimize.results_dir = str(tmp_path / "optimize")
    settings.optimize.max_combinations = optimize_overrides.pop("max_combinations", 4)
    settings.optimize.min_trades = optimize_overrides.pop("min_trades", 1)
    settings.optimize.timeout_seconds = optimize_overrides.pop("timeout_seconds", 60)
    settings.optimize.objective = optimize_overrides.pop("objective", "sharpe")
    settings.backtest.walk_forward_folds = optimize_overrides.pop("walk_forward_folds", 2)
    settings.backtest.max_candles = optimize_overrides.pop("max_candles", 200)
    settings.db_path = str(tmp_path / "opt.db")
    assert not optimize_overrides, f"unknown overrides {optimize_overrides}"
    return settings


def _candles(n=180, interval="1h"):
    """A deterministic wave that gives a trend strategy something to trade."""
    out = []
    price = 100.0
    for i in range(n):
        price *= 1.004 if (i // 20) % 2 == 0 else 0.996
        out.append(
            Candle(
                symbol="BTC/USDT", interval=interval,
                ts=1_600_000_000_000 + i * HOUR_MS,
                open=price, high=price * 1.002, low=price * 0.998,
                close=price, volume=1.0,
            )
        )
    return out


def _candidate(params, value, **kwargs):
    return Candidate(
        params=dict(params),
        ok=True,
        objective_value=value,
        metrics=kwargs.pop("metrics", {"n_trades": 12, "worst_fold_drawdown_pct": 5.0}),
        consistency=kwargs.pop("consistency", {"folds": 2, "profitable_folds": 2,
                                               "verdict": "most folds profitable"}),
    )


def _result(**overrides):
    defaults = dict(
        strategy_name="ema_crossover",
        symbol="BTC/USDT",
        interval="1h",
        objective="sharpe",
        seed=42,
        n_bars=400,
        min_trades=10,
        folds=2,
        n_candidates=3,
        n_evaluated=3,
        duration_seconds=1.5,
        ranked=[
            _candidate({"fast_period": 9, "slow_period": 21}, 1.75),
            _candidate({"fast_period": 5, "slow_period": 21}, 0.5),
        ],
        rejected=[
            Candidate(params={"fast_period": 2, "slow_period": 200}, ok=False,
                      reason="min_trades: 3 closed trades < 10 across 2 folds")
        ],
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


# --- grid generation ----------------------------------------------------------


def test_generate_grid_returns_the_full_product_when_it_fits():
    space = {"a": [1, 2, 3], "b": [10, 20]}
    grid = generate_grid(space, max_combinations=100, seed=1)
    assert len(grid) == 6
    assert {tuple(sorted(c.items())) for c in grid} == {
        (("a", 1), ("b", 10)), (("a", 1), ("b", 20)),
        (("a", 2), ("b", 10)), (("a", 2), ("b", 20)),
        (("a", 3), ("b", 10)), (("a", 3), ("b", 20)),
    }


def test_generate_grid_caps_and_is_seeded():
    space = {"a": list(range(10)), "b": list(range(10)), "c": list(range(10))}
    capped = generate_grid(space, max_combinations=17, seed=42)
    assert len(capped) == 17, "never exceed the cap"
    assert len({json.dumps(c, sort_keys=True) for c in capped}) == 17, "no duplicates"
    assert capped == generate_grid(space, max_combinations=17, seed=42)
    assert capped != generate_grid(space, max_combinations=17, seed=43)
    # Every candidate is a real point of the declared space.
    for combo in capped:
        assert all(value in space[key] for key, value in combo.items())


def test_generate_grid_rejects_nonsense_arguments():
    with pytest.raises(ValueError):
        generate_grid({}, 10, 1)
    with pytest.raises(ValueError):
        generate_grid({"a": [1]}, 0, 1)
    with pytest.raises(ValueError):
        generate_grid({"a": []}, 10, 1)


def test_generate_grid_tolerates_unhashable_values():
    """Ensemble components are dicts: dedupe must not require hashability."""
    one = [{"name": "ema_crossover", "params": {}}]
    two = [{"name": "rsi_mean_reversion", "params": {}}]
    grid = generate_grid({"components": [one, two, one]}, max_combinations=10, seed=1)
    assert len(grid) == 2


def test_default_param_space_stays_inside_the_declared_schema():
    space = default_param_space("ema_crossover")
    assert set(space) == {"fast_period", "slow_period", "position_pct"}
    assert 12 in space["fast_period"], "the shipped default must be searched"
    assert all(2 <= v <= 200 for v in space["fast_period"])
    assert all(5 <= v <= 400 for v in space["slow_period"])
    assert all(0.01 <= v <= 0.5 for v in space["position_pct"])
    assert len(space["fast_period"]) <= search_mod.MAX_VALUES_PER_PARAM


def test_default_param_space_rejects_an_unknown_strategy():
    with pytest.raises(ValueError):
        default_param_space("not_a_strategy")


# --- objective + ranking ------------------------------------------------------


def test_objective_value_implements_the_three_objectives():
    metrics = {"sharpe": 1.5, "total_pnl": 250.0, "worst_fold_drawdown": 50.0}
    assert objective_value("sharpe", metrics) == pytest.approx(1.5)
    assert objective_value("total_pnl", metrics) == pytest.approx(250.0)
    # PnL per unit of the WORST fold drawdown.
    assert objective_value("pnl_drawdown", metrics) == pytest.approx(5.0)
    # No drawdown -> no score, rather than an infinite one.
    assert objective_value("pnl_drawdown", {"total_pnl": 10.0, "worst_fold_drawdown": 0.0}) is None
    assert objective_value("sharpe", {"sharpe": None}) is None
    with pytest.raises(ValueError):
        objective_value("magic", metrics)


def test_run_search_ranks_by_the_objective_and_keeps_rejections():
    result = run_search(
        _candles(), "ema_crossover",
        {"fast_period": [3, 5, 9], "slow_period": [12, 21]},
        "sharpe",
        max_combinations=3, min_trades=1, folds=2, seed=7,
        settings=_settings_for_search(),
    )
    assert result.n_candidates == 3
    assert result.n_evaluated == 3
    assert result.objective == "sharpe"
    values = [c.objective_value for c in result.ranked]
    assert values == sorted(values, reverse=True), "ranked best first"
    assert all(c.consistency for c in result.ranked), "consistency travels with the score"


def _settings_for_search():
    """Settings for an in-memory search: no artifacts, tiny budgets."""
    settings = load_settings().model_copy(deep=True)
    settings.backtest.walk_forward_folds = 2
    settings.optimize.min_trades = 1
    settings.optimize.max_combinations = 3
    return settings


def test_run_search_caps_candidates_from_settings():
    settings = _settings_for_search()
    settings.optimize.max_combinations = 2
    result = run_search(
        _candles(), "ema_crossover",
        {"fast_period": [3, 5, 9], "slow_period": [12, 21]},
        settings=settings,
    )
    assert result.n_candidates == 2
    assert sum(len(v) for v in result.param_space.values()) > 2


def test_run_search_rejects_a_candidate_below_min_trades():
    result = run_search(
        _candles(), "ema_crossover",
        {"fast_period": [3], "slow_period": [12]},
        "total_pnl", max_combinations=1, min_trades=10_000, folds=2,
        settings=_settings_for_search(),
    )
    assert result.ranked == []
    assert result.n_rejected == 1
    assert "min_trades" in result.rejected[0].reason
    assert "best=none" in result.summary_line()


def test_run_search_reports_unbuildable_params_as_rejections():
    result = run_search(
        _candles(), "ema_crossover",
        {"fast_period": [20], "slow_period": [5]},  # fast >= slow: refused
        "sharpe", max_combinations=1, min_trades=0, folds=2,
        settings=_settings_for_search(),
    )
    assert result.ranked == []
    assert "unbuildable" in result.rejected[0].reason


def test_run_search_marks_unevaluated_candidates_on_deadline():
    import time

    result = run_search(
        _candles(), "ema_crossover",
        {"fast_period": [3, 5, 9], "slow_period": [12, 21]},
        "sharpe", max_combinations=4, min_trades=1, folds=2,
        settings=_settings_for_search(), deadline=time.monotonic() - 1,
    )
    assert result.timed_out is True
    assert result.n_evaluated == 0
    assert all("timeout" in c.reason for c in result.rejected)


def test_run_search_uses_the_configured_objective_and_folds():
    settings = _settings_for_search()
    settings.optimize.objective = "total_pnl"
    result = run_search(
        _candles(), "ema_crossover", {"fast_period": [3], "slow_period": [12]},
        max_combinations=1, min_trades=1, settings=settings,
    )
    assert result.objective == "total_pnl"
    assert result.folds == settings.backtest.walk_forward_folds


# --- artifact round-trip ------------------------------------------------------


def test_search_result_json_round_trip():
    result = _result()
    payload = json.loads(json.dumps(result.to_dict()))  # must be JSON-safe
    restored = SearchResult.from_dict(payload)
    assert restored.best().params == result.best().params
    assert restored.best().objective_value == pytest.approx(result.best().objective_value)
    assert restored.n_rejected == result.n_rejected
    assert "min_trades" in restored.rejected[0].reason
    assert restored.param_space == result.param_space
    assert restored.to_dict()["ranked"][0]["rank"] == 1


def test_runner_writes_a_json_artifact_from_stored_candles(tmp_path):
    settings = _settings(tmp_path, max_combinations=3)
    init_db(settings.db_path)
    with get_session_factory(settings.db_path)() as session:
        CandleStore(session).upsert(_candles())

    code, artifact, path = runner_mod.run(
        strategy="ema_crossover",
        symbol="BTC/USDT",
        interval="1h",
        grid={"fast_period": [3, 5], "slow_period": [12, 21]},
        out=tmp_path / "artifact.json",
        timeout=60,
        settings=settings,
        quiet=True,
    )
    assert code == runner_mod.EXIT_OK
    assert path == tmp_path / "artifact.json" and path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["source"] == "algotrading.optimize.runner"
    # 2 x 2 = 4 combinations, capped at settings.optimize.max_combinations = 3.
    assert on_disk["search"]["n_candidates"] == 3
    assert on_disk["search"]["strategy_name"] == "ema_crossover"
    assert on_disk == artifact
    restored = SearchResult.from_dict(on_disk["search"])
    assert restored.n_candidates == 3


def test_runner_uses_the_auto_param_space_and_reports_missing_data(tmp_path):
    settings = _settings(tmp_path, max_combinations=1)
    # No database at all: the child must exit with a code, not a traceback.
    code, artifact, path = runner_mod.run(
        strategy="ema_crossover", settings=settings, quiet=True
    )
    assert code == runner_mod.EXIT_NO_DATA
    assert artifact is None and path is None

    init_db(settings.db_path)
    code, artifact, _ = runner_mod.run(
        strategy="ema_crossover", settings=settings, quiet=True
    )
    assert code == runner_mod.EXIT_NO_DATA, "an empty candles table is not data"

    with get_session_factory(settings.db_path)() as session:
        CandleStore(session).upsert(_candles())
    code, artifact, path = runner_mod.run(
        strategy="ema_crossover", settings=settings, out=tmp_path / "auto.json",
        quiet=True,
    )
    assert code == runner_mod.EXIT_OK
    assert artifact["search"]["n_candidates"] == 1
    assert path.name == "auto.json"


def test_runner_refuses_when_the_optimizer_is_disabled(tmp_path):
    settings = _settings(tmp_path)
    settings.optimize.enabled = False
    code, artifact, path = runner_mod.run(settings=settings, quiet=True)
    assert code == runner_mod.EXIT_USAGE
    assert artifact is None


def test_runner_defaults_its_artifact_into_the_results_dir(tmp_path):
    settings = _settings(tmp_path, max_combinations=1)
    init_db(settings.db_path)
    with get_session_factory(settings.db_path)() as session:
        CandleStore(session).upsert(_candles())
    code, _artifact, path = runner_mod.run(settings=settings, quiet=True)
    assert code == runner_mod.EXIT_OK
    assert path.parent == tmp_path / "optimize"
    assert path.name.startswith("search_ema_crossover_BTC-USDT_1h_")


# --- proposal: PENDING, never applied ----------------------------------------


def test_propose_creates_a_pending_recommendation_and_does_not_apply_it(tmp_path):
    db_path = str(tmp_path / "proposal.db")
    init_db(db_path)
    session = get_session_factory(db_path)()
    try:
        session.add(
            Strategy(name="ema_crossover", version=1, status="active",
                     params=json.dumps({"fast_period": 12, "slow_period": 26}))
        )
        session.commit()
        rec = propose_from_result(session, _result())

        assert rec.status == "pending"
        assert rec.kind == "param_change"
        assert rec.strategy_name == "ema_crossover"
        assert "sharpe=1.7500" in rec.rationale
        assert "Pending human approval" in rec.rationale

        content = json.loads(rec.content_json)
        assert content["params"] == {"fast_period": 9, "slow_period": 21}

        evidence = json.loads(rec.backtest_json)
        assert evidence["objective"] == "sharpe"
        assert evidence["best"]["params"] == {"fast_period": 9, "slow_period": 21}
        assert evidence["top"][0]["rank"] == 1
        assert evidence["n_candidates"] == 3
        assert evidence["rejected_sample"], "rejections are part of the evidence"
        assert evidence["invariants"]

        # NOT applied: the active version is untouched and no new version exists.
        versions = session.execute(select(Strategy)).scalars().all()
        assert len(versions) == 1
        assert versions[0].status == "active"
        assert json.loads(versions[0].params)["fast_period"] == 12
        assert session.get(AIRecommendation, rec.id).status == "pending"
        assert rec.reviewed_at is None
    finally:
        session.close()


def test_propose_refuses_an_empty_search(tmp_path):
    db_path = str(tmp_path / "empty.db")
    init_db(db_path)
    session = get_session_factory(db_path)()
    try:
        result = _result(ranked=[], rejected=[
            Candidate(params={"fast_period": 20, "slow_period": 5}, ok=False,
                      reason="unbuildable")
        ])
        with pytest.raises(ValueError):
            propose_from_result(session, result)
        assert session.execute(select(AIRecommendation)).scalars().all() == []
    finally:
        session.close()


def test_propose_writes_no_settings_and_imports_no_apply_path():
    """The invariant, checked against the source rather than promised in prose."""
    source = _source(proposal_mod)
    imports = [
        line.strip() for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    forbidden = ("strategy_versions", "promote_to_active", "RecommendationStore")
    joined = " ".join(imports)
    assert not any(name in joined for name in forbidden), imports
    assert "apply(" not in source.replace("RecommendationStore.apply", "")


# --- spawn: separate process, nice, timeout, lock -----------------------------


class _FakeProc:
    """Stands in for a child process; `hang` makes it ignore its timeout."""

    hang = False

    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.pid = 424242
        self.killed = False

    def wait(self, timeout=None):
        if _FakeProc.hang:
            raise subprocess.TimeoutExpired(cmd=self.command, timeout=timeout)
        return 0

    def kill(self):
        self.killed = True


@pytest.fixture()
def fake_popen(monkeypatch):
    _FakeProc.hang = False
    monkeypatch.setattr(spawn_mod.subprocess, "Popen", _FakeProc)
    return _FakeProc


def test_build_command_is_niced_and_runs_the_child_module(tmp_path):
    settings = _settings(tmp_path)
    settings.optimize.nice = 15
    settings.optimize.trainer_python = "/usr/bin/python3.14"
    command = build_command(strategy="ema_crossover", symbol="BTC/USDT",
                            interval="1h", combinations=25, settings=settings)
    assert command[0] == "nice" and "-n" in command and "15" in command
    assert command[command.index("-m") + 1] == "algotrading.optimize.runner"
    assert command[0 if command[0] != "nice" else 3] == "/usr/bin/python3.14"
    assert command[command.index("--strategy") + 1] == "ema_crossover"
    assert command[command.index("--symbol") + 1] == "BTC/USDT"
    assert command[command.index("--interval") + 1] == "1h"
    assert command[command.index("--combinations") + 1] == "25"


def test_start_search_subprocess_writes_the_lock_and_refuses_a_second_search(
    tmp_path, fake_popen
):
    settings = _settings(tmp_path)
    handle = start_search_subprocess(strategy="ema_crossover", settings=settings)
    try:
        assert handle.out_path.parent == tmp_path / "optimize"
        assert handle.lock_path.exists()
        payload = json.loads(handle.lock_path.read_text())
        assert payload["pid"] > 0
        assert handle.proc.command[:2] == ["nice", "-n"] or handle.proc.command[0].endswith("python")
        # A second search while the first is running is refused, not queued.
        with pytest.raises(SearchBusyError):
            start_search_subprocess(strategy="ema_crossover", settings=settings)
    finally:
        release_lock(handle.lock_path)


def test_stale_lock_is_replaced(tmp_path):
    settings = _settings(tmp_path)
    lock = spawn_mod.lock_path(settings)
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": 999_999_999, "started_at": 0}), encoding="utf-8")
    acquired = acquire_lock(settings)
    assert acquired == lock
    assert json.loads(lock.read_text())["pid"] != 999_999_999
    release_lock(lock)
    assert not lock.exists()


def test_wait_search_kills_at_the_timeout_and_releases_the_lock(tmp_path, fake_popen):
    settings = _settings(tmp_path)
    settings.optimize.timeout_seconds = 1
    fp = fake_popen
    handle = start_search_subprocess(strategy="ema_crossover", settings=settings)
    assert handle.lock_path.exists()
    fp.hang = True
    code = wait_search(handle, settings=settings)
    assert code == -9, "a killed search reports -9, never success"
    assert handle.proc.killed is True
    assert not handle.lock_path.exists(), "the lock must not survive a killed search"


def test_wait_search_reports_the_exit_code_on_success(tmp_path, fake_popen):
    settings = _settings(tmp_path)
    handle = start_search_subprocess(strategy="ema_crossover", settings=settings)
    assert wait_search(handle, settings=settings) == 0
    assert not handle.lock_path.exists()


def test_spawn_uses_an_absolute_out_path_inside_the_results_dir(tmp_path, fake_popen):
    settings = _settings(tmp_path)
    handle = start_search_subprocess(
        strategy="ema_crossover", out="relative.json", settings=settings
    )
    try:
        assert handle.out_path == tmp_path / "optimize" / "relative.json"
    finally:
        release_lock(handle.lock_path)


# --- ordered-module hygiene ---------------------------------------------------


def _source(module):
    with open(module.__file__, "r", encoding="utf-8") as handle:
        return handle.read()


@pytest.mark.parametrize(
    "module", [search_mod, runner_mod, spawn_mod, proposal_mod],
)
def test_optimize_modules_never_import_bot_side_effects(module):
    imports = [
        line.strip() for line in _source(module).splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    joined = "\n".join(imports)
    for forbidden in ("algotrading.telegram", "algotrading.scheduler",
                      "algotrading.api", "algotrading.execution"):
        assert forbidden not in joined, f"{module.__name__} must not import {forbidden}"
    if module is not spawn_mod:
        assert "subprocess" not in joined, "only spawn may start a process"


def test_search_never_places_orders_and_stays_pure():
    source = _source(search_mod)
    for forbidden in ("place_order", "create_order", "ExecutionEngine", "session.add"):
        assert forbidden not in source


def test_scripts_optimize_propose_writes_a_pending_row(tmp_path):
    """The CLI's --propose path: artifact in, PENDING row out, nothing applied."""
    import importlib.util

    settings = _settings(tmp_path)
    settings.optimize.results_dir = str(tmp_path / "optimize")
    init_db(settings.db_path)
    session = get_session_factory(settings.db_path)()
    session.add(Strategy(name="ema_crossover", version=1, status="active", params="{}"))
    session.commit()
    session.close()

    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text(json.dumps({"search": _result().to_dict()}), encoding="utf-8")

    script = pathlib.Path(runner_mod.__file__).resolve().parent.parent.parent / "scripts" / "optimize.py"
    spec = importlib.util.spec_from_file_location("scripts_optimize_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._propose(settings, artifact_path) == 0

    session = get_session_factory(settings.db_path)()
    try:
        recs = session.execute(select(AIRecommendation)).scalars().all()
        assert len(recs) == 1
        assert recs[0].status == "pending"
        assert json.loads(recs[0].content_json)["params"] == {"fast_period": 9, "slow_period": 21}
        # Nothing was activated: still exactly the one seeded version.
        assert len(session.execute(select(Strategy)).scalars().all()) == 1
    finally:
        session.close()
    # A missing artifact is reported, not proposed.
    assert module._propose(settings, tmp_path / "nope.json") == 1


def test_scripts_optimize_cli_is_thin_and_proposes_only():
    path = pathlib.Path(runner_mod.__file__).resolve().parent.parent.parent / "scripts" / "optimize.py"
    source = path.read_text(encoding="utf-8")
    assert "--propose" in source and "--strategy" in source
    assert "start_search_subprocess" in source
    assert "telegram" not in source
    assert ".apply(" not in source


def test_run_backtest_contract_used_by_the_search_is_unchanged():
    """The optimizer consumes the same result shape the bot does."""
    result = run_backtest(_candles(60), "ema_crossover", {"fast_period": 3, "slow_period": 8})
    assert {"n_trades", "total_pnl", "max_drawdown", "final_balance"} <= set(result.to_dict())
