"""Grid/random search over strategy parameters, ranked by a chosen objective.

Design decisions worth knowing before reading the numbers this produces:

  * Candidates are scored with the WALK-FORWARD layer, never with a single
    whole-series replay. A parameter set that only works in one segment of the
    data scores badly here on purpose — that is the whole point of the exercise.
  * A candidate below `optimize.min_trades` total closed trades is rejected, not
    ranked. A Sharpe computed from two trades is noise wearing a decimal point.
  * The candidate cap is hard: the full grid is enumerated only when it fits in
    `max_combinations`; otherwise a SEEDED random sample of exactly that size is
    drawn. Same seed + same space = same candidate list, on any machine.
  * Nothing here applies anything. `run_search` returns data; turning it into a
    PENDING recommendation is `algotrading.optimize.proposal`'s job.
"""
from __future__ import annotations

import itertools
import json
import logging
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from algotrading.backtest.metrics import walk_forward

log = logging.getLogger(__name__)

OBJECTIVES = ("sharpe", "pnl_drawdown", "total_pnl")
DEFAULT_OBJECTIVE = "sharpe"
DEFAULT_MAX_COMBINATIONS = 200
DEFAULT_MIN_TRADES = 10
DEFAULT_FOLDS = 4
DEFAULT_SEED = 42

# Parameter-space shaping: at most this many values per parameter, so a strategy
# with eight tunables does not explode into a 5^8 grid that the cap then
# truncates arbitrarily (a truncated product grid is biased toward the first
# parameters — a random sample is not).
MAX_VALUES_PER_PARAM = 4


@dataclass
class Candidate:
    """One parameter set and what the replay said about it."""

    params: dict[str, Any]
    ok: bool = True
    reason: str = ""
    objective_value: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    consistency: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, rank: int | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {"params": dict(self.params), "ok": self.ok}
        if rank is not None:
            out["rank"] = rank
        if not self.ok:
            out["reason"] = self.reason
            return out
        out["objective_value"] = self.objective_value
        out["metrics"] = self.metrics
        out["consistency"] = self.consistency
        return out


@dataclass
class SearchResult:
    """Ranked candidates, the rejections with reasons, and the run metadata."""

    strategy_name: str
    symbol: str
    interval: str
    objective: str
    seed: int
    n_bars: int = 0
    min_trades: int = DEFAULT_MIN_TRADES
    folds: int = DEFAULT_FOLDS
    n_candidates: int = 0
    n_evaluated: int = 0
    duration_seconds: float = 0.0
    timed_out: bool = False
    param_space: dict[str, list[Any]] = field(default_factory=dict)
    ranked: list[Candidate] = field(default_factory=list)
    rejected: list[Candidate] = field(default_factory=list)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    def best(self) -> Candidate | None:
        """The top-ranked candidate, or None when nothing survived the filters."""
        return self.ranked[0] if self.ranked else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable artifact (what lands in `settings.optimize.results_dir`)."""
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "interval": self.interval,
            "objective": self.objective,
            "seed": self.seed,
            "n_bars": self.n_bars,
            "min_trades": self.min_trades,
            "folds": self.folds,
            "n_candidates": self.n_candidates,
            "n_evaluated": self.n_evaluated,
            "n_rejected": self.n_rejected,
            "duration_seconds": round(self.duration_seconds, 3),
            "timed_out": self.timed_out,
            "param_space": {key: list(values) for key, values in self.param_space.items()},
            "ranked": [c.to_dict(rank=i + 1) for i, c in enumerate(self.ranked)],
            "rejected": [c.to_dict() for c in self.rejected],
            "assumptions": [
                "Scores come from sequential (non-purged) walk-forward folds with "
                "per-fold balance resets; see metrics.walk_forward for the caveats.",
                "Rejected candidates are kept with their reason so a report can "
                "distinguish 'no edge' from 'not enough trades'.",
                "Ranking is in-sample: the top candidate is a hypothesis, and any "
                "change derived from it stays PENDING until a human approves it.",
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResult":
        """Rebuild a result from its JSON artifact (round-trip for the CLI)."""
        result = cls(
            strategy_name=data.get("strategy_name", ""),
            symbol=data.get("symbol", ""),
            interval=data.get("interval", ""),
            objective=data.get("objective", DEFAULT_OBJECTIVE),
            seed=int(data.get("seed", DEFAULT_SEED)),
            n_bars=int(data.get("n_bars", 0)),
            min_trades=int(data.get("min_trades", DEFAULT_MIN_TRADES)),
            folds=int(data.get("folds", DEFAULT_FOLDS)),
            n_candidates=int(data.get("n_candidates", 0)),
            n_evaluated=int(data.get("n_evaluated", 0)),
            duration_seconds=float(data.get("duration_seconds", 0.0)),
            timed_out=bool(data.get("timed_out", False)),
            param_space={k: list(v) for k, v in (data.get("param_space") or {}).items()},
        )
        for entry in data.get("ranked") or []:
            result.ranked.append(
                Candidate(
                    params=dict(entry.get("params") or {}),
                    ok=True,
                    objective_value=entry.get("objective_value"),
                    metrics=dict(entry.get("metrics") or {}),
                    consistency=dict(entry.get("consistency") or {}),
                )
            )
        for entry in data.get("rejected") or []:
            result.rejected.append(
                Candidate(
                    params=dict(entry.get("params") or {}),
                    ok=False,
                    reason=entry.get("reason", ""),
                )
            )
        return result

    def summary_line(self) -> str:
        """One line for a log or a CLI print; never a table dump."""
        best = self.best()
        head = (
            f"{self.strategy_name} {self.symbol} {self.interval} | "
            f"objective={self.objective} candidates={self.n_candidates} "
            f"ranked={len(self.ranked)} rejected={self.n_rejected} "
            f"{'TIMED OUT ' if self.timed_out else ''}in {self.duration_seconds:.1f}s"
        )
        if best is None:
            return head + " | best=none (every candidate was rejected)"
        value = best.objective_value
        shown = "n/a" if value is None else f"{value:.4f}"
        return f"{head} | best={shown} params={best.params}"


def _dedupe(values: Iterable[Any]) -> list[Any]:
    """Order-preserving dedupe that tolerates unhashable values (lists, dicts)."""
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, sort_keys=True, default=repr)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def generate_grid(
    param_space: dict[str, Sequence[Any]],
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Candidate parameter combinations, never more than `max_combinations`.

    The full cartesian product is returned when it fits under the cap (so the
    result is exhaustive and order-independent). Above the cap, a seeded random
    sample is drawn instead: a truncated product grid would systematically omit
    whatever comes after the cut, which biases the search toward the first
    parameter's values.

    Deterministic for a given (space, cap, seed): the same call returns the same
    list, which is what makes a search artifact reproducible.
    """
    if max_combinations < 1:
        raise ValueError("max_combinations must be >= 1")
    keys = list(param_space)
    if not keys:
        raise ValueError("param_space must not be empty")
    values = [_dedupe(param_space[key]) for key in keys]
    if any(not options for options in values):
        empty = keys[next(i for i, options in enumerate(values) if not options)]
        raise ValueError(f"param_space[{empty!r}] has no candidate values")

    total = 1
    for options in values:
        total *= len(options)

    if total <= max_combinations:
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(out) < max_combinations:
        combo = tuple(rng.choice(options) for options in values)
        key = json.dumps(combo, sort_keys=True, default=repr)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(zip(keys, combo)))
    return out


def objective_value(objective: str, metrics: dict[str, Any]) -> float | None:
    """Score a candidate's aggregate metrics, or None when it is undefined.

    `pnl_drawdown` divides by the WORST fold drawdown, so a strategy that pays
    for its return with one deep fold cannot hide behind a calm average. A
    sample with no drawdown at all scores None rather than infinity: "no
    measured risk" is not "risk-free".
    """
    if objective == "sharpe":
        value = metrics.get("sharpe")
    elif objective == "total_pnl":
        value = metrics.get("total_pnl")
    elif objective == "pnl_drawdown":
        drawdown = float(metrics.get("worst_fold_drawdown") or 0.0)
        total_pnl = float(metrics.get("total_pnl") or 0.0)
        value = (total_pnl / drawdown) if drawdown > 0 else None
    else:
        raise ValueError(f"unknown objective {objective!r}; known: {OBJECTIVES}")
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def aggregate_folds(fold_report: dict[str, Any]) -> dict[str, Any]:
    """Flatten a `walk_forward` report into one metric set for ranking.

    Sums where a sum means something (PnL, trades, fees), a mean for per-fold
    ratios (Sharpe/Sortino — averaging fold Sharpes is a stabler statistic than a
    single number computed on a concatenated curve), and the WORST case for
    drawdown. Documented because a "total Sharpe" across folds is a choice, not a
    law of nature.
    """
    folds = [f for f in fold_report.get("per_fold", []) if not f.get("skipped")]
    metrics = [f["metrics"] for f in folds]
    trades = [m["n_trades"] for m in metrics]
    wins = [m["n_wins"] for m in metrics]
    losses = [m["n_losses"] for m in metrics]

    def _mean_of(key: str) -> float | None:
        values = [m[key] for m in metrics if m.get(key) is not None]
        return round(statistics.fmean(values), 8) if values else None

    total_bars = sum(m["n_bars"] for m in metrics)
    weighted_exposure = (
        sum(m["exposure"] * m["n_bars"] for m in metrics) / total_bars
        if total_bars
        else 0.0
    )
    total_trades = sum(trades)
    returns = [f.get("return_pct", 0.0) for f in folds]
    return {
        "n_folds": len(folds),
        "n_trades": total_trades,
        "n_wins": sum(wins),
        "n_losses": sum(losses),
        "win_rate": round(sum(wins) / total_trades, 6) if total_trades else 0.0,
        "total_pnl": round(sum(m["total_pnl"] for m in metrics), 8),
        "return_pct": round(sum(returns), 6),
        "sharpe": _mean_of("sharpe"),
        "sortino": _mean_of("sortino"),
        "worst_fold_drawdown": round(max((m["max_drawdown"] for m in metrics), default=0.0), 8),
        "worst_fold_drawdown_pct": round(
            max((m["max_drawdown_pct"] for m in metrics), default=0.0), 6
        ),
        "fees_paid": round(sum(m["fees_paid"] for m in metrics), 8),
        "exposure": round(weighted_exposure, 6),
        "best_fold_return_pct": round(max(returns), 6) if returns else None,
        "worst_fold_return_pct": round(min(returns), 6) if returns else None,
    }


def default_param_space(strategy_name: str, settings: Any = None) -> dict[str, list[Any]]:
    """A small, sensible search space derived from the strategy's own schema.

    Values come from `algotrading.strategy.catalog` (the same source the chat
    prompt and `/strategies` use), so a plugin strategy is searchable without
    the optimizer knowing anything about it: integer ranges enumerate when they
    are short (<= 8) and step otherwise, floats get min/default/max, enums use
    their declared values.

    Booleans and free strings are deliberately left at their defaults: flipping
    every boolean doubles the grid per flag, and a random sample of a huge grid
    is worse than a focused one. Pass an explicit space to `run_search` to
    override any of this.
    """
    from algotrading.strategy.catalog import catalog_entries

    entry = next((e for e in catalog_entries() if e.name == strategy_name), None)
    if entry is None:
        raise ValueError(f"unknown strategy {strategy_name!r} (no catalog entry)")

    space: dict[str, list[Any]] = {}
    for spec in entry.params:
        values: list[Any] = []
        if spec.enum:
            values = list(spec.enum)[:MAX_VALUES_PER_PARAM]
        elif spec.type == "int" and spec.min is not None and spec.max is not None:
            values = _int_values(int(spec.min), int(spec.max), spec.default)
        elif spec.type == "float" and spec.min is not None and spec.max is not None:
            low, high = float(spec.min), float(spec.max)
            values = [low]
            if spec.default is not None:
                values.append(float(spec.default))
            values.append(high)
        values = _dedupe(values)[:MAX_VALUES_PER_PARAM]
        if len(values) >= 2:
            space[spec.name] = values
    if not space:
        raise ValueError(
            f"strategy {strategy_name!r} exposes no ranged parameters to search; "
            "pass an explicit param_space"
        )
    return space


def _int_values(low: int, high: int, default: Any) -> list[int]:
    """Grid values for an integer parameter.

    Short ranges are enumerated (there is nothing to sample). Wide ranges step
    around the DEFAULT rather than across the declared span: `fast_period` is
    declared 2..200, but an EMA pair at 134/398 is not a strategy anyone would
    approve — doubling and halving the shipped default keeps the search inside
    the region the strategy was designed for, and still leaves the extremes to an
    explicit `param_space`.
    """
    if high < low:
        low, high = high, low
    span = high - low
    if span <= 8:
        return list(range(low, high + 1))
    centre = int(default) if isinstance(default, (int, float)) and default else (low + high) // 2
    centre = max(low, min(high, centre))
    candidates = {centre, max(low, centre // 2), min(high, centre * 2), min(high, centre * 3)}
    return sorted(candidates)


def _load_settings() -> Any:
    from algotrading.config import load_settings

    return load_settings()


def run_search(
    candles: Sequence[Any],
    strategy_name: str,
    param_space: dict[str, Sequence[Any]],
    objective: str | None = None,
    *,
    folds: int | None = None,
    max_combinations: int | None = None,
    min_trades: int | None = None,
    seed: int | None = None,
    settings: Any = None,
    deadline: float | None = None,
    run_kwargs: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> SearchResult:
    """Rank parameter combinations by `objective` using walk-forward metrics.

    Args:
        candles: bar series, oldest -> newest (never synthesized here).
        strategy_name: registered strategy name.
        param_space: ``{param: [values]}``; see `default_param_space`.
        objective: 'sharpe' | 'pnl_drawdown' | 'total_pnl' (None -> settings).
        folds: walk-forward fold count (None -> settings.backtest).
        max_combinations: hard candidate cap (None -> settings.optimize).
        min_trades: rejection floor on total closed trades (None -> settings.optimize).
        seed: RNG seed for the sampled grid (None -> settings.backtest.random_seed).
        settings: injected settings (defaults to `load_settings()`).
        deadline: `time.monotonic()` value after which remaining candidates are
            recorded as rejected (`timeout`) instead of evaluated — the runner
            uses this to respect `optimize.timeout_seconds` even when the parent
            process is about to kill it.
        run_kwargs: overrides forwarded to `run_backtest` (fee/slippage/risk).
        progress: optional `(done, total)` callback, called after each candidate.

    Returns:
        SearchResult with `ranked` (best first) and `rejected` (with reasons).
        Nothing is applied, persisted or written here.
    """
    candles = list(candles)
    if not candles:
        raise ValueError("run_search requires at least one candle")
    settings = settings if settings is not None else _load_settings()
    opt = getattr(settings, "optimize", None)
    bt = getattr(settings, "backtest", None)

    objective = objective or getattr(opt, "objective", None) or DEFAULT_OBJECTIVE
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; known: {OBJECTIVES}")
    max_combinations = int(
        max_combinations
        if max_combinations is not None
        else getattr(opt, "max_combinations", DEFAULT_MAX_COMBINATIONS)
    )
    min_trades = int(
        min_trades if min_trades is not None else getattr(opt, "min_trades", DEFAULT_MIN_TRADES)
    )
    folds = int(folds if folds is not None else getattr(bt, "walk_forward_folds", DEFAULT_FOLDS))
    if seed is None:
        seed = int(getattr(bt, "random_seed", DEFAULT_SEED))

    combos = generate_grid(param_space, max_combinations, seed)
    symbol = candles[0].symbol
    interval = candles[0].interval
    started = time.monotonic()

    result = SearchResult(
        strategy_name=strategy_name,
        symbol=symbol,
        interval=interval,
        objective=objective,
        seed=seed,
        n_bars=len(candles),
        min_trades=min_trades,
        folds=folds,
        n_candidates=len(combos),
        param_space={key: list(values) for key, values in param_space.items()},
    )

    for index, params in enumerate(combos):
        if deadline is not None and time.monotonic() > deadline:
            result.timed_out = True
            result.rejected.append(
                Candidate(params=dict(params), ok=False, reason="timeout: not evaluated")
            )
            continue
        candidate = _evaluate_candidate(
            candles, strategy_name, params, objective,
            folds=folds, min_trades=min_trades, settings=settings,
            run_kwargs=run_kwargs,
        )
        result.n_evaluated += 1
        if candidate.ok:
            result.ranked.append(candidate)
        else:
            result.rejected.append(candidate)
        if progress is not None:
            progress(index + 1, len(combos))

    # Best first; ties break on the params repr so the order is stable across
    # runs and platforms (a shuffled "best" makes a proposal irreproducible).
    result.ranked.sort(
        key=lambda c: (-(c.objective_value or 0.0), json.dumps(c.params, sort_keys=True, default=repr))
    )
    result.duration_seconds = time.monotonic() - started
    best = result.best()
    log.info(
        "optimize: %s candidates=%d ranked=%d rejected=%d best=%s",
        strategy_name, result.n_candidates, len(result.ranked), result.n_rejected,
        best.objective_value if best is not None else None,
    )
    return result


def _evaluate_candidate(
    candles: Sequence[Any],
    strategy_name: str,
    params: dict[str, Any],
    objective: str,
    *,
    folds: int,
    min_trades: int,
    settings: Any,
    run_kwargs: dict[str, Any] | None,
) -> Candidate:
    """Walk-forward replay one candidate and score it, or explain the rejection."""
    try:
        fold_report = walk_forward(
            candles,
            strategy_name,
            dict(params),
            folds,
            settings=settings,
            run_kwargs=run_kwargs,
        )
    except ValueError as exc:
        # A parameter set the strategy itself refuses (fast >= slow, bad enum)
        # is a rejected candidate, not a crash: the grid is expected to contain
        # nonsense and the report must say so.
        return Candidate(params=dict(params), ok=False, reason=f"unbuildable: {exc}")

    if not fold_report.get("replayed_folds"):
        # Every fold failed to build/replay: a parameter set the strategy refuses
        # (fast >= slow, unknown enum) is a rejection with a reason, not a crash
        # and not a candidate silently scored as zero.
        reason = next(
            (f.get("reason") for f in fold_report.get("per_fold", []) if f.get("reason")),
            "no fold could be replayed",
        )
        return Candidate(params=dict(params), ok=False, reason=f"unbuildable: {reason}")

    metrics = aggregate_folds(fold_report)
    if metrics["n_trades"] < min_trades:
        return Candidate(
            params=dict(params),
            ok=False,
            reason=(
                f"min_trades: {metrics['n_trades']} closed trades < {min_trades} "
                f"across {metrics['n_folds']} folds"
            ),
            metrics=metrics,
        )
    value = objective_value(objective, metrics)
    if value is None:
        return Candidate(
            params=dict(params),
            ok=False,
            reason=f"objective {objective!r} undefined for this candidate",
            metrics=metrics,
        )
    consistency = dict(fold_report.get("consistency", {}))
    return Candidate(
        params=dict(params),
        ok=True,
        objective_value=round(value, 8),
        metrics=metrics,
        consistency=consistency,
    )


__all__ = [
    "Candidate",
    "OBJECTIVES",
    "SearchResult",
    "aggregate_folds",
    "default_param_space",
    "generate_grid",
    "objective_value",
    "run_search",
]
