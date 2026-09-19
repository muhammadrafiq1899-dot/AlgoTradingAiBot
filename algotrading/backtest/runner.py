"""Shadow backtest: replay stored candles through a strategy, honestly.

Used to evaluate an AI proposal BEFORE it gets user approval: instead of
trusting a new strategy version blindly, we replay historical candles through
it and report how it would have performed. This keeps the controlled-release
path deterministic — the backtest never touches the exchange and never mutates
live state.

The runner operates on normalized `Candle` objects (oldest -> newest) and a
strategy instance built from `algotrading.strategy.registry`. It simulates a
single-symbol spot position:

  - BUY  when the strategy emits a buy signal and we are flat.
  - SELL when the strategy emits a sell signal and we hold.
  - exit early when the entry signal carried protective levels (see below).

HONEST, NOT FLATTERING
    A backtest that ignores costs and the risk layer is a sales pitch, not a
    measurement. So this replay charges fees *and* adverse slippage per side,
    can simulate the live risk guards, and states in `BacktestResult.assumptions`
    what it still does not model. Read that list before quoting a number.

WHAT IT ASSUMES (the same class of assumptions Freqtrade documents for its own
backtester, spelled out so nobody has to call them "details"):

  - Fills happen at the SIGNAL CANDLE'S CLOSE, not at the intrabar price that
    produced the signal. Entry fills are made worse by `slippage_pct` (buy
    higher), exit fills worse too (sell lower); slippage is a percentage of the
    fill, applied on top of the fill price.
  - A protective stop is assumed to fill AT THE STOP PRICE even if the bar's low
    was lower; a take-profit is assumed to fill AT THE TARGET PRICE even if the
    bar's high was higher. Real stops gap through on illiquid bars, so this
    flatters the strategy and is exactly why it is written down here.
  - When one bar touches both the stop and the target, the STOP is assumed to
    trigger first (the pessimistic reading).
  - Protective levels can only trigger from the bar AFTER the fill bar: the fill
    happens at the previous bar's close, and that bar's high/low already
    happened before we were in the market.
  - Equity marks use the bar close, net of the exit fee; slippage is not applied
    to marks (it is applied to actual fills only).
  - No partial fills, no order-book depth, no market impact, no funding/borrow
    cost, no exchange downtime, one symbol at a time.
  - `apply_risk_checks=True` simulates the LIVE guards it can (entry cooldown,
    `max_position_pct` sizing cap, the daily-loss entry block). It does NOT
    simulate `max_open_positions` beyond the trivial single-symbol case, and it
    does not model `trailing_stop_pct` (that one is stateful in the engine).

The strategy's own `evaluate` is still treated as a pure function of the window
it is handed, so a strategy that carries state internally will look better here
than it does live.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Iterator, Sequence, overload

from algotrading.market.base import Candle
from algotrading.strategy.registry import UnknownStrategyError, build_strategy

log = logging.getLogger(__name__)

# Flat fee applied per side (entry + exit) as a fraction of notional,
# approximating taker fees. Mirrors the paper gateway's cost model.
DEFAULT_FEE_RATE = 0.001

# Fallback when settings cannot be loaded (unit tests, a bare interpreter).
DEFAULT_SLIPPAGE_PCT = 0.0
DEFAULT_INITIAL_BALANCE = 10_000.0
DEFAULT_COOLDOWN_SECONDS = 300
DEFAULT_MAX_POSITION_PCT = 20.0
DEFAULT_MAX_DAILY_LOSS_PCT = 3.0

# Exit reasons recorded on a trade, so a report can separate "the strategy said
# sell" from "the stop saved us".
EXIT_SIGNAL = "signal"
EXIT_STOP = "stop"
EXIT_TAKE_PROFIT = "take_profit"

# Guards that skip an entry, counted per signal so a report can say *why* a
# strategy's trade count in a backtest is lower than its signal count.
GUARD_COOLDOWN = "cooldown"
GUARD_DAILY_LOSS = "daily_loss"
GUARD_POSITION_SIZE = "position_size"
_GUARD_KEYS = (GUARD_COOLDOWN, GUARD_DAILY_LOSS, GUARD_POSITION_SIZE)


class CandleWindow(Sequence[Candle]):
    """Lazy read-only view of ``candles[0 : end + 1]``.

    The replay hands a growing window to `strategy.evaluate` on every bar. The
    old implementation built ``list(candles[: i + 1])`` each time, which is O(n)
    of copying per bar and O(n^2) over a run — on a phone that is the difference
    between a sub-second and a multi-second backtest (and an optimizer runs this
    hundreds of times).

    Sequence-compatible on purpose: strategies are written against
    ``Sequence[Candle]`` and use ``len()``, indexing (including ``-1``),
    slicing and iteration. Slicing returns a real list, because that is what a
    list would do and some strategies slice to compute rolling windows.

    The window is mutable through `advance_to` (the runner reuses one instance
    for the whole replay instead of allocating per bar). It is valid only for
    the duration of the `evaluate` call — a strategy that stores it would be
    storing live state, which the strategy contract forbids anyway.
    """

    __slots__ = ("_candles", "_end")

    def __init__(self, candles: Sequence[Candle], end: int | None = None) -> None:
        self._candles = candles
        self._end = (len(candles) - 1) if end is None else end

    def advance_to(self, end: int) -> "CandleWindow":
        """Grow/shrink the window in place; returns self for chaining."""
        self._end = end
        return self

    def __len__(self) -> int:
        return self._end + 1

    @overload
    def __getitem__(self, index: int) -> Candle: ...

    @overload
    def __getitem__(self, index: slice) -> list[Candle]: ...

    def __getitem__(self, index: int | slice) -> Candle | list[Candle]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self._candles[i] for i in range(start, stop, step)]
        if index < 0:
            index += len(self)
        if index < 0 or index > self._end:
            raise IndexError(index)
        return self._candles[index]

    def __iter__(self) -> Iterator[Candle]:
        # islice over the underlying list: C-speed, no per-bar copy of the
        # elements themselves.
        return islice(self._candles, 0, self._end + 1)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CandleWindow(len={len(self)})"


@dataclass
class BacktestTrade:
    """A single simulated round trip inside a backtest."""

    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    pnl: float = 0.0
    # --- parity additions (defaulted so older constructions keep working) ---
    qty: float = 0.0
    fee: float = 0.0
    exit_reason: str = EXIT_SIGNAL
    bars_held: int = 0
    entry_ref_price: float = 0.0   # the bar close, before slippage
    exit_ref_price: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_ts": self.entry_ts,
            "entry_price": round(self.entry_price, 8),
            "exit_ts": self.exit_ts,
            "exit_price": round(self.exit_price, 8),
            "pnl": round(self.pnl, 8),
            "qty": round(self.qty, 12),
            "fee": round(self.fee, 8),
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
        }


@dataclass
class BacktestResult:
    """Summary of a strategy replay over stored candles.

    The first block of fields is the original contract (`to_dict` keys are
    consumed by `algotrading/telegram/bot.py` and
    `algotrading/store/recommendations.py`); everything after `trades` is
    additive, so existing constructors and readers keep working.
    """

    strategy_name: str
    params: dict[str, Any]
    symbol: str
    interval: str
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    final_balance: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    # --- parity additions ---------------------------------------------------
    initial_balance: float = 0.0
    # Realized cash, i.e. what is left after fees on every closed round trip.
    # `final_balance` is the liquidation VALUE (cash + open position marked at
    # the last close), so the two differ by exactly the unrealized PnL of a
    # position still open at the end (`position_open_at_end`).
    final_cash: float = 0.0
    n_bars: int = 0
    equity_curve: list[float] = field(default_factory=list)
    exposure: float = 0.0
    position_bars: int = 0
    fees_paid: float = 0.0
    slippage_pct: float = 0.0
    fee_rate: float = DEFAULT_FEE_RATE
    apply_risk_checks: bool = False
    skipped_by_guard: dict[str, int] = field(default_factory=dict)
    capped_entries: int = 0
    position_open_at_end: bool = False
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for `backtest_json` persistence and Telegram display.

        Keeps every key the pre-parity version emitted (callers parse it),
        plus the honesty fields. The raw `equity_curve` is deliberately NOT
        included: a 20k-candle replay would put ~400 KB of floats into a DB
        text column that Telegram also formats. Use the result object (or
        `algotrading.backtest.metrics.compute_metrics`) when you need the
        curve.
        """
        return {
            # --- original contract, unchanged ---
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "interval": self.interval,
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "final_balance": round(self.final_balance, 4),
            # --- parity additions ---
            "initial_balance": round(self.initial_balance, 4),
            "final_cash": round(self.final_cash, 4),
            "n_bars": self.n_bars,
            "exposure": round(self.exposure, 4),
            "position_bars": self.position_bars,
            "fees_paid": round(self.fees_paid, 4),
            "slippage_pct": self.slippage_pct,
            "fee_rate": self.fee_rate,
            "apply_risk_checks": self.apply_risk_checks,
            "skipped_by_guard": dict(self.skipped_by_guard),
            "capped_entries": self.capped_entries,
            "position_open_at_end": self.position_open_at_end,
            "assumptions": list(self.assumptions),
        }


@dataclass(frozen=True)
class _RiskParams:
    """The subset of `settings.risk` the replay can honestly simulate."""

    cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS
    max_position_pct: float = DEFAULT_MAX_POSITION_PCT
    max_daily_loss_pct: float = DEFAULT_MAX_DAILY_LOSS_PCT
    enforce_daily_loss: bool = True
    paper_initial_balance: float = DEFAULT_INITIAL_BALANCE


def _settings_defaults() -> tuple[float, float, bool, _RiskParams]:
    """Resolve the config defaults, falling back to module constants.

    The runner must stay usable without the config tree (a unit test, a bare
    interpreter in the optimizer child). `load_settings()` is cached and raises
    if `config/settings.yaml` is missing, so it is wrapped rather than trusted.
    """
    try:
        from algotrading.config import load_settings

        settings = load_settings()
    except Exception as exc:  # pragma: no cover - only without a config tree
        log.debug("backtest: settings unavailable (%s); using module defaults", exc)
        return (
            DEFAULT_SLIPPAGE_PCT,
            DEFAULT_FEE_RATE,
            False,
            _RiskParams(),
        )
    risk = settings.risk
    return (
        float(settings.backtest.slippage_pct),
        float(settings.backtest.fee_rate),
        bool(settings.backtest.apply_risk_checks),
        _RiskParams(
            cooldown_seconds=int(risk.cooldown_seconds),
            max_position_pct=float(risk.max_position_pct),
            max_daily_loss_pct=float(risk.max_daily_loss_pct),
            enforce_daily_loss=bool(getattr(risk, "enforce_daily_loss", True)),
            paper_initial_balance=float(risk.paper_initial_balance),
        ),
    )


def _protection_levels(
    risk: dict[str, Any], entry_fill: float
) -> tuple[float | None, float | None]:
    """Read protective levels off a signal's `risk` dict.

    The `Signal` contract (see `algotrading/strategy/base.py`) is an open dict,
    and today no shipped strategy fills in `stop_pct` / `take_profit_pct` (they
    all use `position_pct` only, or fold a stop into the rationale). Support is
    implemented here so a strategy that *does* declare them is replayed with
    them, and so AI-authored strategies have a documented way to declare a stop.

    Both percentage and absolute forms are accepted:

      - ``stop_pct`` / ``take_profit_pct``: percent below/above the ENTRY FILL.
      - ``stop_price`` / ``take_profit_price``: absolute price levels.

    Returns ``(stop_price, target_price)``; either may be None when absent or
    nonsensical (a take-profit below the entry is ignored, not inverted).
    """
    stop: float | None = None
    target: float | None = None

    stop_pct = risk.get("stop_pct")
    if stop_pct is not None:
        try:
            pct = float(stop_pct)
        except (TypeError, ValueError):
            pct = 0.0
        if pct > 0:
            stop = entry_fill * (1 - pct / 100.0)

    target_pct = risk.get("take_profit_pct")
    if target_pct is not None:
        try:
            pct = float(target_pct)
        except (TypeError, ValueError):
            pct = 0.0
        if pct > 0:
            target = entry_fill * (1 + pct / 100.0)

    raw_stop = risk.get("stop_price")
    if raw_stop is not None:
        try:
            value = float(raw_stop)
        except (TypeError, ValueError):
            value = 0.0
        if 0 < value < entry_fill:
            stop = value

    raw_target = risk.get("take_profit_price")
    if raw_target is not None:
        try:
            value = float(raw_target)
        except (TypeError, ValueError):
            value = 0.0
        if value > entry_fill:
            target = value

    return stop, target


def _build_assumptions(
    *,
    slippage_pct: float,
    fee_rate: float,
    apply_risk_checks: bool,
    position_capped: bool,
    position_open_at_end: bool = False,
) -> list[str]:
    """The list a report must print next to any number this replay produced."""
    assumptions = [
        (
            "Signals fill at the signal candle's close, not at the intrabar price "
            "that produced them; there is no lookahead beyond the signal bar."
        ),
        (
            f"Entry and exit fills are moved against the position by "
            f"{slippage_pct:g}% (buy higher, sell lower) and charged "
            f"{fee_rate:g} of notional per side."
        ),
        (
            "A protective stop is assumed to fill AT THE STOP PRICE even if the "
            "bar's low was lower, and a take-profit AT THE TARGET even if the "
            "high was higher; when one bar touches both, the stop is assumed first."
        ),
        (
            "Protective levels declared by a signal's `risk` dict (stop_pct / "
            "take_profit_pct / stop_price / take_profit_price) only trigger from "
            "the bar after the fill bar."
        ),
        (
            "No partial fills, no order-book depth or market impact, no funding "
            "cost, no exchange downtime, one symbol at a time."
        ),
        (
            "Equity is marked at the bar close net of the exit fee; slippage is "
            "applied to fills only, so marks are slightly optimistic."
        ),
        (
            "The strategy is assumed pure: internal state that a live engine "
            "would carry between ticks is rebuilt from the window each bar."
        ),
    ]
    if apply_risk_checks:
        assumptions.append(
            "Risk checks simulate entry cooldown, the max_position_pct sizing cap "
            "and the daily-loss entry block only; max_open_positions is trivial on "
            "one symbol and trailing_stop_pct is not modelled."
        )
    else:
        assumptions.append(
            "No live risk layer is simulated: every entry is all-in on this "
            "symbol, with no cooldown, position cap or daily-loss block."
        )
    if position_capped:
        assumptions.append(
            "Entries are sized as a fraction of balance (capped), so PnL is a "
            "portfolio return, not an all-in return."
        )
    if position_open_at_end:
        assumptions.append(
            "A position is still open at the last bar: its PnL is unrealized and "
            "excluded from total_pnl, while final_balance marks it at the last "
            "close (final_cash is the realized part)."
        )
    return assumptions


def run_backtest(
    candles: Sequence[Candle],
    strategy_name: str,
    params: dict[str, Any],
    initial_balance: float | None = None,
    fee_rate: float | None = None,
    *,
    slippage_pct: float | None = None,
    apply_risk_checks: bool | None = None,
    risk_pct: float | None = None,
) -> BacktestResult:
    """Replay `candles` through a strategy and return the equity summary.

    Every override that is None falls back to `settings.backtest`
    (`slippage_pct`, `fee_rate`, `apply_risk_checks`) or
    `settings.risk.paper_initial_balance` (`initial_balance`), so production
    callers get the configured cost model while a test can pin exact numbers.
    The two positional parameters stay positional: existing calls like
    ``run_backtest(candles, name, params, 5_000.0, 0.002)`` keep meaning what
    they meant.

    Position sizing follows the live engine rather than the wishful reading:
    all-in per round trip by default (the historical behaviour this replay was
    built on), or `min(requested, risk.max_position_pct)` of balance when
    `apply_risk_checks` is on. The requested fraction comes from the signal's
    ``risk["position_pct"]`` when present; `risk_pct` pins it explicitly for
    every entry (and is still clamped when the risk checks are on). The
    cooldown and daily-loss guards are only simulated with
    `apply_risk_checks=True`, so an explicit `risk_pct` alone cannot silently
    block entries.

    Args:
        candles: normalized Candle list, oldest -> newest, same symbol+interval.
        strategy_name: registered strategy name (e.g. "ema_crossover").
        params: strategy parameters.
        initial_balance: starting cash (None -> settings.risk.paper_initial_balance).
        fee_rate: per-side cost as a fraction of notional (None -> settings).
        slippage_pct: adverse fill as a PERCENT of price (None -> settings).
        apply_risk_checks: simulate the live guards (None -> settings).
        risk_pct: fraction of balance per entry, overriding all-in sizing.

    Returns:
        BacktestResult with per-trade records, the equity curve, exposure and
        the `assumptions` list. PnL is net of both fees and slippage.

    Raises:
        ValueError: no candles, or the strategy cannot be built.
    """
    if not candles:
        raise ValueError("run_backtest requires at least one candle")

    default_slippage, default_fee, default_risk_checks, risk_cfg = _settings_defaults()
    if initial_balance is None:
        initial_balance = risk_cfg.paper_initial_balance
    if fee_rate is None:
        fee_rate = default_fee
    if slippage_pct is None:
        slippage_pct = default_slippage
    if apply_risk_checks is None:
        apply_risk_checks = default_risk_checks

    if initial_balance <= 0:
        raise ValueError("initial_balance must be > 0")
    if fee_rate < 0:
        raise ValueError("fee_rate must be >= 0")
    if slippage_pct < 0:
        raise ValueError("slippage_pct must be >= 0")

    try:
        strat = build_strategy(strategy_name, dict(params))
    except (UnknownStrategyError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot build strategy {strategy_name!r}: {exc}") from exc

    symbol = candles[0].symbol
    interval = candles[0].interval

    slip = slippage_pct / 100.0
    all_in = risk_pct is None and not apply_risk_checks
    max_pct = risk_cfg.max_position_pct / 100.0
    cooldown_ms = risk_cfg.cooldown_seconds * 1000
    daily_limit_pct = risk_cfg.max_daily_loss_pct

    cash = float(initial_balance)
    peak = float(initial_balance)
    max_dd = 0.0

    holding = False
    qty = 0.0
    entry_fill = 0.0
    entry_ref = 0.0
    entry_ts: int | None = None
    entry_bar = 0
    stop_price: float | None = None
    target_price: float | None = None

    trades: list[BacktestTrade] = []
    equity_curve: list[float] = []
    position_bars = 0
    fees_paid = 0.0
    capped_entries = 0
    position_capped = False
    skipped = {key: 0 for key in _GUARD_KEYS}
    last_entry_ts: int | None = None
    realized_by_day: dict[int, float] = {}
    blocked_days: set[int] = set()

    window = CandleWindow(candles, -1)

    def record(close: float) -> None:
        """Append the bar's liquidation value and update drawdown.

        One mark per bar, computed once: the equity curve and the drawdown
        tracker read the same number, and a replay must stay cheap enough to
        run hundreds of times inside the optimizer.
        """
        nonlocal peak, max_dd
        eq = cash + qty * close * (1 - fee_rate) if holding else cash
        if eq > peak:
            peak = eq
        elif peak - eq > max_dd:
            max_dd = peak - eq
        equity_curve.append(eq)

    def close_position(*, ts: int, ref_close: float, bar: int, reason: str,
                       fill_price: float | None = None) -> None:
        """Sell the lot at `ref_close`, after adverse slippage.

        `fill_price` overrides the raw price for stops/targets, which are
        assumed to fill at their own level (then slipped, like any fill).
        """
        nonlocal cash, holding, qty, entry_fill, entry_ts, fees_paid
        nonlocal stop_price, target_price
        price = (fill_price if fill_price is not None else ref_close) * (1 - slip)
        proceeds = qty * price
        fee = proceeds * fee_rate
        fees_paid += fee
        cash += proceeds - fee
        cost = qty * entry_fill * (1 + fee_rate)
        entry_fee = qty * entry_fill * fee_rate
        trades.append(
            BacktestTrade(
                entry_ts=entry_ts or 0,
                entry_price=entry_fill,
                exit_ts=ts,
                exit_price=price,
                pnl=(proceeds - fee) - cost,
                qty=qty,
                fee=fee + entry_fee,
                exit_reason=reason,
                bars_held=bar - entry_bar,
                entry_ref_price=entry_ref,
                exit_ref_price=ref_close,
            )
        )
        day = ts // 86_400_000
        realized_by_day[day] = realized_by_day.get(day, 0.0) + trades[-1].pnl
        if risk_cfg.enforce_daily_loss and daily_limit_pct > 0 and cash > 0:
            loss_pct = (-realized_by_day[day] / cash) * 100.0
            if loss_pct >= daily_limit_pct:
                # Same semantics as RiskManager.daily_loss_breached: a breach
                # blocks NEW entries for the rest of that UTC day, never exits.
                blocked_days.add(day)
        holding = False
        qty = 0.0
        entry_fill = 0.0
        entry_ts = None
        stop_price = target_price = None

    def open_position(*, ts: int, ref_close: float, bar: int, sig: Any) -> None:
        """Buy at `ref_close` (slipped) using the configured sizing policy."""
        nonlocal cash, holding, qty, entry_fill, entry_ref, entry_ts, entry_bar
        nonlocal fees_paid, capped_entries, position_capped, last_entry_ts
        nonlocal stop_price, target_price
        if all_in and risk_pct is None:
            pct = 1.0
        else:
            requested = risk_pct
            if requested is None:
                raw = (sig.risk or {}).get("position_pct") if sig is not None else None
                try:
                    requested = float(raw) if raw is not None else 1.0
                except (TypeError, ValueError):
                    requested = 1.0
            pct = requested
            if not pct > 0 or pct == float("inf"):
                # The live engine records a qty<=0 sizing request as
                # risk_skipped; here it simply cannot open. `not pct > 0` also
                # catches NaN, which would otherwise poison every later PnL.
                skipped[GUARD_POSITION_SIZE] += 1
                return
            # Spot: a requested fraction above 1.0 would mean borrowing, which
            # this bot cannot do, so it is capped at the whole balance.
            pct = min(pct, 1.0)
            if apply_risk_checks:
                capped = min(pct, max_pct)
                if capped < pct:
                    capped_entries += 1
                    position_capped = True
                pct = capped
        fill = ref_close * (1 + slip)
        notional = cash * pct
        if notional <= 0 or fill <= 0:
            return
        # Buy including the fee: cost = qty * fill * (1 + fee_rate) == notional.
        qty = notional / (fill * (1 + fee_rate))
        fees_paid += qty * fill * fee_rate
        cash -= notional
        entry_fill = fill
        entry_ref = ref_close
        entry_ts = ts
        entry_bar = bar
        holding = True
        last_entry_ts = ts
        stop_price, target_price = _protection_levels(
            (sig.risk or {}) if sig is not None else {}, entry_fill
        )

    n = len(candles)
    for i in range(n):
        candle = candles[i]
        close = candle.close
        ts = candle.ts
        window.advance_to(i)
        day = ts // 86_400_000

        # 1) Protective levels fire intrabar, before any strategy decision, and
        #    only from the bar after the fill (see module docstring).
        if holding and i > entry_bar:
            if stop_price is not None and candle.low <= stop_price:
                close_position(
                    ts=ts, ref_close=candle.low, bar=i, reason=EXIT_STOP,
                    fill_price=stop_price,
                )
            elif target_price is not None and candle.high >= target_price:
                close_position(
                    ts=ts, ref_close=candle.high, bar=i, reason=EXIT_TAKE_PROFIT,
                    fill_price=target_price,
                )

        # 2) The strategy sees the same growing window the engine would hand it.
        sig = strat.evaluate(symbol, window)

        if sig is not None and sig.side == "buy" and not holding:
            blocked = False
            if apply_risk_checks:
                if last_entry_ts is not None and ts - last_entry_ts < cooldown_ms:
                    skipped[GUARD_COOLDOWN] += 1
                    blocked = True
                elif day in blocked_days:
                    skipped[GUARD_DAILY_LOSS] += 1
                    blocked = True
            if not blocked:
                open_position(ts=ts, ref_close=close, bar=i, sig=sig)
        elif sig is not None and sig.side == "sell" and holding:
            close_position(ts=ts, ref_close=close, bar=i, reason=EXIT_SIGNAL)

        if holding:
            position_bars += 1
        record(close)

    # A position still open at the end is marked, not liquidated: the loop's last
    # `record` already did exactly that (the live bot would still be holding, and
    # inventing an exit would invent PnL).

    pnls = [t.pnl for t in trades]
    n_wins = sum(1 for p in pnls if p > 0)
    n_losses = sum(1 for p in pnls if p < 0)

    return BacktestResult(
        strategy_name=strategy_name,
        params=dict(params),
        symbol=symbol,
        interval=interval,
        n_trades=len(trades),
        n_wins=n_wins,
        n_losses=n_losses,
        total_pnl=sum(pnls),
        max_drawdown=max_dd,
        win_rate=(n_wins / len(trades)) if trades else 0.0,
        final_balance=equity_curve[-1] if equity_curve else cash,
        trades=trades,
        initial_balance=float(initial_balance),
        final_cash=cash,
        n_bars=n,
        equity_curve=equity_curve,
        exposure=(position_bars / n) if n else 0.0,
        position_bars=position_bars,
        fees_paid=fees_paid,
        slippage_pct=float(slippage_pct),
        fee_rate=float(fee_rate),
        apply_risk_checks=bool(apply_risk_checks),
        skipped_by_guard=dict(skipped),
        capped_entries=capped_entries,
        position_open_at_end=holding,
        assumptions=_build_assumptions(
            slippage_pct=float(slippage_pct),
            fee_rate=float(fee_rate),
            apply_risk_checks=bool(apply_risk_checks),
            position_capped=position_capped,
            position_open_at_end=holding,
        ),
    )


__all__ = [
    "CandleWindow",
    "BacktestResult",
    "BacktestTrade",
    "run_backtest",
    "DEFAULT_FEE_RATE",
    "EXIT_SIGNAL",
    "EXIT_STOP",
    "EXIT_TAKE_PROFIT",
    "GUARD_COOLDOWN",
    "GUARD_DAILY_LOSS",
    "GUARD_POSITION_SIZE",
]
