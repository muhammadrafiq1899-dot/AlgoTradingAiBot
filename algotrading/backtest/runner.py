"""Shadow backtest: replay stored candles through a strategy.

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

It does NOT model live risk checks (cooldown, position sizing, exposure caps)
— those belong to the execution engine. The backtest only answers "does this
strategy produce edge on the data we have".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from algotrading.market.base import Candle
from algotrading.strategy.registry import UnknownStrategyError, build_strategy

log = logging.getLogger(__name__)

# Flat fee applied per side (entry + exit) as a fraction of notional,
# approximating taker fees. Mirrors the paper gateway's cost model.
DEFAULT_FEE_RATE = 0.001


@dataclass
class BacktestTrade:
    """A single simulated round trip inside a backtest."""

    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Summary of a strategy replay over stored candles."""

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

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for `backtest_json` persistence and Telegram display."""
        return {
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
        }


def run_backtest(
    candles: Sequence[Candle],
    strategy_name: str,
    params: dict[str, Any],
    initial_balance: float = 10_000.0,
    fee_rate: float = DEFAULT_FEE_RATE,
) -> BacktestResult:
    """Replay `candles` through a strategy and return the equity summary.

    The simulated position is all-in per round trip on the single symbol: BUY
    spends the whole balance at the signal close, SELL liquidates at the exit
    close. Fees are charged per side against notional.

    Args:
        candles: normalized Candle list, oldest -> newest, all same symbol+interval.
        strategy_name: registered strategy name (e.g. "ema_crossover").
        params: strategy parameters (position_pct is ignored for sizing here).
        initial_balance: starting cash.
        fee_rate: per-side cost fraction of notional.

    Returns:
        BacktestResult with per-trade records and aggregate metrics.
    """
    if not candles:
        raise ValueError("run_backtest requires at least one candle")

    try:
        strat = build_strategy(strategy_name, dict(params))
    except (UnknownStrategyError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot build strategy {strategy_name!r}: {exc}") from exc

    symbol = candles[0].symbol
    interval = candles[0].interval

    balance = initial_balance  # equity including any open position
    peak = initial_balance
    max_dd = 0.0

    holding = False
    qty = 0.0
    entry_price = 0.0
    entry_ts: int | None = None
    trades: list[BacktestTrade] = []

    def equity_at(close: float) -> float:
        if holding:
            return qty * close - qty * entry_price * fee_rate  # mark-to-market
        return balance

    def track_equity(close: float) -> None:
        nonlocal peak, max_dd
        eq = equity_at(close)
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)

    for i in range(len(candles)):
        window = list(candles[: i + 1])
        close = window[-1].close
        sig = strat.evaluate(symbol, window)

        if sig is not None and sig.side == "buy" and not holding:
            # Open a full position at this close.
            entry_price = close
            entry_ts = window[-1].ts
            qty = balance / (entry_price + entry_price * fee_rate)  # buy incl. fee
            holding = True
        elif sig is not None and sig.side == "sell" and holding:
            # Close: proceeds = qty * close - fee.
            exit_ts = window[-1].ts
            fee = qty * close * fee_rate
            proceeds = qty * close - fee
            cost = qty * entry_price + qty * entry_price * fee_rate
            pnl = proceeds - cost
            trades.append(
                BacktestTrade(
                    entry_ts=entry_ts or 0,
                    entry_price=entry_price,
                    exit_ts=exit_ts,
                    exit_price=close,
                    pnl=pnl,
                )
            )
            balance = proceeds
            holding = False
            qty = 0.0
            track_equity(close)

        if holding:
            track_equity(close)

    # Mark any still-open position to the last close.
    if holding:
        track_equity(candles[-1].close)

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
        final_balance=balance,
        trades=trades,
    )
