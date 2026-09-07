"""Performance metrics computed from the closed-trade ledger.

Pure functions over a list of Trade-like objects so the math is unit-testable
without a DB. Metrics align with the plan: win rate, expectancy, profit factor,
MAE/MFE (via trade feature windows) and loss tagging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# Loss tags are coarse, deterministic rules; they intentionally don't require
# price reconstruction (which lives in a later reconciliation feature).
TREND_REVERSAL = "trend_reversal"
SLIPPAGE = "slippage"
PREMATURE_EXIT = "premature_exit"
REGIME_MISMATCH = "regime_mismatch"


@dataclass
class Metrics:
    period: str = "30m"
    strategy_id: int | None = None
    symbol: str = "ALL"
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0.0
    expectancy: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    loss_tags: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "win_rate": round(self.win_rate, 4),
            "expectancy": round(self.expectancy, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_pnl": round(self.total_pnl, 4),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "loss_tags": self.loss_tags,
        }


def compute_metrics(trades: Iterable, period: str = "30m",
                    strategy_id: int | None = None, symbol: str = "ALL") -> Metrics:
    """Aggregate metrics over a list of closed trades."""
    trades = list(trades)
    m = Metrics(period=period, strategy_id=strategy_id, symbol=symbol)
    m.n_trades = len(trades)

    pnls = []
    wins = []
    losses = []
    tags: dict[str, int] = {}
    equity = 0.0
    peak = 0.0

    for t in trades:
        pnl = t.realized_pnl or 0.0
        pnls.append(pnl)
        equity += pnl
        peak = max(peak, equity)
        m.max_drawdown = max(m.max_drawdown, peak - equity)

        if pnl > 0:
            m.n_wins += 1
            wins.append(pnl)
        elif pnl < 0:
            m.n_losses += 1
            losses.append(pnl)
            for tag in _loss_tags(t):
                tags[tag] = tags.get(tag, 0) + 1

    m.total_pnl = sum(pnls)
    if m.n_trades:
        m.win_rate = m.n_wins / m.n_trades
        m.expectancy = m.total_pnl / m.n_trades

    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    m.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_win > 0 else 0.0)

    m.avg_win = (gross_win / m.n_wins) if m.n_wins else 0.0
    m.avg_loss = (gross_loss / m.n_losses) if m.n_losses else 0.0
    m.loss_tags = tags
    return m


def _loss_tags(trade) -> list[str]:
    """Heuristic loss tags from a closed trade's stored fields."""
    tags = []
    try:
        import json

        explicit = json.loads(trade.loss_reasons or "[]")
    except ValueError:
        explicit = []
    tags.extend(t for t in explicit if t)

    pnl = trade.realized_pnl or 0.0
    if pnl >= 0:
        return tags  # only tag losing trades

    entry = trade.entry_avg_price or 0.0
    exit_ = trade.exit_avg_price or 0.0
    if entry > 0 and exit_ > 0:
        # Closed for a loss far from entry -> trend went against the position.
        move = (exit_ - entry) / entry
        if abs(move) > 0.03:
            tags.append(TREND_REVERSAL)
    if (trade.fees or 0.0) > 0 and pnl < 0 and abs(pnl) <= trade.fees:
        # Loss is dominated by costs -> slippage/fee drag.
        tags.append(SLIPPAGE)
    return tags
