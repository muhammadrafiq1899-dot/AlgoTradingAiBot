"""Telegram message/formatting helpers and inline keyboards.

Keeps presentation logic (concise, one-line-per-item status blocks, inline
approval keyboards) out of the command handlers so handlers stay thin and
testable.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from algotrading.db.models import Position, Strategy, Trade, TradeIntent


def _fmt_ts(ts) -> str:
    if ts is None:
        return "-"
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts, tz=timezone.utc)
    elif ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone().strftime("%m-%d %H:%M")


def _fmt_price(x, digits: int = 2) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _s(row: Strategy) -> str:
    try:
        import json

        params = json.loads(row.params or "{}")
    except ValueError:
        params = {}
    param_str = ", ".join(f"{k}={v}" for k, v in params.items()) or "-"
    return f"• {row.name} v{row.version} [{row.status}] — {param_str}"


def _i(row: TradeIntent) -> str:
    digits = 6 if row.symbol.endswith("/USDT") and row.qty < 1 else 2
    side_icon = "▲" if row.side == "buy" else "▼"
    return (
        f"{side_icon} {row.symbol} {row.side.upper()} qty={_fmt_price(row.qty, digits)}"
        f" @ {_fmt_price(row.avg_fill_price or row.ref_price)} [{row.status}] {_fmt_ts(row.ts)}"
    )


def _t(row: Trade) -> str:
    pnl = row.realized_pnl or 0.0
    icon = "🟢" if pnl >= 0 else "🔴"
    return (
        f"{icon} {row.symbol} {_fmt_price(row.entry_qty, 6)} @ "
        f"{_fmt_price(row.entry_avg_price)} → {_fmt_price(row.exit_avg_price)} "
        f"pnl {_fmt_price(pnl)} ({_fmt_ts(row.closed_at)})"
    )


def _p(row: Position) -> str:
    return (
        f"• {row.symbol}: {_fmt_price(row.qty, 6)} @ {_fmt_price(row.avg_price)} "
        f"(uPNL ~{_fmt_price(_unrealized(row))})"
    )


def _unrealized(pos: Position, mark: float | None = None) -> float:
    if mark is None or pos.qty <= 0:
        return 0.0
    return (mark - pos.avg_price) * pos.qty


def format_status(
    settings,
    session,
    strategies: list[Strategy],
    positions: list[Position],
    intents: list[TradeIntent],
    trades: list[Trade],
    prices: dict[str, float] | None = None,
) -> str:
    """Build the /status block: mode, active strategy, positions, recent activity."""
    prices = prices or {}
    lines: list[str] = []

    active = next((s for s in strategies if s.status == "active"), None)
    mode = settings.mode if hasattr(settings, "mode") else "paper"
    lines.append(f"*AlgoTrading — mode: {mode}*")
    if active:
        lines.append(f"Active strategy: {active.name} v{active.version}")

    # Positions with live mark-to-market
    if positions:
        lines.append("\n*Positions*")
        for p in positions:
            mark = prices.get(p.symbol)
            u = _unrealized(p, mark)
            lines.append(
                f"• {p.symbol}: {_fmt_price(p.qty, 6)} @ {_fmt_price(p.avg_price)}"
                + (f" (uPNL {_fmt_price(u)})" if mark else "")
            )
    else:
        lines.append("\nNo open positions")

    # Recent fills / intents
    if intents:
        lines.append("\n*Recent activity*")
        lines.extend(_i(x) for x in intents[:8])

    # Closed trades today
    if trades:
        lines.append("\n*Today's closed trades*")
        lines.extend(_t(x) for x in trades[:5])

    return "\n".join(lines)


def format_risk(settings, risk_cfg, session) -> str:
    """Compact risk-limit summary for /risk."""
    lines = [
        "*Risk limits*",
        f"• Risk per trade: {risk_cfg.risk_per_trade_pct}%",
        f"• Max position: {risk_cfg.max_position_pct}% of equity",
        f"• Max open positions: {risk_cfg.max_open_positions}",
        f"• Cooldown: {risk_cfg.cooldown_seconds}s",
        f"• Slippage model: {risk_cfg.slippage_pct}%",
        f"• Mode: {settings.mode}",
    ]
    open_positions = session.execute(
        select(Position).where(Position.qty > 0)
    ).scalars().all()
    lines.append(f"• Open positions now: {len(open_positions)}")
    return "\n".join(lines)


def format_strategies(strategies: list[Strategy]) -> str:
    if not strategies:
        return "No strategies configured yet."
    return "*Strategies*\n" + "\n".join(_s(s) for s in strategies)


def format_summary(summaries, session) -> str:
    """Analytics summaries (M5); placeholder until service lands."""
    if not summaries:
        return "No analytics summaries yet.\n\nUse `/summary` again after a few trades."
    lines = ["*Analytics summaries*"]
    for s in summaries:
        try:
            import json

            m = json.loads(s.metrics_json or "{}")
        except ValueError:
            m = {}
        lines.append(
            f"• {s.period} [{s.strategy_id or '-'}]: "
            f"win_rate={m.get('win_rate', '-')} expectancy={m.get('expectancy', '-')}"
        )
    return "\n".join(lines)


# --- Inline approval keyboards (used by M6 recommendation flow) ---

def approval_keyboard(recommendation_id: int) -> InlineKeyboardMarkup:
    """Inline ✅/❌ buttons for approving/rejecting an AI recommendation."""
    keyboard = [
        [
            InlineKeyboardButton("✅ Approve", callback_data=f"approve:{recommendation_id}"),
            InlineKeyboardButton("❌ Reject", callback_data=f"reject:{recommendation_id}"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def escape(text: str) -> str:
    """Escape Telegram Markdown special characters for plain text."""
    for ch in ("_", "*", "[", "]", "(", ")", "~", "`", ">", "#", "+", "-", "=", "|", "{", "}", ".", "!"):
        text = text.replace(ch, "\\" + ch)
    return text
