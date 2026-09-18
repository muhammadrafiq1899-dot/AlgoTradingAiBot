"""Telegram message/formatting helpers and inline keyboards.

Keeps presentation logic (concise, one-line-per-item status blocks, inline
approval keyboards) out of the command handlers so handlers stay thin and
testable.
"""
from __future__ import annotations

import html
import math
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
    lines.append(f"<b>AlgoTrading — mode: {mode}</b>")
    if active:
        lines.append(f"Active strategy: {active.name} v{active.version}")

    # Positions with live mark-to-market
    if positions:
        lines.append("\n<b>Positions</b>")
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
        lines.append("\n<b>Recent activity</b>")
        lines.extend(_i(x) for x in intents[:8])

    # Closed trades today
    if trades:
        lines.append("\n<b>Today's closed trades</b>")
        lines.extend(_t(x) for x in trades[:5])

    return "\n".join(lines)


def format_risk(settings, risk_cfg, session) -> str:
    """Compact risk-limit summary for /risk."""
    lines = [
        "<b>Risk limits</b>",
        f"• Risk per trade: {risk_cfg.risk_per_trade_pct}%",
        f"• Max position: {risk_cfg.max_position_pct}% of equity",
        f"• Max open positions: {risk_cfg.max_open_positions}",
        f"• Cooldown: {risk_cfg.cooldown_seconds}s",
        f"• Slippage model: {risk_cfg.slippage_pct}%",
        f"• Mode: {settings.mode}",
    ]
    trailing = getattr(risk_cfg, "trailing_stop_pct", 0) or 0
    lines.append(
        f"• Trailing stop: {trailing}% from the high"
        if trailing > 0
        else "• Trailing stop: off"
    )
    open_positions = session.execute(
        select(Position).where(Position.qty > 0)
    ).scalars().all()
    lines.append(f"• Open positions now: {len(open_positions)}")
    return "\n".join(lines)


def format_strategies(strategies: list[Strategy]) -> str:
    if not strategies:
        return "No strategies configured yet."
    return "<b>Strategies</b>\n" + "\n".join(_s(s) for s in strategies)


# --- strategy catalog (/strategies) -----------------------------------------

# Telegram caps messages at 4096 chars; leave margin for the header line.
CATALOG_TEXT_LIMIT = 3800


def _one_line(text: str, limit: int = 160) -> str:
    collapsed = " ".join((text or "").split())
    if not collapsed:
        return ""
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1].rstrip() + "…"


def _param_line(param) -> str:
    """Render one catalog ParamSpec as an HTML line fragment."""
    bits = [f"{html.escape(param.name)}: {html.escape(param.type)}"]
    if param.default is not None and not isinstance(param.default, (list, dict)):
        bits.append(f"default {html.escape(str(param.default))}")
    bounds = param.range_text()
    if bounds:
        bits.append(f"({html.escape(bounds)})")
    if param.enum:
        bits.append("one of " + "/".join(html.escape(str(v)) for v in param.enum))
    return " · ".join(bits)


def _score_of(result) -> float | None:
    """Risk-adjusted score of a BacktestResult, or None if unscorable."""
    if result is None:
        return None
    try:
        return _risk_adjusted(result)
    except Exception:  # noqa: BLE001 - scoring must never break the catalog
        return None


def _fmt_score(value: float | None) -> str:
    """Compact risk-adjusted score for the catalog list."""
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "\u221e" if value > 0 else "-\u221e"
    return f"{value:.2f}"


def rank_by_score(entries, scores: dict | None = None) -> list:
    """Order entries best-first by risk-adjusted score; unscored ones last.

    Stable, so ties and the no-scores case keep the catalog's own order. Returns
    a new list; callers should use the same ordering for text and buttons so the
    ranking and the keyboard agree.
    """
    if not scores:
        return list(entries)

    def key(entry):
        score = _score_of(scores.get(entry.name))
        # Unscored sorts last; otherwise higher score sorts first.
        return (score is None, -(score if score is not None else 0.0))

    return sorted(entries, key=key)


def format_strategy_catalog(entries, scores: dict | None = None) -> str:
    """Render the live strategy catalog (built-ins + plugins) for /strategies.

    `entries` is a list of `algotrading.strategy.catalog.StrategyEntry`, so this
    stays in sync with whatever the registry can actually build right now. When
    `scores` (name → BacktestResult) is supplied, each entry is numbered and
    labelled with its risk-adjusted score; pass entries through `rank_by_score`
    first for a ranked list.
    """
    if not entries:
        return "No strategies available."
    scores = scores or {}

    header = "<b>Available strategies</b>"
    if scores:
        header += " \u2014 ranked by PnL \u00f7 max drawdown"
    lines = [header]

    for index, entry in enumerate(entries, 1):
        lines.append("")
        label = html.escape(entry.name)
        if scores:
            label = f"#{index} {label}"
            score = _fmt_score(_score_of(scores.get(entry.name)))
            lines.append(f"<b>{label}</b> [{html.escape(entry.kind)}] \u00b7 score {score}")
        else:
            lines.append(f"<b>{label}</b> [{html.escape(entry.kind)}]")
        summary = _one_line(entry.description)
        if summary:
            lines.append(summary)
        if entry.params:
            lines.extend(f"  \u2022 {_param_line(p)}" for p in entry.params)
        else:
            lines.append("  \u2022 no params")
    return _fit(lines)


# Telegram caps callback_data at 64 bytes; skip anything that can't fit.
CALLBACK_DATA_LIMIT = 64
# Keep the keyboard to a sane height even when many plugins are loaded.
BACKTEST_BUTTON_LIMIT = 24


def backtest_keyboard(entries) -> InlineKeyboardMarkup | None:
    """One button per strategy; tapping it backtests the strategy's params.

    Returns None when there is nothing to show, so callers can pass the result
    straight to ``reply_text(reply_markup=...)``.
    """
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for entry in entries[:BACKTEST_BUTTON_LIMIT]:
        data = f"bt:{entry.name}"
        if len(data.encode("utf-8")) > CALLBACK_DATA_LIMIT:
            continue
        label = f"\U0001f4c8 {entry.name}"
        if len(label) > 40:
            label = label[:39] + "…"
        row.append(InlineKeyboardButton(label, callback_data=data))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows) if rows else None


def _fit(lines: list[str], limit: int = CATALOG_TEXT_LIMIT) -> str:
    """Join lines, truncating so the message stays under Telegram's cap."""
    out: list[str] = []
    size = 0
    for line in lines:
        if size + len(line) + 1 > limit:
            out.append("… (list truncated)")
            break
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)


# --- backtest result / comparison -------------------------------------------

def _param_value(value) -> str:
    """Render a parameter value compactly for a monospace table."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, dict)):
        text = repr(value)
        return text if len(text) <= 40 else text[:39] + "…"
    return str(value)


def _changed_params(current: dict, default: dict) -> list[tuple[str, str, str]]:
    """Only the params that differ, as (name, current, default)."""
    keys = sorted(set(current) | set(default))
    return [
        (key, _param_value(current.get(key)), _param_value(default.get(key)))
        for key in keys
        if current.get(key) != default.get(key)
    ]


def _mono_table(rows: list[tuple[str, ...]], headers: tuple[str, ...]) -> str:
    """Align rows into a Telegram <pre> block (monospace, so columns line up)."""
    first_width = max(len(headers[0]), *(len(r[0]) for r in rows)) if rows else len(headers[0])
    rest_width = max(
        [len(h) for h in headers[1:]]
        + [len(cell) for row in rows for cell in row[1:]]
    )
    lines = [f"{headers[0]:<{first_width}}  " + "  ".join(f"{h:>{rest_width}}" for h in headers[1:])]
    for row in rows:
        lines.append(
            f"{row[0]:<{first_width}}  " + "  ".join(f"{cell:>{rest_width}}" for cell in row[1:])
        )
    return "<pre>" + html.escape("\n".join(lines)) + "</pre>"


def _risk_adjusted(result) -> float | None:
    """PnL earned per unit of max drawdown (recovery factor).

    This is the risk-adjusted score the verdict uses, so a run that earns more
    but suffers a much deeper drawdown does not automatically "win".

    Returns None when there is nothing to score (no trades / flat run). A
    profitable run with zero drawdown scores +inf; a losing one with zero
    drawdown scores -inf (a safety net — a realized loss implies drawdown).
    """
    if result.max_drawdown and result.max_drawdown > 0:
        return result.total_pnl / result.max_drawdown
    if not result.total_pnl:
        return None
    return math.inf if result.total_pnl > 0 else -math.inf


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    if math.isinf(value):
        return "\u221e" if value > 0 else "-\u221e"
    return f"{value:.2f}"


def _risk_verdict(current_result, default_result) -> str:
    """Verdict comparing the two runs on risk-adjusted terms, not raw PnL."""
    metric = "risk-adjusted (PnL \u00f7 max drawdown)"
    cur = _risk_adjusted(current_result)
    dflt = _risk_adjusted(default_result)

    if cur is None and dflt is None:
        return f"{metric}: no trades in either run \u2014 nothing to compare"
    if cur is None:
        return f"{metric}: current n/a, default {_fmt_ratio(dflt)} \u2014 default is the only scored run"
    if dflt is None:
        return f"{metric}: current {_fmt_ratio(cur)}, default n/a \u2014 current is the only scored run"
    if cur == dflt:
        return f"{metric}: both {_fmt_ratio(cur)} \u2014 equal risk-adjusted"

    winner = "current" if cur > dflt else "default"
    lines = [
        f"{metric}: current {_fmt_ratio(cur)} \u00b7 default {_fmt_ratio(dflt)}",
        f"<b>{winner} is better risk-adjusted</b>",
    ]
    # Call out the case that motivates risk-adjustment: higher raw PnL, worse risk.
    cur_pnl, dflt_pnl = current_result.total_pnl, default_result.total_pnl
    if cur_pnl != dflt_pnl:
        pnl_winner = "current" if cur_pnl > dflt_pnl else "default"
        if pnl_winner != winner:
            other = "default" if winner == "current" else "current"
            lines[-1] += f" \u2014 {other} has higher raw PnL but a worse drawdown"
    return "\n".join(lines)


def _metric_values(result) -> list[tuple[str, str]]:
    """(label, formatted value) for one backtest result."""
    return [
        ("trades", str(result.n_trades)),
        ("win rate", f"{result.win_rate:.1%}"),
        ("total pnl", f"{result.total_pnl:.2f}"),
        ("max drawdown", f"{result.max_drawdown:.2f}"),
        ("pnl/drawdown", _fmt_ratio(_risk_adjusted(result))),
        ("final balance", f"{result.final_balance:.2f}"),
    ]


def _comparison_rows(current_result, default_result) -> list[tuple[str, str, str]]:
    """(label, current, default) for the side-by-side results table."""
    return [
        (label, cur, dflt)
        for (label, cur), (_, dflt) in zip(
            _metric_values(current_result), _metric_values(default_result)
        )
    ]


def format_backtest_comparison(
    name: str,
    symbol: str,
    interval: str,
    n_candles: int,
    current_params: dict,
    default_params: dict,
    current_result,
    default_result,
) -> str:
    """Side-by-side replay of a strategy's current params vs catalog defaults.

    When the two param sets are identical (the common case for a seeded built-in
    or an unreleased plugin) there is nothing to compare, so a single result is
    shown with a note instead of a duplicated table.
    """
    header = (
        f"<b>Backtest: {html.escape(name)}</b>\n"
        f"{html.escape(symbol)} {html.escape(interval)} · {n_candles} candles"
    )

    changed = _changed_params(current_params, default_params)
    if not changed:
        table = _mono_table(_metric_values(current_result), ("metric", "current"))
        return f"{header}\n\n{table}\n\n<i>current params match the catalog defaults</i>"

    params_table = _mono_table(changed, ("param", "current", "default"))
    results_table = _mono_table(
        _comparison_rows(current_result, default_result),
        ("metric", "current", "default"),
    )

    verdict = _risk_verdict(current_result, default_result)

    return (
        f"{header}\n\n"
        f"<b>Params</b> (current vs default; only differences)\n{params_table}\n\n"
        f"<b>Results</b>\n{results_table}\n\n{verdict}"
    )


def format_summary(summaries, session) -> str:
    """Analytics summaries (M5); placeholder until service lands."""
    if not summaries:
        return "No analytics summaries yet.\n\nUse `/summary` again after a few trades."
    lines = ["<b>Analytics summaries</b>"]
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


#: How the bot should read a batch of images. Mirrors the ``imgflow:<token>:<mode>``
#: callback actions handled in ``telegram/commands.py``.
IMAGE_FLOW_MODES = ("combine", "separate")


def image_flow_keyboard(token: str) -> InlineKeyboardMarkup:
    """Choice buttons for a batch of images: one strategy, or one per image.

    ``token`` identifies the buffered uploads (callback data is capped at 64
    bytes, so paths cannot travel through it).
    """
    keyboard = [
        [InlineKeyboardButton(
            "🧩 One strategy from all",
            callback_data=f"imgflow:{token}:combine")],
        [InlineKeyboardButton(
            "🧱 Separate strategies → ensemble",
            callback_data=f"imgflow:{token}:separate")],
    ]
    return InlineKeyboardMarkup(keyboard)



