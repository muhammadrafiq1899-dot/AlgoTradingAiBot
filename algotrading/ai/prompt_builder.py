"""Deterministic prompt assembly for the AI assistant.

The assistant is strictly advisory: it sees analytics summaries, recent candle
feature windows and (when enabled) decision-log lessons + public news headlines
(read-only analytical inputs) and is asked to emit a structured
recommendation. The prompt is built entirely from local data so the same inputs
always produce the same prompt (no time-of-day drift beyond what the data itself
changes).

Two optional blocks, both **data, never instructions**:

* ``lessons`` — the assistant's own past decisions and their realized outcomes
  (``ai/decision_log.build_lessons``), so it can avoid repeating a change that
  already failed. It is memory, not a permission: nothing in it can make the
  assistant act.
* ``headlines`` — public RSS titles, rendered behind an explicit untrusted-data
  header (``ai/news.UNTRUSTED_HEADER``, mirroring chat hard rule 5). Only ever
  passed in when ``ai.news_enabled`` is set.
"""
from __future__ import annotations

from typing import Any, Sequence

from algotrading.ai.news import format_headlines
from algotrading.market.base import Candle

SYSTEM_PROMPT = (
    "You are the strategy analyst for an algorithmic spot-trading bot. "
    "You propose changes to trading strategies. "
    "You are ADVISORY ONLY: you never execute trades, never touch the "
    "exchange, and never change the live strategy yourself. "
    "Your output must be a single JSON object (no markdown fences) of the "
    "form: "
    '{"kind": "param_change|new_strategy|hypothesis|failure_analysis", '
    '"strategy_name": "ema_crossover|rsi_reversion", '
    '"content": {"params": {...}, "rationale": "...", "position_pct": 0.2}, '
    '"rationale": "one-line why"}'
)


def _fmt_float(x: Any) -> str:
    return f"{float(x):.4f}" if x is not None else "n/a"


def build_feature_window(candles: Sequence[Candle], max_rows: int = 24) -> list[str]:
    """Render the most recent candles as compact text rows.

    Uses the last `max_rows` candles (excluding the very latest, which is the
    signal candle) so the model sees the state *before* deciding.
    """
    rows = []
    for c in list(candles)[-(max_rows + 1): -1]:
        rows.append(
            f"{c.ts} {c.symbol} {c.interval} "
            f"O={_fmt_float(c.open)} H={_fmt_float(c.high)} "
            f"L={_fmt_float(c.low)} C={_fmt_float(c.close)} V={c.volume}"
        )
    return rows


def _summaries_block(summaries: Sequence[Any]) -> str:
    if not summaries:
        return "  (no analytics summaries available)"
    lines = []
    for s in summaries:
        metrics = getattr(s, "metrics_json", None)
        try:
            import json

            m = json.loads(metrics) if metrics else {}
        except (ValueError, TypeError):
            m = {}
        period = getattr(s, "period", "?")
        lines.append(
            f"  period={period} n_trades={m.get('n_trades', 0)} "
            f"win_rate={_fmt_float(m.get('win_rate'))} "
            f"total_pnl={_fmt_float(m.get('total_pnl'))} "
            f"profit_factor={_fmt_float(m.get('profit_factor'))} "
            f"max_drawdown={_fmt_float(m.get('max_drawdown'))} "
            f"loss_tags={m.get('loss_tags', {})}"
        )
    return "\n".join(lines)


def _lessons_block(lessons: Sequence[str]) -> str:
    """Render the advisory decision-log lines (deterministic, no LLM call)."""
    lines = [f"- {line}" for line in lessons if str(line or "").strip()]
    if not lines:
        return ""
    return (
        "Lessons from your own past decisions (realized outcomes; advisory "
        "memory only — you still cannot trade or apply anything):\n"
        + "\n".join(lines)
    )


def _headlines_block(headlines: Any) -> str:
    """Render news headlines as untrusted data, or "" when there are none."""
    if not headlines:
        return ""
    if isinstance(headlines, str):
        return headlines.strip()
    return format_headlines(headlines)


def build_prompt(
    symbol: str,
    interval: str,
    summaries: Sequence[Any],
    candles: Sequence[Candle],
    strategy_name: str,
    params: dict[str, Any],
    max_feature_rows: int = 24,
    lessons: Sequence[str] | None = None,
    headlines: Any = None,
) -> list[dict[str, str]]:
    """Build the chat messages for the LLM.

    Args:
        symbol: trading symbol (e.g. "BTC/USDT").
        interval: candle interval (e.g. "1h").
        summaries: AnalyticsSummary rows (period + metrics_json).
        candles: recent Candle rows (oldest -> newest).
        strategy_name: currently active strategy.
        params: current strategy params (the "parameter diff" baseline).
        lessons: decision-log lines (see ``ai/decision_log.build_lessons``) or
            None/[] to omit the block entirely.
        headlines: ``Headline`` objects, plain strings, or an already-formatted
            block. Only passed by the caller when ``ai.news_enabled`` is true —
            this function never fetches anything.
    """
    feature_rows = build_feature_window(candles, max_rows=max_feature_rows)
    feature_block = "\n".join(feature_rows) if feature_rows else "  (no candles available)"

    user_prompt = (
        f"Current active strategy: {strategy_name}\n"
        f"Current params: {params}\n"
        f"Symbol: {symbol}  Interval: {interval}\n\n"
        f"Recent analytics summaries:\n{_summaries_block(summaries)}\n\n"
        f"Recent price candles (oldest -> newest, latest excluded):\n{feature_block}\n\n"
    )

    lessons_block = _lessons_block(list(lessons or []))
    if lessons_block:
        user_prompt += f"{lessons_block}\n\n"

    headlines_block = _headlines_block(headlines)
    if headlines_block:
        user_prompt += f"{headlines_block}\n\n"

    user_prompt += (
        "Propose at most one change. If you believe the current strategy is "
        "fine, return kind=\"hypothesis\" with an empty params diff and a "
        "rationale explaining why. Return ONLY the JSON object."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
