"""Chat orchestrator: the LLM as the bridge between the user and the bot.

Plain-text Telegram messages (from allowlisted users) are answered by the LLM,
which has four tools:

  - get_status      current bot state (mode, strategy, positions, activity, analytics)
  - get_market      recent candles for research
  - backtest        shadow-backtest a strategy+params on stored candles (deterministic)
  - propose_change  create a PENDING strategy-change recommendation

Safety model (unchanged): the agent NEVER trades, never changes the active
strategy, risk limits, or mode. The only way to alter the bot is
`propose_change`, which persists a PENDING recommendation that requires the
user's explicit Approve tap (existing approval flow). The trading engine stays
fully deterministic; the LLM only reads state and proposes.

The agent is a plain sync function (`run_agent`) so the Telegram handler can
run it in a worker thread without blocking the event loop.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from sqlalchemy import select

from algotrading.ai.client import AIClient, RecommendationError
from algotrading.config import Settings
from algotrading.db.models import (
    AnalyticsSummary,
    AIRecommendation,
    Position,
    Strategy,
    Trade,
    TradeIntent,
)
from algotrading.market.candles import CandleStore
from algotrading.store.recommendations import create_pending_recommendation

log = logging.getLogger(__name__)

MAX_TOOL_TURNS = 3

CHAT_SYSTEM_PROMPT = """You are the conversational control layer of AlgoTrading, an algorithmic spot-trading bot running on Binance. The user talks to you in natural language; you are the bridge between the user and the bot.

You have these tools (call ONE per turn):
- get_status: current bot state. No args.
- get_market: recent candles for research. Args: {"symbol": "BTC/USDT" (optional), "interval": "1h"|"1m" (optional)}.
- backtest: replay historical candles through a strategy. Args: {"strategy_name": "...", "params": {...}}.
- propose_change: create a strategy-change proposal the user must approve. Args: {"kind": "param_change"|"new_strategy"|"hypothesis"|"failure_analysis", "strategy_name": "...", "params": {...}, "rationale": "..."}.

Known strategies and params:
- ema_crossover: fast_period (int), slow_period (int, > fast), position_pct (float 0.01-0.5)
- rsi_mean_reversion: period (int), oversold (float), overbought (float, > oversold), position_pct (float 0.01-0.5)

HARD RULES:
1. You are ADVISORY ONLY. You never trade, never change the active strategy, risk limits, or mode. The ONLY way to change anything is propose_change, which stays PENDING until the user taps Approve.
2. Before proposing a param change, run a backtest and mention its result in your reply.
3. Answer questions about the bot from get_status / get_market / backtest results. Be concise, plain text, no markdown, no emoji spam.
4. Respond ONLY with a JSON object: {"action": "reply", "text": "..."} or {"action": "tool", "name": "...", "args": {...}}."""


def _active_strategy(session) -> Strategy | None:
    return session.execute(
        select(Strategy)
        .where(Strategy.status == "active")
        .order_by(Strategy.version.desc())
        .limit(1)
    ).scalar_one_or_none()


def _latest_analytics(session) -> AnalyticsSummary | None:
    return session.execute(
        select(AnalyticsSummary).order_by(AnalyticsSummary.ts.desc()).limit(1)
    ).scalar_one_or_none()


# --- tools ------------------------------------------------------------------

def tool_get_status(session, settings: Settings) -> dict[str, Any]:
    """Compact snapshot of bot state, used both as chat context and as a tool."""
    active = _active_strategy(session)
    positions = session.execute(
        select(Position).where(Position.qty > 0)
    ).scalars().all()
    intents = session.execute(
        select(TradeIntent).order_by(TradeIntent.ts.desc()).limit(5)
    ).scalars().all()
    trades = session.execute(
        select(Trade).order_by(Trade.closed_at.desc()).limit(5)
    ).scalars().all()
    summary = _latest_analytics(session)
    metrics = {}
    if summary is not None:
        try:
            metrics = json.loads(summary.metrics_json or "{}")
        except ValueError:
            metrics = {}
    return {
        "mode": settings.mode,
        "symbols": settings.market.symbols,
        "active_strategy": {
            "name": active.name,
            "version": active.version,
            "params": json.loads(active.params or "{}"),
        } if active else None,
        "open_positions": [
            {"symbol": p.symbol, "qty": p.qty, "avg_price": p.avg_price}
            for p in positions
        ],
        "recent_intents": [
            {"symbol": i.symbol, "side": i.side, "status": i.status, "qty": i.qty}
            for i in intents
        ],
        "recent_trades": [
            {"symbol": t.symbol, "pnl": t.realized_pnl, "closed_at": str(t.closed_at)}
            for t in trades
        ],
        "latest_analytics": metrics,
    }


def tool_get_market(session, settings: Settings, symbol: str | None = None,
                    interval: str | None = None) -> str:
    """Recent candles as compact text rows (research input for the LLM)."""
    from algotrading.ai.prompt_builder import build_feature_window
    from algotrading.market.base import Candle

    sym = symbol or (settings.market.symbols or ["BTC/USDT"])[0]
    iv = interval or ("1h" if "1h" in settings.market.intervals else "1m")
    rows = CandleStore(session).get(sym, iv, limit=60)
    candles = [
        Candle(symbol=r.symbol, interval=r.interval, ts=r.ts, open=r.open,
               high=r.high, low=r.low, close=r.close, volume=r.volume)
        for r in rows
    ]
    if not candles:
        return f"no candles stored for {sym} {iv}"
    return f"{sym} {iv} ({len(candles)} candles):\n" + "\n".join(
        build_feature_window(candles, max_rows=24)
    )


def tool_backtest(session, settings: Settings, strategy_name: str,
                  params: dict[str, Any]) -> str:
    """Deterministic shadow backtest on stored candles; returns a summary."""
    from algotrading.backtest.runner import run_backtest
    from algotrading.market.base import Candle

    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    sym = (settings.market.symbols or ["BTC/USDT"])[0]
    iv = "1h" if "1h" in settings.market.intervals else "1m"
    rows = CandleStore(session).get(sym, iv, limit=500)
    candles = [
        Candle(symbol=r.symbol, interval=r.interval, ts=r.ts, open=r.open,
               high=r.high, low=r.low, close=r.close, volume=r.volume)
        for r in rows
    ]
    if not candles:
        return "no candles stored — cannot backtest yet"
    result = run_backtest(candles, strategy_name, dict(params))
    return (
        f"Backtest {strategy_name} on {sym} {iv}: {result.n_trades} trades, "
        f"win_rate={result.win_rate:.2%}, total_pnl={result.total_pnl:.2f}, "
        f"max_drawdown={result.max_drawdown:.2f}, final_balance={result.final_balance:.2f}"
    )


def tool_propose_change(session, settings: Settings, **args: Any) -> dict[str, Any]:
    """Create a PENDING recommendation (never applies anything)."""
    rec = create_pending_recommendation(
        session,
        kind=args["kind"],
        strategy_name=args["strategy_name"],
        params=args["params"],
        rationale=args.get("rationale", ""),
    )
    return {
        "proposal_id": rec.id,
        "kind": rec.kind,
        "strategy_name": rec.strategy_name,
        "rationale": rec.rationale,
    }


# All tools share the uniform call signature: fn(session, settings, **args).
# Unknown/extra args surface as TypeError -> fed back to the LLM as a tool error.
TOOLS: dict[str, Callable[..., Any]] = {
    "get_status": tool_get_status,
    "get_market": tool_get_market,
    "backtest": tool_backtest,
    "propose_change": tool_propose_change,
}


# --- agent loop -------------------------------------------------------------

def run_agent(session_factory: Callable[[], Any], settings: Settings, user_text: str,
              client: AIClient | None = None, max_turns: int = MAX_TOOL_TURNS) -> dict[str, Any]:
    """Answer a plain-text user message through the LLM orchestrator.

    Returns {"text": str, ...} — plus proposal_id/kind/strategy_name/rationale
    when a propose_change tool created a PENDING recommendation, so the caller
    can attach the approval keyboard. Sync: call from a worker thread.

    Never raises for LLM/tool failures — returns a friendly reply instead.
    """
    if not settings.ai.enabled:
        return {"text": "🤖 AI assistant is not configured (set AI_API_KEY in .env and restart)."}
    client = client or AIClient(config=settings.ai)

    session = session_factory()
    try:
        context = tool_get_status(session, settings)
        context_text = "Current bot state:\n" + json.dumps(context, default=str)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {"role": "user", "content": f"{context_text}\n\nUser message: {user_text}"},
        ]
        proposal: dict[str, Any] | None = None
        for _ in range(max_turns):
            data = client.complete_json(messages)
            action = data.get("action")
            if action == "reply":
                out: dict[str, Any] = {
                    "text": str(data.get("text", "")) or "(empty reply)"
                }
                if proposal is not None:
                    out.update(proposal)  # proposal_id/kind/... for the approval card
                return out
            if action != "tool":
                raise RecommendationError(f"unexpected action {action!r}")

            name = data.get("name", "")
            args = data.get("args") or {}
            fn = TOOLS.get(name)
            if fn is None:
                result = f"unknown tool {name!r}; known tools: {sorted(TOOLS)}"
            else:
                try:
                    if not isinstance(args, dict):
                        raise TypeError("args must be a JSON object")
                    result = fn(session, settings, **args)
                except Exception as exc:  # noqa: BLE001 - tool errors are LLM feedback
                    log.warning("chat tool %s failed: %s", name, exc)
                    result = f"tool {name} failed: {exc}"
            if isinstance(result, dict) and result.get("proposal_id") is not None:
                proposal = result
            messages.append({"role": "assistant", "content": json.dumps(data, default=str)})
            messages.append({"role": "user", "content": f"Tool result: {result}"})
        out = {"text": "🤖 I couldn't finish in time — please try again or rephrase."}
        if proposal is not None:
            out.update(proposal)
        return out
    except RecommendationError as exc:
        log.warning("chat agent LLM error: %s", exc)
        return {"text": f"🤖 The LLM call failed: {exc}. Check AI_API_KEY / AI_BASE_URL / AI_MODEL."}
    finally:
        session.close()