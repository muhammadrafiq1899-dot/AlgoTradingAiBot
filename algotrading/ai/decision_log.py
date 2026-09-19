"""Advisory decision log: what the AI proposed, and what actually happened.

Every APPLIED recommendation gets one ``ai_decision_log`` row. Once its review
horizon has elapsed, ``evaluate_due`` fills in the *realized* outcome from the
closed-trade ledger (plus a benchmark move from stored candles), and
``build_lessons`` turns those rows into a few short lines that go back into the
next advisory prompt.

WHY this exists: without it the daily review proposes changes with no memory of
its own track record. With it, the model sees "the same kind of tweak on this
strategy lost 2% over a week while the market fell 0.5%" and can talk itself out
of a repeat.

Invariants it must never break:

* **Advisory only.** This module is read by ``build_lessons`` and nothing else.
  No execution code path imports it, so a lesson can never influence an order.
* **No LLM calls.** Every text here is deterministic Python over stored rows —
  cheap on a phone, and reproducible in tests.
* **Realized, not projected.** Outcomes come from closed trades inside the
  window; a window with no trades is reported as ``flat`` with ``n_trades=0``
  rather than being dressed up as a win.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from algotrading.db.models import AIDecisionLog, AIRecommendation, Candle, Trade

log = logging.getLogger(__name__)

#: Outcome vocabulary. ``pending`` means the horizon has not elapsed yet.
PENDING = "pending"
WIN = "win"
LOSS = "loss"
FLAT = "flat"

#: Default review horizon: one week is long enough for a strategy tweak to show
#: up in realized trades and short enough to keep a usable sample.
DEFAULT_HORIZON_DAYS = 7

#: |PnL%| below this counts as "flat": fees and slippage make a 0.0% window
#: meaningless to call a win.
FLAT_BAND_PCT = 0.05


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; every comparison here is UTC."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


# --- recording ---------------------------------------------------------------


def record_applied(
    session: Session,
    recommendation: AIRecommendation,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    now: datetime | None = None,
) -> AIDecisionLog | None:
    """Open a log row for an APPLIED recommendation (idempotent).

    Returns the existing row when the recommendation was already logged, or
    None when the recommendation is not APPLIED yet (nothing has been applied,
    so there is nothing to evaluate later).

    NOTE: the apply path lives in ``algotrading/store/recommendations.py``,
    which this change does not own. Rather than edit a file owned by another
    agent, the scheduler syncs applied recommendations here via
    :func:`sync_applied` — same rows, same time source, one daily tick later.
    """
    if recommendation is None or recommendation.id is None:
        return None
    if getattr(recommendation, "status", None) != "applied":
        log.debug(
            "decision log: recommendation %s is %r, not applied; not logging",
            recommendation.id,
            getattr(recommendation, "status", None),
        )
        return None

    existing = session.execute(
        select(AIDecisionLog).where(AIDecisionLog.recommendation_id == recommendation.id)
    ).scalars().first()
    if existing is not None:
        return existing

    # reviewed_at is stamped when the human approval was applied; fall back to
    # the row's creation timestamp for older rows.
    applied_at = _aware(recommendation.reviewed_at) or _aware(recommendation.ts) or (now or _utcnow())
    entry = AIDecisionLog(
        recommendation_id=recommendation.id,
        strategy_name=recommendation.strategy_name or "",
        kind=recommendation.kind or "",
        applied_at=applied_at,
        horizon_days=max(1, int(horizon_days)),
        outcome=PENDING,
    )
    session.add(entry)
    session.commit()
    log.info(
        "decision log: recorded %s on %s (horizon %sd)",
        entry.kind,
        entry.strategy_name,
        entry.horizon_days,
    )
    return entry


def sync_applied(
    session: Session,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    limit: int = 50,
) -> list[AIDecisionLog]:
    """Log every applied recommendation that has no log row yet.

    This is the hook the scheduler runs, so an approval made over Telegram is
    captured even though ``RecommendationStore.apply`` (owned elsewhere) does
    not call into this module.
    """
    logged = set(
        session.execute(
            select(AIDecisionLog.recommendation_id).where(
                AIDecisionLog.recommendation_id.is_not(None)
            )
        ).scalars()
    )
    applied = session.execute(
        select(AIRecommendation)
        .where(AIRecommendation.status == "applied")
        .order_by(AIRecommendation.ts.asc())
        .limit(limit)
    ).scalars().all()
    out: list[AIDecisionLog] = []
    for rec in applied:
        if rec.id in logged:
            continue
        entry = record_applied(session, rec, horizon_days=horizon_days)
        if entry is not None:
            out.append(entry)
    return out


# --- evaluation --------------------------------------------------------------


def _closed_trades(session: Session) -> list[Trade]:
    """All closed trades, oldest first.

    Filtering happens in Python on purpose: SQLite stores these datetimes as
    strings, so a timezone-aware bind parameter compared in SQL can silently
    miss rows. The ledger is small (a phone bot, not a market maker), and the
    whole list is loaded once per evaluation run rather than per decision.
    """
    return list(
        session.execute(
            select(Trade).where(Trade.closed_at.is_not(None)).order_by(Trade.closed_at.asc())
        ).scalars().all()
    )


def _window_trades(trades: Sequence[Trade], applied_at: datetime, until: datetime) -> list[Trade]:
    """Closed trades in the window (``closed_at`` between the two instants)."""
    return [t for t in trades if applied_at <= (_aware(t.closed_at) or applied_at) <= until]


def _strategy_pnl_pct(trades: Sequence[Trade]) -> tuple[float, float, dict[str, Any]]:
    """(pnl_pct, total_pnl, metrics) from realized closed trades.

    Percentage base is the entry cost of the trades in the window
    (``entry_qty * entry_avg_price``), so a 100-unit trade and a 1000-unit trade
    in the same window are weighted by what they actually risked. No trades ⇒
    0.0% — the window made no money, which is honest, not a guess.
    """
    total_pnl = sum(float(t.realized_pnl or 0.0) for t in trades)
    cost = sum(abs(float(t.entry_qty or 0.0) * float(t.entry_avg_price or 0.0)) for t in trades)
    wins = sum(1 for t in trades if (t.realized_pnl or 0.0) > 0)
    losses = sum(1 for t in trades if (t.realized_pnl or 0.0) < 0)
    pnl_pct = (total_pnl / cost * 100.0) if cost > 0 else 0.0
    metrics = {
        "n_trades": len(trades),
        "n_wins": wins,
        "n_losses": losses,
        "total_pnl": round(total_pnl, 4),
        "entry_cost": round(cost, 4),
    }
    return pnl_pct, total_pnl, metrics


def _benchmark_pct(
    session: Session, symbol: str, start: datetime, end: datetime, interval: str = "1h"
) -> float | None:
    """Move (%) of ``symbol`` between the close at/before ``start`` and ``end``.

    Returns None when the candles are not stored (the symbol was not tracked, or
    the window predates the candle history) — "if the candles are available",
    never a made-up number.
    """
    if not symbol:
        return None
    start_ms, end_ms = _to_ms(start), _to_ms(end)
    stmt = select(Candle).where(Candle.symbol == symbol, Candle.ts <= end_ms)
    if interval:
        # Candles are keyed by interval; mixing timeframes would compare a 1h
        # close with a 1d close.
        stmt = stmt.where(Candle.interval == interval)
    rows = session.execute(stmt.order_by(Candle.ts.asc())).scalars().all()
    if not rows:
        return None
    base = None
    last = None
    for row in rows:
        if row.ts <= start_ms:
            base = row
        else:
            last = row  # ordered ascending: the final one wins
    if base is None or last is None or base.ts >= last.ts or not base.close:
        return None
    return (last.close - base.close) / base.close * 100.0


def _outcome(pnl_pct: float) -> str:
    if pnl_pct > FLAT_BAND_PCT:
        return WIN
    if pnl_pct < -FLAT_BAND_PCT:
        return LOSS
    return FLAT


def evaluate_due(
    session: Session,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    now: datetime | None = None,
    *,
    benchmark_symbol: str | None = None,
    benchmark_interval: str = "1h",
) -> list[AIDecisionLog]:
    """Fill in every log row whose horizon has elapsed.

    Rows still inside their horizon are left ``pending`` (the whole point of the
    horizon). Benchmark symbol defaults to the most-traded symbol in the window,
    then to any symbol present in the candles table — callers with settings
    (the scheduler) pass their primary market symbol explicitly.
    """
    now = _aware(now) or _utcnow()
    rows = session.execute(
        select(AIDecisionLog)
        .where(AIDecisionLog.outcome == PENDING)
        .order_by(AIDecisionLog.applied_at.asc())
    ).scalars().all()
    closed = _closed_trades(session)

    evaluated: list[AIDecisionLog] = []
    for row in rows:
        applied_at = _aware(row.applied_at)
        if applied_at is None:
            continue
        horizon = max(1, int(row.horizon_days or horizon_days))
        due_at = applied_at + timedelta(days=horizon)
        if due_at > now:
            continue  # horizon not elapsed — leave it pending

        trades = _window_trades(closed, applied_at, due_at)
        pnl_pct, _total_pnl, metrics = _strategy_pnl_pct(trades)
        symbol = benchmark_symbol or _infer_benchmark(session, trades)
        benchmark_pct = _benchmark_pct(session, symbol, applied_at, due_at, benchmark_interval)
        if symbol:
            metrics["benchmark_symbol"] = symbol

        row.evaluated_at = now
        row.pnl_pct = round(pnl_pct, 4)
        row.benchmark_pct = None if benchmark_pct is None else round(benchmark_pct, 4)
        row.outcome = _outcome(pnl_pct)
        row.metrics_json = json.dumps(metrics)
        row.reflection = _reflection(row)
        evaluated.append(row)

    if evaluated:
        session.commit()
        log.info("decision log: evaluated %d decision(s)", len(evaluated))
    return evaluated


def _infer_benchmark(session: Session, trades: Sequence[Trade]) -> str:
    """Most-traded symbol in the window, else the first symbol we hold candles for."""
    if trades:
        counts: dict[str, int] = {}
        for trade in trades:
            symbol = trade.symbol or ""
            if symbol:
                counts[symbol] = counts.get(symbol, 0) + 1
        if counts:
            return max(sorted(counts), key=lambda s: counts[s])
    row = session.execute(select(Candle.symbol).order_by(Candle.symbol.asc()).limit(1)).first()
    return row[0] if row else ""


# --- lessons + reflection ----------------------------------------------------


def _reflection(row: AIDecisionLog) -> str:
    """One deterministic sentence about a decision's outcome (no LLM call)."""
    outcome = row.outcome
    horizon = max(1, int(row.horizon_days or DEFAULT_HORIZON_DAYS))
    pnl = row.pnl_pct if row.pnl_pct is not None else 0.0
    kind = (row.kind or "change").replace("_", " ")
    strategy = row.strategy_name or "the active strategy"
    if outcome == PENDING:
        return f"{kind} on {strategy} is still inside its {horizon}-day window."
    verdict = {
        WIN: "worked",
        LOSS: "did not work",
        FLAT: "made no measurable difference",
    }.get(outcome, "was inconclusive")
    tail = ""
    if row.benchmark_pct is not None:
        comparison = "outperformed" if pnl > row.benchmark_pct else "underperformed"
        tail = (
            f" — it {comparison} the benchmark by "
            f"{abs(pnl - row.benchmark_pct):.1f} points"
        )
    return (
        f"The {kind} on {strategy} {verdict}: {pnl:+.1f}% over {horizon} days{tail}. "
        "Treat it as one sample, not proof."
    )


def reflect(session: Session, limit: int = 20) -> list[AIDecisionLog]:
    """(Re)write the deterministic reflection on the most recently evaluated rows.

    Kept separate from ``evaluate_due`` so a wording change can backfill older
    rows without re-running the evaluation, and so nothing here needs a network
    call or a model.
    """
    rows = session.execute(
        select(AIDecisionLog)
        .where(AIDecisionLog.evaluated_at.is_not(None))
        .order_by(AIDecisionLog.evaluated_at.desc())
        .limit(max(1, int(limit)))
    ).scalars().all()

    changed: list[AIDecisionLog] = []
    for row in rows:
        text = _reflection(row)
        if row.reflection != text:
            row.reflection = text
            changed.append(row)
    if changed:
        session.commit()
    return changed


def _fmt_pct(value: float | None) -> str:
    return "nan" if value is None else f"{value:+.1f}%"


def build_lessons(session: Session, limit: int = 5) -> list[str]:
    """Short deterministic lesson lines for the advisory prompt.

    Shape: ``2026-09-01 param_change on ema_crossover: 7d -2.1% vs BTC/USDT
    -0.5% (loss)``. Newest first, capped by ``limit``. Deterministic: same rows
    in, same lines out — no timestamps of "now", no LLM.
    """
    if limit <= 0:
        return []
    rows = session.execute(
        select(AIDecisionLog)
        .where(AIDecisionLog.evaluated_at.is_not(None))
        .order_by(AIDecisionLog.applied_at.desc())
        .limit(int(limit))
    ).scalars().all()

    lessons: list[str] = []
    for row in rows:
        applied = _aware(row.applied_at)
        stamp = applied.strftime("%Y-%m-%d") if applied else "unknown date"
        horizon = max(1, int(row.horizon_days or DEFAULT_HORIZON_DAYS))
        pnl = row.pnl_pct if row.pnl_pct is not None else 0.0
        benchmarks = _benchmark_labels(row)
        comparison = f" vs {benchmarks}" if benchmarks else ""
        lessons.append(
            f"{stamp} {row.kind or 'change'} on {row.strategy_name or '-'}: "
            f"{horizon}d {_fmt_pct(pnl)}{comparison} ({row.outcome})"
        )
    return lessons


def _benchmark_labels(row: AIDecisionLog) -> str:
    """``BTC/USDT -0.5%`` when a benchmark move is known, else ""."""
    if row.benchmark_pct is None:
        return ""
    try:
        symbol = json.loads(row.metrics_json or "{}").get("benchmark_symbol") or ""
    except (ValueError, TypeError):
        symbol = ""
    return f"{symbol} {_fmt_pct(row.benchmark_pct)}".strip()
