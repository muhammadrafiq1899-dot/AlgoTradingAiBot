"""Exporters: turn DB state into CSV/JSON files a human or another tool can read.

Why this exists: the bot's state is locked inside SQLite on a phone. Getting it
out used to mean copying the whole database. These exporters are read-only,
pure-Python (stdlib ``csv``/``json`` only) and take a *path*, so the same code
serves the ``algobot export`` CLI, ``scripts/export_data.py``, and the
``/export/...`` HTTP endpoints (which reuse the text builders below instead of
touching the disk at all).

Rules kept on purpose:

* **Read-only.** Nothing here writes to the database.
* **Never outside the given path.** One file per call, written via a temp file
  in the same directory and renamed onto the target — an interrupted export
  leaves no half-file and cannot clobber anything else.
* **ISO-8601 timestamps, UTC.** SQLite hands back naive datetimes; treating them
  as UTC (rather than exporting them bare) keeps the timezone unambiguous.
* **Deterministic order.** Trades are emitted oldest -> newest so an equity
  curve and a diff of two exports line up.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from algotrading.db.models import (
    AnalyticsSummary,
    AIRecommendation,
    Position,
    Signal,
    Strategy,
    Trade,
    TradeIntent,
)

log = logging.getLogger(__name__)

# Column order for the trade CSV. Stable on purpose: a spreadsheet template or a
# downstream script can rely on it.
TRADE_COLUMNS: tuple[str, ...] = (
    "id", "symbol", "entry_qty", "entry_avg_price", "exit_avg_price",
    "realized_pnl", "fees", "opened_at", "closed_at", "strategy_id",
    "loss_reasons",
)

# Column order for the equity curve CSV.
EQUITY_COLUMNS: tuple[str, ...] = (
    "closed_at", "symbol", "trade_id", "realized_pnl", "cumulative_pnl", "equity",
)


# ---------------------------------------------------------------------------
# plain row builders (shared by the file exporters and the HTTP endpoints)
# ---------------------------------------------------------------------------

def _iso(value: datetime | None) -> str | None:
    """ISO-8601 UTC string, or None.

    Naive datetimes come straight from SQLite (the columns are declared
    timezone-aware but SQLite stores no offset); they are interpreted as UTC
    instead of being exported without one.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _loss_reasons(raw: str | None) -> list[str]:
    """Parse the `loss_reasons` JSON column into a list (never raises)."""
    try:
        value = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def trade_rows(session: Session) -> list[dict[str, Any]]:
    """Every closed/partial trade, oldest first (open trades last).

    `closed_at` is NULL for a trade that has not been closed yet; those sort
    last so the equity curve (closed-only) can simply skip them.
    """
    stmt = (
        select(Trade)
        .order_by(Trade.closed_at.is_(None), Trade.closed_at.asc(), Trade.id.asc())
    )
    out: list[dict[str, Any]] = []
    for t in session.execute(stmt).scalars():
        out.append(
            {
                "id": t.id,
                "symbol": t.symbol,
                "entry_qty": t.entry_qty,
                "entry_avg_price": t.entry_avg_price,
                "exit_avg_price": t.exit_avg_price,
                "realized_pnl": t.realized_pnl,
                "fees": t.fees,
                "opened_at": _iso(t.opened_at),
                "closed_at": _iso(t.closed_at),
                "strategy_id": t.strategy_id,
                "loss_reasons": _loss_reasons(t.loss_reasons),
            }
        )
    return out


def equity_rows(
    session: Session, initial_balance: float = 10_000.0
) -> list[dict[str, Any]]:
    """Equity curve derived from closed trades, in time order.

    `equity` = `initial_balance` + cumulative realized PnL, so the curve starts
    at the configured starting balance and moves only on closed trades (an open
    position is unrealized by definition). Fees are already inside
    `realized_pnl`; they are not subtracted twice.

    Args:
        session: DB session.
        initial_balance: starting equity (``risk.paper_initial_balance`` for
            paper mode).
    """
    equity = float(initial_balance)
    out: list[dict[str, Any]] = []
    for t in trade_rows(session):
        if t["closed_at"] is None:
            continue
        pnl = float(t["realized_pnl"] or 0.0)
        equity += pnl
        out.append(
            {
                "closed_at": t["closed_at"],
                "symbol": t["symbol"],
                "trade_id": t["id"],
                "realized_pnl": pnl,
                "cumulative_pnl": equity - float(initial_balance),
                "equity": equity,
            }
        )
    return out


def _latest_analytics(session: Session, period: str) -> dict[str, Any] | None:
    """Most recent analytics summary for a period ("30m" | "daily")."""
    row = session.execute(
        select(AnalyticsSummary)
        .where(AnalyticsSummary.period == period)
        .order_by(AnalyticsSummary.ts.desc(), AnalyticsSummary.id.desc())
        .limit(1)
    ).scalars().first()
    if row is None:
        return None
    try:
        metrics = json.loads(row.metrics_json or "{}")
    except (TypeError, ValueError):
        metrics = {}
    return {
        "period": row.period,
        "ts": _iso(row.ts),
        "symbol": row.symbol,
        "metrics": metrics,
    }


def _active_strategy(session: Session) -> dict[str, Any] | None:
    row = session.execute(
        select(Strategy).where(Strategy.status == "active").order_by(Strategy.version.desc())
    ).scalars().first()
    if row is None:
        return None
    try:
        params = json.loads(row.params or "{}")
    except (TypeError, ValueError):
        params = {}
    return {
        "name": row.name,
        "version": row.version,
        "status": row.status,
        "params": params,
    }


def _count(session: Session, model: Any) -> int:
    return int(session.execute(select(func.count()).select_from(model)).scalar() or 0)


def summary_dict(
    session: Session,
    *,
    mode: str = "",
    initial_balance: float = 10_000.0,
    strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Snapshot of everything a user asks "how is it doing?" about.

    Deliberately excludes a candle row count: `candles` is the one table that
    grows into the hundreds of thousands of rows, and `COUNT(*)` on it would
    scan the whole table on every export just to print a number.
    """
    trades = trade_rows(session)
    closed = [t for t in trades if t["closed_at"] is not None]
    realized = sum(float(t["realized_pnl"] or 0.0) for t in closed)
    fees = sum(float(t["fees"] or 0.0) for t in trades)
    open_positions = list(
        session.execute(select(Position).where(Position.qty > 0)).scalars()
    )
    pending = int(
        session.execute(
            select(func.count())
            .select_from(AIRecommendation)
            .where(AIRecommendation.status == "pending")
        ).scalar()
        or 0
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "strategy": strategy if strategy is not None else _active_strategy(session),
        "counts": {
            "trades": len(trades),
            "closed_trades": len(closed),
            "positions": _count(session, Position),
            "open_positions": len(open_positions),
            "trade_intents": _count(session, TradeIntent),
            "signals": _count(session, Signal),
            "pending_recommendations": pending,
        },
        "notes": [
            "candles row count is omitted on purpose: COUNT(*) on the one table "
            "that grows into hundreds of thousands of rows would scan it all."
        ],
        "totals": {
            "realized_pnl": round(realized, 8),
            "fees": round(fees, 8),
            "initial_balance": float(initial_balance),
            "equity": round(float(initial_balance) + realized, 8),
        },
        "open_positions": [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_price": p.avg_price,
                "trailing_stop_price": p.trailing_stop_price,
                "opened_at": _iso(p.opened_at),
            }
            for p in open_positions
        ],
        "analytics": {
            "30m": _latest_analytics(session, "30m"),
            "daily": _latest_analytics(session, "daily"),
        },
    }


# ---------------------------------------------------------------------------
# text builders
# ---------------------------------------------------------------------------

def _csv_text(columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> str:
    """Render rows as CSV with a header. `\\n` line endings (not `\\r\\n`)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(list(columns))
    for row in rows:
        writer.writerow([row[c] for c in columns])
    return buf.getvalue()


def trades_csv_text(rows: list[dict[str, Any]]) -> str:
    """Trades as CSV; `loss_reasons` is written as its JSON list text."""
    flat = [
        {**row, "loss_reasons": json.dumps(row["loss_reasons"])} for row in rows
    ]
    return _csv_text(TRADE_COLUMNS, flat)


def trades_json_text(rows: list[dict[str, Any]]) -> str:
    """Trades as a pretty-printed JSON array (oldest first)."""
    return json.dumps(rows, indent=2, sort_keys=True) + "\n"


def equity_csv_text(rows: list[dict[str, Any]]) -> str:
    return _csv_text(EQUITY_COLUMNS, rows)


def summary_json_text(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


# ---------------------------------------------------------------------------
# file exporters
# ---------------------------------------------------------------------------

def _write_atomic(path: str | Path, text: str) -> None:
    """Temp file next to the target + rename: no half-written exports.

    Same directory on purpose — ``os.replace`` is atomic only within one
    filesystem, and on Android the app data dir and the CWD can differ.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:  # pragma: no cover - best-effort cleanup
                log.warning("could not remove temp file %s", tmp)


def export_trades_csv(session: Session, path: str | Path) -> int:
    """Write all trades to `path` as CSV. Returns the number of trades."""
    rows = trade_rows(session)
    _write_atomic(path, trades_csv_text(rows))
    log.info("exported %s trades to %s", len(rows), path)
    return len(rows)


def export_trades_json(session: Session, path: str | Path) -> int:
    """Write all trades to `path` as a JSON array. Returns the number of trades."""
    rows = trade_rows(session)
    _write_atomic(path, trades_json_text(rows))
    log.info("exported %s trades to %s", len(rows), path)
    return len(rows)


def export_equity_csv(
    session: Session, path: str | Path, *, initial_balance: float = 10_000.0
) -> int:
    """Write the closed-trade equity curve to `path` as CSV. Returns row count."""
    rows = equity_rows(session, initial_balance)
    _write_atomic(path, equity_csv_text(rows))
    log.info("exported equity curve (%s points) to %s", len(rows), path)
    return len(rows)


def export_summary_json(
    session: Session,
    path: str | Path,
    *,
    mode: str = "",
    initial_balance: float = 10_000.0,
    strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the summary snapshot to `path`. Returns the dict that was written."""
    summary = summary_dict(
        session, mode=mode, initial_balance=initial_balance, strategy=strategy
    )
    _write_atomic(path, summary_json_text(summary))
    log.info("exported summary to %s", path)
    return summary
