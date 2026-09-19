"""Read-only HTML dashboard for the running bot.

Server-rendered on purpose: this process also runs the trading loop, so the
dashboard is one DB session, one string, one response — no JS framework, no
external assets, nothing to fail on a phone that only has loopback. It is
**read-only by construction**: there are no forms and no POST handlers, so
nothing on this page can change bot state (invariant §11.3/§11.5 of the
project map).

Trust model: strategy names come from AI-authored plugins, symbols from config
and the exchange, and error text from arbitrary exceptions. Everything dynamic
is therefore passed through :func:`html.escape` before it reaches the page —
the dashboard is the one place where bot state is rendered as HTML.

Two layers so the page is testable without parsing HTML:
:func:`collect_state` builds a plain dict from a DB session,
:func:`render_dashboard` turns that dict into the page.
"""
from __future__ import annotations

import html
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from algotrading.alerts import get_alerter
from algotrading.db.models import Candle, Position, Strategy, Trade
from algotrading.execution.engine import DAILY_LOSS_LOOKBACK_ROWS
from algotrading.execution.risk import RiskManager
from algotrading.market.candles import INTERVAL_MS

log = logging.getLogger(__name__)

# The page reloads itself; 30s keeps a phone browser from hammering the API
# while still looking "live" next to the 60s market tick.
REFRESH_SECONDS = 30

# How many closed trades the page shows.
RECENT_TRADES = 20


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def _daily_realized_pnl(session: Session, now: datetime) -> float:
    """Realized PnL of trades closed since 00:00 UTC — the guard's window.

    Mirrors ``ExecutionEngine._daily_realized_pnl`` (same UTC day, same row
    cap) so the number the dashboard prints is the number the guard acted on.
    Filtering happens in Python because SQLite returns naive datetimes and a SQL
    comparison against a tz-aware boundary is a footgun.
    """
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rows = session.execute(
        select(Trade.closed_at, Trade.realized_pnl)
        .where(Trade.closed_at.is_not(None))
        .order_by(Trade.closed_at.desc())
        .limit(DAILY_LOSS_LOOKBACK_ROWS)
    ).all()
    total = 0.0
    for closed_at, pnl in rows:
        if closed_at is None or pnl is None:
            continue
        ts = closed_at if closed_at.tzinfo else closed_at.replace(tzinfo=timezone.utc)
        if ts >= day_start:
            total += float(pnl)
    return total


def _total_realized_pnl(session: Session) -> float:
    """All-time realized PnL — the basis for the guard's equity denominator."""
    total = session.execute(select(func.sum(Trade.realized_pnl))).scalar()
    return float(total or 0.0)


def _mark_price(session: Session, symbol: str) -> tuple[float | None, int | None]:
    """Newest stored close for a symbol, across every interval.

    Any interval will do for a mark: this is the most recent price the bot has
    written, not an analysis input.
    """
    row = session.execute(
        select(Candle.close, Candle.ts)
        .where(Candle.symbol == symbol)
        .order_by(Candle.ts.desc())
        .limit(1)
    ).first()
    if row is None:
        return None, None
    return float(row[0]), int(row[1])


def _active_strategy(session: Session) -> dict[str, Any] | None:
    row = session.execute(
        select(Strategy).where(Strategy.status == "active").order_by(Strategy.version.desc())
    ).scalars().first()
    if row is None:
        return None
    try:
        params = json.loads(row.params or "{}")
    except (TypeError, ValueError):
        params = {"_raw": row.params}
    return {"name": row.name, "version": row.version, "status": row.status, "params": params}


def _iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()


def collect_state(
    session: Session,
    settings: Any,
    health: Any | None = None,
    alerter: Any | None = None,
    *,
    now_ms: int | None = None,
    uptime_seconds: int | None = None,
) -> dict[str, Any]:
    """Gather everything the dashboard shows, in one pass, read-only.

    Args:
        session: DB session (the caller owns it; the API opens one per request).
        settings: validated `Settings`.
        health: `HealthMonitor` for the heartbeat age (None = unknown).
        alerter: alert channel (defaults to ``get_alerter()``, the process-wide
            instance configured from settings at startup).
        now_ms: epoch-ms "now" override (tests).
        uptime_seconds: how long this API process has been serving.

    Returns:
        A JSON-serializable dict; `render_dashboard` is the only consumer.
    """
    now = datetime.fromtimestamp((now_ms or time.time() * 1000) / 1000, tz=timezone.utc)
    mode = str(getattr(settings, "mode", "") or "")
    market_cfg = settings.market
    risk_cfg = settings.risk
    alerter = alerter if alerter is not None else get_alerter()

    # --- positions (with a mark-to-market PnL) ---
    positions: list[dict[str, Any]] = []
    for p in session.execute(select(Position).where(Position.qty > 0)).scalars():
        price, price_ts = _mark_price(session, p.symbol)
        unrealized = None
        unrealized_pct = None
        if price is not None and p.avg_price:
            unrealized = (price - p.avg_price) * p.qty
            unrealized_pct = (price / p.avg_price - 1.0) * 100.0
        positions.append(
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_price": p.avg_price,
                "mark_price": price,
                "mark_ts": price_ts,
                "unrealized_pnl": unrealized,
                "unrealized_pct": unrealized_pct,
                "trailing_stop_price": p.trailing_stop_price,
                "opened_at": p.opened_at.isoformat() if p.opened_at else None,
            }
        )

    # --- last ~20 closed trades, newest first ---
    trades: list[dict[str, Any]] = []
    recent = session.execute(
        select(Trade)
        .where(Trade.closed_at.is_not(None))
        .order_by(Trade.closed_at.desc(), Trade.id.desc())
        .limit(RECENT_TRADES)
    ).scalars()
    for t in recent:
        trades.append(
            {
                "id": t.id,
                "symbol": t.symbol,
                "entry_qty": t.entry_qty,
                "entry_avg_price": t.entry_avg_price,
                "exit_avg_price": t.exit_avg_price,
                "realized_pnl": float(t.realized_pnl or 0.0),
                "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            }
        )

    # --- daily loss guard (same inputs the execution engine hands RiskManager) ---
    daily_pnl = _daily_realized_pnl(session, now)
    basis_balance = float(risk_cfg.paper_initial_balance) + _total_realized_pnl(session)
    breached, loss_pct = RiskManager(risk_cfg).daily_loss_breached(daily_pnl, basis_balance)

    # --- market data freshness per configured symbol x interval ---
    now_ms_int = int(now.timestamp() * 1000)
    market: list[dict[str, Any]] = []
    for symbol in market_cfg.symbols:
        for interval in market_cfg.intervals:
            row = session.execute(
                select(Candle.ts)
                .where(Candle.symbol == symbol, Candle.interval == interval)
                .order_by(Candle.ts.desc())
                .limit(1)
            ).scalar()
            age_s = int((now_ms_int - int(row)) / 1000) if row is not None else None
            market.append(
                {
                    "symbol": symbol,
                    "interval": interval,
                    "latest_ts": int(row) if row is not None else None,
                    "latest_at": _iso(int(row)) if row is not None else None,
                    "age_seconds": age_s,
                    "fresh": age_s is not None and age_s <= market_cfg.max_staleness_seconds,
                    "known_interval": interval in INTERVAL_MS,
                }
            )

    # --- heartbeat / scheduler ---
    last_beat = None
    if health is not None:
        try:
            last_beat = health.last_beat()
        except Exception as exc:  # noqa: BLE001 - a status page never raises
            log.warning("dashboard: heartbeat unavailable: %s", exc)
    heartbeat_age = int(time.time() - last_beat) if last_beat else None
    tick = int(getattr(settings.schedule, "market_tick_seconds", 60) or 60)
    scheduler_ok = heartbeat_age is not None and heartbeat_age < tick * 2

    try:
        alert_status = dict(alerter.status())
    except Exception as exc:  # noqa: BLE001 - status of a status page
        alert_status = {"error": str(exc)}

    return {
        "generated_at": now.isoformat(),
        "refresh_seconds": REFRESH_SECONDS,
        "mode": mode,
        "venue": "testnet" if getattr(market_cfg, "use_testnet", False) else "mainnet",
        "strategy": _active_strategy(session),
        "positions": positions,
        "trades": trades,
        "today": {
            "date": now.date().isoformat(),
            "realized_pnl": daily_pnl,
        },
        "guard": {
            "enabled": bool(risk_cfg.enforce_daily_loss),
            "limit_pct": float(risk_cfg.max_daily_loss_pct),
            "today_loss_pct": loss_pct,
            "breached": breached,
            "basis_balance": basis_balance,
        },
        "market": market,
        "max_staleness_seconds": int(market_cfg.max_staleness_seconds),
        "heartbeat": {
            "age_seconds": heartbeat_age,
            "status": "ok" if scheduler_ok else "stale",
        },
        "scheduler": {
            "status": "ok" if scheduler_ok else "stale",
            "market_tick_seconds": tick,
        },
        "api": {
            "status": "ok",
            "uptime_seconds": uptime_seconds,
        },
        "alerts": alert_status,
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def esc(value: Any) -> str:
    """Escape any dynamic value for HTML (quotes included)."""
    return html.escape("" if value is None else str(value), quote=True)


def _money(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return esc(f"{float(value):,.2f}")
    except (TypeError, ValueError):
        return esc(value)


def _num(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    try:
        return esc(f"{float(value):,.{digits}f}")
    except (TypeError, ValueError):
        return esc(value)


def _pct(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return esc(f"{float(value):.2f}%")
    except (TypeError, ValueError):
        return esc(value)


def _age(seconds: Any) -> str:
    if seconds is None:
        return "never"
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return esc(seconds)
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m {s % 60}s ago"
    return f"{s // 3600}h {(s % 3600) // 60}m ago"


def _pnl_class(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "flat"
    if v > 0:
        return "pos"
    if v < 0:
        return "neg"
    return "flat"


def _table(headers: list[str], rows: list[list[str]], empty: str = "none") -> str:
    if not rows:
        return f'<p class="empty">{esc(empty)}</p>'
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_dashboard(state: dict[str, Any]) -> str:
    """Render the page. Every value from `state` goes through :func:`esc`.

    Kept as one function with local helpers: this page is deliberately a single
    self-contained document (inline CSS, no assets, no JS), and splitting it
    into templates would mean pulling in a template engine the Termux
    dependency budget does not allow for.
    """
    strategy = state.get("strategy") or {}
    guard = state.get("guard") or {}
    alerts = state.get("alerts") or {}
    heartbeat = state.get("heartbeat") or {}
    api = state.get("api") or {}
    today = state.get("today") or {}

    strategy_line = (
        f"{esc(strategy.get('name'))} v{esc(strategy.get('version'))} "
        f"({esc(strategy.get('status'))})"
        if strategy
        else "none active"
    )
    params = strategy.get("params") or {}
    params_line = esc(json.dumps(params, sort_keys=True)) if params else "—"

    mode = esc(state.get("mode") or "?")
    venue = esc(state.get("venue") or "?")
    breached = bool(guard.get("breached"))
    guard_class = "neg" if breached else ("pos" if guard.get("enabled") else "flat")
    guard_text = (
        "BREACHED — new entries blocked"
        if breached
        else ("armed" if guard.get("enabled") else "disabled")
    )

    position_rows = [
        [
            esc(p.get("symbol")),
            _num(p.get("qty"), 8),
            _money(p.get("avg_price")),
            _money(p.get("mark_price")),
            f'<span class="{_pnl_class(p.get("unrealized_pnl"))}">'
            f'{_money(p.get("unrealized_pnl"))}</span>',
            f'<span class="{_pnl_class(p.get("unrealized_pnl"))}">'
            f'{_pct(p.get("unrealized_pct"))}</span>',
            _money(p.get("trailing_stop_price")),
        ]
        for p in state.get("positions") or []
    ]

    trade_rows = [
        [
            esc(t.get("closed_at")),
            esc(t.get("symbol")),
            _num(t.get("entry_qty"), 8),
            _money(t.get("entry_avg_price")),
            _money(t.get("exit_avg_price")),
            f'<span class="{_pnl_class(t.get("realized_pnl"))}">'
            f'{_money(t.get("realized_pnl"))}</span>',
        ]
        for t in state.get("trades") or []
    ]

    market_rows = [
        [
            esc(m.get("symbol")),
            esc(m.get("interval")),
            esc(m.get("latest_at") or "never"),
            esc(_age(m.get("age_seconds"))),
            # A configured interval the code does not know about is worth
            # flagging: it will simply never produce candles.
            f'<span class="{"pos" if m.get("fresh") else "neg"}">'
            f'{"fresh" if m.get("fresh") else ("unknown interval" if not m.get("known_interval") else "stale")}</span>',
        ]
        for m in state.get("market") or []
    ]

    alert_cell = (
        f'enabled={esc(alerts.get("enabled"))} '
        f'url={esc(alerts.get("url_configured"))} '
        f'sent={esc(alerts.get("sent"))} '
        f'suppressed={esc(alerts.get("suppressed"))}'
        if "error" not in alerts
        else esc(alerts.get("error"))
    )

    uptime = api.get("uptime_seconds")
    api_cell = (
        f"serving · uptime {esc(_age(uptime)).replace(' ago', '')}"
        if uptime is not None
        else "serving"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{int(state.get("refresh_seconds") or REFRESH_SECONDS)}">
<title>AlgoTrading dashboard</title>
<style>
:root {{ color-scheme: dark; }}
body {{ background:#0f1115; color:#e6e8eb; font:14px/1.45 -apple-system,system-ui,"Segoe UI",Roboto,sans-serif;
        margin:0; padding:16px; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
h2 {{ font-size:14px; margin:20px 0 8px; color:#9aa4b2; text-transform:uppercase; letter-spacing:.06em; }}
.sub {{ color:#9aa4b2; margin-bottom:8px; }}
.cards {{ display:flex; flex-wrap:wrap; gap:8px; }}
.card {{ background:#171a21; border:1px solid #232833; border-radius:8px; padding:10px 12px; min-width:150px; flex:1 1 150px; }}
.card .k {{ color:#9aa4b2; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }}
.card .v {{ font-size:16px; margin-top:4px; word-break:break-word; }}
table {{ width:100%; border-collapse:collapse; background:#171a21; border:1px solid #232833; border-radius:8px; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid #232833; font-variant-numeric:tabular-nums; }}
th {{ color:#9aa4b2; font-size:11px; text-transform:uppercase; letter-spacing:.06em; }}
tr:last-child td {{ border-bottom:none; }}
.pos {{ color:#3ddc84; }}
.neg {{ color:#ff6b6b; }}
.flat {{ color:#9aa4b2; }}
.empty {{ color:#6b7480; font-style:italic; }}
.foot {{ color:#6b7480; margin-top:18px; font-size:12px; }}
code {{ background:#171a21; border:1px solid #232833; border-radius:4px; padding:1px 5px; word-break:break-all; }}
</style>
</head>
<body>
<h1>AlgoTrading</h1>
<div class="sub">read-only · mode <b>{mode}</b> · venue <b>{venue}</b> · generated {esc(state.get("generated_at"))}</div>

<div class="cards">
<div class="card"><div class="k">Strategy</div><div class="v">{strategy_line}<br><span class="flat">{params_line}</span></div></div>
<div class="card"><div class="k">Today's realized PnL ({esc(today.get("date"))} UTC)</div>
  <div class="v"><span class="{_pnl_class(today.get("realized_pnl"))}">{_money(today.get("realized_pnl"))}</span></div></div>
<div class="card"><div class="k">Daily loss guard</div>
  <div class="v"><span class="{guard_class}">{guard_text}</span><br>
  limit {_pct(guard.get("limit_pct"))} · today used {_pct(guard.get("today_loss_pct"))}<br>
  <span class="flat">basis {_money(guard.get("basis_balance"))} · breached {esc(guard.get("breached"))}</span></div></div>
<div class="card"><div class="k">Heartbeat</div>
  <div class="v"><span class="{"pos" if heartbeat.get("status") == "ok" else "neg"}">{esc(heartbeat.get("status"))}</span>
  <br><span class="flat">{esc(_age(heartbeat.get("age_seconds")))}</span></div></div>
<div class="card"><div class="k">Scheduler</div>
  <div class="v"><span class="{"pos" if (state.get("scheduler") or {}).get("status") == "ok" else "neg"}">{esc((state.get("scheduler") or {}).get("status"))}</span>
  <br><span class="flat">tick every {esc((state.get("scheduler") or {}).get("market_tick_seconds"))}s</span></div></div>
<div class="card"><div class="k">API</div><div class="v">{esc(api_cell)}</div></div>
<div class="card"><div class="k">Alerts</div><div class="v">{alert_cell}</div></div>
</div>

<h2>Open positions ({len(state.get("positions") or [])})</h2>
{_table(["Symbol", "Qty", "Avg price", "Mark", "Unrealized", "Unrealized %", "Trailing stop"],
        position_rows, "No open positions.")}

<h2>Last {RECENT_TRADES} trades</h2>
{_table(["Closed (UTC)", "Symbol", "Qty", "Entry", "Exit", "Realized PnL"],
        trade_rows, "No closed trades yet.")}

<h2>Market data freshness</h2>
{_table(["Symbol", "Interval", "Latest candle (UTC)", "Age", "Status"],
        market_rows, "No market data configured.")}

<p class="foot">Read-only page · stale after {esc(state.get("max_staleness_seconds"))}s without a candle ·
exports: <code>/export/trades.csv</code> <code>/export/trades.json</code> <code>/export/summary.json</code>
(all require the API bearer token)</p>
</body>
</html>
"""
