# AlgoTrading — AI-assisted algorithmic trading bot for Termux

A modular-monolith spot-trading bot that runs on Android via **Termux**, is
controlled from **Telegram**, and keeps all trade execution deterministic in
Python. The AI layer is strictly advisory: it reads analytics summaries and
proposes strategy changes that require **explicit human approval** before they
become active. Execution, risk control, and record-keeping never depend on the
AI.

**Status: M0–M6 complete, M7 (Termux hardening + live) complete.** 61+ unit
tests pass (`pytest tests/`).

- Exchange: Binance Spot (lightweight REST client — no ccxt, so it builds on
  Termux/Android)
- Modes: `paper` (simulated fills vs live prices, no keys) and `live`
  (real orders, requires keys + explicit review)
- `--demo-data`: fully offline synthetic feed for testing the whole pipeline
- SQLite in WAL mode; the trade lifecycle is event-sourced (immutable
  `order_events` log, positions/trades rebuilt from events on startup)

## Quickstart

```bash
# 1. One-time setup (Termux): packages, venv, deps, .env, DB init, wake-lock
bash scripts/setup_termux.sh

# 2. Configure secrets
nano .env          # TELEGRAM_BOT_TOKEN at minimum (from @BotFather)

# 3. Run (supervised: auto-restart on crash + heartbeat watchdog)
bash scripts/run_bot.sh

# or run directly:
.venv/bin/python -m algotrading.main                # paper mode vs live prices
.venv/bin/python -m algotrading.main --demo-data    # offline demo feed
.venv/bin/python -m algotrading.main --no-telegram  # headless (logs only)
```

Desktop development works too: `python -m venv .venv && .venv/bin/pip install
-r requirements.txt`, then the same commands (wake-lock is skipped when
`termux-wake-lock` is absent).

## Architecture

```
scheduler (APScheduler) ──► market tick: fetch candles → store → strategy engine → execution engine
  ├─ analytics (30m + daily metrics snapshots)
  ├─ ai_review (daily advisory proposal → PENDING recommendation)
  ├─ reconcile (local intents vs exchange open orders, every 15m)
  └─ heartbeat (touch data/heartbeat for the watchdog)

telegram (python-telegram-bot, polling) ── control surface:
  /status /start_bot /stop_bot /strategy /risk /summary + inline ✅/❌ approval

api (FastAPI, optional) ── /health (open) + /status (bearer-token guarded)
```

`algotrading/main.py` wires everything in one async event loop. The
scheduler's jobs run in a worker thread; each job opens its own DB session, so
no session is ever shared across threads. Startup order: restore DB → seed
starter strategies on first run → rebuild positions from events → initial
reconcile → resume.

## Configuration

- `config/settings.yaml` — mode, universe, intervals, schedule, risk limits, AI, internal API
- `config/strategies.yaml` — starter strategy parameter schemas (`ema_crossover`, `rsi_mean_reversion`)
- `.env` (from `.env.example`, gitignored) — Telegram token, Binance keys, AI key, `API_TOKEN`

| Env var | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated numeric user IDs (strict allowlist) |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Required for live mode only |
| `AI_API_KEY` / `AI_BASE_URL` / `AI_MODEL` | Optional AI assistant (any OpenAI-compatible endpoint) |
| `API_TOKEN` | Bearer token for the internal API's `/status` |

## Telegram commands

`/help` · `/status` (mode, active strategy, positions, recent fills) ·
`/start_bot` / `/stop_bot` (pause/resume the scheduler) · `/strategy`
(versioned strategies) · `/risk` (current limits) · `/summary` (analytics) ·
inline ✅/❌ buttons on AI proposals. Users outside the allowlist are silently
ignored.

## Safety model (non-negotiable)

1. **Intent-before-order** — a `trade_intent` with a unique idempotency key is
   persisted before any order is sent; retries can never place duplicates.
2. **Event-sourced ledger** — every transition is an immutable `order_event`;
   positions/trades are derived views rebuilt from events on startup.
3. **Risk gates** — exposure caps, cooldowns, duplicate-signal guard; failures
   are recorded as `risk_skipped` events.
4. **Stale-data freeze** — if market data is older than
   `max_staleness_seconds`, no new entries are made.
5. **AI is advisory** — proposals are PENDING until a human approves; the
   assistant never touches the exchange or the active strategy. If the LLM
   fails, the bot keeps running on the last approved strategy.
6. **Live is explicit** — `mode: live` in settings.yaml, and the bot refuses
   to start without Binance keys.
7. **Reconciliation** — a periodic job (and a startup pass) compares local
   intents with exchange open orders and reports drift.

## Live mode

```bash
# config/settings.yaml
mode: live
# .env
BINANCE_API_KEY=...
BINANCE_API_SECRET=...
```

The bot will refuse to start live without keys. **Do a separate code review
before flipping to live**, and consider starting with small
`risk.max_position_pct` / `risk.risk_per_trade_pct` values.

## Running as a Termux service (runit)

```bash
mkdir -p ~/.termux/services/algotrading
cat > ~/.termux/services/algotrading/run <<'EOF'
#!/data/data/com.termux/files/usr/bin/sh
exec /data/data/com.termux/files/home/AlgoTrading/scripts/run_bot.sh
EOF
chmod +x ~/.termux/services/algotrading/run
# restart termux-services:  sv up algotrading  (see termux-services docs)
```

`scripts/run_bot.sh` already supervises: it restarts the bot on crash with
exponential backoff and kills+restarts it if the heartbeat goes stale while
the process is alive.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Covers indicator math, risk gates, analytics metrics, the paper execution
flow end-to-end (signal → intent → fill → ledger events → rebuild), Telegram
auth/approvals, the AI recommendation flow (M6c), and M7 wiring (scheduler
jobs, market tick against a demo feed, stale-data freeze, internal API auth).