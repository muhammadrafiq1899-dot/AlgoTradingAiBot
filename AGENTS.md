# AlgoTradingAiBot

AI-assisted spot-trading bot for Android (Termux) or any computer, controlled via Telegram. Trades Binance with paper mode by default, live trading explicit. AI suggests strategy changes but cannot execute without approval.

## Dev environment

**Prerequisites:** Python 3.11+, Telegram bot token, your Telegram user ID. (Optional: Binance API keys for live, AI API key for assistant.)

**Setup (Termux):**
```bash
bash scripts/setup_termux.sh   # installs deps, creates .venv, .env, DB, installs algobot command
algobot setup                  # guided wizard to fill .env (or edit .env directly)
```

**Run the bot:**
```bash
algobot start                  # paper mode, real Binance data
algobot start --demo-data      # offline demo, no network
algobot stop                   # stop cleanly
algobot status                 # show mode, strategy, positions
algobot logs                   # recent logs (algobot logs 200 for more)
```

**Foreground run (for debugging):**
```bash
.venv/bin/python -m algotrading.main                # paper mode
.venv/bin/python -m algotrading.main --demo-data    # offline practice
.venv/bin/python -m algotrading.main --no-telegram  # logs only
```

**Test:**
```bash
.venv/bin/python -m pytest tests/ -q                # 173 tests
bash scripts/check_project_map.sh                   # verify docs in sync
.venv/bin/python -m compileall -q algotrading       # byte‑compile check
```

## Conventions

- **Configuration:** `config/settings.yaml` (YAML) for non‑secret settings; `.env` (gitignored) for secrets (TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, BINANCE_*, AI_*, API_TOKEN).
- **Strategies:** Python files in `strategies/`; each defines `evaluate(symbol, candles)` and a `STRATEGY` alias. Hot‑reloaded every 300s.
- **Modules:** Plug‑in capabilities (market, execution, strategy, analytics, control, api) selected via `modules.enabled` in settings.yaml. Locked capabilities (execution) cannot be overridden.
- **Event sourcing:** Orders write `trade_intent` before exchange call; fills append to immutable `order_events`; positions/trades rebuilt from events on start.
- **AI assistant:** Advisory only. Can `propose_change` (parameter tweak, new strategy, etc.) but never executes. Enable with `AI_API_KEY` (external LLM) **or** `USE_HERMES=true` (local Hermes Agent CLI, no key).
- **Naming:** Strategy files snake_case (e.g., `ema_crossover.py`). Module names in settings.yaml use dots (e.g., `market.binance`).
- **Commits:** Keep `PROJECT_MAP.md` updated; `scripts/check_project_map.sh` fails if new source files undocumented.

## Pitfalls

- **Secrets:** `.env` must be created (copy from `.env.example`) and filled; never commit it.
- **Database:** `data/algotrading.db` holds state; backups in `data/backups/` (7‑day retention). Do not edit while bot running.
- **Heartbeat:** `data/heartbeat` touched every 60s; supervisor loop kills bot if stale >300s. Do not delete manually.
- **Single instance:** Supervisor loop uses `data/run_bot.lock`; only one `run_bot.sh` may run. `algobot start` is safe.
- **Demo mode:** `--demo-data` uses synthetic prices; ignores live Binance and disables live gateway.
- **AI availability:** Without `AI_API_KEY` *and* without `USE_HERMES=true`, assistant disabled; all other features work. With `USE_HERMES=true` the bot shells out to `hermes chat -q` (needs the CLI on PATH; keep the default terminal-free toolset).
- **Termux specifics:** `algobot` command installed to `$PREFIX/bin` by setup script; relies on `.venv` path. If moving repo, reinstall or run via `.venv/bin/python -m algotrading.main`.
- **Trade safety:** Bot persists intent before order; retry uses same idempotency key. Never sends duplicate order.
- **Live trading:** Requires `mode: live` in settings.yaml AND Binance keys; bot refuses to start otherwise.