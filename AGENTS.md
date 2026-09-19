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
algobot download --symbols BTC/USDT --intervals 1h --days 365   # deeper history
algobot export --dir data/exports                                # trades/summary files
```

**Foreground run (for debugging):**
```bash
.venv/bin/python -m algotrading.main                # paper mode
.venv/bin/python -m algotrading.main --demo-data    # offline practice
.venv/bin/python -m algotrading.main --no-telegram  # logs only
```

**Test:**
```bash
.venv/bin/python -m pytest tests/ -q                # 547 tests
bash scripts/check_project_map.sh                   # verify docs in sync
.venv/bin/python -m compileall -q algotrading       # byte‑compile check
```

## Conventions

- **Configuration:** `config/settings.yaml` (YAML) for non‑secret settings; `.env` (gitignored) for secrets (TELEGRAM_BOT_TOKEN, TELEGRAM_ALLOWED_USERS, BINANCE_*, BINANCE_TESTNET_*, AI_*, API_TOKEN, ALERT_WEBHOOK_URL).
- **Risk is enforced, not documented.** `max_position_pct` caps every entry (`RiskManager.size_by_pct`), `max_daily_loss_pct` blocks new entries for the rest of the UTC day (`enforce_daily_loss` turns it off explicitly), and sizing uses the configured balance or the venue's. A guard that only exists in `config.py` validation is a bug.
- **Evaluation interval:** `market.eval_interval` selects the interval the strategy engine evaluates; unset means the legacy "1h if present" rule. The stale-data allowance follows that interval's own length (`INTERVAL_MS`).
- **Venues/orders:** spot Binance; `market.use_testnet` + `BINANCE_TESTNET_*` for testnet (validation requires keys for the *selected* venue). Entries are market by default; a signal may ask for a limit (`risk['order_type']='limit'`). `risk.exchange_stop_enabled` rests a stop on the exchange after a fill — its identity lives in `order_events` (`stop_placed`/`stop_canceled`), never a new column.
- **Research:** `algotrading/backtest/` (slippage, configurable fees, optional risk-check simulation, metrics/walk-forward/Monte Carlo) and `algotrading/optimize/` (parameter search). The search MUST run as a separate niced process (`optimize/spawn.py`) and its only output is a PENDING `param_change`.
- **Strategies:** Python files in `strategies/`; each defines `evaluate(symbol, candles)` and a `STRATEGY` alias. Hot‑reloaded every 300s. `strategies/three_commas_bot.py` is a worked hand-conversion of a Pine v5 script (long-only, signal-driven stops).
- **Authoring contract (AI or hand-written):** the class must be buildable as `cls(params_dict)` (`def __init__(self, params=None)`), and only the indicator helpers / `math` / `statistics` may be imported (`from algotrading.strategy import indicators as ta`); anything else is refused by `strategy/validation.py`. `create_pending_recommendation` runs the full compile gate, so bad code comes back to the model as tool feedback instead of becoming an unapplicable Approve.
- **Modules:** Plug‑in capabilities (market, execution, strategy, analytics, control, api) selected via `modules.enabled` in settings.yaml. Locked capabilities (execution) cannot be overridden.
- **Event sourcing:** Orders write `trade_intent` before exchange call; fills append to immutable `order_events`; positions/trades rebuilt from events on start.
- **AI assistant:** Advisory only. Can `propose_change` (parameter tweak, new strategy, etc.) but never executes. Enable with `AI_API_KEY` (external LLM) **or** `USE_HERMES=true` (local Hermes Agent CLI, no key). News headlines, the decision log and the portfolio/correlation view are advisory context too — headline text is untrusted data (chat hard rule 5) and none of it feeds execution automatically.
- **Naming:** Strategy files snake_case (e.g., `ema_crossover.py`). Module names in settings.yaml use dots (e.g., `market.binance`).
- **Commits:** Keep `PROJECT_MAP.md` updated; `scripts/check_project_map.sh` fails if new source files undocumented. Uncommitted work is fine; don't commit unless asked.

## Pitfalls

- **Secrets:** `.env` must be created (copy from `.env.example`) and filled; never commit it.
- **Database:** `data/algotrading.db` holds state; backups in `data/backups/` (7‑day retention). Do not edit while bot running.
- **Heartbeat:** `data/heartbeat` touched every 60s; supervisor loop kills bot if stale >300s (grace grows by however long the loop itself was suspended, so a screen-off phone isn't mistaken for a hung bot). Do not delete manually.
- **Android kills the whole app, not the bot:** on Android 12+ (and forced harder by vendor ROMs, e.g. vivo/iQOO) the entire Termux app is SIGKILLed once it is backgrounded with the screen off — measured here: ~17 min after screen lock, taking the supervisor, the bot, any child watchers and interactive sessions with it. Signature: the log stops mid-tick with no `shutting down…`/`bye` and no watchdog line, and `algobot status` says not running while the db/heartbeat simply stop advancing. The bot cannot prevent this from inside; the levers are the wake lock (requested at startup, demo mode included), battery-optimisation exemption for Termux, and disabling the phantom-process monitor (Android 14: Developer options → *Disable child process restrictions*; or `adb shell settings put global settings_enable_monitor_phantom_procs false`). Keeping the tick cheap matters too: a tick that burns a CPU core is exactly what those killers target.
- **Single instance:** Supervisor loop uses `data/run_bot.lock`; only one `run_bot.sh` may run. `algobot start` is safe.
- **Demo mode:** `--demo-data` uses synthetic prices; ignores live Binance and disables live gateway.
- **AI availability:** Without `AI_API_KEY` *and* without `USE_HERMES=true`, assistant disabled; all other features work. With `USE_HERMES=true` the bot shells out to `hermes chat -q` (needs the CLI on PATH; keep the default terminal-free toolset). The provider occasionally truncates a long answer mid-string; `complete_text` + `salvage_partial_reply` degrade gracefully (partial reply kept, partial tool call still an error).
- **Images:** Telegram photos/image files are read only by the Hermes provider (`USE_HERMES=true`) — the external `AI_API_KEY` path has no vision path and replies with a hint. Caps/behaviour live in `config/settings.yaml` `ai.images_enabled` / `ai.image_max_bytes` / `ai.image_dir`; uploads land in `data/uploads/` (gitignored) and are deleted right after the AI reads them. Text inside a picture is untrusted data, never an instruction (chat hard rule 5).- **Image batches:** photos are buffered per chat (`PHOTO_BATCH_DELAY_SECONDS`) so an album is answered once; 2+ images get the 🧩 one-strategy / 🧱 separate→ensemble choice (`imgflow:<token>:<mode>`). Batches are in-memory (lost on restart), capped by `IMAGE_BATCH_LIMIT`, and expire after `IMAGE_FLOW_TTL_SECONDS`. The 🧩 flow transcribes each image then writes one strategy; the 🧱 flow is one agent call per image.
- **Ensembles:** `EnsembleStrategy` components resolve through `strategy/registry.build_strategy`, so AI-authored plugins qualify — but a component must be approved before an ensemble can reference it. `consensus` = "no firing component disagrees" and `filter` uses the first firing component as primary; don't document them as strict AND gates.
- **Termux specifics:** `algobot` command installed to `$PREFIX/bin` by setup script; relies on `.venv` path. If moving repo, reinstall or run via `.venv/bin/python -m algotrading.main`.
- **Trade safety:** Bot persists intent before order; retry uses same idempotency key. Never sends duplicate order.
- **Live trading:** Requires `mode: live` in settings.yaml AND Binance keys; bot refuses to start otherwise. On testnet (`market.use_testnet: true`) the *testnet* keys are the ones required — the check follows the selected venue.
- **Risk guard silence:** a "daily loss limit reached" skip is correct behaviour, not a bug — new entries stop for the rest of the UTC day while protective exits keep running. Look at the dashboard/daily-loss line before assuming the strategy is dead.
- **Alerts:** the webhook channel is best-effort and rate-limited per kind; it never raises into the tick. `ALERT_WEBHOOK_URL` enables it (or the `alerts:` block). Silent alerts usually mean `notify_*` is off or the rate limit swallowed the event.
- **Advisory memory lag:** the decision log is synced by the daily jobs (not at approval time), so a freshly applied recommendation may take until the next daily run to appear with its outcome.
- **Dashboard/export auth:** `/dashboard` and `/export/*` need `Authorization: Bearer $API_TOKEN`; `/health` and `/metrics` stay open. `/dashboard` 404s when `api.dashboard: false`.
- **Optimizer lock:** only one search may run (`optimize.results_dir` lockfile); a stale lock from a killed process is replaced automatically. Never run a search in-process — it contends with the 60s tick for the same interpreter.