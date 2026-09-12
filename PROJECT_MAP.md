# PROJECT_MAP — AlgoTrading

> **What this is:** a table of contents / road-sign for AI agents working on this repo.
> Read this first. It tells you where everything lives, how the data flows, where to
> find code for a feature, and where to look when something breaks.
>
> **Update rule (IMPORTANT):** whenever you add, remove, rename, or repurpose a module,
> table, job, command, or data-flow step, **update this file in the same change**.
> If this map is wrong, fix it as part of your work. Keep it accurate and short enough
> to skim in one minute.
>
> **Enforcement:** `scripts/check_project_map.sh` fails when this map is missing,
> omits any source file (algotrading/, scripts/, config/, tests/), or is older than a
> tracked change. Install it as a pre-commit hook once
> (`bash scripts/install_hooks.sh`) so out-of-sync commits are blocked, or call the
> same script from CI (`bash scripts/check_project_map.sh`). See §10.

---

## 1. Project snapshot

AI-assisted **spot-trading bot** for Binance that runs on Android/Termux and is
controlled from Telegram. Modular monolith, one Python process, one asyncio event loop.

- **Execution is deterministic Python; AI is advisory only** (proposals need human approval).
- **Modes:** `paper` (simulated fills vs live prices, no keys) / `live` (real orders, requires keys).
- **Data:** SQLite (WAL), event-sourced trade lifecycle (`order_events` is the source of truth).
- **Stack constraints (Termux/Android):** no ccxt (pulls Rust `cryptography`), no `openai` SDK
  (pulls Rust `jiter`), pydantic **v1** only, pure-Python indicators (no numpy/pandas).
- Version: `algotrading/__init__.py` → `__version__ = "0.1.0"`.
- Tests: `pytest tests/` (~60+, all M0–M7 milestones).

## 2. How to run

```bash
bash scripts/setup_termux.sh            # one-time: pkgs, venv, deps, .env, DB init, wake-lock
.venv/bin/python -m algotrading.main               # paper mode vs live Binance prices
.venv/bin/python -m algotrading.main --demo-data   # offline synthetic feed (forces paper)
.venv/bin/python -m algotrading.main --no-telegram # headless
bash scripts/run_bot.sh                 # supervised loop: auto-restart + heartbeat watchdog
.venv/bin/python -m pytest tests/ -q    # test suite
```

⚠️ **Gotcha:** `python -m algotrading.main` must run from the repo root (or any dir — `main.py`
fixes `sys.path` itself, but `-m` does not). Running `.venv/bin/python -m algotrading/main`
fails with `No module named algotrading/main` and creates stray `bot.log`-style artifacts.

## 3. Directory map

```
AlgoTrading/
├── algotrading/              # the whole application (package)
│   ├── main.py               # entry point: wires scheduler + telegram + API in one loop
│   ├── config.py             # pydantic Settings from config/*.yaml + .env secrets
│   ├── cli_setup.py          # .env read/write, setup wizard, algobot start/stop/status/logs
│   ├── market/               # price data: providers + candle store
│   ├── strategy/             # pure strategies, indicators, signal evaluation engine
│   ├── execution/            # risk checks, trade intents, order gateways (paper/live)
│   ├── ledger/               # event-sourced trade lifecycle (fills → positions/trades)
│   ├── db/                   # SQLAlchemy models, engine/session, seeding
│   ├── analytics/            # metrics from closed trades + periodic summaries
│   ├── ai/                   # advisory LLM: client, prompt builder, assistant
│   ├── store/                # recommendation lifecycle + versioned strategy release
│   ├── telegram/             # control surface: bot bootstrap, commands, UI formatting
│   ├── api/                  # optional FastAPI: /health (open), /status (bearer)
│   ├── scheduler/            # APScheduler job registry (the heartbeat of the bot)
│   ├── supervisor/           # Termux wake-lock, heartbeat file, order reconciliation
│   └── backtest/             # shadow backtest for AI proposals
├── .github/workflows/       # CI (ci.yml): pytest + PROJECT_MAP freshness check
├── scripts/                  # setup_termux.sh, run_bot.sh, init_db.py, setup.py, algobot,
│                             #   check_project_map.sh (map freshness), install_hooks.sh
├── config/                   # settings.yaml (everything tunable), strategies.yaml (schemas)
├── tests/                    # pytest suite, one file per subsystem
├── data/                     # runtime: algotrading.db, heartbeat, algobot.pid (gitignored)
└── logs/                     # algotrading.log (rotating, gitignored)
```

## 4. Data flow (read this before touching anything)

**Main loop (every `market_tick_seconds`, default 60s):**

```
scheduler.market_tick (algotrading/scheduler/jobs.py)
  → market.BinanceMarketProvider|DemoProvider.fetch_klines  (algotrading/market/)
  → CandleStore.upsert → DB table `candles`
  → stale-data freeze check (max_staleness_seconds)
  → strategy.StrategyEngine.evaluate(snapshot)   → DB table `signals` (candidate|skipped)
  → execution.ExecutionEngine.execute(signal_id)
      → risk.RiskManager.check_buy/check_sell   (reject → `risk_skipped` event)
      → persist TradeIntent (UNIQUE idempotency_key) BEFORE order  ← safety rule #1
      → gateway.place_market_order (PaperGateway | LiveGateway)
      → ledger.Ledger.mark_filled/mark_sent/mark_failed  → immutable `order_events`
      → positions/trades derived views updated incrementally
```

**Other scheduled jobs** (`scheduler/jobs.py` + `config/settings.yaml` `schedule:`):
`analytics` (30m metrics snapshot), `analytics_daily` (UTC midnight), `ai_review`
(daily advisory LLM proposal → PENDING recommendation), `reconcile` (15m local
intents vs exchange open orders), `heartbeat` (60s touch of `data/heartbeat`).

**Control surfaces (do NOT drive the pipeline):** Telegram commands (`/status`,
`/strategy`, `/risk`, `/summary`, `/start_bot`, `/stop_bot`, inline ✅/❌ approval),
plain-text **chat** (`telegram/chat.py`: the LLM orchestrator answers questions and
can call `get_status` / `get_market` / `backtest` / `propose_change` tools), and the
internal API (`/health`, `/status`). AI approval path:
`ai_review` (daily) OR chat `propose_change` → `AIRecommendation(pending)` →
human ✅ → `RecommendationStore.apply` → shadow `backtest.runner` →
`store.strategy_versions.create_new_version` + `promote_to_active` (retires old
active). The chat agent never trades or changes anything directly — proposals only.

**Threading rule:** scheduler jobs run in a worker thread; **each job opens its own
DB session** (`ctx.session_factory()`), never share sessions across threads. The
Telegram bot runs in the event loop with one session.

## 5. Module reference

| Path | Purpose | Key symbols |
|---|---|---|
| `algotrading/main.py` | Entry point, wiring, lifecycle | `parse_args`, `build_context`, `AppController`, `run`, `main` (shutdown waits for in-flight scheduler jobs) |
| `algotrading/config.py` | YAML + .env → validated settings | `Settings`, `RiskConfig`, `MarketConfig`, `ScheduleConfig`, `AIConfig`, `ApiConfig`, `LogConfig`, `load_settings` (cached), `validate_settings`, `get_secret`, `load_strategy_definitions` |
| `algotrading/logging_config.py` | Structured JSON/text logging + correlation IDs | `setup_logging`, `JSONFormatter`, `TextFormatter`, `get_correlation_id`, `set_correlation_id` |
| `algotrading/db/models.py` | ORM tables (event-sourced core) | `Base`, `Strategy`, `Signal`, `TradeIntent`, `OrderEvent`, `Position`, `Trade`, `Candle`, `AnalyticsSummary`, `AIRecommendation`, `Meta` |
| `algotrading/db/__init__.py` | Engine, WAL pragmas, sessions | `get_engine`, `get_session_factory`, `init_db`, `get_schema_version`, `get_session`, `SCHEMA_VERSION` |
| `algotrading/db/seed.py` | Seed starter strategies (idempotent) | `ensure_seeded`, `seed_strategies` |
| `algotrading/market/base.py` | Provider interface + candle model | `Candle` (dataclass), `MarketDataProvider` |
| `algotrading/market/binance_rest.py` | Minimal Binance REST client (no ccxt) with retry/backoff | `BinanceRestClient` (klines/ticker public; account/orders signed), `BinanceError`, `with_retry` decorator, `_is_retryable_error`, `_retry_delay` |
| `algotrading/market/binance_provider.py` | Adapter: REST → `Candle` list with circuit breaker | `BinanceMarketProvider` (uses `CircuitBreaker` for fault tolerance) |
| `algotrading/market/circuit_breaker.py` | Circuit breaker pattern for external services | `CircuitBreaker`, `CircuitOpenError`, `CircuitState`, `get_circuit`, `reset_all_circuits` |
| `algotrading/market/demo.py` | Deterministic synthetic feed | `DemoProvider` |
| `algotrading/market/candles.py` | Persist/query candles, backfill | `CandleStore` (upsert/get/latest_ts/prune), `backfill` |
| `algotrading/strategy/base.py` | Strategy protocol + signal model | `Signal`, `Strategy` (stateless, pure) |
| `algotrading/strategy/indicators.py` | Pure-Python TA (no numpy) | `sma`, `ema`, `rsi`, `atr`, `bollinger_bands`, `macd`, `supertrend`, `vwap`, `closes/highs/lows`, `last_valid` |
| `algotrading/strategy/starters.py` | The built-in strategies (8 total) | `EMACrossover`, `RSIMeanReversion`, `BBMeanReversion`, `MACDTrend`, `SuperTrendStrategy`, `VWAPReclaim`, `MultiTFEMA`, `EnsembleStrategy`, `STRATEGIES` registry dict |
| `algotrading/strategy/registry.py` | Name → class lookup + param validation | `build_strategy`, `known_names`, `UnknownStrategyError` |
| `algotrading/strategy/engine.py` | Evaluate snapshot → persisted signals with optional hot-reload | `StrategyEngine` (`evaluate`, `get_active_strategy`, `_maybe_reload_strategy`, duplicate guard) |
| `algotrading/execution/base.py` | Gateway protocol + result type | `ExchangeGateway`, `OrderResult` |
| `algotrading/execution/risk.py` | Sizing + guards + trailing stops, pure logic | `RiskManager` (`check_buy`, `check_sell`, `size_position`, `compute_trailing_stop_price`, `check_trailing_stop`), `RiskDecision` |
| `algotrading/execution/engine.py` | Signal → intent → order → event + trailing stops | `ExecutionEngine.execute` (intent-before-order, idempotency key), `update_trailing_stops`, `_init_trailing_stop` |
| `algotrading/execution/paper_gateway.py` | Simulated fills w/ slippage + 0.1% fee | `PaperGateway` |
| `algotrading/execution/live_gateway.py` | Real Binance market orders | `LiveGateway` (status `unknown` on ambiguous failure) |
| `algotrading/ledger/store.py` | Event-sourced lifecycle + rebuild | `Ledger` (`mark_filled/mark_sent/mark_failed/mark_risk_skipped`, `rebuild_positions`) |
| `algotrading/analytics/metrics.py` | Pure metric math | `compute_metrics`, `Metrics`, loss tags |
| `algotrading/analytics/service.py` | Snapshot metrics into `analytics_summaries` | `AnalyticsService.run`, `run_daily` |
| `algotrading/ai/client.py` | OpenAI-compatible chat client (requests) | `AIClient.complete_json`, `extract_json`, `RecommendationError` |
| `algotrading/ai/prompt_builder.py` | Deterministic prompt from local data | `build_prompt`, `build_feature_window` |
| `algotrading/ai/assistant.py` | LLM → validated PENDING recommendation | `Assistant.propose`, `_validate` |
| `algotrading/store/recommendations.py` | Recommendation CRUD + human apply + pending creation | `RecommendationStore` (`pending`, `mark_rejected`, `apply`), `create_pending_recommendation`, `ALLOWED_KINDS` / `ALLOWED_STRATEGY_NAMES` |
| `algotrading/store/strategy_versions.py` | Controlled single-active release | `latest_version`, `create_new_version`, `promote_to_active` |
| `algotrading/telegram/bot.py` | PTB Application bootstrap + factories | `build_application`, approve/reject callbacks |
| `algotrading/telegram/commands.py` | Command handlers + allowlist auth + rate limiting + chat handler wiring | `build_handlers`, `_auth_decorator`, `_rate_limited`, `_check_rate_limit` |
| `algotrading/telegram/chat.py` | LLM orchestrator (plain-text chat): tool-using agent | `run_agent`, `tool_get_status/get_market/backtest/propose_change`, `CHAT_SYSTEM_PROMPT` |
| `algotrading/telegram/ui.py` | Formatting + inline keyboards | `format_status/risk/strategies/summary`, `approval_keyboard` |
| `algotrading/api/app.py` | FastAPI factory with enriched health + metrics | `build_api` (`/health`, `/metrics`, `/status`) |
| `algotrading/api/metrics.py` | Prometheus metrics (optional) | `init_metrics`, `get_metrics`, `record_tick`, `record_signal`, `record_order`, `record_fill`, `tick_timer`, `market_timer`, `execution_timer` |
| `algotrading/api/auth.py` | Bearer guard (constant-time) | `require_token` |
| `algotrading/scheduler/jobs.py` | All periodic jobs + context | `BotContext`, `market_tick`, `analytics_tick`, `analytics_daily`, `ai_review`, `reconcile_tick`, `heartbeat_tick`, `build_scheduler` |
| `algotrading/supervisor/health.py` | Heartbeat file + wake-lock | `HealthMonitor`, `ensure_wake_lock` |
| `algotrading/supervisor/reconcile.py` | Intents vs exchange drift check | `reconcile`, `reconcile_and_report` |
| `algotrading/backtest/runner.py` | Replay candles through a strategy | `run_backtest`, `BacktestResult`, `BacktestTrade` |
| `algotrading/cli_setup.py` | .env + wizard + `algobot` subcommands | `read_env/write_env`, `prompt_for_fields`, `start/stop/status/show_logs/menu/run_setup` |
| `scripts/setup_termux.sh` | Full Termux bootstrap (installs `algobot`) | — |
| `scripts/run_bot.sh` | Supervisor loop: restart + heartbeat watchdog (300s grace period before it can kill; single-instance lock via `data/run_bot.lock`); auto SQLite backup before each start (`data/backups/`, 7-day retention) | — |
| `scripts/init_db.py` | Create DB + seed (idempotent) | — |
| `scripts/setup.py`, `scripts/algobot` | Thinnest wrappers over `cli_setup` | — |
| `config/settings.yaml` | All tunables: mode, universe, risk, schedule, AI, API | — |
| `config/strategies.yaml` | Strategy param schemas (seeds DB, validates AI diffs) | — |

## 6. Database tables (SQLite, `data/algotrading.db`)

| Table | Purpose | Truth? |
|---|---|---|
| `strategies` | Versioned strategy rows: `draft/approved/active/retired`, params JSON | source |
| `signals` | Candidate entries/exits from the strategy engine (`candidate/sent/skipped`) | source |
| `trade_intents` | Every order attempt w/ UNIQUE `idempotency_key`; status `pending/sent/filled/canceled/failed/skipped` | source |
| `order_events` | **Immutable append-only event log** (`accepted/fill/partial/canceled/rejected/risk_skipped`) | **source of truth** |
| `positions` | Derived open positions, rebuilt from events at startup | derived |
| `trades` | Derived closed trades w/ PnL + loss tags | derived |
| `candles` | OHLCV (PK: symbol+interval+ts) | source |
| `analytics_summaries` | Periodic metric snapshots (30m/daily) | source |
| `ai_recommendations` | AI proposals: `pending/approved/rejected/applied` + backtest_json | source |
| `meta` | Key/value: `schema_version`, `strategies_seeded` | source |

Migration hook: `SCHEMA_VERSION` in `algotrading/db/__init__.py` (bump + add migration in `init_db`).

## 7. Feature map — "I want to…" → where to go

| Task | Start here |
|---|---|
| Add a new strategy | `strategy/starters.py` (class + register in `STRATEGIES`) → `config/strategies.yaml` (schema) → `store/recommendations.py` `ALLOWED_STRATEGY_NAMES` → tests: `test_indicators.py`, `test_backtest.py` |
| **Built-in strategies** (ready to use) | `ema_crossover`, `rsi_mean_reversion`, `bb_mean_reversion`, `macd_trend`, `supertrend`, `vwap_reclaim`, `multi_tf_ema`, `ensemble` |
| Add a new indicator | `strategy/indicators.py` (pure lists, oldest→newest, NaN padding) |
| Change risk limits / sizing | `execution/risk.py` + `config/settings.yaml` `risk:` block |
| Change position sizing in execution | `execution/engine.py` `_qty_for` (currently fixed-fraction; ATR-aware noted as future slot) |
| Add a Telegram command | `telegram/commands.py` `build_handlers` (decorate with `@auth`) → `telegram/ui.py` formatter |
| Add a scheduled job | `scheduler/jobs.py`: function(ctx) + register in `build_scheduler` + `ScheduleConfig` in `config.py`/`settings.yaml` |
| Change tick interval / schedule | `config/settings.yaml` `schedule:` (validated in `config.py`) |
| Change market timeframes | `config/settings.yaml` `market.intervals:` (1m,5m,15m,30m,1h,4h,1d) |
| Add an API endpoint | `api/app.py` `build_api` (+ `require_token` for anything sensitive) |
| New AI recommendation kind | `store/recommendations.py` `ALLOWED_KINDS` → `ai/assistant.py` + `telegram/chat.py` prompts → `apply` → approval flow |
| Chat with the bot in natural language | `telegram/chat.py` (`run_agent` + tools); handler wired in `telegram/commands.py`; requires `AI_API_KEY` |
| Add a DB table / column | `db/models.py` + bump `SCHEMA_VERSION` in `db/__init__.py` |
| Add a Binance endpoint | `market/binance_rest.py` (public vs signed helpers) |
| Change fill/order behavior | `execution/paper_gateway.py` / `live_gateway.py` (keep the `ExchangeGateway` protocol) |
| Tune the AI prompt | `ai/prompt_builder.py` (deterministic — same inputs ⇒ same prompt) |
| Change log format (text/JSON) | `config/settings.yaml` `log:` block → `logging_config.py` handles formatting |
| Config validation on startup | `config.py` `validate_settings()` → called in `main.py` before `build_context` |
| Automated SQLite backup | `scripts/run_bot.sh` → creates `data/backups/algotrading-YYYY-MM-DD-HHMMSS.db` before each start, 7-day retention |
| Binance API retry with backoff | `market/binance_rest.py` `with_retry` decorator on `_get`/`_post`/`_delete` (3 attempts, exponential + jitter) |
| Market data circuit breaker | `market/circuit_breaker.py` + `binance_provider.py` (5 failures -> open, 60s -> half-open, auto-recovery) |
| Enriched /health endpoint | `api/app.py` → DB latency, market freshness, scheduler heartbeat age |
| Telegram command rate limiting | `telegram/commands.py` `_rate_limited` decorator (10 req/60s per user, in-memory sliding window) |
| Strategy param hot-reload | `strategy/engine.py` `_maybe_reload_strategy()` + `config/settings.yaml` `strategy.hot_reload` (gated, off by default) |
| Prometheus /metrics endpoint | `api/metrics.py` + `api/app.py` `/metrics` (optional, gated by `metrics.enabled`, requires prometheus-client) |
| Trailing stop support | `execution/risk.py` + `execution/engine.py` + `config/settings.yaml` `risk.trailing_stop_pct` (optional, gated, off by default) |
| Ensemble/filter strategies | `strategy/starters.py` `EnsembleStrategy` + `config/strategies.yaml` `ensemble` + `store/recommendations.py` `ensemble_strategy`/`filter_strategy` kinds (consensus/any/filter/weighted modes) |

## 8. Debugging guide

**Logs:** `logs/algotrading.log` (rotating, 5 MB × 3). Supervisor prints to stdout of
`run_bot.sh`; `algobot logs [N]` tails it. Heartbeat: `data/heartbeat` (mtime = last beat).

**Log formats:** Set `log.format: json` in `settings.yaml` for structured JSON logs
with correlation IDs (request/trace tracking). JSON fields configurable via
`log.json_fields`. Default is human-readable text with `[correlation_id]` prefix.

**Startup order (main.py `run`):** init_db → ensure_seeded → `Ledger.rebuild_positions`
(replay events) → startup `reconcile` → scheduler.start → telegram → API.

**Common failure signatures:**

| Symptom | Likely cause / where to look |
|---|---|
| "no candles fetched" each tick | provider failing; check `market/binance_rest.py` network errors, demo mode off |
| "data stale; freezing new entries" | candle ts older than interval + `max_staleness_seconds`; time/clock drift or provider stalled — freeze is intentional |
| signal exists but never executes | status `skipped` + `risk_skipped` event (`RiskManager` guard: cooldown, max positions, no position to close) |
| "duplicate signal" (`[dup]` rationale) | `StrategyEngine._skip_duplicate` — already holding the symbol, buy skipped |
| order status `unknown` | `LiveGateway` ambiguous network/fill failure — reconcile job must confirm; never blindly retry |
| "reconcile drift" | local intent has no matching exchange order (matched by `idempotency_key`); check `supervisor/reconcile.py` |
| bot killed ~10s after start, in a loop | old watchdog kill window — a slow boot (wake-lock/network stall) can exceed it before the first heartbeat. Current `run_bot.sh` has a 300s grace period and a single-instance lock (`data/run_bot.lock`); remove a stale lock only when no `run_bot.sh` is running |
| bot running but never replies | check `logs/algotrading.log` for `Ignoring non-allowlisted user <id>` — your Telegram user ID must be in `TELEGRAM_ALLOWED_USERS`; verify your ID via @userinfobot |
| bot receives messages but replies fail ("No error handlers are registered" in log) | MarkdownV2 parse errors on unescaped `_ ( ) .` — the telegram layer now uses HTML parse mode (`TEXT_MARKDOWN = "HTML"` in `commands.py`); HTML only treats `< > &` specially, which our texts never contain |
| AI never proposes | `ai.enabled` false (no `AI_API_KEY`), or PENDING rec already exists (one-open-question rule), or `_validate` rejected output |
| strategy eval broken after release | version `retired/active` mismatch in `store/strategy_versions.py`; params JSON invalid → engine logs "Cannot build active strategy" |
| `No module named algotrading/main` | bot started from wrong directory (see §2) |
| schema/migration issues | `SCHEMA_VERSION` + `meta` table; derived tables are rebuilt from `order_events`, never hand-edit |

**Trace one trade end-to-end:** signal row → `trade_intents` (idempotency_key) →
`order_events` for that intent → resulting `positions`/`trades` row. All transitions
are recorded; if state disagrees with events, `Ledger.rebuild_positions()` is the
deterministic fix.

## 9. Tests map

| File | Covers |
|---|---|
| `test_indicators.py` | SMA/EMA/RSI/ATR math |
| `test_risk.py` | sizing, exposure caps, cooldowns |
| `test_execution_flow.py` | e2e paper: signal → intent → fill → ledger events → rebuild |
| `test_scheduler.py` | M7 job wiring: market tick, stale-data freeze, demo feed |
| `test_analytics.py` | metric math + summary persistence |
| `test_ai.py` | AI client parsing + prompt determinism |
| `test_m6c.py` | recommendation store + apply, assistant validation, versioned release |
| `test_telegram.py` | allowlist auth, /status formatting, approval flow |
| `test_api.py` | /health open, /status bearer-guarded |
| `test_live_gateway.py` | live gateway + supervisor units |
| `test_backtest.py` | shadow backtest replay + equity summary |

## 10. Keeping this map in sync (git hook / CI)

- **One-time install:** `bash scripts/install_hooks.sh` writes `.git/hooks/pre-commit`,
  which runs `scripts/check_project_map.sh` on every commit. Uninstall:
  `rm .git/hooks/pre-commit` (a pre-existing foreign hook is backed up to
  `.git/hooks/pre-commit.bak` at install time).
- **CI:** `.github/workflows/ci.yml` runs on every push and pull request: it installs
  the dependencies (pydantic v1 pinned explicitly, mirroring the Termux-tested
  environment), runs `bash scripts/check_project_map.sh`, then `pytest tests/ -q`.
  The check is exit-1-on-failure, so it gates merges the same way the hook gates
  local commits.
- **What the check verifies:**
  1. The map exists and has the expected structure (anchor lines).
  2. **Every source file** in the git index under `algotrading/`, `scripts/`, `config/`
     and `tests/` is mentioned in the map **by name** — a new module that isn't
     documented fails the commit regardless of mtimes (`__init__.py` markers exempt).
  3. No tracked file that **actually differs from HEAD** (a real content change, not a
     `git restore` mtime bump) is newer than the map.
- **Limits of the check:** it verifies *presence*, not *accuracy*. Renaming a function
  inside an existing file, changing a data-flow step, or altering a schedule won't be
  caught — that part stays the human/agent rule at the top of this file. When you
  add/rename/remove anything, update this map; the hook catches the cases where a file
  was added or changed and the map was left untouched.

## 11. Non-negotiable invariants (do not break)

1. **Intent-before-order:** persist `trade_intent` w/ unique `idempotency_key` before any order; retries reuse it.
2. **Event-sourced ledger:** `order_events` immutable; positions/trades are derived views.
3. **AI advisory only:** proposals PENDING until human approval; AI never touches exchange or active strategy. This includes chat: the LLM orchestrator (`telegram/chat.py`) can read state, research, and backtest, but the only way it can change anything is `propose_change` (PENDING).
4. **One active strategy version** at a time (`store/strategy_versions.py`).
5. **Stale-data freeze:** no new entries when candles too old (protective exits only).
6. **Live requires keys + explicit mode;** bot refuses to start otherwise.
7. **Thread-safety:** each job/thread opens its own DB session.
8. **Termux compatibility:** keep deps dependency-light (no ccxt/openai-sdk/numpy/pandas; pydantic v1; pure-Python math).