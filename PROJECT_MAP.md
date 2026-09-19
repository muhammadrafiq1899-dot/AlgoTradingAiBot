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

- **Plug-and-play modules:** every non-execution subsystem (market data, strategy
  sources, analytics, control surface, API) is a module selected from
  `config/settings.yaml` `modules:` and composed by `algotrading/modules/`. Adding,
  replacing, or disabling one is a config change — see §5.
- **Execution is a locked module:** the order gateway can only be a built-in module
  and cannot be disabled or replaced by config/external code (deterministic order path).
- **Strategies are files:** user/AI-authored strategies live in `strategies/*.py`,
  are AST-validated on load, and go live only after human approval. They are
  first-class ensemble components too, so several approved strategies can be
  combined (`ensemble` mode consensus/any/filter/weighted) into one active strategy.
- **Execution is deterministic Python; AI is advisory only** (proposals need human approval).
- **Advisory LLM provider:** either an external OpenAI-compatible API (`AI_API_KEY`)
  or the **local Hermes Agent CLI** (`USE_HERMES=true`) — see `algotrading/hermes_ai/`.
  Either one sets `settings.ai.enabled`; the nested Hermes call runs with a
  terminal-free toolset so it cannot act on the machine. The Hermes provider can
  also **read images** (a Telegram photo/screenshot attached with `--image`); the
  API-key provider cannot, and says so instead of ignoring the picture.
- **Modes:** `paper` (simulated fills vs live prices, no keys) / `live` (real orders, requires keys).
- **Data:** SQLite (WAL), event-sourced trade lifecycle (`order_events` is the source of truth).
- **Stack constraints (Termux/Android):** no ccxt (pulls Rust `cryptography`), no `openai` SDK
  (pulls Rust `jiter`), pydantic **v1** only, pure-Python indicators (no numpy/pandas).
- Version: `algotrading/__init__.py` → `__version__ = "0.1.0"`.
- Tests: `pytest tests/` (256 passing, all M0–M7 milestones; includes a full non-technical-user scenario in `tests/test_scenario_end_to_end.py`).

## 2. How to run

```bash
bash scripts/setup_termux.sh            # one-time: pkgs, venv, deps, .env, DB init, wake-lock
.venv/bin/python -m algotrading.main               # paper mode vs live Binance prices
.venv/bin/python -m algotrading.main --demo-data   # offline synthetic feed (forces paper)
.venv/bin/python -m algotrading.main --no-telegram # headless
bash scripts/run_bot.sh                 # supervised loop: auto-restart + sleep-aware heartbeat watchdog
.venv/bin/python -m pytest tests/ -q    # test suite
```

⚠️ **Gotcha:** `python -m algotrading.main` must run from the repo root (or any dir — `main.py`
fixes `sys.path` itself, but `-m` does not). Running `.venv/bin/python -m algotrading/main`
fails with `No module named algotrading/main` and creates stray `bot.log`-style artifacts.

**Android reality — why the bot "force closes" when the screen goes off.** Android 12+
(and vendor ROMs such as vivo/iQOO on top of it) kill the **whole Termux app**, not just
this bot, once it is backgrounded with the screen off — measured on this device: the app
was SIGKILLed ~17 minutes after the screen was locked, taking the supervisor, the bot,
the child watchers and any interactive session with it. Tell-tale signature in the log:
lines stop mid-tick with **no** `shutting down…` / `bye` and **no** watchdog message, and
every Termux session is gone when you come back. Nothing in the bot can prevent that from
inside, so the levers are: keep the wake lock held (the bot requests it at startup, demo
mode included), exempt Termux from battery optimisation, and disable the phantom-process
monitor (Developer options → *Disable child process restrictions*, or `adb shell settings
put global settings_enable_monitor_phantom_procs false`). Keep the bot's CPU/IO footprint
low so the device has no reason to single it out — that is what the incremental fetch and
batched upsert in §4 are for.

## 3. Directory map

```
AlgoTrading/
├── algotrading/              # the whole application (package)
│   ├── main.py               # entry point: composes modules + scheduler, no hard-coded wiring
│   ├── config.py             # pydantic Settings from config/*.yaml + .env secrets
│   ├── cli_setup.py          # .env read/write, setup wizard, algobot start/stop/status/logs
│   ├── modules/              # plug-and-play module framework (base/registry/manager)
│   │   └── builtin/          # shipped modules: market, gateway, strategy, analytics, telegram, api
│   ├── market/               # price data: providers + candle store
│   ├── strategy/             # pure strategies, indicators, signal evaluation engine
│   │                         #   + validation.py (code safety) + plugins.py (strategy files)
│   ├── execution/            # risk checks, trade intents, order gateways (paper/live)
│   ├── ledger/               # event-sourced trade lifecycle (fills → positions/trades)
│   ├── db/                   # SQLAlchemy models, engine/session, seeding
│   ├── analytics/            # metrics from closed trades + periodic summaries
│   ├── ai/                   # advisory LLM: client, prompt builder, assistant
│   ├── hermes_ai/            # advisory LLM via the local Hermes Agent CLI (USE_HERMES)
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
├── strategies/               # user/AI-authored strategy plugins (.py), hot-reloaded
├── tests/                    # pytest suite, one file per subsystem
├── data/                     # runtime: algotrading.db, heartbeat, algobot.pid (gitignored)
└── logs/                     # algotrading.log (rotating, gitignored)
```

## 4. Data flow (read this before touching anything)

**Composition (`algotrading/modules/manager.py`):** `main.py` builds the
`BotContext` + `ModuleManager`, then `ModuleManager.setup` runs each enabled
module in capability order (market → execution → strategy → analytics → control
→ api), installing `ctx.provider` / `ctx.gateway` and capability keys into
`ctx.services`. Modules contribute scheduled jobs via `Module.jobs(ctx)`, merged
into `build_scheduler` as `ctx.extra_jobs`; async control surfaces start after
the scheduler. `--no-telegram` disables `control.telegram`; `--demo-data`
selects `market.demo` + `gateway.paper`.

**Stock plugin loading (`strategy.plugins` module):** the loader scans
`modules.strategy_plugin_paths` (`strategies/*.py`), AST-validates each file, and
registers classes into the strategy registry alongside the built-ins. A periodic
job (`strategy_plugins_reload`) picks up hand-edits without a restart.

**Main loop (every `market_tick_seconds`, default 60s):**

```
scheduler.market_tick (algotrading/scheduler/jobs.py)
  → market.BinanceMarketProvider|DemoProvider.fetch_klines(since_ms=newest stored)
  → CandleStore.upsert (incremental: skips rows older than the stored latest, one
    batched INSERT ... ON CONFLICT DO UPDATE) → DB table `candles`
  → protective exits first: execution.update_trailing_stops (+ execute the sell signals)
  → stale-data freeze check (max_staleness_seconds)
  → strategy.StrategyEngine.evaluate(snapshot)   → DB table `signals` (candidate|skipped)
  → execution.ExecutionEngine.execute(signal_id)
      → risk.RiskManager.check_buy/check_sell   (reject → `risk_skipped` event)
      → persist TradeIntent (UNIQUE idempotency_key) BEFORE order  ← safety rule #1
      → gateway.place_market_order (PaperGateway | LiveGateway)
      → ledger.Ledger.mark_filled/mark_sent/mark_failed  → immutable `order_events`
      → positions/trades derived views updated incrementally
```

**Core scheduled jobs** (`scheduler/jobs.py`, always on): `market_tick`,
`reconcile` (15m local intents vs exchange open orders), `heartbeat` (60s touch
of `data/heartbeat`). **Module-contributed jobs** (from `Module.jobs(ctx)`):
`analytics` (30m metrics snapshot), `analytics_daily` (UTC midnight), `ai_review`
(daily advisory LLM proposal → PENDING recommendation) from the analytics module,
and `strategy_plugins_reload` from the strategy module.

**Control surfaces (do NOT drive the pipeline):** Telegram commands (`/status`,
`/strategy`, `/risk`, `/summary`, `/start_bot`, `/stop_bot`, inline ✅/❌ approval),
plain-text **chat** (`telegram/chat.py`: the LLM orchestrator answers questions and
can call `get_status` / `get_market` / `backtest` / `propose_change` tools),
**image chat** (Telegram photos are buffered per chat for `PHOTO_BATCH_DELAY_SECONDS`;
one picture is answered immediately with the image attached, several at once get a
choice keyboard — 🧩 *one strategy from all* (each image read by a vision pass, then
ONE strategy written over the transcripts) or 🧱 *separate strategies → ensemble*
(one agent call per image, follow-up points at the ensemble step) — handled by
`commands.image_cmd` / `_flush_photo_batch` / `_run_image_flow` and the
`imgflow:<token>:<mode>` callback; uploads are deleted once read), and the internal
API (`/health`, `/status`). AI approval path:
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
| `algotrading/main.py` | Entry point, module composition, lifecycle | `parse_args`, `build_context` → `(BotContext, ModuleManager)`, `AppController`, `run`, `main` (shutdown waits for in-flight scheduler jobs) |
| `algotrading/config.py` | YAML + .env → validated settings | `Settings`, `RiskConfig`, `MarketConfig`, `ScheduleConfig`, `AIConfig` (incl. `images_enabled`, `image_max_bytes`, `image_dir` — resolved to an absolute path), `ApiConfig`, `LogConfig`, `ModulesConfig`, `load_settings` (cached), `validate_settings`, `get_secret`, `load_strategy_definitions` |
| `algotrading/modules/base.py` | Module contract, capabilities, execution lock | `Module`, `ModuleSpec`, `ServiceBag`, `CAPABILITY_MARKET/EXECUTION/STRATEGY/ANALYTICS/CONTROL/API/SCHEDULER_CONTROL`, `LOCKED_CAPABILITIES`, `BUILD_ORDER`, `sort_modules`, `ModuleError` |
| `algotrading/modules/registry.py` | Name → class + config-driven resolution | `register_module`, `get_module`, `all_modules`, `module_names`, `load_external_modules`, `default_enabled_names`, `resolve_modules`, `ModuleSpec` |
| `algotrading/modules/manager.py` | Module lifecycle (setup/start/stop) + job collection | `ModuleManager` (`resolve`, `setup`, `start`, `stop`, `collect_jobs`, `describe`), `build_manager` |
| `algotrading/modules/builtin/market_provider.py` | `market` capability (config-selectable) | `BinanceMarketModule` (`market.binance`), `DemoMarketModule` (`market.demo`) |
| `algotrading/modules/builtin/execution_gateway.py` | `execution` capability — **LOCKED** | `PaperGatewayModule` (`gateway.paper`), `LiveGatewayModule` (`gateway.live`, keys required) |
| `algotrading/modules/builtin/strategy_source.py` | `strategy` capability + plugin reload job | `StrategyPluginModule` (`strategy.plugins`), `reload_strategy_plugins`, `RELOAD_JOB_ID` |
| `algotrading/modules/builtin/analytics_jobs.py` | `analytics` capability + metrics/AI jobs | `AnalyticsModule` (`analytics.default`) |
| `algotrading/modules/builtin/telegram_control.py` | `control` capability (async start/stop) | `TelegramControlModule` (`control.telegram`), `make_recommendation_pusher` |
| `algotrading/modules/builtin/http_api.py` | `api` capability (optional) | `HttpApiModule` (`api.http`) |
| `algotrading/logging_config.py` | Structured JSON/text logging + correlation IDs | `setup_logging`, `JSONFormatter`, `TextFormatter`, `get_correlation_id`, `set_correlation_id` |
| `algotrading/db/models.py` | ORM tables (event-sourced core) | `Base`, `Strategy`, `Signal`, `TradeIntent`, `OrderEvent`, `Position`, `Trade`, `Candle`, `AnalyticsSummary`, `AIRecommendation`, `Meta` |
| `algotrading/db/__init__.py` | Engine, WAL pragmas, sessions | `get_engine`, `get_session_factory`, `init_db`, `get_schema_version`, `get_session`, `SCHEMA_VERSION` |
| `algotrading/db/seed.py` | Seed starter strategies (idempotent) | `ensure_seeded`, `seed_strategies` |
| `algotrading/market/base.py` | Provider interface + candle model | `Candle` (dataclass), `MarketDataProvider` |
| `algotrading/market/binance_rest.py` | Minimal Binance REST client (no ccxt) with retry/backoff | `BinanceRestClient` (klines/ticker public; account/orders signed), `BinanceError`, `with_retry` decorator, `_is_retryable_error`, `_retry_delay` |
| `algotrading/market/binance_provider.py` | Adapter: REST → `Candle` list with circuit breaker | `BinanceMarketProvider` (uses `CircuitBreaker` for fault tolerance) |
| `algotrading/market/circuit_breaker.py` | Circuit breaker pattern for external services | `CircuitBreaker`, `CircuitOpenError`, `CircuitState`, `get_circuit`, `reset_all_circuits` |
| `algotrading/market/demo.py` | Deterministic synthetic feed | `DemoProvider` |
| `algotrading/market/candles.py` | Persist/query candles, backfill | `CandleStore` (upsert/get/latest_ts/prune), `backfill`. `upsert` only writes rows at/after the newest stored candle and does it with **one batched `INSERT ... ON CONFLICT DO UPDATE`** — the old per-row `execute(delete(...))` + `add()` loop let autoflush flush the pending batch on every iteration (O(n²): ~10s per 1000 candles, which pinned a full CPU core inside every 60s tick on a phone) |
| `algotrading/strategy/base.py` | Strategy protocol + signal model | `Signal`, `Strategy` (stateless, pure) |
| `algotrading/strategy/indicators.py` | Pure-Python TA (no numpy) + indicator registry | `sma`, `ema`, `rsi`, `atr`, `bollinger_bands`, `macd`, `supertrend`, `vwap`, `closes/highs/lows`, `last_valid`, `register_indicator`, `create_indicator` |
| `algotrading/strategy/validation.py` | One safety gate for AI/user-authored code | `validate_strategy_code`, `validate_indicator_code`, `compile_strategy` (also probes the constructor — see below), `compile_indicator`, `safe_namespace`, `_safe_import`, `CodeValidationError`. Imports allowed: the indicator helpers, `math`, `statistics` — `from algotrading.strategy import indicators as ta`, bare `from indicators import ...`, `import math`; `_import_allowed` is the single rule read by both the AST check and the runtime import hook, and sibling access (`from algotrading.strategy import registry`) is refused. `_check_constructor` rejects a class that cannot be built as `cls(params_dict)` |
| `algotrading/strategy/plugins.py` | Load/write/hot-reload strategy files on disk | `StrategyPluginLoader`, `StrategyPlugin`, `StrategyPluginError`, `get_default_loader`, `configure_default_loader`, `DEFAULT_PLUGIN_DIR` |
| `algotrading/strategy/starters.py` | The built-in strategies (8 total) | `EMACrossover`, `RSIMeanReversion`, `BBMeanReversion`, `MACDTrend`, `SuperTrendStrategy`, `VWAPReclaim`, `MultiTFEMA`, `EnsembleStrategy` (components resolved through the registry via `_build_component`, so AI-authored plugin strategies combine too; note `consensus` = "no firing component disagrees"), `STRATEGIES` registry dict |
| `algotrading/strategy/registry.py` | Name → class lookup (built-ins + plugins) | `build_strategy`, `known_names`, `builtin_names`, `plugin_names`, `is_builtin`, `is_plugin`, `reload_plugins`, `UnknownStrategyError` |
| `algotrading/strategy/catalog.py` | Live strategy catalog (schemas + ranges) for prompts/UI | `catalog_entries`, `default_params`, `StrategyEntry`, `ParamSpec` (`range_text`, `compact`), `BUILTIN`/`PLUGIN` |
| `algotrading/strategy/engine.py` | Evaluate snapshot → persisted signals with optional hot-reload | `StrategyEngine` (`evaluate`, `get_active_strategy`, `_maybe_reload_strategy`, duplicate guard) |
| `algotrading/execution/base.py` | Gateway protocol + result type | `ExchangeGateway`, `OrderResult` |
| `algotrading/execution/risk.py` | Sizing + guards + trailing stops, pure logic | `RiskManager` (`check_buy`, `check_sell`, `size_position`, `compute_trailing_stop_price`, `check_trailing_stop`), `RiskDecision` |
| `algotrading/execution/engine.py` | Signal → intent → order → event + trailing stops | `ExecutionEngine.execute` (intent-before-order, idempotency key), `update_trailing_stops` (returns protective sell signals the caller must execute; `highest_price` may be NULL, so it falls back to entry price), `_init_trailing_stop` |
| `algotrading/execution/paper_gateway.py` | Simulated fills w/ slippage + 0.1% fee | `PaperGateway` |
| `algotrading/execution/live_gateway.py` | Real Binance market orders | `LiveGateway` (status `unknown` on ambiguous failure) |
| `algotrading/ledger/store.py` | Event-sourced lifecycle + rebuild | `Ledger` (`mark_filled/mark_sent/mark_failed/mark_risk_skipped`, `rebuild_positions`) |
| `algotrading/analytics/metrics.py` | Pure metric math | `compute_metrics`, `Metrics`, loss tags |
| `algotrading/analytics/service.py` | Snapshot metrics into `analytics_summaries` | `AnalyticsService.run`, `run_daily` |
| `algotrading/ai/client.py` | OpenAI-compatible chat client (requests) | `AIClient.complete_json`, `extract_json`, `RecommendationError` |
| `algotrading/hermes_ai/client.py` | Advisory LLM via the local Hermes Agent CLI (`USE_HERMES=true`); prompts ship as `-q` args in a terminal-free, rule-free invocation | `HermesAgentClient` (`complete_json(messages, image=None)`, `complete_text` — verbatim answer, no JSON requirement, for the per-image transcription pass; `enabled`), `parse_stream_json` (reads the `stream-json` `result` event — plain output echoes the query, so a naive text parse returns the *prompt's* JSON), `salvage_partial_reply` (a provider-cut answer that is clearly a `reply` still yields its partial text; truncated tool calls stay errors), `_ask`/`_command(prompt, image=...)` (`--image PATH`, `--max-turns 2` when an image is attached so the CLI-injected `vision_analyze` turn fits), `SAFE_TOOLSET`, `MAX_TURNS`/`IMAGE_MAX_TURNS`, env knobs `HERMES_CLI`/`HERMES_TOOLSETS`/`HERMES_TIMEOUT` |
| `algotrading/hermes_ai/__init__.py` | Package export | `HermesAgentClient`, `parse_stream_json` |
| `algotrading/ai/prompt_builder.py` | Deterministic prompt from local data | `build_prompt`, `build_feature_window` |
| `algotrading/ai/assistant.py` | LLM → validated PENDING recommendation (client chosen from `settings.ai`: Hermes Agent or external API) | `Assistant.propose`, `_validate`, `_validate_ast` (runs the same `compile_strategy` gate, so the daily review can't file a proposal that fails to build) |
| `algotrading/store/recommendations.py` | Recommendation CRUD + human apply + pending creation | `RecommendationStore` (`pending`, `mark_rejected`, `apply`, `_shadow_backtest`), `create_pending_recommendation`, `allowed_strategy_names`, `ALLOWED_KINDS` (incl. `new_strategy` / `edit_strategy` / `new_indicator`); `create_pending_recommendation` runs the full `compile_strategy` gate on authored templates *before* a PENDING row exists (bad code comes back to the model as tool feedback instead of a dead Approve); `apply` authors `new_strategy`/`edit_strategy` code *before* the shadow backtest (the name doesn't resolve otherwise) and skips it for `new_indicator` (a function, not a strategy) |
| `algotrading/store/strategy_versions.py` | Controlled single-active release + strategy file writes | `latest_version`, `active_version`, `create_new_version`, `create_new_strategy` (removes the file when the write fails to load, so the name isn't poisoned), `update_strategy_code`, `promote_to_active` (retires every other active row — exactly one active strategy overall) |
| `algotrading/telegram/bot.py` | PTB Application bootstrap + factories | `build_application`, `_make_approve`, `_make_reject`, `_make_backtest` (current vs default comparison, via `_current_params`), `_make_catalog_scores` (ranked scores, cached, threaded), `_stored_candles`, `run_bot` |
| `algotrading/telegram/commands.py` | Command handlers + allowlist auth + rate limiting + chat/image batching + image-flow choice | `build_handlers` (`on_start/on_stop/on_approve/on_reject/on_backtest/on_catalog_scores`, `chat_cmd`, `image_cmd`), `collect_image` (photo/document → local file, gated *before* download: assistant enabled, `ai.images_enabled`, `use_hermes`, `image_max_bytes`), `_uploads_dir`, `_image_suffix`, `_image_path`, `_prune_uploads` (24h sweep of crash leftovers), `_delete_uploads`, `_deliver_agent_result` (answer + approval card, via `reply_text` or `bot.send_message`), image batching: `_photo_batches`/`_photo_batch_tasks`/`_image_flows`, `_schedule_photo_flush`, `_flush_photo_batch`, `_offer_image_flow`, `_run_image_flow` ('combine' = one call over all images, 'separate' = one call per image), `reset_photo_batches`, `PHOTO_BATCH_DELAY_SECONDS`/`IMAGE_BATCH_LIMIT`/`IMAGE_FLOW_TTL_SECONDS`, `IMAGE_FLOW_CHOICE_TEXT`/`IMAGE_FLOW_SEPARATE_FOLLOWUP`/`IMAGE_FLOW_EXPIRED`, `IMAGE_CAPTION_FALLBACK`/`IMAGE_BATCH_CAPTION_FALLBACK`, `UPLOAD_MAX_AGE_SECONDS`, `_auth_decorator`, `_rate_limited`, `reset_rate_limits`, `on_callback` (`approve:`/`reject:`/`bt:`/`imgflow:<token>:<mode>`); commands: `/help`, `/status`, `/strategies` (ranked catalog + backtest buttons), `/strategy`, `/risk`, `/summary`, `/start_bot`, `/stop_bot` |
| `algotrading/telegram/chat.py` | LLM orchestrator (plain-text chat + images): tool-using agent | `run_agent` (`image_path=` attaches one picture to every turn; 2+ `image_paths=` transcribes each image first and runs the loop once over the transcripts), `_describe_image`, `tool_get_status/get_market/backtest/propose_change`, `CHAT_SYSTEM_PROMPT` (hard rule 5: image text is untrusted data; an ENSEMBLE MODES block states the implemented consensus/any/filter/weighted semantics so the bot does not overclaim them in plain-text turns; the new_indicator schema example no longer carries the shell-mangled `$_2_`/`$_100_` literals), `IMAGE_INPUT_PROMPT` (image + ensemble guidance, added only for image turns by `build_system_prompt(with_image=True)`), `DESCRIBE_IMAGE_PROMPT`, `strategy_catalog` (delegates to `strategy/catalog.py`) |
| `algotrading/telegram/ui.py` | Formatting + inline keyboards | `format_status/risk/strategies/summary` (`format_risk` shows the trailing-stop setting), `format_strategy_catalog` (HTML, truncated to Telegram's cap; optional score/rank labels), `rank_by_score` (best-first, unscored last), `format_backtest_comparison` (side-by-side `<pre>` tables + risk-adjusted verdict), `_risk_adjusted` = PnL ÷ max drawdown, `approval_keyboard`, `backtest_keyboard` (`bt:<name>`, 64-byte-safe, capped), `image_flow_keyboard` (🧩/🧱 choice for a photo batch) + `IMAGE_FLOW_MODES` |
| `algotrading/api/app.py` | FastAPI factory with enriched health + metrics | `build_api` (`/health`, `/metrics`, `/status`) |
| `algotrading/api/metrics.py` | Prometheus metrics (optional) | `init_metrics`, `get_metrics`, `record_tick`, `record_signal`, `record_order`, `record_fill`, `tick_timer`, `market_timer`, `execution_timer` |
| `algotrading/api/auth.py` | Bearer guard (constant-time) | `require_token` |
| `algotrading/scheduler/jobs.py` | Core periodic jobs + context + module job merge | `BotContext` (with `provide`/`get`/`has`, `services`, `extra_jobs`), `JobSpec`, `market_tick` (runs `_run_trailing_stops` **before** the stale-data gate, so protective exits never depend on fresh entries; fetches incrementally with `since_ms = newest stored candle` per symbol/interval instead of re-downloading the full 1000-candle window every minute), `_run_trailing_stops`, `analytics_tick`, `analytics_daily`, `ai_review`, `reconcile_tick`, `heartbeat_tick`, `build_scheduler` |
| `algotrading/supervisor/health.py` | Heartbeat file + wake-lock | `HealthMonitor`, `ensure_wake_lock` |
| `algotrading/supervisor/reconcile.py` | Intents vs exchange drift check | `reconcile`, `reconcile_and_report` |
| `algotrading/backtest/runner.py` | Replay candles through a strategy | `run_backtest`, `BacktestResult`, `BacktestTrade` |
| `algotrading/backtest/stored.py` | Load stored candles for a backtest (shared DB adapter) | `load_candles` → `(symbol, interval, candles)`, `resolve_market`, `DEFAULT_LIMIT` |
| `algotrading/cli_setup.py` | .env + wizard + `algobot` subcommands | `read_env/write_env`, `prompt_for_fields`, `start/stop/status/show_logs/menu/run_setup`. `stop()` SIGTERMs the process group, waits `STOP_GRACE_SECONDS` (20s) and then SIGKILLs it — a tick stuck in an exchange call used to keep the python child alive after the supervisor died (a "stopped" bot still burning CPU, then two bots on one DB after the next start) |
| `scripts/setup_termux.sh` | Full Termux bootstrap (installs `algobot`) | — |
| `scripts/run_bot.sh` | Supervisor loop: restart + heartbeat watchdog (300s grace period before it can kill, extended by however long this loop itself was suspended — a screen-off phone must not look like a hung bot; single-instance lock via `data/run_bot.lock`); auto SQLite backup before each start (`data/backups/`, 7-day retention) | — |
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
| Add / replace a bot part (module) | `algotrading/modules/builtin/` — new `Module` subclass with a `ModuleSpec` + `@register_module`; import it in `builtin/__init__.py`; select it via `config/settings.yaml` `modules:`; tests: `test_modules.py` |
| Choose which parts run | `config/settings.yaml` `modules.enabled/disabled/params` (execution is locked: cannot be disabled or externalised) |
| Run headless (no Telegram) | `--no-telegram`, or `modules.disabled: [control.telegram]` |
| Load a third-party module | `modules.external: ["pkg.mod:MyModule"]` (locked capabilities are refused) |
| Add a new strategy (built-in) | `strategy/starters.py` (class + register in `STRATEGIES`) → `config/strategies.yaml` (schema) → tests: `test_indicators.py`, `test_backtest.py` |
| Add a strategy as a file (user/AI) | `strategies/<name>.py` (convention in `strategies/README.md`); validated by `strategy/validation.py`, loaded by `strategy/plugins.py`. Contract: `def __init__(self, params=None)` (built as `cls(params_dict)`) and only the indicator/`math`/`statistics` imports |
| Let the AI create / edit a strategy | `new_strategy` / `edit_strategy` recommendation kinds → human ✅ → `store/strategy_versions.py` writes the plugin file + a new version |
| Change what code safety allows | `strategy/validation.py` (single gate for AI/user code; used by plugins, recommendations, indicators) |
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
| Chat with the bot in natural language | `telegram/chat.py` (`run_agent` + tools); handler wired in `telegram/commands.py`; requires the assistant to be enabled (`AI_API_KEY` **or** `USE_HERMES=true`) |
| Send a screenshot/image and have the AI read it | `telegram/commands.py` `image_cmd` → `collect_image` (photo/document → `data/uploads/`, deleted after the call). One picture: `chat.run_agent(image_path=...)` → `hermes_ai/client.py` `--image` (`--max-turns 2`). Several at once: buffered (`PHOTO_BATCH_DELAY_SECONDS`), then a 🧩/🧱 choice keyboard (`image_flow_keyboard`, `imgflow:<token>:<mode>`) — 🧩 `image_paths=` transcribes each image and writes ONE strategy, 🧱 one call per image then the ensemble step. Hermes provider only, sizes capped by `ai.image_max_bytes`; tune wording in `IMAGE_INPUT_PROMPT`/`DESCRIBE_IMAGE_PROMPT` |
| Combine several strategies into one (ensemble) | `strategy/starters.py` `EnsembleStrategy` (components via `_build_component` → registry, so AI-authored plugins qualify) + `store/recommendations.py` `ensemble_strategy`/`filter_strategy` kinds; the chat drafts it from the catalog once the components are approved and active |
| Run the advisory AI with no API key (local Hermes Agent) | `.env` `USE_HERMES=true` (sets `ai.enabled` in `config.py` `load_settings`) → `algotrading/hermes_ai/client.py` shells out to `hermes chat -q --format stream-json -t safe`; picked up by `telegram/chat.py` `run_agent` and `scheduler/jobs.py` `ai_review` |
| Add a DB table / column | `db/models.py` + bump `SCHEMA_VERSION` in `db/__init__.py` |
| Add a Binance endpoint | `market/binance_rest.py` (public vs signed helpers) |
| Change fill/order behavior | `execution/paper_gateway.py` / `live_gateway.py` (keep the `ExchangeGateway` protocol) |
| Tune the AI prompt | `ai/prompt_builder.py` (deterministic — same inputs ⇒ same prompt) |
| Change which strategies the chat model knows about | `strategy/catalog.py` `catalog_entries()` builds it from `config/strategies.yaml` + plugin metadata, so AI-created plugins appear automatically; `config.py` `StrategyParam.type` accepts int/float/bool/str/list |
| Show the available strategies / param ranges in Telegram | `/strategies` → `strategy/catalog.py` `catalog_entries()` → `telegram/ui.py` `format_strategy_catalog()` |
| Rank strategies by risk-adjusted score | `/strategies` → `telegram/bot.py` `_make_catalog_scores` (backtests each strategy with its current params, threaded + 60s cached) → `telegram/ui.py` `rank_by_score` |
| Backtest a strategy from Telegram | `/strategies` inline `bt:<name>` buttons → `telegram/commands.py` `on_callback` → `telegram/bot.py` `_make_backtest` (`_current_params` vs `strategy/catalog.py` `default_params`, replayed via `backtest/stored.py` + `runner.run_backtest`, formatted by `telegram/ui.py` `format_backtest_comparison`; the verdict scores PnL ÷ max drawdown, not raw PnL) |
| Change log format (text/JSON) | `config/settings.yaml` `log:` block → `logging_config.py` handles formatting |
| Config validation on startup | `config.py` `validate_settings()` → called in `main.py` before `build_context` |
| Automated SQLite backup | `scripts/run_bot.sh` → creates `data/backups/algotrading-YYYY-MM-DD-HHMMSS.db` before each start, 7-day retention |
| Binance API retry with backoff | `market/binance_rest.py` `with_retry` decorator on `_get`/`_post`/`_delete` (3 attempts, exponential + jitter) |
| Market data circuit breaker | `market/circuit_breaker.py` + `binance_provider.py` (5 failures -> open, 60s -> half-open, auto-recovery) |
| Enriched /health endpoint | `api/app.py` → DB latency, market freshness, scheduler heartbeat age |
| Telegram command rate limiting | `telegram/commands.py` `_rate_limited` decorator (10 req/60s per user, in-memory sliding window) |
| Strategy param hot-reload | `strategy/engine.py` `_maybe_reload_strategy()` + `config/settings.yaml` `strategy.hot_reload` (gated, off by default) |
| Prometheus /metrics endpoint | `api/metrics.py` + `api/app.py` `/metrics` (optional, gated by `metrics.enabled`, requires prometheus-client) |
| Trailing stop support | `execution/risk.py` + `execution/engine.py` + `config/settings.yaml` `risk.trailing_stop_pct` (optional, gated); `scheduler/jobs.py` `_run_trailing_stops` runs it every tick before the stale gate and executes the returned protective sells |
| Convert a Pine Script strategy into the bot | `strategies/three_commas_bot.py` is a worked example: same loader/validator/approval path as any plugin (`strategies/README.md` has the contract — `cls(params_dict)`, `evaluate(symbol, candles)`, indicator/math imports only). Long-only spot: map Pine shorts to exits, and turn broker stop/limit orders into sell signals |
| Verify the whole product as a user | `tests/test_scenario_end_to_end.py` — the end-to-end journey (setup → trade → every command → AI approval → API → restart) |
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
| trailing stop never moves / never exits | must run **before** the stale gate in `market_tick` (`_run_trailing_stops`) and its returned sell signals must be executed; a `highest_price` of NULL means no tick has run since entry (fall back to entry price) |
| "duplicate signal" (`[dup]` rationale) | `StrategyEngine._skip_duplicate` — already holding the symbol, buy skipped |
| order status `unknown` | `LiveGateway` ambiguous network/fill failure — reconcile job must confirm; never blindly retry |
| "reconcile drift" | local intent has no matching exchange order (matched by `idempotency_key`); check `supervisor/reconcile.py` |
| bot killed ~10s after start, in a loop | old watchdog kill window — a slow boot (wake-lock/network stall) can exceed it before the first heartbeat. Current `run_bot.sh` has a 300s grace period and a single-instance lock (`data/run_bot.lock`); remove a stale lock only when no `run_bot.sh` is running |
| bot running but never replies | check `logs/algotrading.log` for `Ignoring non-allowlisted user <id>` — your Telegram user ID must be in `TELEGRAM_ALLOWED_USERS`; verify your ID via @userinfobot |
| bot receives messages but replies fail ("No error handlers are registered" in log) | MarkdownV2 parse errors on unescaped `_ ( ) .` — the telegram layer now uses HTML parse mode (`TEXT_MARKDOWN = "HTML"` in `commands.py`); HTML only treats `< > &` specially, which our texts never contain |
| AI never proposes | `ai.enabled` false (no `AI_API_KEY` and `USE_HERMES` unset), or PENDING rec already exists (one-open-question rule), or `_validate` rejected output |
| "AI assistant is not configured" even though `USE_HERMES=true` | the bot was started before the `.env` edit (settings are read once at startup), or the value isn't one of `true/1/yes/on` | restart the bot; `USE_HERMES=true` must set `settings.ai.enabled` in `config.py` `load_settings` |
| "USE_HERMES=true but the `hermes` command was not found on PATH" | `shutil.which("hermes")` failed from the bot's environment (`hermes_ai/client.py`); `algobot start` inherits the shell's PATH | run `hermes --version` in the same shell (or pin `HERMES_CLI=/abs/path/hermes`), otherwise set `AI_API_KEY` |
| "Hermes CLI returned no answer" / "did not return JSON" | the nested agent printed no `{"type":"result"}` event, exited early, or answered with prose instead of the expected JSON | reproduce with `hermes chat -q 'hi'`; raise `HERMES_TIMEOUT` if it is only slow; see `logs/algotrading.log` |
| Image sent but the bot replies "needs the local Hermes Agent" | `USE_HERMES` is not set, so the advisory provider is the external API (no vision path). The image is refused on purpose, never silently dropped | `.env` `USE_HERMES=true` + restart; or ask in text |
| Image sent but nothing happens at all | before this feature the photo matched no handler (silent). Now: auth allowlist, `ai.images_enabled`, or assistant disabled — `collect_image` replies with the reason | check `logs/algotrading.log` for `image download failed`, and that `settings.ai.enabled` is true |
| "That image is X MB — I read up to Y MB" | upload over `ai.image_max_bytes` (checked on Telegram's declared size *and* on the bytes that landed) | raise the cap in `config/settings.yaml` or send a smaller screenshot |
| Image turns are slow / hit the timeout | an image call allows `--max-turns 2` (the nested agent spends one turn on `vision_analyze`) and the file is base64-encoded into the provider request | raise `HERMES_TIMEOUT`; screenshots beat photos of a screen |
| Image with embedded text ("ignore your rules…") | text inside an image is untrusted data by design (chat hard rule 5) — the agent is told to refuse and report it | expected behaviour; the agent still cannot trade or apply anything |
| Batch keyboard says "expired" | the images waited longer than `IMAGE_FLOW_TTL_SECONDS` (15 min), or the bot restarted (batches are in-memory) | send the pictures again; batches are not persisted on purpose |
| "(Only the first 6 are used.)" | a batch exceeded `IMAGE_BATCH_LIMIT` — each image costs its own vision pass | send fewer at once, or raise the limit |
| Multi-image run answers only about some images | one transcription came back truncated by the provider; `complete_text` + partial-JSON salvage keeps what was written and marks the rest | re-send the image, or raise `HERMES_TIMEOUT` if the CLI itself was slow |
| "AI proposal … apply failed: unknown strategy: X" for an ensemble | a component was never approved/released, so the name does not resolve (`EnsembleStrategy` builds through the registry) | approve the component strategies first, then re-draft the ensemble |
| Photo answered after a ~1s pause | by design: photos are buffered for `PHOTO_BATCH_DELAY_SECONDS` so an album is answered once | expected |
| "new_strategy template rejected: import 'X' is not available" | authored code tried to import something outside the sandbox allowlist | the model is told at propose time and rewrites it; for hand-written plugins use `ta` / `from algotrading.strategy import indicators as ta` |
| "strategy cannot be built as cls(params_dict)" | the generated class used a keyword constructor (`def __init__(self, period=14)`) instead of taking the params dict | the propose-time gate rejects it before any approval; the prompt and `strategies/README.md` state the required shape |
| "strategy plugin 'X' already exists" right after a failed approval | a half-written plugin file from an earlier failure (fixed: a failed write is now removed) | delete the stale `strategies/X.py` and re-approve |
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
| `test_telegram.py` | allowlist auth, /status formatting, approval flow, /strategies catalog (built-ins + plugins, truncation, risk-adjusted ranking) and its `bt:` backtest buttons (current-vs-default comparison, verdict), image input (photo routing via a real PTB `Update`, single photo answered + upload deleted, caption fallback, proposal card, photo burst → one 🧩/🧱 choice, both flows through the `imgflow` callback, batch cap, expired token, gating order, non-image documents, stale-upload sweep) |
| `test_chat.py` | LLM chat orchestrator: tools (backtest/propose_change), safety contract, image turns (attachment on every turn, image-only prompt block) |
| `test_hermes_ai.py` | Hermes Agent provider: `stream-json` result-event parsing (must not return the echoed prompt's JSON), terminal-free toolset invocation, `USE_HERMES` enabling the assistant with no `AI_API_KEY`, chat gating when the CLI is missing, image attachment (`--image`, `--max-turns 2`, text turns unchanged, missing file refused before the CLI runs), truncated answers (`salvage_partial_reply`: partial reply kept, partial tool call still an error), verbatim `complete_text` for transcription |
| `test_api.py` | /health open, /status bearer-guarded |
| `test_live_gateway.py` | live gateway + supervisor units |
| `test_backtest.py` | shadow backtest replay + equity summary |
| `test_ema_percentage_strategy.py` | EMA + percentage-threshold strategy signals |
| `test_modules.py` | module resolution, external loading, lifecycle, job contribution, execution lock |
| `test_strategy_plugins.py` | plugin load/write/edit, code-safety rejection (including the import allowlist: documented form works, bare `indicators` is aliased, siblings/os are refused), constructor shape (`cls(params_dict)` probe), propose-time gate for uncompilable code, failed-write cleanup, hot-reload, AI authoring flow, ensemble components: an AI-authored plugin combines (and an unknown name still fails) |
| `test_scenario_end_to_end.py` | **Full non-technical-user scenario**: setup wizard, module composition, a real entry + trailing-stop exit (regression: protective exits run with no candidates, NULL `highest_price`, `CandleStore.latest` never existed), every Telegram command + backtest button + approve/reject, chat-driven `new_strategy` approval, analytics, API, restart rebuild |
| `test_three_commas_bot.py` | The Pine v5 "3Commas Bot" port in `strategies/three_commas_bot.py`: real-loader contract, catalog metadata/ranges, entry + MA-cross exit, ATR swing stop, R:R target, ATR trailing exit (and that it beats the static stop on a reversal), rr_exit arming, session (incl. wrap-around) and date filters, all nine MA types, and activation through an approved `param_change` |

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
3. **AI advisory only:** proposals PENDING until human approval; AI never touches exchange or active strategy. This includes chat: the LLM orchestrator (`telegram/chat.py`) can read state, research, and backtest, but the only way it can change anything is `propose_change` (PENDING). When the provider is the local Hermes Agent, the nested call runs terminal-free (`hermes chat -q --format stream-json -t safe --ignore-rules`), so the agent cannot run commands or edit files either. An attached image does not change this: `--image` adds a picture to the same terminal-free call (the CLI injects its `vision_analyze` tool for the attachment only), and text *inside* an image is untrusted data the agent is told to refuse as instructions.
4. **One active strategy version** at a time (`store/strategy_versions.py`): `promote_to_active` retires EVERY other active row, not just the same name's, because `StrategyEngine.get_active_strategy` orders by version without filtering by name — otherwise a fresh approval could silently not take over.
   An ensemble is a version too: approving `ensemble_strategy` makes the ensemble the
   active strategy, and its components must already exist (approved plugins included).
5. **Stale-data freeze:** no new entries when candles too old (protective exits only).
6. **Live requires keys + explicit mode;** bot refuses to start otherwise.
7. **Thread-safety:** each job/thread opens its own DB session.
8. **Termux compatibility:** keep deps dependency-light (no ccxt/openai-sdk/numpy/pandas; pydantic v1; pure-Python math).
9. **Execution is locked:** the `execution` capability may only be provided by built-in modules, and `modules.disabled`/`modules.enabled` cannot remove or replace it. All AI/user-authored code (strategies, indicators) goes through `strategy/validation.py` before it can run, and can never mutate `ctx.gateway` or the ledger.