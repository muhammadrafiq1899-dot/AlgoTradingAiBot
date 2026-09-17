# AlgoTrading — an AI-assisted trading bot you run from your phone

A spot-trading bot that runs on Android via **Termux** (or any computer), is
controlled entirely from **Telegram**, and keeps every trade decision in plain,
deterministic Python. An optional AI assistant reads your results and *suggests*
strategy changes, but it can never place an order or change anything by itself —
you tap **Approve** or it doesn't happen.

> **The short version:** the bot watches prices, runs a strategy, and places
> paper (or real) trades on Binance. You talk to it in Telegram. You can ask the
> AI for a new strategy in plain English and it drafts one for your approval.

**Status:** all milestones complete · **173 tests pass** (`pytest tests/`).
Version `0.1.0`.

---

## Table of contents

**Part 1 — For the trader (no coding required)**
1. [What this bot actually does](#1-what-this-bot-actually-does)
2. [What you need](#2-what-you-need)
3. [Setting it up, step by step](#3-setting-it-up-step-by-step)
4. [Starting, stopping, and checking the bot](#4-starting-stopping-and-checking-the-bot)
5. [Talking to the bot in Telegram](#5-talking-to-the-bot-in-telegram)
6. [The commands, explained plainly](#6-the-commands-explained-plainly)
7. [Choosing a strategy (and comparing them)](#7-choosing-a-strategy-and-comparing-them)
8. [Using the AI assistant](#8-using-the-ai-assistant)
9. [Creating your own strategy by describing it](#9-creating-your-own-strategy-by-describing-it)
10. [Settings you might want to change](#10-settings-you-might-want-to-change)
11. [The safety rules, in plain English](#11-the-safety-rules-in-plain-english)
12. [If something goes wrong](#12-if-something-goes-wrong)
13. [Going live with real money](#13-going-live-with-real-money)
14. [Glossary](#14-glossary)

**Part 2 — For the engineer**
15. [Architecture](#15-architecture)
16. [The plug-and-play module system](#16-the-plug-and-play-module-system)
17. [Strategies as files](#17-strategies-as-files)
18. [Configuration reference](#18-configuration-reference)
19. [Data model & event sourcing](#19-data-model--event-sourcing)
20. [Telegram + chat internals](#20-telegram--chat-internals)
21. [The HTTP API](#21-the-http-api)
22. [Extending the bot](#22-extending-the-bot)
23. [Running as a service on Termux](#23-running-as-a-service-on-termux)
24. [Testing](#24-testing)
25. [Non-negotiable invariants](#25-non-negotiable-invariants)

---
---

# Part 1 — For the trader (no coding required)

## 1. What this bot actually does

Think of it as a tireless assistant that does four things, in a loop, forever:

1. **Watches the market.** Every minute it downloads the latest prices for the
   coins you chose (BTC, ETH by default) and saves them.
2. **Looks for an opportunity.** It runs a *strategy* — a rule like "buy when the
   fast average price crosses above the slow average price" — over those prices.
3. **Places the order.** If the rule says buy and the risk checks allow it, it
   places a trade and writes down exactly what it did and why.
4. **Protects the position.** If you enabled a trailing stop, it follows the
   price up and closes the position automatically if the price falls back.

Meanwhile it can:
- **Message you** on Telegram whenever there's something to report.
- **Answer questions** in plain English when you enable the AI assistant.
- **Draft new strategies** for you to approve — it never activates one on its own.
- **Backtest** any strategy against your saved price history on demand.

By default everything runs in **paper mode**: real market prices, but simulated
trades. No money is at risk. You switch to real trading deliberately, later, by
editing one line.

## 2. What you need

- **A phone (Android) or any computer with Python 3.11+.** The instructions below
  assume Android/Termux, because that's the target. On a computer the same steps
  apply without the Termux-specific bits.
- **A Telegram account** and a bot token (you'll make one in 30 seconds with
  Telegram's @BotFather).
- **Your Telegram numeric user ID** so only *you* can control the bot.
- *(Optional, later)* Binance API keys if you want to trade real money.
- *(Optional)* An AI API key (any OpenAI-compatible service) if you want the
  assistant. Without it, everything else still works.

## 3. Setting it up, step by step

### Step 1 — Create your Telegram bot

1. Open Telegram and message **@BotFather**.
2. Send `/newbot` and follow the prompts (name + username ending in `bot`).
3. BotFather replies with a **token** that looks like `123456789:ABCdef...`.
   Copy it. This is your bot's password — don't share it.

### Step 2 — Find your Telegram user ID

Message **@userinfobot**. It replies with your numeric ID (e.g. `123456789`).
Copy it. This is how the bot knows it's really you.

### Step 3 — Install everything

In Termux:

```bash
bash scripts/setup_termux.sh
```

This installs the system packages, creates the Python environment, installs the
dependencies, creates your `.env` secrets file, and initialises the database. It
is safe to run more than once.

### Step 4 — Configure it

You have two ways to do the same thing.

**Option A — the guided wizard (easiest):**

```bash
bash scripts/setup.py
```

It asks for each value and tells you what it wants. Press Enter to keep a value
that's already shown; type `skip` to leave one blank.

**Option B — edit the file directly:**

```bash
nano .env
```

Fill in at minimum:

```
TELEGRAM_BOT_TOKEN=123456789:ABCdef...
TELEGRAM_ALLOWED_USERS=123456789
```

Optional extras:

|| Setting | What it does |
|---|---|
|| `AI_API_KEY` | Enables the AI assistant via an external LLM (when `USE_HERMES=false`). Leave blank to skip. |
|| `AI_BASE_URL` | The AI service address. Default is OpenAI. Change for other providers. |
|| `AI_MODEL` | Which AI model to use. Default `gpt-4o-mini`. |
|| `USE_HERMES` | Set to `true` to run the assistant on the local Hermes Agent instead of an external LLM (no API key needed). |
|| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Only needed for live trading. |
|| `API_TOKEN` | Only if you enable the optional internal status web server. |

### Step 5 — Start it

```bash
bash scripts/run_bot.sh
```

Or, if you installed the `algobot` shortcut (done by the setup script):

```bash
algobot start
```

Then open your bot in Telegram and send `/help`. If it answers, you're live in
paper mode.

## 4. Starting, stopping, and checking the bot

The `algobot` command is your control panel:

| Command | What it does |
|---|---|
| `algobot` | Opens a menu (start / stop / status / logs / setup / exit) |
| `algobot start` | Starts the bot in the background |
| `algobot stop` | Stops it cleanly |
| `algobot status` | Shows mode, whether it's running, Telegram/AI status, and which coins |
| `algobot logs` | Shows the last 40 log lines (`algobot logs 200` for more) |
| `algobot setup` | Re-runs the configuration wizard |

You can also run it in the foreground to watch the logs live:

```bash
.venv/bin/python -m algotrading.main                # paper mode against real prices
.venv/bin/python -m algotrading.main --demo-data    # offline practice mode
.venv/bin/python -m algotrading.main --no-telegram  # no Telegram, logs only
```

> **Tip:** `--demo-data` generates pretend prices. Use it to watch the whole
> thing work before you trust it with real market prices.

## 5. Talking to the bot in Telegram

Two ways to interact:

- **Commands** — start with `/`, listed in `/help`.
- **Plain messages** — if you set `AI_API_KEY`, just type a sentence. Examples:
  - *"How is my bot doing?"*
  - *"Backtest a faster EMA crossover — 5 and 20."*
  - *"That last loss looks like it bought too late. Can we fix that?"*
  - *"Create a strategy that buys after three red candles."*

The assistant can read your bot's state, look at price history, run backtests,
and **draft** changes. It will hand you an Approve / Reject card for anything
that would change how the bot trades.

## 6. The commands, explained plainly

| Command | What it shows you |
|---|---|
| `/help` | The list of commands. |
| `/status` | The essentials: mode (paper/live), the active strategy and its settings, open positions, and recent orders. Start here. |
| `/strategies` | **Every strategy the bot can run**, ranked best-first by a risk-adjusted score, with its full parameter list and allowed ranges. Buttons underneath let you backtest any one of them. |
| `/strategy` | Your *saved versions* — which release is active and which are waiting, with their parameters. |
| `/risk` | Your safety limits: risk per trade, max position size, max number of positions, cooldown between entries, slippage, and whether the trailing stop is on. |
| `/summary` | Recent performance summaries (win rate, expectancy) from closed trades. |
| `/start_bot` | Pauses/restarts the scheduler so the bot stops opening new trades. |
| `/stop_bot` | Resumes it. |

### The buttons under `/strategies`

Each strategy has a **📈 button**. Tap it and the bot will:

1. Replay your saved price history through that strategy using its **current**
   settings.
2. Replay it again using the **shipped default** settings.
3. Show you both results side by side — trades, win rate, profit, worst
   drawdown, and profit-per-unit-of-risk — and tell you which one won on a
   **risk-adjusted** basis (profit ÷ worst drawdown), not just raw profit.

That last part matters: a setting can make more money and still be worse if it
suffers much bigger swings. The comparison says so explicitly.

## 7. Choosing a strategy (and comparing them)

Out of the box you get nine ready-made strategies:

| Name | Plain-English idea |
|---|---|
| `ema_crossover` | Buy when the fast average price crosses above the slow one. |
| `ema_percentage_strategy` | Like the above but waits for a minimum percentage move to confirm. |
| `rsi_mean_reversion` | Buy when the market looks oversold and starts to recover. |
| `bb_mean_reversion` | Buy when price dips below the lower Bollinger band. |
| `macd_trend` | Buy when the MACD momentum indicator turns up. |
| `supertrend` | Follow the trend, flip when it reverses. |
| `vwap_reclaim` | Buy when price reclaims the volume-weighted average price. |
| `multi_tf_ema` | Compare EMAs across several timeframes before buying. |
| `ensemble` | Combine several of the above and act only when they agree. |

You don't have to guess which is best — open `/strategies` and look at the
ranking, or tap a button to compare settings.

**How strategies go live:** exactly one strategy version is *active* at a time.
When a new version is approved, the old one is retired. This applies whether the
change came from the AI or from you.

## 8. Using the AI assistant

Enable it one of two ways, then restart:

- Set `AI_API_KEY` (plus optionally `AI_BASE_URL` / `AI_MODEL`) to use an
  external OpenAI-compatible service, **or**
- Set `USE_HERMES=true` to run the assistant through a locally installed
  [Hermes Agent](https://hermes-agent.nousresearch.com/docs) — no API key, no
  cloud service. The bot calls it with a terminal-free toolset, so the assistant
  still can't touch anything by itself.

That enables two things:

1. **A daily review.** Once a day the assistant looks at your recent trades and
   analytics and, if it has an idea, sends you a proposal card on Telegram. You
   approve or reject it.
2. **Free-form chat.** Ask anything about your bot in plain language.

**What it can propose:**

| Proposal type | What it means |
|---|---|
| `param_change` | Tune an existing strategy's numbers. |
| `new_strategy` | Write an entirely new strategy in Python. |
| `edit_strategy` | Rewrite the code of a strategy the AI (or you) created earlier. |
| `new_indicator` | Add a new technical indicator the strategies can use. |
| `hypothesis` / `failure_analysis` | Plain-language reasoning about performance. |
| `ensemble_strategy` / `filter_strategy` | Combine strategies, or filter their signals. |

**What it can never do:** place a trade, change the active strategy, alter your
risk limits, or switch to live mode. Every change waits for your tap.

## 9. Creating your own strategy by describing it

Just ask, in Telegram:

> *"Create a strategy that buys when the price drops for three days in a row and
> sells when it gains 5%."*

The assistant will write it, and you'll get an Approve / Reject card. Tap
Approve and the bot:

- Saves the strategy as a readable Python file in the `strategies/` folder.
- Records a new version in the database and makes it active.
- Picks it up immediately — the next chat message already knows it exists.

If you don't like how it behaves, ask the AI to edit it, or open the file in
`strategies/` and change it yourself. The bot checks the file for safety and
reloads it automatically.

Full instructions and the file format live in
[`strategies/README.md`](strategies/README.md).

## 10. Settings you might want to change

Everything tunable is in **`config/settings.yaml`** — it's plain text with
comments. The ones people change most:

```yaml
market:
  symbols:            # which coins to trade
    - BTC/USDT
    - ETH/USDT

risk:
  paper_initial_balance: 10000.0   # pretend starting money in paper mode
  risk_per_trade_pct: 1.0          # % of balance risked per trade
  max_position_pct: 20.0           # biggest single position, % of balance
  max_open_positions: 3            # how many positions at once
  cooldown_seconds: 300            # wait this long between entries on a coin
  slippage_pct: 0.05               # paper-mode fill slippage
  trailing_stop_pct: 2.0           # 0 = off; else % below the high that exits

schedule:
  market_tick_seconds: 60          # how often it checks the market
```

After editing, restart the bot (`algobot stop` then `algobot start`) so the new
values take effect.

## 11. The safety rules, in plain English

These aren't optional — they're built in so the bot can't hurt you even if
something goes wrong:

1. **Nothing is sent to the exchange before it's written down.** The bot records
   its intention *first*, with a unique reference. If the network hiccups and it
   retries, the exchange sees the same reference and won't double-fill.
2. **Every action is written to an unchangeable log.** Your positions and trade
   history are *rebuilt* from that log when the bot restarts, so they can never
   drift out of sync.
3. **Risk limits are checked before every order.** Too many positions already
   open? Too soon after the last trade? The order is refused and the reason is
   recorded.
4. **Stale data freezes new entries.** If the price feed goes quiet, the bot
   stops opening positions — but it can still close them for protection.
5. **The AI is an advisor, never a trader.** It can only file proposals.
6. **Live mode is explicit.** Real trading requires you to change a setting *and*
   provide API keys; the bot refuses to start otherwise.
7. **Drift is checked.** Every 15 minutes (and at startup) the bot compares its
   records to the exchange and tells you if they disagree.

## 12. If something goes wrong

| What you see | What's probably happening | What to do |
|---|---|---|
| The bot never replies | Your Telegram ID isn't in the allowlist | Check `TELEGRAM_ALLOWED_USERS` in `.env` (get your ID from @userinfobot) |
| "AI assistant is not configured" | Neither provider is switched on | Set `AI_API_KEY` **or** `USE_HERMES=true` in `.env`, then restart |
| "USE_HERMES=true but the `hermes` command was not found" | Hermes Agent isn't on the bot's PATH | Install Hermes Agent (`hermes --version` in the same shell), or set `AI_API_KEY` instead |
| "The LLM call failed: Hermes CLI returned no answer" | The local agent produced no parsable answer | Run `hermes chat -q 'hi'` in the same shell to check it works; raise `HERMES_TIMEOUT` in `.env` if it's just slow |
| "No candles stored yet" | It hasn't collected prices yet | Wait a minute, or check the logs |
| "data stale; freezing new entries" | The price feed stopped | Usually temporary; check your connection |
| "no candles fetched" every minute | Can't reach Binance | Check internet / whether Binance is blocked in your region |
| A signal exists but no trade happened | A risk rule blocked it | `/risk` shows your limits; the log records the reason |
| The bot keeps restarting | Something crashes on startup | `algobot logs 200` and read the bottom |
| It replies but formatting looks wrong | Telegram rejected the message | Check the log for parse errors; the bot uses HTML formatting |

The complete technical troubleshooting guide is in
[`PROJECT_MAP.md`](PROJECT_MAP.md) §8.

## 13. Going live with real money

**Don't rush this.** Before flipping the switch:

1. Let it run in **paper mode for a while**. Watch the trades it *would* have
   made. Confirm you're happy with how often it trades and how much it risks.
2. Create Binance API keys with **spot trading only**, and ideally restrict them
   by IP.
3. Put them in `.env` as `BINANCE_API_KEY` / `BINANCE_API_SECRET`.
4. Change one line in `config/settings.yaml`:
   ```yaml
   mode: live
   ```
5. **Lower your risk.** Start with `max_position_pct: 2` and
   `risk_per_trade_pct: 0.5` until you trust it.
6. Restart and watch `/status` closely for the first day.

The bot refuses to start in live mode without keys, and logs a loud warning when
real orders are enabled.

## 14. Glossary

| Term | Meaning |
|---|---|
| **Paper mode** | Trading with real prices but fake money. Nothing is risked. |
| **Live mode** | Real orders with real money. |
| **Strategy** | The rule that decides when to buy and sell. |
| **Parameter** | A number you can tune inside a strategy (e.g. "fast average = 12"). |
| **Indicator** | A calculation over prices (an average, RSI, MACD, …) used by strategies. |
| **Backtest** | Replaying saved history to see how a strategy *would* have done. |
| **Drawdown** | How far your account fell from its peak — a measure of pain. |
| **Risk-adjusted** | Judging profit *relative to* the risk taken. We use profit ÷ worst drawdown. |
| **Trailing stop** | An automatic exit that follows the price up and sells if it falls back. |
| **Position** | A coin you currently hold. |
| **Tick** | One pass of the bot's market check loop. |
| **Recommendation** | A proposed change that waits for your approval. |

---
---

# Part 2 — For the engineer

## 15. Architecture

One Python process, one asyncio event loop, a modular monolith:

```
                         ┌──────────────────────────────────────┐
                         │  ModuleManager (config-driven)       │
                         │  market → execution → strategy →     │
                         │  analytics → control → api           │
                         └───────────────┬──────────────────────┘
                                         │ installs into
                                         ▼
   ┌─────────────────────── BotContext (service bag) ────────────────────────┐
   │  provider  gateway  services{capability: impl}  extra_jobs  health       │
   └───────────────┬─────────────────────────────────────────┬───────────────┘
                   │                                         │
      APScheduler (worker thread)                   Control surfaces (event loop)
      ├─ market_tick (60s)                          ├─ Telegram commands + chat
      │    fetch → store → stale-gate →             ├─ inline approve/reject/backtest
      │    protective exits → strategy → execute    └─ FastAPI /health /status /metrics
      ├─ reconcile (15m)  ├─ heartbeat (60s)
      └─ module jobs: analytics, analytics_daily, ai_review, strategy_plugins_reload
```

The trading core — market tick, strategy engine, deterministic execution, and the
event-sourced ledger — is fixed. Everything around it is a **module** selected
from `config/settings.yaml`.

**Startup order** (`algotrading/main.py`): `init_db` → seed strategies on first
run → `Ledger.rebuild_positions()` (replay events) → startup reconcile →
`manager.setup(ctx)` → `build_scheduler` → `manager.start(ctx)`.

**Threading:** scheduler jobs run in APScheduler's worker thread(s); each job
opens its own session from `ctx.session_factory`. The Telegram bot runs on the
event loop. Sessions are never shared across threads.

## 16. The plug-and-play module system

A module declares a **capability**, optional dependencies, and wires itself into
the shared context.

```python
from algotrading.modules.base import CAPABILITY_MARKET, Module, ModuleSpec
from algotrading.modules.registry import register_module

@register_module
class MyFeed(Module):
    spec = ModuleSpec(name="market.mine", capability=CAPABILITY_MARKET,
                      description="My price feed", builtin=False)

    def setup(self, ctx):          # synchronous, before the scheduler starts
        ctx.provider = MyFeedClient()
        ctx.provide(CAPABILITY_MARKET, ctx.provider)

    def jobs(self, ctx):           # optional scheduled work
        return [JobSpec("mine_sync", my_job, "interval", {"minutes": 5})]
```

Select it in `config/settings.yaml`:

```yaml
modules:
  enabled: [market.mine, gateway.paper, strategy.plugins, analytics.default]
  # or external code:
  # external: ["my_pkg.my_module:MyFeed"]
  params:
    market.mine: {api_key: "..."}
```

**Shipped modules**

| Module | Capability | Notes |
|---|---|---|
| `market.binance` | `market` | Public Binance REST klines/ticker. |
| `market.demo` | `market` | Deterministic synthetic feed; no network. |
| `gateway.paper` | `execution` | Simulated fills with slippage + 0.1% fee. **Locked.** |
| `gateway.live` | `execution` | Real Binance orders; requires keys. **Locked.** |
| `strategy.plugins` | `strategy` | Loads `strategies/*.py`; schedules hot-reload. |
| `analytics.default` | `analytics` | Metrics snapshots, daily summary, AI review jobs. |
| `control.telegram` | `control` | Telegram polling app; approval push. |
| `api.http` | `api` | FastAPI health/status/metrics; opt-in. |

**Selection rules** (`algotrading/modules/registry.py`):

- `enabled: null` → derive from mode/demo (`market.demo` in demo/paper, `gateway.live` in live, plus the default control surfaces).
- `enabled: [names]` → exactly those; unknown names fail loudly.
- `disabled: [names]` → removed from either set.
- Duplicate capabilities and unsatisfied `requires` are rejected at resolve time.
- `--no-telegram` disables `control.telegram`; `--demo-data` forces `market.demo` + `gateway.paper` and refuses `gateway.live`.

**The execution lock.** `LOCKED_CAPABILITIES = {"execution"}`. A locked module
must be `builtin=True`, cannot be disabled by config, and external modules
claiming it are refused at import. A locked module's `setup`/`start` failure is
fatal; optional modules degrade gracefully.

## 17. Strategies as files

Strategy *code* is a file; the *database* owns release state (version/active/
retired). Files are inspectable, diffable, and hand-editable.

```
strategies/
  momentum_nudge.py     # class with evaluate(symbol, candles) + STRATEGY alias
  README.md             # the authoring guide
```

- On load, each file goes through `strategy/validation.py` — a single AST gate
  that whitelists imports/names and forbids `eval`, `exec`, `open`, dunder
  escapes, etc. (`compile_strategy`, `compile_indicator`, `safe_namespace`).
- `strategy/plugins.py` loads, writes, and hot-reloads them (`strategy_plugins_reload`
  job, default every 300s).
- `strategy/registry.py` merges built-ins with plugins, so `known_names()`,
  `build_strategy()`, the chat prompt, and `/strategies` all see a new plugin
  immediately — no restart.

**AI authoring flow:** `new_strategy` / `edit_strategy` proposal → human ✅ →
`store/strategy_versions.create_new_strategy` / `update_strategy_code` writes the
file and cuts a `draft` version → shadow backtest (now that the name resolves) →
`promote_to_active`. Built-ins are deliberately *not* editable this way; change
their behaviour with `param_change`.

## 18. Configuration reference

### `config/settings.yaml`

| Block | Keys |
|---|---|
| `mode` | `paper` \| `live` |
| `market` | `exchange`, `symbols`, `intervals`, `backfill_days`, `poll_seconds`, `max_staleness_seconds` |
| `risk` | `paper_initial_balance`, `risk_per_trade_pct`, `max_position_pct`, `max_open_positions`, `cooldown_seconds`, `max_daily_loss_pct`, `slippage_pct`, `trailing_stop_pct` |
| `schedule` | `market_tick_seconds`, `analytics_minutes`, `daily_analytics_hour`, `ai_review_hour`, `reconcile_minutes` |
| `ai` | `enabled`, `base_url`, `model`, `temperature`, `max_recommendations_per_review` |
| `api` | `enabled`, `host`, `port` |
| `log_level`, `log_format`, `log_json_fields` | text/json logging |
| `strategy.hot_reload` | watch config files for changes |
| `modules` | `enabled`, `disabled`, `external`, `params`, `strategy_plugin_paths`, `autoload_strategies` |
| `metrics.enabled` | expose Prometheus `/metrics` |

### `.env` (secrets, gitignored)

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather. |
| `TELEGRAM_ALLOWED_USERS` | Comma-separated numeric IDs; strict allowlist. |
| `BINANCE_API_KEY` / `BINANCE_API_SECRET` | Live mode only. |
| `BINANCE_TESTNET_API_KEY` / `BINANCE_TESTNET_API_SECRET` | Optional testnet keys. |
| `AI_API_KEY` / `AI_BASE_URL` / `AI_MODEL` | Any OpenAI-compatible endpoint. Presence of the key enables the assistant. |
| `API_TOKEN` | Bearer token for `/status`; empty keeps the API off. |

`config.py` merges `.env` onto the YAML, resolves relative paths against the
project root (not CWD — a Termux gotcha), and `validate_settings()` runs before
the bot starts.

## 19. Data model & event sourcing

SQLite in WAL mode at `data/algotrading.db`.

| Table | Role |
|---|---|
| `strategies` | Versioned rows: `draft/approved/active/retired` + params JSON. |
| `signals` | Candidate entries/exits (`candidate/sent/skipped`). |
| `trade_intents` | Every order attempt, with a UNIQUE `idempotency_key`. |
| `order_events` | **Immutable append-only log — the source of truth.** |
| `positions` / `trades` | Derived views, rebuilt from events at startup. |
| `candles` | OHLCV, PK `symbol+interval+ts`. |
| `analytics_summaries` | Periodic metric snapshots. |
| `ai_recommendations` | Proposals + status + backtest JSON. |
| `meta` | `schema_version`, `strategies_seeded`. |

**Order path (fixed):** load signal → risk checks → persist `trade_intent`
(`status=pending`, unique key) **before** sending → `gateway.place_market_order`
→ record an `order_event` and update the intent → derive position/trade.

Because the intent exists before the order, a retry after a network error cannot
double-fill. Because events are immutable, `Ledger.rebuild_positions()` is the
deterministic recovery: it deletes derived rows and replays fills in timestamp
order.

## 20. Telegram + chat internals

`telegram/commands.py` `build_handlers(...)` returns PTB handlers with a strict
allowlist decorator and per-user rate limiting (10 requests / 60s):

- Commands: `/help`, `/status`, `/strategies`, `/strategy`, `/risk`, `/summary`, `/start_bot`, `/stop_bot`.
- Callbacks: `approve:<id>`, `reject:<id>`, `bt:<strategy_name>`.
- Plain text → the chat orchestrator.

**`/strategies`** renders `strategy/catalog.py` `catalog_entries()` (built-ins
from `config/strategies.yaml` + plugin metadata) through
`telegram/ui.py` `format_strategy_catalog()`, HTML-escaped and truncated to
Telegram's 4096-char cap. `rank_by_score()` sorts best-first by PnL ÷ max
drawdown; scores come from `_make_catalog_scores`, which backtests each strategy
with its current params in a worker thread, memoised for 60s and capped to 25
strategies.

**Backtest buttons** (`_make_backtest`) replay stored candles twice — current
params vs catalog defaults — via `backtest/stored.py` + `backtest/runner.py`, and
`ui.format_backtest_comparison()` renders the param diff, a `<pre>` results
table, a `pnl/drawdown` row, and a risk-adjusted verdict. It never places an
order.

**Chat orchestrator** (`telegram/chat.py`): a sync `run_agent()` with four tools
— `get_status`, `get_market`, `backtest`, `propose_change` — a max of 3 tool
turns, and a friendly-failure contract (LLM/tool errors become replies, never
crashes). The system prompt's strategy catalog is rebuilt every message from the
live registry, so an AI-created plugin is visible to the very next turn. The
only mutating tool persists a PENDING recommendation.

## 21. The HTTP API

Off by default; enable with `api.enabled: true` + `API_TOKEN`.

| Endpoint | Auth | Returns |
|---|---|---|
| `/health` | none | `status`, `mode`, `version`, `uptime_s`, DB latency, market freshness, heartbeat age |
| `/metrics` | none | Prometheus text (if `prometheus-client` is installed) |
| `/status` | `Authorization: Bearer <API_TOKEN>` | active strategy, open positions, recent intents, pending recommendation count |

Keep `api.host` on loopback. `/status` uses a constant-time bearer comparison.

## 22. Extending the bot

| I want to… | Go to |
|---|---|
| Add/replace a bot part | `algotrading/modules/builtin/` — new `Module` + `@register_module`, import in `builtin/__init__.py`, select in `settings.yaml` |
| Add a built-in strategy | `strategy/starters.py` + `config/strategies.yaml` schema |
| Add a user/AI strategy | Drop a validated `strategies/<name>.py` (see `strategies/README.md`) |
| Add an indicator | `strategy/indicators.py` (pure lists, oldest→newest, NaN padding) |
| Change code-safety rules | `strategy/validation.py` (the single gate) |
| Add a Telegram command | `telegram/commands.py` `build_handlers` + a formatter in `telegram/ui.py` |
| Add a scheduled job | `scheduler/jobs.py` `function(ctx)` + a `JobSpec` from the owning module |
| Add an API endpoint | `api/app.py` `build_api` (+ `require_token` if sensitive) |
| Add a recommendation kind | `store/recommendations.py` `ALLOWED_KINDS` + both prompts + `apply` |
| Add a DB table/column | `db/models.py` + bump `SCHEMA_VERSION` in `db/__init__.py` |

`PROJECT_MAP.md` is the detailed file-by-file map and is **enforced**:
`scripts/check_project_map.sh` fails CI/pre-commit if a source file isn't
mentioned in it.

## 23. Running as a service on Termux

`scripts/run_bot.sh` already supervises: it restarts on crash with exponential
backoff, enforces a single instance via `data/run_bot.lock`, backs up SQLite
before each start (`data/backups/`, 7-day retention), and restarts the bot if the
heartbeat goes stale (300s grace period).

To also survive reboots, register it with `termux-services`:

```bash
mkdir -p ~/.termux/services/algotrading
cat > ~/.termux/services/algotrading/run <<'EOF'
#!/data/data/com.termux/files/usr/bin/sh
exec /data/data/com.termux/files/home/AlgoTrading/scripts/run_bot.sh
EOF
chmod +x ~/.termux/services/algotrading/run
```

## 24. Testing

```bash
.venv/bin/python -m pytest tests/ -q          # 173 tests
bash scripts/check_project_map.sh             # map freshness
.venv/bin/python -m compileall -q algotrading # byte-compile check
```

| File | Covers |
|---|---|
| `test_indicators.py` | SMA/EMA/RSI/ATR math |
| `test_risk.py` | Sizing, exposure caps, cooldowns |
| `test_execution_flow.py` | Signal → intent → fill → events → rebuild |
| `test_scheduler.py` | Job wiring, stale-data freeze, demo feed |
| `test_analytics.py` | Metric math + summary persistence |
| `test_ai.py` | AI client parsing, prompt determinism |
| `test_m6c.py` | Recommendation store, assistant validation, versioned release |
| `test_telegram.py` | Allowlist, formatting, approvals, `/strategies` ranking + backtest buttons |
| `test_chat.py` | Chat orchestrator tools + safety contract |
| `test_api.py` | `/health` open, `/status` bearer-guarded |
| `test_live_gateway.py` | Live gateway + supervisor units |
| `test_backtest.py` | Backtest replay + equity summary |
| `test_ema_percentage_strategy.py` | EMA % strategy signals |
| `test_modules.py` | Module resolution, external loading, lifecycle, execution lock |
| `test_strategy_plugins.py` | Plugin load/write/edit, safety rejection, hot-reload, AI authoring |
| **`test_scenario_end_to_end.py`** | **The whole product as a non-technical trader would use it** — setup wizard, startup, a real entry + trailing-stop exit, every Telegram command, chat-driven strategy creation and approval, analytics, API, and restart recovery |

## 25. Non-negotiable invariants

1. **Intent-before-order** — persist `trade_intent` (unique key) before sending; retries reuse it.
2. **Event-sourced ledger** — `order_events` is immutable; positions/trades are derived.
3. **AI is advisory only** — proposals stay PENDING until a human approves; the AI never trades or edits the active strategy. The chat agent can only `propose_change`.
4. **One active strategy version** at a time.
5. **Stale-data freeze** — no new entries on stale candles; protective exits still run.
6. **Live is explicit** — requires `mode: live` *and* keys, or the bot refuses to start.
7. **Thread safety** — each job/thread opens its own DB session.
8. **Termux compatibility** — no ccxt, no `openai` SDK, no numpy/pandas; pydantic v1; pure-Python math.
9. **Execution is locked** — only built-in modules may hold the `execution` capability; config can't disable it and external code can't claim it; all AI/user code passes `strategy/validation.py` before it can run.
