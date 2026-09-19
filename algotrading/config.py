"""Configuration loading and validation.

Loads settings.yaml + strategies.yaml from the config/ directory and merges
secrets from .env (via python-dotenv). Produces pydantic models so every
component gets validated, typed configuration.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Project root = two levels up from this file (algotrading/config.py -> repo root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_SETTINGS = CONFIG_DIR / "settings.yaml"
DEFAULT_STRATEGIES = CONFIG_DIR / "strategies.yaml"


class MarketConfig(BaseModel):
    exchange: str = "binance"
    symbols: list[str] = ["BTC/USDT", "ETH/USDT"]
    intervals: list[str] = ["1m", "1h"]
    # Interval the strategy engine evaluates. None = legacy behaviour ("1h" when
    # configured, else the first interval). Set it explicitly for scalping.
    eval_interval: str | None = None
    backfill_days: int = 30
    poll_seconds: int = 60
    max_staleness_seconds: int = 300  # freeze new entries if data older than this
    # Venue selection: testnet keeps the live code path but against Binance's
    # spot testnet (BINANCE_TESTNET_API_KEY/SECRET). base_url overrides entirely.
    use_testnet: bool = False
    base_url: str = ""


class RiskConfig(BaseModel):
    paper_initial_balance: float = 10_000.0
    risk_per_trade_pct: float = 1.0          # fraction of balance risked per trade
    max_position_pct: float = 20.0           # max single-position fraction of balance
    max_open_positions: int = 3
    cooldown_seconds: int = 300              # min gap between entries on a symbol
    max_daily_loss_pct: float = 3.0          # stop new entries if daily loss exceeds this
    enforce_daily_loss: bool = True          # set false to disable the guard explicitly
    slippage_pct: float = 0.05               # paper fill slippage
    trailing_stop_pct: float = 0.0           # trailing stop % (0 = disabled)
    # Resting exchange-side stop (live only). Protective stops then survive the
    # bot process dying, instead of being re-derived from a signal every tick.
    exchange_stop_enabled: bool = False
    exchange_stop_limit_offset_pct: float = 0.1  # limit offset below a stop-loss-limit


class BacktestConfig(BaseModel):
    fee_rate: float = 0.001          # per-side cost as a fraction of notional
    slippage_pct: float = 0.05       # adverse fill on entry and exit
    apply_risk_checks: bool = False  # simulate cooldown/max positions/caps
    walk_forward_folds: int = 4
    monte_carlo_runs: int = 200
    bootstrap_runs: int = 200
    random_seed: int = 42
    max_candles: int = 20_000        # row cap for one replay


class OptimizeConfig(BaseModel):
    enabled: bool = True
    max_combinations: int = 200      # hard cap on grid/random candidates
    timeout_seconds: int = 900       # wall-clock budget for the child process
    min_trades: int = 10             # candidates below this are rejected outright
    objective: Literal["sharpe", "pnl_drawdown", "total_pnl"] = "sharpe"
    trainer_python: str = ""         # interpreter for the search ("" = this one)
    nice: int = 10                   # child niceness, so the tick keeps its CPU
    results_dir: str = "data/optimize"  # repo-relative; resolved at load time


class AlertsConfig(BaseModel):
    enabled: bool = False
    webhook_url: str = ""            # from ALERT_WEBHOOK_URL in .env
    min_interval_seconds: int = 60   # rate limit per event kind
    notify_signals: bool = False
    notify_fills: bool = True
    notify_risk: bool = True
    notify_errors: bool = True


class AnalyticsConfig(BaseModel):
    portfolio_bars: int = 200        # candles behind the exposure/correlation view


class ScheduleConfig(BaseModel):
    market_tick_seconds: int = 60
    analytics_minutes: int = 30
    daily_analytics_hour: int = 0            # UTC hour for daily summary
    ai_review_hour: int = 6                  # UTC hour for daily AI proposal job
    reconcile_minutes: int = 15


class AIConfig(BaseModel):
    enabled: bool = False
    use_hermes: bool = False  # Use Hermes Agent instead of external LLM
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_recommendations_per_review: int = 1
    # Image input (Telegram photo/document -> advisory LLM). Only the local
    # Hermes Agent CLI can read images; the external API path replies with a
    # "set USE_HERMES=true" hint instead.
    images_enabled: bool = True
    image_max_bytes: int = 5_000_000
    image_dir: str = "data/uploads"   # repo-relative; resolved at load time
    # Market news headlines for the advisory context (public RSS, no API key).
    news_enabled: bool = False
    news_feed_urls: list[str] = Field(
        default_factory=lambda: [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
        ]
    )
    news_max_items: int = 8
    news_timeout_seconds: int = 10
    # Advisory decision log: what the AI proposed, what was approved, and how it
    # performed afterwards — fed back into the next review as lessons.
    decision_log_enabled: bool = True
    decision_log_lessons: int = 5


class ApiConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    dashboard: bool = True  # read-only HTML dashboard at "/"


class LogConfig(BaseModel):
    level: str = "INFO"
    format: Literal["text", "json"] = "text"
    json_fields: list[str] = Field(
        default_factory=lambda: [
            "timestamp",
            "level",
            "logger",
            "message",
            "correlation_id",
        ]
    )


class StrategyConfig(BaseModel):
    hot_reload: bool = False


class MetricsConfig(BaseModel):
    enabled: bool = False


class ModulesConfig(BaseModel):
    """Plug-and-play module selection (see algotrading/modules/).

    ``enabled=None`` means "derive the set from mode/demo". ``params`` passes
    per-module constructor arguments, keyed by module name.
    """
    enabled: list[str] | None = None
    disabled: list[str] = Field(default_factory=list)
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    external: list[str] = Field(default_factory=list)
    # Directories scanned for user/AI-authored strategy plugin files.
    strategy_plugin_paths: list[str] = Field(default_factory=lambda: ["strategies"])
    autoload_strategies: bool = True


class Settings(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    market: MarketConfig = Field(default_factory=MarketConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    optimize: OptimizeConfig = Field(default_factory=OptimizeConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    analytics: AnalyticsConfig = Field(default_factory=AnalyticsConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    modules: ModulesConfig = Field(default_factory=ModulesConfig)
    telegram_allowed_users: list[int] = Field(default_factory=list)
    data_dir: str = "data"
    log_dir: str = "logs"
    db_path: str = ""            # resolved at load time from data_dir

    class Config:
        # Live is always explicit; never inferred. Kept for clarity/safety.
        validate_assignment = True


@lru_cache(maxsize=1)
def load_settings(
    settings_path: str | Path = DEFAULT_SETTINGS,
    strategies_path: str | Path = DEFAULT_STRATEGIES,
) -> Settings:
    """Load and validate settings, merging .env secrets. Cached per process."""
    load_dotenv(PROJECT_ROOT / ".env")

    raw = _load_yaml(settings_path)
    settings = Settings.parse_obj(raw)

    # Resolve absolute paths relative to project root
    settings.data_dir = str(PROJECT_ROOT / settings.data_dir)
    settings.log_dir = str(PROJECT_ROOT / settings.log_dir)
    settings.db_path = str(Path(settings.data_dir) / "algotrading.db")
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)

    # Uploaded images land next to the runtime data, not in the CWD (Termux
    # gotcha: the bot is started from wherever the supervisor happens to be).
    ai_dir = Path(settings.ai.image_dir)
    settings.ai.image_dir = str(ai_dir if ai_dir.is_absolute() else PROJECT_ROOT / ai_dir)

    # Strategy plugin dirs are relative to the project root, not the CWD, so the
    # bot finds them regardless of where it was launched from (Termux gotcha).
    settings.modules.strategy_plugin_paths = [
        str(p if Path(p).is_absolute() else PROJECT_ROOT / p)
        for p in settings.modules.strategy_plugin_paths
    ]

    # Optimizer artifact dir: same Termux rule (never relative to the CWD).
    opt_dir = Path(settings.optimize.results_dir)
    settings.optimize.results_dir = str(
        opt_dir if opt_dir.is_absolute() else PROJECT_ROOT / opt_dir
    )

    # Merge .env secrets
    if os.getenv("TELEGRAM_ALLOWED_USERS"):
        ids = [
            int(x.strip())
            for x in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",")
            if x.strip()
        ]
        settings.telegram_allowed_users = ids

    ai = settings.ai
    ai.api_key = os.getenv("AI_API_KEY", ai.api_key)
    ai.base_url = os.getenv("AI_BASE_URL", ai.base_url)
    ai.model = os.getenv("AI_MODEL", ai.model)
    # Advisory LLM provider: external OpenAI-compatible API (AI_API_KEY) or the
    # local Hermes Agent CLI (USE_HERMES=true). Either one enables the assistant.
    ai.use_hermes = os.getenv("USE_HERMES", "false").strip().lower() in (
        "true", "1", "yes", "on",
    )
    ai.enabled = bool(ai.api_key) or ai.use_hermes

    # Binance keys (needed for live only; paper ignores them)
    _secrets["binance_api_key"] = os.getenv("BINANCE_API_KEY", "")
    _secrets["binance_api_secret"] = os.getenv("BINANCE_API_SECRET", "")
    _secrets["binance_testnet_key"] = os.getenv("BINANCE_TESTNET_API_KEY", "")
    _secrets["binance_testnet_secret"] = os.getenv("BINANCE_TESTNET_API_SECRET", "")
    _secrets["telegram_token"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
    _secrets["api_token"] = os.getenv("API_TOKEN", "")
    _secrets["alert_webhook_url"] = os.getenv("ALERT_WEBHOOK_URL", "")

    # A configured webhook turns the alert channel on; an explicit
    # `alerts.enabled: false` in settings.yaml is still honoured.
    if _secrets["alert_webhook_url"] and not settings.alerts.enabled:
        settings.alerts.enabled = True
    settings.alerts.webhook_url = os.getenv(
        "ALERT_WEBHOOK_URL", settings.alerts.webhook_url
    )

    return settings


# Convenience properties for logging config (accessed via settings.log)
@property
def log_level(self) -> str:
    return self.log.level


@property
def log_format(self) -> str:
    return self.log.format


@property
def log_json_fields(self) -> list[str]:
    return self.log.json_fields


Settings.log_level = log_level
Settings.log_format = log_format
Settings.log_json_fields = log_json_fields


# Runtime secret bucket, kept out of the settings model (not serialized/logged).
_secrets: dict[str, str] = {}


def get_secret(name: str) -> str:
    """Return a secret by name ('' if unset). Loads settings first if needed."""
    load_settings()
    return _secrets.get(name, "")


class StrategyParam(BaseModel):
    name: str
    # YAML schemas use str (intervals, modes) and list (ensemble components) in
    # addition to the numeric/bool types; all must parse for seeding + the
    # dynamic strategy catalog the chat prompt is built from.
    type: Literal["int", "float", "bool", "str", "list"]
    # Values use Any: YAML already yields native int/float/bool types, and
    # pydantic v1's `int | float` union would truncate 0.2 -> 0. Preserve them.
    default: Any
    min: Any = None
    max: Any = None
    # Allowed values for enum-style params (e.g. ensemble `mode`).
    enum: list[Any] | None = None


class StrategyDef(BaseModel):
    name: str
    description: str = ""
    params: list[StrategyParam] = Field(default_factory=list)
    # Extended fields for AI-generated strategies
    template: str = ""  # Python code template with {{param_name}} placeholders
    indicator_deps: list[str] = Field(default_factory=list)  # Required indicator functions
    validation: dict[str, Any] = Field(default_factory=dict)  # AST whitelist/blacklist rules
    test_template: str = ""  # Unit test template for validation


def load_strategy_definitions(path: str | Path = DEFAULT_STRATEGIES) -> list[StrategyDef]:
    raw = _load_yaml(path)
    strategies = []
    for s in raw.get("strategies", []):
        # Ensure backward compatibility - add empty extended fields if not present
        if "template" not in s:
            s["template"] = ""
        if "indicator_deps" not in s:
            s["indicator_deps"] = []
        if "validation" not in s:
            s["validation"] = {}
        if "test_template" not in s:
            s["test_template"] = ""
        strategies.append(StrategyDef.parse_obj(s))
    return strategies


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {p}")
    return data


def validate_settings(settings: Settings) -> None:
    """Validate settings for runtime safety. Raises RuntimeError on failure.

    Checks:
    - Live mode requires Binance API keys
    - Risk parameters are within sane bounds
    - Schedule intervals don't conflict
    - Log format is valid
    - AI config is valid when enabled
    - API config is valid when enabled
    """
    # Live mode requires keys for the venue actually selected. A testnet-only
    # setup must not be forced to hold mainnet credentials just to boot.
    if settings.mode == "live":
        if settings.market.use_testnet:
            binance_key = _secrets.get("binance_testnet_key", "")
            binance_secret = _secrets.get("binance_testnet_secret", "")
            required = "BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET"
        else:
            binance_key = _secrets.get("binance_api_key", "")
            binance_secret = _secrets.get("binance_api_secret", "")
            required = "BINANCE_API_KEY and BINANCE_API_SECRET"
        if not binance_key or not binance_secret:
            venue = "testnet (market.use_testnet: true)" if settings.market.use_testnet else "mainnet"
            raise RuntimeError(
                f"LIVE mode on {venue} requires {required} in .env"
            )

    # Risk parameter bounds
    risk = settings.risk
    if not 0 < risk.risk_per_trade_pct <= 100:
        raise ValueError("risk.risk_per_trade_pct must be in (0, 100]")
    if not 0 < risk.max_position_pct <= 100:
        raise ValueError("risk.max_position_pct must be in (0, 100]")
    if risk.max_open_positions < 1:
        raise ValueError("risk.max_open_positions must be >= 1")
    if risk.cooldown_seconds < 0:
        raise ValueError("risk.cooldown_seconds must be >= 0")
    if not 0 < risk.max_daily_loss_pct <= 100:
        raise ValueError("risk.max_daily_loss_pct must be in (0, 100]")
    if not 0 <= risk.slippage_pct <= 10:
        raise ValueError("risk.slippage_pct must be in [0, 10]")
    if not 0 <= risk.trailing_stop_pct <= 100:
        raise ValueError("risk.trailing_stop_pct must be in [0, 100]")
    if not 0 <= risk.exchange_stop_limit_offset_pct <= 10:
        raise ValueError("risk.exchange_stop_limit_offset_pct must be in [0, 10]")
    if risk.paper_initial_balance <= 0:
        raise ValueError("risk.paper_initial_balance must be > 0")

    # Schedule intervals
    sched = settings.schedule
    if sched.market_tick_seconds < 10:
        raise ValueError("schedule.market_tick_seconds must be >= 10")
    if sched.analytics_minutes < 1:
        raise ValueError("schedule.analytics_minutes must be >= 1")
    if not 0 <= sched.daily_analytics_hour <= 23:
        raise ValueError("schedule.daily_analytics_hour must be in [0, 23]")
    if not 0 <= sched.ai_review_hour <= 23:
        raise ValueError("schedule.ai_review_hour must be in [0, 23]")
    if sched.reconcile_minutes < 1:
        raise ValueError("schedule.reconcile_minutes must be >= 1")

    # Log format
    if settings.log.format not in ("text", "json"):
        raise ValueError("log.format must be 'text' or 'json'")

    # AI config
    if settings.ai.enabled:
        if not settings.ai.use_hermes and not settings.ai.api_key:
            raise ValueError(
                "ai.enabled=true and ai.use_hermes=false requires ai.api_key (AI_API_KEY in .env)"
            )
        if settings.ai.temperature < 0 or settings.ai.temperature > 2:
            raise ValueError("ai.temperature must be in [0, 2]")
        if settings.ai.max_recommendations_per_review < 1:
            raise ValueError("ai.max_recommendations_per_review must be >= 1")
        if settings.ai.image_max_bytes < 1:
            raise ValueError("ai.image_max_bytes must be >= 1")

    # API config
    if settings.api.enabled:
        if not _secrets.get("api_token"):
            raise ValueError("api.enabled=true requires API_TOKEN in .env")
        if not 1 <= settings.api.port <= 65535:
            raise ValueError("api.port must be in [1, 65535]")

    # Market config
    market = settings.market
    if not market.symbols:
        raise ValueError("market.symbols must not be empty")
    if not market.intervals:
        raise ValueError("market.intervals must not be empty")
    if market.poll_seconds < 1:
        raise ValueError("market.poll_seconds must be >= 1")
    if market.max_staleness_seconds < market.poll_seconds:
        raise ValueError(
            "market.max_staleness_seconds must be >= market.poll_seconds"
        )
    if market.backfill_days < 1:
        raise ValueError("market.backfill_days must be >= 1")
    if market.eval_interval is not None:
        from algotrading.market.candles import INTERVAL_MS

        if market.eval_interval not in market.intervals:
            raise ValueError(
                "market.eval_interval must be one of market.intervals "
                f"({market.intervals})"
            )
        if market.eval_interval not in INTERVAL_MS:
            raise ValueError(
                f"market.eval_interval {market.eval_interval!r} is not a known "
                f"interval ({sorted(INTERVAL_MS)})"
            )

    # Backtest / optimizer budgets
    bt = settings.backtest
    if not 0 <= bt.fee_rate < 0.1:
        raise ValueError("backtest.fee_rate must be in [0, 0.1)")
    if not 0 <= bt.slippage_pct <= 10:
        raise ValueError("backtest.slippage_pct must be in [0, 10]")
    if not 2 <= bt.walk_forward_folds <= 20:
        raise ValueError("backtest.walk_forward_folds must be in [2, 20]")
    for name in ("monte_carlo_runs", "bootstrap_runs"):
        if getattr(bt, name) < 0:
            raise ValueError(f"backtest.{name} must be >= 0")
    if bt.max_candles < 100:
        raise ValueError("backtest.max_candles must be >= 100")

    opt = settings.optimize
    if opt.max_combinations < 1:
        raise ValueError("optimize.max_combinations must be >= 1")
    if opt.timeout_seconds < 10:
        raise ValueError("optimize.timeout_seconds must be >= 10")
    if opt.min_trades < 0:
        raise ValueError("optimize.min_trades must be >= 0")
    if not 0 <= opt.nice <= 19:
        raise ValueError("optimize.nice must be in [0, 19]")

    # Alerts
    alerts = settings.alerts
    if alerts.enabled and not alerts.webhook_url:
        raise ValueError(
            "alerts.enabled=true requires alerts.webhook_url or ALERT_WEBHOOK_URL in .env"
        )
    if alerts.webhook_url and not alerts.webhook_url.startswith(("http://", "https://")):
        raise ValueError("alerts.webhook_url must be an http(s) URL")
    if alerts.min_interval_seconds < 0:
        raise ValueError("alerts.min_interval_seconds must be >= 0")

    # Portfolio analytics
    if settings.analytics.portfolio_bars < 10:
        raise ValueError("analytics.portfolio_bars must be >= 10")

    # Telegram
    if not _secrets.get("telegram_token") and settings.telegram_allowed_users:
        raise ValueError(
            "telegram_allowed_users set but TELEGRAM_BOT_TOKEN not configured"
        )

    # Modules: names must exist, no duplicate capability, locked ones stay on.
    # Imported lazily: algotrading.modules imports config at module load.
    from algotrading.modules.registry import get_module, module_names

    modules = settings.modules
    for name in list(modules.enabled or []) + list(modules.disabled):
        if get_module(name) is None:
            raise ValueError(f"unknown module {name!r}; known: {module_names()}")
    for name in modules.disabled:
        if get_module(name).spec.locked:
            raise ValueError(
                f"module {name!r} provides a locked capability and cannot be disabled"
            )
    if modules.enabled is not None:
        caps: dict[str, str] = {}
        for name in modules.enabled:
            capability = get_module(name).spec.capability
            if capability in caps:
                raise ValueError(
                    f"modules {caps[capability]!r} and {name!r} both provide "
                    f"capability {capability!r}"
                )
            caps[capability] = name
    if not modules.strategy_plugin_paths:
        raise ValueError("modules.strategy_plugin_paths must not be empty")
