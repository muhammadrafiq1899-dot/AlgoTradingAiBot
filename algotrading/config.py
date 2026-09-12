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
    backfill_days: int = 30
    poll_seconds: int = 60
    max_staleness_seconds: int = 300  # freeze new entries if data older than this


class RiskConfig(BaseModel):
    paper_initial_balance: float = 10_000.0
    risk_per_trade_pct: float = 1.0          # fraction of balance risked per trade
    max_position_pct: float = 20.0           # max single-position fraction of balance
    max_open_positions: int = 3
    cooldown_seconds: int = 300              # min gap between entries on a symbol
    max_daily_loss_pct: float = 3.0          # stop bot if daily loss exceeds this
    slippage_pct: float = 0.05               # paper fill slippage
    trailing_stop_pct: float = 0.0           # trailing stop % (0 = disabled)


class ScheduleConfig(BaseModel):
    market_tick_seconds: int = 60
    analytics_minutes: int = 30
    daily_analytics_hour: int = 0            # UTC hour for daily summary
    ai_review_hour: int = 6                  # UTC hour for daily AI proposal job
    reconcile_minutes: int = 15


class AIConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_recommendations_per_review: int = 1


class ApiConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8000


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


class Settings(BaseModel):
    mode: Literal["paper", "live"] = "paper"
    market: MarketConfig = Field(default_factory=MarketConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
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
    ai.enabled = bool(ai.api_key)

    # Binance keys (needed for live only; paper ignores them)
    _secrets["binance_api_key"] = os.getenv("BINANCE_API_KEY", "")
    _secrets["binance_api_secret"] = os.getenv("BINANCE_API_SECRET", "")
    _secrets["binance_testnet_key"] = os.getenv("BINANCE_TESTNET_API_KEY", "")
    _secrets["binance_testnet_secret"] = os.getenv("BINANCE_TESTNET_API_SECRET", "")
    _secrets["telegram_token"] = os.getenv("TELEGRAM_BOT_TOKEN", "")
    _secrets["api_token"] = os.getenv("API_TOKEN", "")

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
    type: Literal["int", "float", "bool"]
    # Values use Any: YAML already yields native int/float/bool types, and
    # pydantic v1's `int | float` union would truncate 0.2 -> 0. Preserve them.
    default: Any
    min: Any = None
    max: Any = None


class StrategyDef(BaseModel):
    name: str
    description: str = ""
    params: list[StrategyParam] = Field(default_factory=list)


def load_strategy_definitions(path: str | Path = DEFAULT_STRATEGIES) -> list[StrategyDef]:
    raw = _load_yaml(path)
    return [StrategyDef.parse_obj(s) for s in raw.get("strategies", [])]


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
    # Live mode requires keys
    if settings.mode == "live":
        binance_key = _secrets.get("binance_api_key", "")
        binance_secret = _secrets.get("binance_api_secret", "")
        if not binance_key or not binance_secret:
            raise RuntimeError(
                "LIVE mode requires BINANCE_API_KEY and BINANCE_API_SECRET in .env"
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
        if not settings.ai.api_key:
            raise ValueError("ai.enabled=true requires ai.api_key (AI_API_KEY in .env)")
        if settings.ai.temperature < 0 or settings.ai.temperature > 2:
            raise ValueError("ai.temperature must be in [0, 2]")
        if settings.ai.max_recommendations_per_review < 1:
            raise ValueError("ai.max_recommendations_per_review must be >= 1")

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

    # Telegram
    if not _secrets.get("telegram_token") and settings.telegram_allowed_users:
        raise ValueError(
            "telegram_allowed_users set but TELEGRAM_BOT_TOKEN not configured"
        )
