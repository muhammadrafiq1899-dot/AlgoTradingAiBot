"""SQLAlchemy ORM models for AlgoTrading.

The trade lifecycle (intents, order events, fills) is event-sourced: state is
rebuilt by replaying immutable events. Positions and closed trades are derived
views for convenience, not independent sources of truth.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    # JSON-encoded {param_name: value}
    params: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(16), default="draft")
    # draft | approved | active | retired
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (UniqueConstraint("name", "version", name="uq_strategy_name_version"),)


class Signal(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategies.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(8))  # buy | sell
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ref_price: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text, default="")
    risk_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(16), default="candidate")
    # candidate | sent | skipped
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (Index("ix_signals_symbol_ts", "symbol", "ts"),)


class TradeIntent(Base):
    __tablename__ = "trade_intents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"), nullable=True)
    # Idempotency key — unique per intent, prevents duplicate orders on retry.
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    symbol: Mapped[str] = mapped_column(String(32))
    side: Mapped[str] = mapped_column(String(8))
    qty: Mapped[float] = mapped_column(Float)
    order_type: Mapped[str] = mapped_column(String(16), default="market")
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # pending | sent | filled | canceled | failed
    exchange_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # For paper mode, the ref price used for the simulated fill.
    ref_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    fee: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class OrderEvent(Base):
    """Immutable event log. Never updated or deleted; append-only."""

    __tablename__ = "order_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    intent_id: Mapped[int | None] = mapped_column(ForeignKey("trade_intents.id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(24))
    # accepted | fill | partial | canceled | rejected | risk_skipped
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Position(Base):
    """Derived open-position view (rebuilt from events)."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True)
    qty: Mapped[float] = mapped_column(Float, default=0.0)
    avg_price: Mapped[float] = mapped_column(Float, default=0.0)
    trailing_stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    highest_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Trade(Base):
    """Closed-trade record (derived from fills; holds PnL + loss tags)."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    entry_qty: Mapped[float] = mapped_column(Float)
    entry_avg_price: Mapped[float] = mapped_column(Float)
    exit_avg_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    loss_reasons: Mapped[str] = mapped_column(Text, default="[]")  # JSON list
    strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategies.id"), nullable=True)

    __table_args__ = (Index("ix_trades_symbol_closed", "symbol", "closed_at"),)


class Candle(Base):
    __tablename__ = "candles"

    symbol: Mapped[str] = mapped_column(String(32), primary_key=True)
    interval: Mapped[str] = mapped_column(String(8), primary_key=True)
    ts: Mapped[int] = mapped_column(Integer, primary_key=True)  # epoch ms, aligned to interval
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    __table_args__ = (Index("ix_candles_sym_interval_ts", "symbol", "interval", "ts"),)


class AnalyticsSummary(Base):
    __tablename__ = "analytics_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    period: Mapped[str] = mapped_column(String(16))  # 30m | daily
    strategy_id: Mapped[int | None] = mapped_column(ForeignKey("strategies.id"), nullable=True)
    symbol: Mapped[str] = mapped_column(String(32), default="ALL")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AIRecommendation(Base):
    __tablename__ = "ai_recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    kind: Mapped[str] = mapped_column(String(24))
    # new_strategy | param_change | hypothesis | failure_analysis
    strategy_name: Mapped[str] = mapped_column(String(64), default="")
    content_json: Mapped[str] = mapped_column(Text, default="{}")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    # pending | approved | rejected | applied
    rationale: Mapped[str] = mapped_column(Text, default="")
    backtest_json: Mapped[str] = mapped_column(Text, default="{}")  # shadow-backtest result
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Meta(Base):
    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
