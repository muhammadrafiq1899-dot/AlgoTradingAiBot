"""Prometheus metrics for AlgoTrading.

Optional dependency on `prometheus-client` (pure Python).
If not installed, metrics are disabled gracefully.
"""
from __future__ import annotations

import time
from typing import Any

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Dummy classes for type hints
    Counter = Gauge = Histogram = object
    generate_latest = lambda: b""
    CONTENT_TYPE_LATEST = "text/plain"

from algotrading.config import Settings


# Global metrics (initialized on first call to init_metrics)
_ticks_total: Counter | None = None
_signals_generated: Counter | None = None
_orders_placed: Counter | None = None
_fills: Counter | None = None
_errors: Counter | None = None
_tick_latency: Histogram | None = None
_market_latency: Histogram | None = None
_execution_latency: Histogram | None = None
_active_positions: Gauge | None = None
_open_orders: Gauge | None = None
_equity: Gauge | None = None
_daily_pnl: Gauge | None = None
_last_tick_ts: Gauge | None = None
_last_heartbeat_ts: Gauge | None = None
_circuit_breaker_state: Gauge | None = None


def init_metrics(settings: Settings) -> None:
    """Initialize Prometheus metrics. Safe to call multiple times."""
    global _ticks_total, _signals_generated, _orders_placed, _fills, _errors
    global _tick_latency, _market_latency, _execution_latency
    global _active_positions, _open_orders, _equity, _daily_pnl
    global _last_tick_ts, _last_heartbeat_ts, _circuit_breaker_state

    if not PROMETHEUS_AVAILABLE:
        return

    if _ticks_total is not None:
        return  # Already initialized

    _ticks_total = Counter(
        "algotrading_ticks_total",
        "Total number of market ticks processed",
        ["status"],  # success, stale, error
    )
    _signals_generated = Counter(
        "algotrading_signals_generated_total",
        "Total number of signals generated",
        ["strategy", "symbol", "side", "status"],  # candidate, skipped, dup
    )
    _orders_placed = Counter(
        "algotrading_orders_placed_total",
        "Total number of orders placed",
        ["symbol", "side", "status"],  # sent, filled, failed
    )
    _fills = Counter(
        "algotrading_fills_total",
        "Total number of order fills",
        ["symbol", "side"],
    )
    _errors = Counter(
        "algotrading_errors_total",
        "Total number of errors",
        ["component", "error_type"],
    )
    _tick_latency = Histogram(
        "algotrading_tick_latency_seconds",
        "Market tick processing latency",
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    )
    _market_latency = Histogram(
        "algotrading_market_fetch_latency_seconds",
        "Market data fetch latency",
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    )
    _execution_latency = Histogram(
        "algotrading_execution_latency_seconds",
        "Order execution latency",
        buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    )
    _active_positions = Gauge(
        "algotrading_active_positions",
        "Number of currently open positions",
    )
    _open_orders = Gauge(
        "algotrading_open_orders",
        "Number of open orders on exchange",
    )
    _equity = Gauge(
        "algotrading_equity_usd",
        "Current portfolio equity in USD",
    )
    _daily_pnl = Gauge(
        "algotrading_daily_pnl_usd",
        "Daily PnL in USD",
    )
    _last_tick_ts = Gauge(
        "algotrading_last_tick_timestamp",
        "Unix timestamp of last successful market tick",
    )
    _last_heartbeat_ts = Gauge(
        "algotrading_last_heartbeat_timestamp",
        "Unix timestamp of last heartbeat",
    )
    _circuit_breaker_state = Gauge(
        "algotrading_circuit_breaker_state",
        "Circuit breaker state (0=closed, 1=half-open, 2=open)",
        ["name"],
    )


def get_metrics() -> bytes:
    """Return Prometheus metrics in text format."""
    if not PROMETHEUS_AVAILABLE:
        return b"# Prometheus client not installed\n"
    return generate_latest()


def metrics_content_type() -> str:
    """Return the content type for Prometheus metrics."""
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


# --- Convenience functions for recording metrics ---

def record_tick(status: str = "success") -> None:
    """Record a market tick completion."""
    if _ticks_total:
        _ticks_total.labels(status=status).inc()
    if _last_tick_ts and status == "success":
        _last_tick_ts.set(time.time())


def record_signal(strategy: str, symbol: str, side: str, status: str) -> None:
    """Record a signal generation."""
    if _signals_generated:
        _signals_generated.labels(strategy=strategy, symbol=symbol, side=side, status=status).inc()


def record_order(symbol: str, side: str, status: str) -> None:
    """Record an order placement."""
    if _orders_placed:
        _orders_placed.labels(symbol=symbol, side=side, status=status).inc()


def record_fill(symbol: str, side: str) -> None:
    """Record an order fill."""
    if _fills:
        _fills.labels(symbol=symbol, side=side).inc()


def record_error(component: str, error_type: str) -> None:
    """Record an error."""
    if _errors:
        _errors.labels(component=component, error_type=error_type).inc()


def record_tick_latency(latency: float) -> None:
    """Record tick processing latency."""
    if _tick_latency:
        _tick_latency.observe(latency)


def record_market_latency(latency: float) -> None:
    """Record market data fetch latency."""
    if _market_latency:
        _market_latency.observe(latency)


def record_execution_latency(latency: float) -> None:
    """Record order execution latency."""
    if _execution_latency:
        _execution_latency.observe(latency)


def set_active_positions(count: int) -> None:
    """Set active positions gauge."""
    if _active_positions:
        _active_positions.set(count)


def set_open_orders(count: int) -> None:
    """Set open orders gauge."""
    if _open_orders:
        _open_orders.set(count)


def set_equity(value: float) -> None:
    """Set equity gauge."""
    if _equity:
        _equity.set(value)


def set_daily_pnl(value: float) -> None:
    """Set daily PnL gauge."""
    if _daily_pnl:
        _daily_pnl.set(value)


def set_heartbeat_ts(ts: float) -> None:
    """Set heartbeat timestamp."""
    if _last_heartbeat_ts:
        _last_heartbeat_ts.set(ts)


def set_circuit_breaker_state(name: str, state: int) -> None:
    """Set circuit breaker state (0=closed, 1=half-open, 2=open)."""
    if _circuit_breaker_state:
        _circuit_breaker_state.labels(name=name).set(state)


class MetricsTimer:
    """Context manager for timing operations."""
    def __init__(self, recorder: callable):
        self._recorder = recorder
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._recorder(time.perf_counter() - self._start)


def tick_timer() -> MetricsTimer:
    return MetricsTimer(record_tick_latency)


def market_timer() -> MetricsTimer:
    return MetricsTimer(record_market_latency)


def execution_timer() -> MetricsTimer:
    return MetricsTimer(record_execution_latency)