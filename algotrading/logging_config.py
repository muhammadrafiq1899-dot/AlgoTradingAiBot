"""Structured JSON logging with correlation ID support.

Provides a JSONFormatter that outputs structured logs with:
- timestamp, level, logger name, message
- correlation_id (from contextvar, for request tracing)
- extra fields passed via logging's `extra` parameter

Usage:
    from algotrading.logging_config import setup_logging, get_correlation_id, set_correlation_id
    setup_logging(settings)
    # In request handlers / scheduler jobs:
    set_correlation_id("abc-123")
    log.info("processing signal", extra={"signal_id": 42})
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from logging.handlers import RotatingFileHandler

# Context variable for correlation ID (request/trace ID)
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)


def get_correlation_id() -> str:
    """Return the current correlation ID (empty string if not set)."""
    return _correlation_id.get()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(cid)


def clear_correlation_id() -> None:
    """Clear the correlation ID."""
    _correlation_id.set("")


class JSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID and extra fields support."""

    def __init__(self, include_fields: list[str] | None = None) -> None:
        super().__init__()
        self.include_fields = include_fields or [
            "timestamp",
            "level",
            "logger",
            "message",
            "correlation_id",
        ]

    def format(self, record: logging.LogRecord) -> str:
        # Base log entry
        log_entry: dict[str, Any] = {}

        if "timestamp" in self.include_fields:
            log_entry["timestamp"] = datetime.fromtimestamp(record.created).isoformat()

        if "level" in self.include_fields:
            log_entry["level"] = record.levelname

        if "logger" in self.include_fields:
            log_entry["logger"] = record.name

        if "message" in self.include_fields:
            log_entry["message"] = record.getMessage()

        if "correlation_id" in self.include_fields:
            log_entry["correlation_id"] = _correlation_id.get()

        # Add extra fields from record.__dict__ (passed via logging.extra)
        for key, value in record.__dict__.items():
            # Skip standard LogRecord attributes
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
                "getMessage",
            }:
                log_entry[key] = value

        # Include exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with correlation ID support."""

    def __init__(self, include_correlation_id: bool = True) -> None:
        fmt = "%(asctime)s %(levelname)s %(name)s"
        if include_correlation_id:
            fmt += " [%(correlation_id)s]"
        fmt += ": %(message)s"
        super().__init__(fmt)

    def format(self, record: logging.LogRecord) -> str:
        # Inject correlation_id into record for formatting
        record.correlation_id = _correlation_id.get()  # type: ignore[attr-defined]
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_format: str = "text",  # "text" or "json"
    json_fields: list[str] | None = None,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
) -> None:
    """Configure root logger with console and rotating file handlers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        log_format: "text" for human-readable, "json" for structured logs
        json_fields: List of fields to include in JSON output (None = defaults)
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep
    """
    root = logging.getLogger()
    root.setLevel(log_level.upper())

    # Clear existing handlers to avoid duplicates on reconfiguration
    root.handlers.clear()

    # Choose formatter
    if log_format == "json":
        formatter = JSONFormatter(include_fields=json_fields)
    else:
        formatter = TextFormatter(include_correlation_id=True)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Rotating file handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        str(log_path / "algotrading.log"),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Reduce noise from third-party loggers
    logging.getLogger("apscheduler").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)