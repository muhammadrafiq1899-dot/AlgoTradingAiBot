"""Termux supervisor: wake-lock and heartbeat.

Keeps the bot alive on a mobile device:
- heartbeat(): touch a file so an external watchdog / run_bot.sh loop knows
  the process is alive.
- ensure_wake_lock(): run `termux-wake-lock` once so Android keeps the CPU
  awake while trading.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class HealthMonitor:
    """Touches a heartbeat file on a schedule; helpers for wake-lock."""

    def __init__(self, heartbeat_path: str = "data/heartbeat") -> None:
        self._path = Path(heartbeat_path)

    def beat(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            datetime.now(timezone.utc).isoformat(), encoding="utf-8"
        )

    def last_beat(self) -> float:
        """Epoch seconds of the last heartbeat, or 0 if none."""
        try:
            return self._path.stat().st_mtime
        except FileNotFoundError:
            return 0.0


def ensure_wake_lock() -> bool:
    """Acquire Termux wake-lock; returns True if it is (or becomes) held.

    No-op when `termux-wake-lock` is unavailable (e.g. desktop dev), where we
    return False so callers can log and continue.
    """
    if shutil.which("termux-wake-lock") is None:
        log.info("termux-wake-lock not found; skipping (desktop/CI)")
        return False
    try:
        subprocess.run(
            ["termux-wake-lock"], check=True, capture_output=True, timeout=10
        )
        return True
    except (subprocess.SubprocessError, OSError) as exc:
        log.warning("termux-wake-lock failed: %s", exc)
        return False
