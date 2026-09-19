"""Outbound alert channel.

One webhook, fire-and-forget, never in the critical path. The bot's job is
trading; if the endpoint is slow or down, the tick must not care. Everything
here therefore swallows its own errors and reports them through the logger.

Configure with `alerts.webhook_url` in settings.yaml or `ALERT_WEBHOOK_URL` in
.env (which also enables the channel). Payloads carry both a human `text` line
(Slack/Discord/ntfy style endpoints render it) and structured `fields`.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

log = logging.getLogger(__name__)

# Alert kinds ↔ settings.alerts.notify_* switches.
KIND_FLAGS = {
    "signal": "notify_signals",
    "fill": "notify_fills",
    "risk": "notify_risk",
    "error": "notify_errors",
}
DEFAULT_TIMEOUT_SECONDS = 10


@dataclass
class Alert:
    """One notification."""

    kind: str
    title: str
    body: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        text = self.title if not self.body else f"{self.title}\n{self.body}"
        return {
            "text": text,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "fields": self.fields,
            "ts": datetime.now(timezone.utc).isoformat(),
        }


class WebhookAlerter:
    """Posts alerts to a webhook with per-kind rate limiting.

    Rate limiting matters because a risk guard can fire on every tick: without
    it a stuck cooldown would post once a minute forever.
    """

    def __init__(
        self,
        url: str = "",
        *,
        enabled: bool = True,
        min_interval_seconds: int = 60,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        notify: dict[str, bool] | None = None,
    ) -> None:
        self.url = url
        self.enabled = bool(enabled and url)
        self.min_interval_seconds = max(0, int(min_interval_seconds))
        self.timeout = timeout
        self._notify = dict(notify or {})
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()
        self.sent: list[Alert] = []      # in-memory trail, useful for tests/status
        self.suppressed = 0              # rate-limited or disabled count

    # --- gating ---

    def kind_enabled(self, kind: str) -> bool:
        flag = KIND_FLAGS.get(kind)
        if flag is None:
            return True
        return bool(self._notify.get(flag, True))

    def _rate_limited(self, kind: str, now: float) -> bool:
        if self.min_interval_seconds <= 0:
            return False
        last = self._last_sent.get(kind)
        return last is not None and (now - last) < self.min_interval_seconds

    # --- sending ---

    def send(self, kind: str, title: str, body: str = "", **fields: Any) -> bool:
        """Post one alert. Returns True when it was actually delivered."""
        alert = Alert(kind=kind, title=title, body=body, fields=fields)
        if not self.enabled or not self.kind_enabled(kind):
            self.suppressed += 1
            return False
        now = time.time()
        with self._lock:
            if self._rate_limited(kind, now):
                self.suppressed += 1
                log.debug("alert suppressed (rate limit): %s", title)
                return False
            self._last_sent[kind] = now
        try:
            response = requests.post(
                self.url, json=alert.to_payload(), timeout=self.timeout
            )
        except Exception as exc:  # noqa: BLE001 - alerts never break the caller
            log.warning("alert webhook failed (%s): %s", exc, title)
            return False
        if response.status_code >= 400:
            # Recorded before the POST (so a failing endpoint cannot be spammed
            # once per tick) but NOT counted as delivered.
            log.warning(
                "alert webhook returned %s: %s", response.status_code, title
            )
            return False
        with self._lock:
            self.sent.append(alert)
        return True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "url_configured": bool(self.url),
            "sent": len(self.sent),
            "suppressed": self.suppressed,
        }


# Module-level default, configured once from settings in main.py / jobs.py.
_alerter: WebhookAlerter = WebhookAlerter(url="", enabled=False)


def configure_alerter(settings: Any) -> WebhookAlerter:
    """Install the process-wide alerter from a `Settings` object."""
    global _alerter
    cfg = getattr(settings, "alerts", None)
    if cfg is None:
        _alerter = WebhookAlerter(url="", enabled=False)
    else:
        _alerter = WebhookAlerter(
            url=getattr(cfg, "webhook_url", ""),
            enabled=getattr(cfg, "enabled", False),
            min_interval_seconds=getattr(cfg, "min_interval_seconds", 60),
            notify={
                "notify_signals": getattr(cfg, "notify_signals", False),
                "notify_fills": getattr(cfg, "notify_fills", True),
                "notify_risk": getattr(cfg, "notify_risk", True),
                "notify_errors": getattr(cfg, "notify_errors", True),
            },
        )
    return _alerter


def get_alerter() -> WebhookAlerter:
    return _alerter


def notify(kind: str, title: str, body: str = "", **fields: Any) -> bool:
    """Send through the process-wide alerter (no-op when unconfigured)."""
    return _alerter.send(kind, title, body, **fields)


def notify_json(kind: str, title: str, payload: dict[str, Any]) -> bool:
    """Convenience wrapper for callers that already have a dict."""
    return _alerter.send(kind, title, "", **payload)


__all__ = [
    "Alert",
    "WebhookAlerter",
    "configure_alerter",
    "get_alerter",
    "notify",
    "notify_json",
    "KIND_FLAGS",
    "json",
]
