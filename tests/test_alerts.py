"""Tests for the outbound alert channel (the P2 'second alert channel').

The channel's contract is "best-effort and silent when unconfigured or broken":
it must never raise into the caller, never post when disabled, and never let a
stuck guard spam the endpoint once per tick.
"""
import json

import pytest

from algotrading.alerts import (
    WebhookAlerter,
    configure_alerter,
    get_alerter,
    notify,
)
from algotrading.alerts import webhook as webhook_module
from algotrading.config import AlertsConfig, Settings


class _Response:
    def __init__(self, status_code=200):
        self.status_code = status_code


class _Recorder:
    """Captures requests.post calls."""

    def __init__(self, response=None, raises=None):
        self.calls = []
        self._response = response or _Response()
        self._raises = raises

    def __call__(self, url, json=None, timeout=None):  # noqa: A002 - matches requests' kwarg
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self._raises:
            raise self._raises
        return self._response


@pytest.fixture()
def restore_alerter():
    """Put the process-wide alerter back the way we found it."""
    original = webhook_module._alerter
    yield
    webhook_module._alerter = original


def test_disabled_alerter_never_posts(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(webhook_module.requests, "post", recorder)
    alerter = WebhookAlerter(url="", enabled=True)          # no url -> off
    assert alerter.enabled is False
    assert alerter.send("fill", "bought BTC") is False
    assert recorder.calls == []
    assert alerter.suppressed == 1


def test_posts_structured_json_payload(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(webhook_module.requests, "post", recorder)
    alerter = WebhookAlerter(url="https://example.test/hook", timeout=3)

    assert alerter.send("fill", "FILL buy BTC/USDT", "qty=0.5", qty=0.5, price=51000.0)

    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["url"] == "https://example.test/hook"
    assert call["timeout"] == 3
    payload = call["json"]
    assert payload["kind"] == "fill"
    assert payload["title"] == "FILL buy BTC/USDT"
    assert payload["fields"] == {"qty": 0.5, "price": 51000.0}
    assert payload["text"].startswith("FILL buy BTC/USDT")
    assert "ts" in payload
    # JSON-serialisable: a webhook consumer must be able to parse it.
    json.dumps(payload)
    assert alerter.status()["sent"] == 1


def test_rate_limit_is_per_kind(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(webhook_module.requests, "post", recorder)
    alerter = WebhookAlerter(url="https://example.test/hook", min_interval_seconds=60)

    assert alerter.send("risk", "cooldown active") is True
    assert alerter.send("risk", "cooldown active again") is False   # same kind, too soon
    assert alerter.send("fill", "a fill") is True                   # other kind unaffected
    assert len(recorder.calls) == 2
    assert alerter.status()["suppressed"] == 1


def test_zero_interval_disables_rate_limiting(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(webhook_module.requests, "post", recorder)
    alerter = WebhookAlerter(url="https://example.test/hook", min_interval_seconds=0)

    assert alerter.send("risk", "one") is True
    assert alerter.send("risk", "two") is True
    assert len(recorder.calls) == 2


def test_kind_switches_gate_delivery(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(webhook_module.requests, "post", recorder)
    alerter = WebhookAlerter(
        url="https://example.test/hook",
        min_interval_seconds=0,
        notify={"notify_signals": False, "notify_fills": True},
    )

    assert alerter.send("signal", "candidate buy") is False
    assert alerter.send("fill", "filled") is True
    assert len(recorder.calls) == 1


def test_transport_failure_is_swallowed(monkeypatch):
    monkeypatch.setattr(
        webhook_module.requests, "post", _Recorder(raises=ConnectionError("no route"))
    )
    alerter = WebhookAlerter(url="https://example.test/hook")

    assert alerter.send("error", "boom") is False   # returns, never raises
    assert alerter.status()["sent"] == 0


def test_http_error_does_not_lock_the_channel_out(monkeypatch):
    """A 5xx returns False but must not silence the channel afterwards."""
    responses = [_Response(500), _Response(200)]
    recorder = _Recorder()

    def post(url, json=None, timeout=None):  # noqa: A002 - matches requests' kwarg
        recorder.calls.append({"url": url, "json": json, "timeout": timeout})
        return responses.pop(0) if responses else _Response(200)

    monkeypatch.setattr(webhook_module.requests, "post", post)
    alerter = WebhookAlerter(url="https://example.test/hook", min_interval_seconds=0)

    assert alerter.send("error", "boom") is False
    assert alerter.send("error", "boom again") is True
    assert len(recorder.calls) == 2
    assert alerter.status()["sent"] == 1


def test_configure_alerter_reads_settings(restore_alerter):
    configured = configure_alerter(
        Settings(
            alerts=AlertsConfig(
                enabled=True,
                webhook_url="https://example.test/hook",
                min_interval_seconds=5,
                notify_signals=True,
                notify_fills=False,
            )
        )
    )
    assert configured.enabled is True
    assert configured.min_interval_seconds == 5
    assert configured.kind_enabled("signal") is True
    assert configured.kind_enabled("fill") is False

    # Settings without an alerts block (or a bare stub) degrade to "off".
    off = configure_alerter(type("S", (), {})())
    assert off.enabled is False


def test_notify_uses_the_process_wide_alerter(monkeypatch, restore_alerter):
    recorder = _Recorder()
    monkeypatch.setattr(webhook_module.requests, "post", recorder)
    configure_alerter(
        Settings(alerts=AlertsConfig(enabled=True, webhook_url="https://example.test/hook"))
    )

    assert notify("fill", "filled BTC/USDT", qty=1.0) is True
    assert get_alerter().status()["sent"] == 1
