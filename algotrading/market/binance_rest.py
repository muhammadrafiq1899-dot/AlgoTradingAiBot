"""Minimal Binance Spot REST client.

Chosen over ccxt because ccxt hard-imports `cryptography`, whose Rust wheels are
unavailable on Termux/Android and cannot build from source. Binance spot signing
only needs HMAC-SHA256 (stdlib hmac/hashlib), so this client is dependency-light
(just `requests`).

Public endpoints (klines, ticker, time) require no API keys and work in both
paper and live modes. Signed endpoints (orders) require keys and are only used
by the execution gateway in live mode.

Docs: https://developers.binance.com/docs/binance-spot-api-docs
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.parse
from typing import Any

import requests

log = logging.getLogger(__name__)

SPOT_BASE = "https://api.binance.com"
TESTNET_BASE = "https://testnet.binance.vision"
DEFAULT_TIMEOUT = 15


class BinanceError(RuntimeError):
    """Raised on a non-2xx Binance response or malformed payload."""


class BinanceRestClient:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = SPOT_BASE,
        testnet: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self._key = api_key
        self._secret = api_secret
        self._base = TESTNET_BASE if testnet else base_url
        self._timeout = timeout
        self._session = requests.Session()

    # ---------- auth/signing helpers ----------

    def _signed_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Add timestamp+signature to params for authenticated endpoints."""
        if not self._key or not self._secret:
            raise BinanceError("API key/secret not configured (required for signed endpoints)")
        p = dict(params)
        p["timestamp"] = int(time.time() * 1000)
        p["recvWindow"] = p.get("recvWindow", 5000)
        query = urllib.parse.urlencode(p)
        p["signature"] = hmac.new(
            self._secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        return p

    def _get(self, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        url = self._base + path
        headers = {"X-MBX-APIKEY": self._key} if signed else {}
        if signed:
            params = self._signed_params(params or {})
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise BinanceError(f"network error on GET {path}: {exc}") from exc
        return self._handle(resp)

    def _post(self, path: str, params: dict[str, Any] | None = None, signed: bool = True) -> Any:
        url = self._base + path
        headers = {"X-MBX-APIKEY": self._key} if signed else {}
        if signed:
            params = self._signed_params(params or {})
        try:
            resp = self._session.post(url, params=params, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise BinanceError(f"network error on POST {path}: {exc}") from exc
        return self._handle(resp)

    def _delete(self, path: str, params: dict[str, Any] | None = None, signed: bool = True) -> Any:
        url = self._base + path
        headers = {"X-MBX-APIKEY": self._key} if signed else {}
        if signed:
            params = self._signed_params(params or {})
        try:
            resp = self._session.delete(url, params=params, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise BinanceError(f"network error on DELETE {path}: {exc}") from exc
        return self._handle(resp)

    @staticmethod
    def _handle(resp: requests.Response) -> Any:
        if resp.status_code >= 400:
            try:
                err = resp.json()
            except ValueError:
                err = {"msg": resp.text[:200]}
            raise BinanceError(f"Binance {resp.status_code}: {err}")
        try:
            return resp.json()
        except ValueError as exc:
            raise BinanceError(f"non-JSON response: {resp.text[:200]}") from exc

    # ---------- public market data ----------

    @staticmethod
    def _sym(symbol: str) -> str:
        """Normalize internal 'BASE/QUOTE' notation to Binance 'BASEQUOTE'."""
        return symbol.replace("/", "").replace("-", "").upper()

    def server_time_ms(self) -> int:
        data = self._get("/api/v3/time")
        return int(data["serverTime"])

    def klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 500,
    ) -> list[list[Any]]:
        """Return raw kline rows. Row: [open_time, o, h, l, c, v, close_time, ...]"""
        params: dict[str, Any] = {
            "symbol": self._sym(symbol),
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        return self._get("/api/v3/klines", params=params)

    def ticker_price(self, symbol: str) -> float:
        data = self._get("/api/v3/ticker/price", params={"symbol": self._sym(symbol)})
        return float(data["price"])

    # ---------- signed account/orders (live only) ----------

    def account_balances(self) -> dict[str, float]:
        """Return {asset: free_balance} for non-zero balances."""
        data = self._get("/api/v3/account", signed=True)
        out: dict[str, float] = {}
        for bal in data.get("balances", []):
            free = float(bal.get("free", 0))
            locked = float(bal.get("locked", 0))
            if free + locked > 0:
                out[bal["asset"]] = free + locked
        return out

    def create_order(
        self,
        symbol: str,
        side: str,  # BUY | SELL
        quantity: float,
        order_type: str = "MARKET",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "symbol": self._sym(symbol),
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        return self._post("/api/v3/order", params=params, signed=True)

    def get_order(self, symbol: str, client_order_id: str | None = None, order_id: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": self._sym(symbol)}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        if order_id is not None:
            params["orderId"] = order_id
        return self._get("/api/v3/order", params=params, signed=True)

    def cancel_order(self, symbol: str, client_order_id: str | None = None, order_id: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": self._sym(symbol)}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        if order_id is not None:
            params["orderId"] = order_id
        return self._delete("/api/v3/order", params=params, signed=True)

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = self._sym(symbol)
        return self._get("/api/v3/openOrders", params=params, signed=True)
