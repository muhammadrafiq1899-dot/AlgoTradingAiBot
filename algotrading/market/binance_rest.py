"""Minimal Binance Spot REST client.

Chosen over ccxt because ccxt hard-imports `cryptography`, whose Rust wheels are
unavailable on Termux/Android and cannot build from source. Binance spot signing
only needs HMAC-SHA256 (stdlib hmac/hashlib), so this client is dependency-light
(just `requests`).

Public endpoints (klines, ticker, time) require no API keys and work in both
paper and live modes. Signed endpoints (orders) require keys and are only used
by the execution gateway in live mode.

Venues (see `resolve_venue` / `from_settings`): the client talks to exactly one
venue, chosen once at construction. ``testnet=True`` or an explicit
``base_url`` select Binance's spot testnet (https://testnet.binance.vision).
There is deliberately **no** fallback from testnet to mainnet: a missing
testnet key fails loudly instead of quietly sending real orders to the live
market. Market **data** must come from the same venue as the orders — testnet
klines exist but have thinner liquidity and a much shorter history than
mainnet, so a testnet run produces testnet-looking prices.

Docs: https://developers.binance.com/docs/binance-spot-api-docs
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import random
import time
import urllib.parse
from typing import Any, Callable, TypeVar

import requests

log = logging.getLogger(__name__)

SPOT_BASE = "https://api.binance.com"
TESTNET_BASE = "https://testnet.binance.vision"
DEFAULT_TIMEOUT = 15

# Binance rejects a clientOrderId longer than this (chars: A-Z a-z 0-9 _ - .).
MAX_CLIENT_ORDER_ID_LEN = 36
ORDER_PATH = "/api/v3/order"
OPEN_ORDERS_PATH = "/api/v3/openOrders"

# Retry configuration
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0      # seconds
RETRY_MAX_DELAY = 30.0      # seconds
RETRY_EXPONENTIAL_BASE = 2.0
RETRY_JITTER = 0.3          # 30% jitter


class BinanceError(RuntimeError):
    """Raised on a non-2xx Binance response or malformed payload."""

    def __init__(self, message: str, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def _is_retryable_error(exc: Exception) -> bool:
    """Determine if an error is retryable."""
    if isinstance(exc, BinanceError):
        # Retry on 429 (rate limit), 5xx (server errors), network errors
        if exc.status_code is not None:
            return exc.status_code == 429 or exc.status_code >= 500
        # Network errors (no status code) are retryable
        return True
    if isinstance(exc, requests.RequestException):
        return True
    return False


def _retry_delay(attempt: int) -> float:
    """Calculate delay with exponential backoff and jitter."""
    delay = min(RETRY_BASE_DELAY * (RETRY_EXPONENTIAL_BASE ** attempt), RETRY_MAX_DELAY)
    jitter = delay * RETRY_JITTER * (random.random() * 2 - 1)  # ±30%
    return max(0, delay + jitter)


T = TypeVar("T")


def with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator that retries a function with exponential backoff on retryable errors."""
    def wrapper(*args, **kwargs) -> T:
        last_exc: Exception | None = None
        for attempt in range(RETRY_MAX_ATTEMPTS):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < RETRY_MAX_ATTEMPTS - 1 and _is_retryable_error(exc):
                    delay = _retry_delay(attempt)
                    log.warning(
                        "Binance request failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1, RETRY_MAX_ATTEMPTS, exc, delay
                    )
                    time.sleep(delay)
                    continue
                # Non-retryable or last attempt - raise
                raise
        # Should not reach here, but just in case
        raise last_exc  # type: ignore[misc]
    return wrapper


def resolve_venue(market: Any) -> tuple[str, bool]:
    """Resolve ``(base_url, testnet)`` from a ``MarketConfig`` or ``Settings``.

    An explicit ``market.base_url`` wins over everything (self-hosted mirrors,
    a local proxy); otherwise ``market.use_testnet`` selects the spot testnet.
    Both values are returned — not just the URL — because the API *keys* differ
    per venue, and the caller must not guess which one it is holding.
    """
    cfg = getattr(market, "market", market)
    use_testnet = bool(getattr(cfg, "use_testnet", False))
    explicit = str(getattr(cfg, "base_url", "") or "").strip()
    if explicit:
        return explicit.rstrip("/"), use_testnet
    return (TESTNET_BASE if use_testnet else SPOT_BASE), use_testnet


class BinanceRestClient:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "",
        testnet: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        self._key = api_key
        self._secret = api_secret
        # Explicit URL > testnet > mainnet. `base_url=""` is the "not set" value
        # the config uses, so an empty string must never become the URL.
        explicit = (base_url or "").strip()
        if explicit:
            self._base = explicit.rstrip("/")
        else:
            self._base = TESTNET_BASE if testnet else SPOT_BASE
        self._testnet = testnet
        self._timeout = timeout
        self._session = requests.Session()

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        *,
        api_key: str = "",
        api_secret: str = "",
        require_keys: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> BinanceRestClient:
        """Build a client for the venue the settings select.

        ``require_keys=True`` (the live order path) raises instead of returning
        a keyless client: an unauthenticated client on the testnet base would
        fail later, and there is no mainnet fallback by design — silently
        trading the real market because a testnet key was missing is the exact
        failure this guards against.
        """
        base, testnet = resolve_venue(settings)
        if require_keys and (not api_key or not api_secret):
            venue = "testnet (BINANCE_TESTNET_API_KEY/SECRET)" if testnet else "mainnet (BINANCE_API_KEY/SECRET)"
            raise BinanceError(
                f"missing Binance API credentials for {venue}; refusing to start "
                "authenticated trading without keys (no mainnet fallback)"
            )
        return cls(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base,
            testnet=testnet,
            timeout=timeout,
        )

    @property
    def base_url(self) -> str:
        return self._base

    @property
    def testnet(self) -> bool:
        return self._testnet

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

    @with_retry
    def _get(self, path: str, params: dict[str, Any] | None = None, signed: bool = False) -> Any:
        url = self._base + path
        headers = {"X-MBX-APIKEY": self._key} if signed else {}
        if signed:
            params = self._signed_params(params or {})
        try:
            resp = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise BinanceError(f"network error on GET {path}: {exc}", retryable=True) from exc
        return self._handle(resp)

    @with_retry
    def _post(self, path: str, params: dict[str, Any] | None = None, signed: bool = True) -> Any:
        url = self._base + path
        headers = {"X-MBX-APIKEY": self._key} if signed else {}
        if signed:
            params = self._signed_params(params or {})
        try:
            resp = self._session.post(url, params=params, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise BinanceError(f"network error on POST {path}: {exc}", retryable=True) from exc
        return self._handle(resp)

    @with_retry
    def _delete(self, path: str, params: dict[str, Any] | None = None, signed: bool = True) -> Any:
        url = self._base + path
        headers = {"X-MBX-APIKEY": self._key} if signed else {}
        if signed:
            params = self._signed_params(params or {})
        try:
            resp = self._session.delete(url, params=params, headers=headers, timeout=self._timeout)
        except requests.RequestException as exc:
            raise BinanceError(f"network error on DELETE {path}: {exc}", retryable=True) from exc
        return self._handle(resp)

    @staticmethod
    def _handle(resp: requests.Response) -> Any:
        if resp.status_code >= 400:
            try:
                err = resp.json()
            except ValueError:
                err = {"msg": resp.text[:200]}
            # Determine if error is retryable
            retryable = resp.status_code == 429 or resp.status_code >= 500
            raise BinanceError(f"Binance {resp.status_code}: {err}", status_code=resp.status_code, retryable=retryable)
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

    def get_balance(self, asset: str = "USDT") -> float:
        """Free + locked balance of one asset (0.0 when the account holds none).

        Deliberately reuses `account_balances()` rather than adding a per-asset
        endpoint: sizing needs the quote balance on every entry, and a second
        signed call per tick would only add request weight (weight limit 6000/min
        on spot; `/api/v3/account` costs 20).
        """
        return float(self.account_balances().get(asset.upper(), 0.0))

    @staticmethod
    def _client_id_param(client_order_id: str | None) -> dict[str, Any]:
        return {"newClientOrderId": client_order_id} if client_order_id else {}

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
        return self._post(ORDER_PATH, params=params, signed=True)

    def place_limit_order(
        self,
        symbol: str,
        side: str,  # buy | sell
        quantity: float,
        price: float,
        client_order_id: str | None = None,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        """POST a resting LIMIT order.

        A limit order is *not* synchronously filled: Binance returns
        ``status=NEW`` (or ``FILLED`` when it crossed on arrival) and the fill
        arrives later, so the caller must treat anything but ``FILLED`` as
        `sent` and let the reconcile path confirm it.
        """
        params: dict[str, Any] = {
            "symbol": self._sym(symbol),
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": quantity,
            "price": price,
        }
        params.update(self._client_id_param(client_order_id))
        return self._post(ORDER_PATH, params=params, signed=True)

    def place_stop_order(
        self,
        symbol: str,
        side: str,  # buy | sell
        quantity: float,
        stop_price: float,
        limit_price: float | None = None,
        client_order_id: str | None = None,
        time_in_force: str = "GTC",
    ) -> dict[str, Any]:
        """POST a resting protective STOP order (STOP_LOSS or STOP_LOSS_LIMIT).

        Binance requires ``stopPrice`` on both types and, for
        ``STOP_LOSS_LIMIT``, a ``price`` as well — ``timeInForce`` is only
        meaningful on the limit variant. The exchange's *response* carries
        ``orderId`` plus ``clientOrderId`` (our idempotency key) and
        ``origQuoteOrderId``; the quote id is a response field, never a request
        parameter, so it cannot be used to place or find an order — we key on
        ``newClientOrderId``.

        With no ``limit_price`` this sends a market ``STOP_LOSS``, which is the
        safest default for protection: once triggered it always fills, whereas
        a ``STOP_LOSS_LIMIT`` can be left unfilled through a gap and leave the
        position unprotected. A limit variant is only worth it when the caller
        is explicitly willing to trade certainty for a price floor/ceiling.
        """
        params: dict[str, Any] = {
            "symbol": self._sym(symbol),
            "side": side.upper(),
            "type": "STOP_LOSS_LIMIT" if limit_price is not None else "STOP_LOSS",
            "quantity": quantity,
            "stopPrice": stop_price,
        }
        if limit_price is not None:
            params["price"] = limit_price
            params["timeInForce"] = time_in_force
        params.update(self._client_id_param(client_order_id))
        return self._post(ORDER_PATH, params=params, signed=True)

    def get_order(self, symbol: str, client_order_id: str | None = None, order_id: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": self._sym(symbol)}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        if order_id is not None:
            params["orderId"] = order_id
        return self._get(ORDER_PATH, params=params, signed=True)

    def cancel_order(self, symbol: str, client_order_id: str | None = None, order_id: int | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"symbol": self._sym(symbol)}
        if client_order_id:
            params["origClientOrderId"] = client_order_id
        if order_id is not None:
            params["orderId"] = order_id
        return self._delete(ORDER_PATH, params=params, signed=True)

    def open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = self._sym(symbol)
        return self._get(OPEN_ORDERS_PATH, params=params, signed=True)

    def list_open_orders(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Alias of `open_orders` matching the gateway's optional-capability name."""
        return self.open_orders(symbol)
