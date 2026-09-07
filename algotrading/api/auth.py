"""Token guard for the internal API.

The API binds to loopback by default and is optional (settings.api.enabled).
/status carries trading data and requires `Authorization: Bearer <token>`;
/health stays open but only reports non-sensitive operational facts.

If no token is configured, guarded endpoints refuse all requests (403) — the
bot never serves trading data without a secret.
"""
from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException


def require_token(token: str):
    """Build a FastAPI dependency that checks the bearer token.

    The token is fixed at startup (from .env API_TOKEN). Comparison is
    constant-time via hmac.compare_digest.
    """

    async def _dep(authorization: str | None = Header(default=None)) -> None:
        if not token:
            raise HTTPException(status_code=403, detail="API token not configured")
        expected = f"Bearer {token}"
        if not authorization or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid or missing token")

    return Depends(_dep)