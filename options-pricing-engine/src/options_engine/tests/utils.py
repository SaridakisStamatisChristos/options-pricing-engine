"""Test utilities for authentication and data generation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from jose import jwt

_FAKE_KID = "test-key"
_FAKE_SECRET = "dev-secret-value-at-least-32-bytes!!!"
_ISS = "https://issuer.test"
_AUD = "options-pricing-engine"


def make_token(*, scopes: list[str] | None = None, expires_in: int = 3600) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": "test-user",
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(seconds=expires_in),
        "iss": _ISS,
        "aud": _AUD,
    }
    if scopes:
        payload["scope"] = " ".join(scopes)
    headers = {"kid": _FAKE_KID, "alg": "HS256"}
    return jwt.encode(payload, _FAKE_SECRET, algorithm="HS256", headers=headers)


__all__ = ["make_token"]
