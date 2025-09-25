"""Test utilities for authentication and data generation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from jose import jwt

from .conftest import FAKE_KID, FAKE_SECRET


def make_token(*, subject: str = "trader", scopes: list[str] | None = None, expires_in: int = 3600) -> str:
    """Create a signed JWT recognised by the test JWKS."""

    issued_at = datetime.now(UTC)
    payload = {
        "sub": subject,
        "iat": issued_at,
        "nbf": issued_at,
        "exp": issued_at + timedelta(seconds=expires_in),
        "iss": "https://issuer.test",
        "aud": "options-pricing-engine",
    }
    if scopes:
        payload["scope"] = " ".join(scopes)
    headers = {"kid": FAKE_KID, "alg": "HS256"}
    return jwt.encode(payload, FAKE_SECRET, algorithm="HS256", headers=headers)


__all__ = ["make_token"]
