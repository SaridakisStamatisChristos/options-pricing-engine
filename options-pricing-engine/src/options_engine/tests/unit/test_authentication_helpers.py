"""Unit tests covering authentication helpers."""
from __future__ import annotations

import time
from typing import Any

import pytest
from jose import jwt

from options_engine.security import oidc


class _FakeJWKSCache:
    def get_key(self, kid: str) -> dict[str, Any]:
        assert kid == "kid-123"
        return {"alg": "RS256"}

    def reset(self) -> None:  # pragma: no cover - behaviour not exercised in this test
        raise AssertionError("reset should not be called")


def test_oidc_authenticator_passes_clock_skew_to_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    authenticator = oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="test-audience",
        jwks_cache=_FakeJWKSCache(),
        clock_skew_seconds=120,
    )

    monkeypatch.setattr(oidc.jwt, "get_unverified_header", lambda token: {"kid": "kid-123"})

    captured: dict[str, Any] = {}

    def _fake_decode(token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "sub": "user-123",
            "scope": "pricing:read",
            "iat": int(time.time()) - 10,
            "nbf": int(time.time()) - 10,
            "exp": int(time.time()) + 10,
        }

    monkeypatch.setattr(oidc.jwt, "decode", _fake_decode)

    claims = authenticator.decode("token")
    assert claims.subject == "user-123"
    assert captured["options"]["leeway"] == 120


def test_development_authenticator_accepts_rotated_secret_with_skew() -> None:
    authenticator = oidc.DevelopmentJWTAuthenticator(
        secrets=("old-secret", "fresh-secret"),
        issuer=None,
        audience=None,
        clock_skew_seconds=90,
    )

    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "local-user",
            "iat": now - 300,
            "nbf": now - 300,
            "exp": now - 30,  # expired but within the configured skew allowance
            "scope": ["pricing:read", "risk:read"],
        },
        "fresh-secret",
        algorithm="HS256",
    )

    claims = authenticator.decode(token)
    assert claims.subject == "local-user"
    assert claims.scopes == frozenset({"pricing:read", "risk:read"})
    assert claims.kid == "development"
