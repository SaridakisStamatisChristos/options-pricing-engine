"""Unit tests covering authentication helpers."""
from __future__ import annotations

import base64
import time
from collections.abc import Mapping
from typing import Any

import pytest
from fastapi import HTTPException, status
from jose import JWTError, jwt

from options_engine.api import config as api_config
from options_engine.api import security as api_security
from options_engine.security import oidc


class _FakeJWKSCache:
    def __init__(self) -> None:
        self.reset_called = False

    def get_key(self, kid: str) -> dict[str, Any]:
        assert kid == "kid-123"
        return {"alg": "RS256"}

    def reset(self) -> None:  # pragma: no cover - behaviour not exercised in this test
        self.reset_called = True


@pytest.fixture(autouse=True)
def _reset_caches() -> None:
    api_config.get_settings.cache_clear()
    api_security._get_authenticator.cache_clear()
    yield
    api_config.get_settings.cache_clear()
    api_security._get_authenticator.cache_clear()


class _RecordingCounter:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def labels(self, **labels: str) -> "_RecordingCounter":
        self.calls.append(labels)
        return self

    def inc(self) -> None:
        if not self.calls:
            self.calls.append({})
        current = self.calls[-1]
        current["count"] = current.get("count", 0) + 1


class _StubAuthenticator:
    def __init__(self, *, result: oidc.OIDCClaims | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls = 0

    def decode(self, token: str) -> oidc.OIDCClaims:
        self.calls += 1
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


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
        secrets=(
            b"old-secret-should-be-at-least-32-bytes!!",
            b"fresh-secret-should-be-at-least-32-bytes!",
        ),
        issuer="https://issuer.test",
        audience="test-audience",
        clock_skew_seconds=90,
    )

    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "local-user",
            "iss": "https://issuer.test",
            "aud": "test-audience",
            "iat": now - 300,
            "nbf": now - 300,
            "exp": now - 30,  # expired but within the configured skew allowance
            "scope": ["pricing:read", "risk:read"],
        },
        b"fresh-secret-should-be-at-least-32-bytes!",
        algorithm="HS256",
    )

    claims = authenticator.decode(token)
    assert claims.subject == "local-user"
    assert claims.scopes == frozenset({"pricing:read", "risk:read"})
    assert claims.kid == "development"


def test_production_rejects_dev_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPE_ENVIRONMENT", "production")
    monkeypatch.setenv("OPE_ALLOWED_HOSTS", "api.example.com")
    monkeypatch.setenv("OIDC_ISSUER", "https://issuer.test")
    monkeypatch.setenv("OIDC_AUDIENCE", "options-pricing-engine")
    monkeypatch.setenv("OIDC_JWKS_URL", "https://issuer.test/jwks")
    secret = base64.urlsafe_b64encode(b"x" * 32).decode().rstrip("=")
    monkeypatch.setenv("DEV_JWT_SECRET", secret)

    with pytest.raises(RuntimeError, match="Development JWT secrets are forbidden"):
        api_config.get_settings()


def test_dev_secret_requires_oidc_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OIDC_ISSUER", raising=False)
    monkeypatch.setenv("OIDC_AUDIENCE", "options-pricing-engine")
    secret = base64.urlsafe_b64encode(b"y" * 32).decode().rstrip("=")
    monkeypatch.setenv("DEV_JWT_SECRET", secret)

    with pytest.raises(RuntimeError, match="requires OIDC_ISSUER and OIDC_AUDIENCE"):
        api_config.get_settings()


def test_dev_secret_strength_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OIDC_ISSUER", "https://issuer.test")
    monkeypatch.setenv("OIDC_AUDIENCE", "options-pricing-engine")
    monkeypatch.setenv("DEV_JWT_SECRET", "too-short")

    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        api_config.get_settings()


def test_jwks_cache_uses_cached_keys_when_refresh_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    timeline = {"now": 0.0}

    monkeypatch.setattr(oidc.time, "monotonic", lambda: timeline["now"])

    attempts = {"count": 0}

    def _fetcher(_: str) -> Mapping[str, Any]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return {"keys": [{"kid": "kid-1", "alg": "RS256"}]}
        raise RuntimeError("jwks down")

    cache = oidc.JWKSCache("https://issuer.test/jwks", refresh_interval_seconds=10, fetcher=_fetcher)

    assert cache.get_key("kid-1")["alg"] == "RS256"
    timeline["now"] = 20.0
    assert cache.get_key("kid-1")["alg"] == "RS256"
    assert attempts["count"] == 2


def test_jwks_cache_cold_start_failure() -> None:
    def _fetcher(_: str) -> Mapping[str, Any]:
        raise RuntimeError("jwks offline")

    cache = oidc.JWKSCache("https://issuer.test/jwks", fetcher=_fetcher)

    with pytest.raises(oidc.JWKSUnavailableError):
        cache.get_key("kid-1")


def test_oidc_header_alg_spoof_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    authenticator = oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="options-pricing-engine",
        jwks_cache=_FakeJWKSCache(),
    )

    monkeypatch.setattr(
        oidc.jwt,
        "get_unverified_header",
        lambda token: {"kid": "kid-123", "alg": "HS256"},
    )

    captured: dict[str, Any] = {}

    def _fake_decode(token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "sub": "user-123",
            "scope": "",
            "iss": "https://issuer.test",
            "aud": "options-pricing-engine",
            "iat": int(time.time()),
            "nbf": int(time.time()),
            "exp": int(time.time()) + 60,
        }

    monkeypatch.setattr(oidc.jwt, "decode", _fake_decode)

    claims = authenticator.decode("token")
    assert claims.subject == "user-123"
    assert captured["algorithms"] == ["RS256"]


def test_oidc_mode_rejects_hs_token_even_with_dev_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    authenticator = oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="options-pricing-engine",
        jwks_cache=_FakeJWKSCache(),
    )
    dev_authenticator = oidc.DevelopmentJWTAuthenticator(
        secrets=(b"dev-secret-value-at-least-32-bytes!!!",),
        issuer="https://issuer.test",
        audience="options-pricing-engine",
    )
    chain = api_security._AuthenticatorChain(primary=authenticator, dev=dev_authenticator)

    monkeypatch.setattr(oidc.jwt, "get_unverified_header", lambda token: {"kid": "kid-123"})

    def _decode(token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
        if isinstance(key, Mapping):
            raise JWTError("invalid signature")
        return {
            "sub": "dev-user",
            "scope": "",
            "iss": "https://issuer.test",
            "aud": "options-pricing-engine",
            "iat": int(time.time()) - 10,
            "nbf": int(time.time()) - 10,
            "exp": int(time.time()) + 60,
        }

    monkeypatch.setattr(oidc.jwt, "decode", _decode)

    with pytest.raises(JWTError):
        chain.decode("token")


def test_authenticator_falls_back_to_dev_when_oidc_unavailable() -> None:
    dev_claims = oidc.OIDCClaims(
        subject="dev-user",
        scopes=frozenset(),
        claims={"sub": "dev-user", "iss": "https://issuer.test", "aud": "options-pricing-engine"},
        kid="development",
    )
    primary = _StubAuthenticator(error=oidc.OIDCUnavailableError("jwks down"))
    dev = _StubAuthenticator(result=dev_claims)
    chain = api_security._AuthenticatorChain(primary=primary, dev=dev)

    assert chain.decode("token") is dev_claims
    assert primary.calls == 1
    assert dev.calls == 1


def test_development_leeway_enforced() -> None:
    secret = b"dev-secret-value-at-least-32-bytes!!!"
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "dev-user",
            "iss": "https://issuer.test",
            "aud": "options-pricing-engine",
            "iat": now,
            "nbf": now + 30,
            "exp": now + 300,
        },
        secret,
        algorithm="HS256",
    )

    tolerant = oidc.DevelopmentJWTAuthenticator(
        secrets=(secret,),
        issuer="https://issuer.test",
        audience="options-pricing-engine",
        clock_skew_seconds=60,
    )
    strict = oidc.DevelopmentJWTAuthenticator(
        secrets=(secret,),
        issuer="https://issuer.test",
        audience="options-pricing-engine",
        clock_skew_seconds=0,
    )

    tolerant.decode(token)
    with pytest.raises(JWTError):
        strict.decode(token)


def test_oidc_leeway_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _FakeJWKSCache()
    authenticator = oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="options-pricing-engine",
        jwks_cache=cache,
        clock_skew_seconds=60,
    )
    strict = oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="options-pricing-engine",
        jwks_cache=cache,
        clock_skew_seconds=0,
    )

    monkeypatch.setattr(oidc.jwt, "get_unverified_header", lambda token: {"kid": "kid-123"})

    claims = {
        "sub": "user-123",
        "scope": "",
        "iss": "https://issuer.test",
        "aud": "options-pricing-engine",
        "iat": int(time.time()),
        "nbf": int(time.time()) + 30,
        "exp": int(time.time()) + 300,
    }

    def _decode(token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
        leeway = kwargs["options"]["leeway"]
        if leeway < 30:
            raise JWTError("Token used too early")
        return claims

    monkeypatch.setattr(oidc.jwt, "decode", _decode)

    assert authenticator.decode("token").subject == "user-123"
    with pytest.raises(JWTError):
        strict.decode("token")


def test_decode_token_returns_503_on_oidc_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingCounter()
    monkeypatch.setattr(api_security, "AUTH_FAILURES", recorder)
    monkeypatch.setattr(
        api_security,
        "_get_authenticator",
        lambda: _StubAuthenticator(error=oidc.OIDCUnavailableError("jwks down")),
    )

    with pytest.raises(HTTPException) as exc:
        api_security._decode_token("token")

    assert exc.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert recorder.calls and recorder.calls[-1]["reason"] == "jwks_unavailable"
