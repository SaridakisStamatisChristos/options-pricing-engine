"""Unit tests covering authentication helpers."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import Mapping
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException, status
from jwt.algorithms import RSAAlgorithm
from jwt.exceptions import (
    ExpiredSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
)
from jwt.exceptions import (
    PyJWTError as JWTError,
)

from options_engine.api import config as api_config
from options_engine.api import security as api_security
from options_engine.security import DevelopmentSignatureError, oidc

_TEST_RSA_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65_537, key_size=2_048)
_TEST_RSA_JWK = json.loads(RSAAlgorithm.to_jwk(_TEST_RSA_PRIVATE_KEY.public_key()))
_TEST_RSA_JWK.update({"alg": "RS256", "kid": "kid-123", "use": "sig"})


class _FakeJWKSCache:
    def __init__(self) -> None:
        self.reset_called = False

    def get_key(self, kid: str, *, force_refresh: bool = False) -> dict[str, Any]:
        assert kid == "kid-123"
        assert isinstance(force_refresh, bool)
        return dict(_TEST_RSA_JWK)

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

    def labels(self, **labels: str) -> _RecordingCounter:
        self.calls.append(labels)
        return self

    def inc(self) -> None:
        if not self.calls:
            self.calls.append({})
        current = self.calls[-1]
        current["count"] = current.get("count", 0) + 1


class _StubAuthenticator:
    def __init__(
        self, *, result: oidc.OIDCClaims | None = None, error: Exception | None = None
    ) -> None:
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

    monkeypatch.setattr(
        oidc.jwt,
        "get_unverified_header",
        lambda token: {"kid": "kid-123", "alg": "RS256"},
    )

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
    assert captured["leeway"] == 120


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


def test_oidc_authenticator_verifies_real_rs256_jwk() -> None:
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "oidc-user",
            "iss": "https://issuer.test",
            "aud": "test-audience",
            "iat": now,
            "nbf": now - 1,
            "exp": now + 300,
            "scope": "pricing:read risk:read",
        },
        _TEST_RSA_PRIVATE_KEY,
        algorithm="RS256",
        headers={"alg": "RS256", "kid": "kid-123"},
    )
    authenticator = oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="test-audience",
        jwks_cache=_FakeJWKSCache(),
    )

    claims = authenticator.decode(token)

    assert claims.subject == "oidc-user"
    assert claims.scopes == frozenset({"pricing:read", "risk:read"})
    assert claims.kid == "kid-123"


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


def test_dev_secret_encoding_is_explicit_and_unambiguous() -> None:
    raw = b"x" * 32
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")

    assert api_config._normalise_dev_secret("SECRET", f"base64:{encoded}") == raw
    assert api_config._normalise_dev_secret("SECRET", f"hex:{raw.hex()}") == raw
    assert api_config._normalise_dev_secret("SECRET", encoded) == encoded.encode()
    with pytest.raises(RuntimeError, match="not valid explicit"):
        api_config._normalise_dev_secret("SECRET", "base64:not!valid")


def test_config_rejects_duplicate_or_orphaned_rotation_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "primary-development-secret-at-least-32-bytes"
    monkeypatch.setenv("DEV_JWT_SECRET", secret)
    monkeypatch.setenv("DEV_JWT_ADDITIONAL_SECRETS", secret)
    with pytest.raises(RuntimeError, match="must be unique"):
        api_config.get_settings()

    api_config.get_settings.cache_clear()
    monkeypatch.delenv("DEV_JWT_SECRET")
    monkeypatch.delenv("OPE_JWT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="require a primary"):
        api_config.get_settings()


def test_production_requires_hosts_and_complete_oidc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPE_ENVIRONMENT", "production")
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.delenv("ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("OPE_ALLOWED_HOSTS", raising=False)
    monkeypatch.delenv("DEV_JWT_SECRET", raising=False)

    with pytest.raises(RuntimeError, match="ALLOWED_HOSTS"):
        api_config.get_settings()

    monkeypatch.setenv("OPE_ALLOWED_HOSTS", "api.example.test")
    with pytest.raises(RuntimeError, match="OIDC configuration"):
        api_config.get_settings()


def test_config_rejects_duplicate_dev_secret_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_JWT_SECRET", "a" * 32)
    monkeypatch.setenv("OPE_JWT_SECRET", "b" * 32)
    with pytest.raises(RuntimeError, match="Only one"):
        api_config.get_settings()


def test_config_rejects_unsafe_environment_and_origin_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ENV", raising=False)
    monkeypatch.setenv("OPE_ENVIRONMENT", "prd-typo")
    with pytest.raises(RuntimeError, match="must be development"):
        api_config.get_settings()

    api_config.get_settings.cache_clear()
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("OPE_ENVIRONMENT", "development")
    with pytest.raises(RuntimeError, match="conflicting environments"):
        api_config.get_settings()

    api_config.get_settings.cache_clear()
    monkeypatch.delenv("ENV")
    monkeypatch.setenv("OPE_ENVIRONMENT", "development")
    monkeypatch.setenv("OPE_ALLOWED_ORIGINS", "*")
    with pytest.raises(RuntimeError, match="Wildcard CORS"):
        api_config.get_settings()


def test_config_rejects_conflicting_aliases_and_malformed_network_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALLOWED_HOSTS", "api-one.example.test")
    monkeypatch.setenv("OPE_ALLOWED_HOSTS", "api-two.example.test")
    with pytest.raises(RuntimeError, match="must not conflict"):
        api_config.get_settings()

    api_config.get_settings.cache_clear()
    monkeypatch.delenv("ALLOWED_HOSTS")
    monkeypatch.setenv("OPE_ALLOWED_HOSTS", "https://api.example.test")
    with pytest.raises(RuntimeError, match="Invalid ALLOWED_HOSTS"):
        api_config.get_settings()

    api_config.get_settings.cache_clear()
    monkeypatch.setenv("OPE_ALLOWED_HOSTS", "api.example.test")
    monkeypatch.setenv("OPE_ALLOWED_ORIGINS", "https://app.example.test:99999")
    with pytest.raises(RuntimeError, match="invalid port"):
        api_config.get_settings()

    api_config.get_settings.cache_clear()
    monkeypatch.setenv("OPE_ALLOWED_ORIGINS", "https://app.example.test")
    monkeypatch.setenv("OIDC_ISSUER", "https://issuer.test?tenant=ambiguous")
    with pytest.raises(RuntimeError, match="must not contain URL parameters"):
        api_config.get_settings()

    api_config.get_settings.cache_clear()
    monkeypatch.setenv("OIDC_ISSUER", "https://issuer.test")
    monkeypatch.setenv("OIDC_AUDIENCE", "invalid audience")
    with pytest.raises(RuntimeError, match="OIDC_AUDIENCE"):
        api_config.get_settings()


def test_production_rejects_wildcard_hosts_and_insecure_oidc_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPE_ENVIRONMENT", "production")
    monkeypatch.setenv("OPE_ALLOWED_HOSTS", "*")
    monkeypatch.setenv("OIDC_ISSUER", "https://issuer.test")
    monkeypatch.setenv("OIDC_AUDIENCE", "options-pricing-engine")
    monkeypatch.setenv("OIDC_JWKS_URL", "https://issuer.test/jwks")
    with pytest.raises(RuntimeError, match="Wildcard ALLOWED_HOSTS"):
        api_config.get_settings()

    api_config.get_settings.cache_clear()
    monkeypatch.setenv("OPE_ALLOWED_HOSTS", "api.example.test")
    monkeypatch.setenv("OIDC_JWKS_URL", "http://localhost/jwks")
    with pytest.raises(RuntimeError, match="must use HTTPS"):
        api_config.get_settings()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("OPE_THREADS", "not-an-int", "must be an integer"),
        ("OPE_THREADS", "0", "must be >= 1"),
        ("OPE_THREAD_QUEUE_TIMEOUT_SECONDS", "not-a-number", "must be a number"),
        ("OPE_THREAD_QUEUE_TIMEOUT_SECONDS", "nan", "must be finite"),
        ("OPE_MONTE_CARLO_SEED", "not-an-int", "must be an integer"),
        ("OPE_MONTE_CARLO_SEED", "-1", "must be within"),
        ("OPE_CORS_ALLOW_CREDENTIALS", "sometimes", "must be a boolean"),
        ("OIDC_CLOCK_SKEW_S", "-1", "must be >= 0"),
        ("OIDC_CLOCK_SKEW_S", "301", "must be <= 300"),
        ("OIDC_JWKS_CACHE_TTL_S", "59", "must be >= 60"),
        ("OIDC_JWKS_MAX_STALE_S", "299", "must be >= 300"),
    ],
)
def test_config_rejects_invalid_numeric_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(RuntimeError, match=message):
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

    cache = oidc.JWKSCache(
        "https://issuer.test/jwks", refresh_interval_seconds=10, fetcher=_fetcher
    )

    assert cache.get_key("kid-1")["alg"] == "RS256"
    timeline["now"] = 20.0
    assert cache.get_key("kid-1")["alg"] == "RS256"
    assert attempts["count"] == 2


def test_jwks_cache_rejects_keys_after_bounded_stale_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline = {"now": 0.0}
    monkeypatch.setattr(oidc.time, "monotonic", lambda: timeline["now"])
    attempts = 0

    def _fetcher(_: str) -> Mapping[str, Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return {"keys": [{"kid": "kid-1", "alg": "RS256"}]}
        raise RuntimeError("jwks down")

    cache = oidc.JWKSCache(
        "https://issuer.test/jwks",
        refresh_interval_seconds=10,
        max_stale_seconds=15,
        fetcher=_fetcher,
    )
    assert cache.get_key("kid-1")["alg"] == "RS256"

    timeline["now"] = 20.0
    with pytest.raises(oidc.JWKSUnavailableError):
        cache.get_key("kid-1")


def test_unknown_kids_cannot_force_unbounded_jwks_refreshes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeline = {"now": 0.0}
    monkeypatch.setattr(oidc.time, "monotonic", lambda: timeline["now"])
    attempts = 0

    def _fetcher(_: str) -> Mapping[str, Any]:
        nonlocal attempts
        attempts += 1
        return {"keys": [{"kid": "known", "alg": "RS256"}]}

    cache = oidc.JWKSCache("https://issuer.test/jwks", fetcher=_fetcher)
    assert cache.get_key("known")["alg"] == "RS256"
    for _ in range(10):
        with pytest.raises(KeyError):
            cache.get_key("attacker-controlled-kid")
    assert attempts == 1

    timeline["now"] = 31.0
    with pytest.raises(KeyError):
        cache.get_key("attacker-controlled-kid")
    assert attempts == 2


def test_jwks_cache_cold_start_failure() -> None:
    def _fetcher(_: str) -> Mapping[str, Any]:
        raise RuntimeError("jwks offline")

    cache = oidc.JWKSCache("https://issuer.test/jwks", fetcher=_fetcher)

    with pytest.raises(oidc.JWKSUnavailableError):
        cache.get_key("kid-1")


def test_jwks_cache_validates_payload_and_unknown_keys() -> None:
    malformed = oidc.JWKSCache(
        "https://issuer.test/jwks",
        fetcher=lambda _: {"keys": "not-a-list"},
    )
    with pytest.raises(oidc.JWKSUnavailableError):
        malformed.get_key("kid-1")

    cache = oidc.JWKSCache(
        "https://issuer.test/jwks",
        fetcher=lambda _: {"keys": [None, {"kid": "known", "kty": "RSA"}]},
    )
    assert cache.get_key("known")["kty"] == "RSA"
    with pytest.raises(KeyError, match="Unknown signing key"):
        cache.get_key("missing")
    with pytest.raises(KeyError, match="kid must be provided"):
        cache.get_key("")

    duplicates = oidc.JWKSCache(
        "https://issuer.test/jwks",
        fetcher=lambda _: {"keys": [{"kid": "same"}, {"kid": "same"}]},
    )
    with pytest.raises(oidc.JWKSUnavailableError):
        duplicates.get_key("same")


def test_jwks_cache_returns_isolated_key_snapshots() -> None:
    cache = oidc.JWKSCache(
        "https://issuer.test/jwks",
        fetcher=lambda _: {"keys": [{"kid": "isolated", "kty": "RSA", "key_ops": ["verify"]}]},
    )

    first = cache.get_key("isolated")
    first["kty"] = "oct"
    first["key_ops"].append("sign")

    assert cache.get_key("isolated") == {
        "kid": "isolated",
        "kty": "RSA",
        "key_ops": ["verify"],
    }


def test_oidc_key_policy_rejects_non_signing_keys() -> None:
    authenticator = oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="test-audience",
        jwks_cache=_FakeJWKSCache(),
    )

    with pytest.raises(JWTError, match="signatures"):
        authenticator._resolve_key_algorithm({"use": "enc", "alg": "RS256"})
    with pytest.raises(JWTError, match="verification"):
        authenticator._resolve_key_algorithm({"key_ops": ["sign"], "alg": "RS256"})
    with pytest.raises(JWTError, match="list of strings"):
        authenticator._resolve_key_algorithm({"key_ops": "verify", "alg": "RS256"})
    assert authenticator._resolve_key_algorithm({"kty": "RSA"}) == "RS256"
    assert authenticator._resolve_key_algorithm({"kty": "EC", "crv": "P-256"}) == "ES256"
    with pytest.raises(JWTError, match="does not match"):
        authenticator._resolve_key_algorithm({"kty": "EC", "alg": "RS256"})
    with pytest.raises(JWTError, match="curve"):
        authenticator._resolve_key_algorithm({"kty": "EC", "alg": "ES256", "crv": "P-384"})
    with pytest.raises(JWTError, match="Unsupported"):
        authenticator._resolve_key_algorithm({"kty": "RSA", "alg": "HS256"})
    with pytest.raises(JWTError, match="duplicates"):
        authenticator._resolve_key_algorithm(
            {"kty": "RSA", "alg": "RS256", "key_ops": ["verify", "verify"]}
        )
    with pytest.raises(JWTError, match="Unsupported"):
        authenticator._resolve_key_algorithm({"kty": "oct"})


def test_oidc_rejects_unsupported_critical_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    authenticator = oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="test-audience",
        jwks_cache=_FakeJWKSCache(),
    )
    monkeypatch.setattr(
        oidc.jwt,
        "get_unverified_header",
        lambda _: {"kid": "kid-123", "alg": "RS256", "crit": ["custom"]},
    )

    with pytest.raises(JWTError, match="critical JOSE"):
        authenticator.decode("token")


def test_oidc_rejects_empty_token_and_missing_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    authenticator = oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="test-audience",
        jwks_cache=_FakeJWKSCache(),
    )
    with pytest.raises(JWTError, match="non-empty"):
        authenticator.decode("")
    with pytest.raises(JWTError, match="byte limit"):
        authenticator.decode("x" * 16_385)

    monkeypatch.setattr(
        oidc.jwt,
        "get_unverified_header",
        lambda _: {"kid": "kid-123", "alg": "RS256"},
    )
    monkeypatch.setattr(oidc.jwt, "decode", lambda *_args, **_kwargs: {})
    with pytest.raises(JWTError, match="sub"):
        authenticator.decode("token")


def test_jwks_fetch_rejects_insecure_remote_url() -> None:
    with pytest.raises(ValueError, match="must use HTTPS"):
        oidc.JWKSCache("http://issuer.example.test/jwks")


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

    with pytest.raises(JWTError, match="does not match"):
        authenticator.decode("token")
    assert captured == {}


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

    monkeypatch.setattr(
        oidc.jwt,
        "get_unverified_header",
        lambda token: {"kid": "kid-123", "alg": "RS256"},
    )

    def _decode(token: str, key: Any, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(key, (bytes, str)):
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


def test_development_rotation_preserves_verified_claim_errors() -> None:
    first_secret = b"first-secret-value-at-least-32-bytes!!"
    second_secret = b"second-secret-value-at-least-32-bytes!"
    now = int(time.time())
    expired = jwt.encode(
        {
            "sub": "dev-user",
            "iss": "https://issuer.test",
            "aud": "options-pricing-engine",
            "iat": now - 600,
            "nbf": now - 600,
            "exp": now - 400,
        },
        first_secret,
        algorithm="HS256",
    )
    authenticator = oidc.DevelopmentJWTAuthenticator(
        secrets=(first_secret, second_secret),
        issuer="https://issuer.test",
        audience="options-pricing-engine",
    )

    with pytest.raises(ExpiredSignatureError):
        authenticator.decode(expired)


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

    monkeypatch.setattr(
        oidc.jwt,
        "get_unverified_header",
        lambda token: {"kid": "kid-123", "alg": "RS256"},
    )

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
        leeway = kwargs["leeway"]
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


def test_decode_token_returns_401_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingCounter()
    monkeypatch.setattr(api_security, "AUTH_FAILURES", recorder)
    monkeypatch.setattr(
        api_security,
        "_get_authenticator",
        lambda: _StubAuthenticator(error=api_security.AuthenticationConfigurationError("missing")),
    )

    with pytest.raises(HTTPException) as exc:
        api_security._decode_token("token")

    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc.value.detail == "Authentication not configured"
    assert recorder.calls and recorder.calls[-1]["reason"] == "not_configured"


def test_decode_token_labels_expired_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingCounter()
    monkeypatch.setattr(api_security, "AUTH_FAILURES", recorder)
    monkeypatch.setattr(
        api_security,
        "_get_authenticator",
        lambda: _StubAuthenticator(error=ExpiredSignatureError("expired")),
    )

    with pytest.raises(HTTPException) as exc:
        api_security._decode_token("token")

    assert exc.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert recorder.calls and recorder.calls[-1]["reason"] == "expired"


def test_decode_token_labels_claim_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingCounter()
    monkeypatch.setattr(api_security, "AUTH_FAILURES", recorder)

    def _claim_error_auth() -> _StubAuthenticator:
        return _StubAuthenticator(error=InvalidAudienceError("Invalid audience"))

    monkeypatch.setattr(api_security, "_get_authenticator", _claim_error_auth)

    with pytest.raises(HTTPException):
        api_security._decode_token("token")

    assert recorder.calls and recorder.calls[-1]["reason"] == "aud"

    recorder.calls.clear()
    monkeypatch.setattr(
        api_security,
        "_get_authenticator",
        lambda: _StubAuthenticator(error=InvalidIssuerError("Invalid issuer")),
    )

    with pytest.raises(HTTPException):
        api_security._decode_token("token")

    assert recorder.calls and recorder.calls[-1]["reason"] == "iss"


def test_decode_token_labels_dev_signature_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingCounter()
    monkeypatch.setattr(api_security, "AUTH_FAILURES", recorder)
    monkeypatch.setattr(
        api_security,
        "_get_authenticator",
        lambda: _StubAuthenticator(error=DevelopmentSignatureError("bad sig")),
    )

    with pytest.raises(HTTPException):
        api_security._decode_token("token")

    assert recorder.calls and recorder.calls[-1]["reason"] == "dev_bad_sig"
