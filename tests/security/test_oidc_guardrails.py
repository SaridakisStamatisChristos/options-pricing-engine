"""Adversarial boundary tests for OIDC, JWKS, and development JWT handling."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from jwt.exceptions import InvalidSignatureError, PyJWTError

from options_engine.security import oidc

_REAL_FETCH_JWKS = oidc._fetch_jwks


class _Response:
    def __init__(
        self,
        body: bytes = b'{"keys":[]}',
        *,
        headers: Mapping[str, str] | None = None,
        chunks: list[bytes] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.headers = dict(headers or {"content-type": "application/json"})
        self._chunks = chunks if chunks is not None else [body]
        self._error = error

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        if self._error is not None:
            raise self._error

    def iter_bytes(self) -> list[bytes]:
        return self._chunks


class _Client:
    def __init__(self, response: _Response) -> None:
        self._response = response

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def stream(self, method: str, url: str, *, headers: Mapping[str, str]) -> _Response:
        assert method == "GET"
        assert url == "https://issuer.test/jwks"
        assert headers == {"Accept": "application/json"}
        return self._response


def _install_response(monkeypatch: pytest.MonkeyPatch, response: _Response) -> None:
    monkeypatch.setattr(
        oidc.httpx,
        "Client",
        lambda *_args, **_kwargs: _Client(response),
    )


class _StaticCache:
    def __init__(self, key: Mapping[str, Any] | None = None) -> None:
        self.key = dict(key or {"kid": "kid-1", "kty": "RSA", "alg": "RS256"})
        self.calls: list[bool] = []

    def get_key(self, _kid: str, *, force_refresh: bool = False) -> Mapping[str, Any]:
        self.calls.append(force_refresh)
        return dict(self.key)


def _authenticator(cache: object | None = None) -> oidc.OIDCAuthenticator:
    return oidc.OIDCAuthenticator(
        issuer="https://issuer.test",
        audience="options-engine",
        jwks_cache=cache or _StaticCache(),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "url",
    [
        "",
        " https://issuer.test",
        "https://issuer.test/path with-space",
        "https://issuer.test/path\tvalue",
        "https://issuer.test/\x7f",
        "https://user:secret@issuer.test/jwks",
        "https:///missing-host",
        "https://issuer.test:99999/jwks",
        "https://issuer.test/jwks#fragment",
        "ftp://issuer.test/jwks",
        "http://issuer.test/jwks",
        "https://issuer.test/" + "x" * 2_100,
        7,
    ],
)
def test_https_url_validation_rejects_ambiguous_boundaries(url: object) -> None:
    with pytest.raises(ValueError):
        oidc._validate_https_url(url, label="issuer")  # type: ignore[arg-type]


def test_https_url_validation_allows_https_query_and_loopback_http() -> None:
    oidc._validate_https_url("https://issuer.test/jwks?tenant=one", label="JWKS")
    oidc._validate_https_url("http://localhost/jwks", label="JWKS")
    with pytest.raises(ValueError, match="query"):
        oidc._validate_https_url(
            "https://issuer.test/path;params?tenant=one",
            label="issuer",
            allow_query=False,
        )


def test_fetch_jwks_accepts_json_and_vendor_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b'{"keys":[{"kid":"one"}]}'
    for content_type in ("application/json; charset=utf-8", "application/jwk-set+json"):
        _install_response(
            monkeypatch,
            _Response(
                body,
                headers={
                    "content-type": content_type,
                    "content-length": str(len(body)),
                },
            ),
        )
        assert _REAL_FETCH_JWKS("https://issuer.test/jwks")["keys"][0]["kid"] == "one"


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({"content-type": "text/html"}, "content type"),
        (
            {"content-type": "application/json", "content-length": "unknown"},
            "Content-Length",
        ),
        (
            {"content-type": "application/json", "content-length": "\u0661"},
            "Content-Length",
        ),
        (
            {
                "content-type": "application/json",
                "content-length": str(oidc._MAX_JWKS_BYTES + 1),
            },
            "size limit",
        ),
        (
            {"content-type": "application/json", "content-length": "1" * 21},
            "size limit",
        ),
    ],
)
def test_fetch_jwks_rejects_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch,
    headers: Mapping[str, str],
    message: str,
) -> None:
    _install_response(monkeypatch, _Response(headers=headers))
    with pytest.raises(RuntimeError, match=message):
        _REAL_FETCH_JWKS("https://issuer.test/jwks")


def test_fetch_jwks_enforces_streaming_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oidc, "_MAX_JWKS_BYTES", 5)
    _install_response(
        monkeypatch,
        _Response(chunks=[b"123", b"456"], headers={"content-type": "application/json"}),
    )
    with pytest.raises(RuntimeError, match="size limit"):
        _REAL_FETCH_JWKS("https://issuer.test/jwks")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"{", "valid JSON"),
        (b'{"keys":[],"keys":[]}', "valid JSON"),
        (b"[]", "missing 'keys'"),
        (b"{}", "missing 'keys'"),
    ],
)
def test_fetch_jwks_rejects_ambiguous_json(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    message: str,
) -> None:
    _install_response(monkeypatch, _Response(body))
    with pytest.raises(RuntimeError, match=message):
        _REAL_FETCH_JWKS("https://issuer.test/jwks")


def test_duplicate_json_key_hook_is_deterministic() -> None:
    assert oidc._reject_duplicate_json_keys([("one", 1), ("two", 2)]) == {
        "one": 1,
        "two": 2,
    }
    with pytest.raises(ValueError, match="duplicate"):
        oidc._reject_duplicate_json_keys([("one", 1), ("one", 2)])


def test_oidc_claims_are_isolated_and_top_level_immutable() -> None:
    raw_claims = {"sub": "user", "nested": {"value": 1}}
    claims = oidc.OIDCClaims(
        subject="user",
        scopes=frozenset({"pricing:read"}),
        claims=raw_claims,
        kid="kid-1",
    )
    raw_claims["nested"]["value"] = 2
    assert claims.claims["nested"]["value"] == 1
    with pytest.raises(TypeError):
        claims.claims["new"] = "value"  # type: ignore[index]


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("subject", "", ValueError),
        ("subject", "x" * 513, ValueError),
        ("subject", 1, ValueError),
        ("kid", "", ValueError),
        ("kid", "x" * 257, ValueError),
        ("kid", 1, ValueError),
        ("scopes", ["pricing:read"], ValueError),
        ("scopes", frozenset({"bad scope"}), ValueError),
        ("scopes", frozenset({1}), ValueError),
        ("claims", [], TypeError),
    ],
)
def test_oidc_claims_validate_every_field(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    values: dict[str, object] = {
        "subject": "user",
        "scopes": frozenset(),
        "claims": {},
        "kid": "kid",
    }
    values[field] = value
    with pytest.raises(exception):
        oidc.OIDCClaims(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("refresh", "stale", "exception"),
    [
        (True, 900, TypeError),
        (0, 900, ValueError),
        (86_401, 86_401, ValueError),
        (10, 9, ValueError),
        (10, 86_401, ValueError),
    ],
)
def test_jwks_cache_configuration_is_bounded(
    refresh: object,
    stale: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        oidc.JWKSCache(
            "https://issuer.test/jwks",
            refresh_interval_seconds=refresh,  # type: ignore[arg-type]
            max_stale_seconds=stale,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"keys": []},
        {"keys": [None, {}, {"kid": ""}, {"kid": "x" * 257}]},
        {"keys": "not-a-list"},
    ],
)
def test_jwks_cache_rejects_payloads_without_usable_keys(payload: Mapping[str, Any]) -> None:
    cache = oidc.JWKSCache("https://issuer.test/jwks", fetcher=lambda _url: payload)
    with pytest.raises(oidc.JWKSUnavailableError):
        cache.get_key("kid")


def test_jwks_cache_rejects_oversized_key_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oidc, "_MAX_JWKS_KEYS", 1)
    cache = oidc.JWKSCache(
        "https://issuer.test/jwks",
        fetcher=lambda _url: {"keys": [{"kid": "one"}, {"kid": "two"}]},
    )
    with pytest.raises(oidc.JWKSUnavailableError):
        cache.get_key("one")


def test_jwks_cache_reset_refetches_keys() -> None:
    calls = 0

    def fetcher(_url: str) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        return {"keys": [{"kid": "key", "version": calls}]}

    cache = oidc.JWKSCache("https://issuer.test/jwks", fetcher=fetcher)
    assert cache.get_key("key")["version"] == 1
    cache.reset()
    assert cache.get_key("key")["version"] == 2


def test_jwks_cache_preserves_previous_keys_during_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    calls = 0
    monkeypatch.setattr(oidc.time, "monotonic", lambda: clock[0])

    def fetcher(_url: str) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        kid = "old" if calls == 1 else "new"
        return {"keys": [{"kid": kid, "version": calls}]}

    cache = oidc.JWKSCache("https://issuer.test/jwks", fetcher=fetcher)
    assert cache.get_key("old")["version"] == 1
    clock[0] = 31.0
    assert cache.get_key("new")["version"] == 2
    assert cache.get_key("old")["version"] == 1


def test_jwks_cache_force_refresh_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    calls = 0
    monkeypatch.setattr(oidc.time, "monotonic", lambda: clock[0])

    def fetcher(_url: str) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        return {"keys": [{"kid": "key", "version": calls}]}

    cache = oidc.JWKSCache("https://issuer.test/jwks", fetcher=fetcher)
    assert cache.get_key("key")["version"] == 1
    assert cache.get_key("key", force_refresh=True)["version"] == 1
    clock[0] = 31.0
    assert cache.get_key("key", force_refresh=True)["version"] == 2
    assert calls == 2


def test_jwks_cache_rejects_expired_private_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oidc.time, "monotonic", lambda: 100.0)
    cache = oidc.JWKSCache(
        "https://issuer.test/jwks",
        refresh_interval_seconds=10,
        max_stale_seconds=20,
        fetcher=lambda _url: {"keys": [{"kid": "key"}]},
    )
    cache._current_keys = {"key": {"kid": "key"}}
    cache._next_refresh = 1_000.0
    cache._last_successful_refresh = 0.0
    with pytest.raises(oidc.JWKSUnavailableError, match="expired"):
        cache.get_key("key")


@pytest.mark.parametrize(
    ("kid", "force", "exception"),
    [
        (None, False, KeyError),
        ("", False, KeyError),
        ("x" * 257, False, KeyError),
        ("kid", 1, TypeError),
    ],
)
def test_jwks_cache_validates_lookup_inputs(
    kid: object,
    force: object,
    exception: type[Exception],
) -> None:
    cache = oidc.JWKSCache(
        "https://issuer.test/jwks", fetcher=lambda _url: {"keys": [{"kid": "kid"}]}
    )
    with pytest.raises(exception):
        cache.get_key(kid, force_refresh=force)  # type: ignore[arg-type]


@pytest.mark.parametrize("audience", ["", " audience", "bad audience", "bad\taud", "x" * 513, 1])
def test_authenticators_reject_noncanonical_audiences(audience: object) -> None:
    with pytest.raises(ValueError, match="audience"):
        oidc.OIDCAuthenticator(
            issuer="https://issuer.test",
            audience=audience,  # type: ignore[arg-type]
            jwks_cache=_StaticCache(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="audience"):
        oidc.DevelopmentJWTAuthenticator(
            secrets=(b"x" * 32,),
            issuer="https://issuer.test",
            audience=audience,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("clock_skew", [True, -1, 301])
def test_authenticator_clock_skew_is_bounded(clock_skew: object) -> None:
    exception = TypeError if clock_skew is True else ValueError
    with pytest.raises(exception):
        oidc._bounded_integer("clock_skew", clock_skew, minimum=0, maximum=300)
    with pytest.raises(exception):
        oidc.OIDCAuthenticator(
            issuer="https://issuer.test",
            audience="audience",
            jwks_cache=_StaticCache(),  # type: ignore[arg-type]
            clock_skew_seconds=clock_skew,  # type: ignore[arg-type]
        )


def test_oidc_decode_maps_key_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unavailable:
        def get_key(self, _kid: str, *, force_refresh: bool = False) -> Mapping[str, Any]:
            raise oidc.JWKSUnavailableError("offline")

    authenticator = _authenticator(Unavailable())
    monkeypatch.setattr(
        oidc.jwt, "get_unverified_header", lambda _token: {"kid": "kid", "alg": "RS256"}
    )
    with pytest.raises(oidc.OIDCUnavailableError):
        authenticator.decode("token")


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ({"alg": "RS256"}, "kid"),
        ({"kid": "", "alg": "RS256"}, "kid"),
        ({"kid": "x" * 257, "alg": "RS256"}, "kid"),
        ({"kid": "kid", "alg": "RS256", "b64": False}, "unencoded"),
    ],
)
def test_oidc_decode_rejects_unsafe_headers(
    monkeypatch: pytest.MonkeyPatch,
    header: Mapping[str, Any],
    message: str,
) -> None:
    authenticator = _authenticator()
    monkeypatch.setattr(oidc.jwt, "get_unverified_header", lambda _token: header)
    with pytest.raises(PyJWTError, match=message):
        authenticator.decode("token")


def test_oidc_decode_refreshes_once_after_invalid_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _StaticCache()
    authenticator = _authenticator(cache)
    monkeypatch.setattr(
        oidc.jwt, "get_unverified_header", lambda _token: {"kid": "kid", "alg": "RS256"}
    )
    monkeypatch.setattr(authenticator, "_prepare_verification_key", lambda *_args: object())
    calls = 0

    def decode(*_args: object, **_kwargs: object) -> Mapping[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise InvalidSignatureError("rotated")
        return {"sub": "user", "scope": "pricing:read"}

    monkeypatch.setattr(oidc.jwt, "decode", decode)
    assert authenticator.decode("token").subject == "user"
    assert cache.calls == [False, True]


def test_oidc_decode_handles_refresh_outage_and_algorithm_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RotatingCache:
        fail = True

        def get_key(self, _kid: str, *, force_refresh: bool = False) -> Mapping[str, Any]:
            if force_refresh and self.fail:
                raise oidc.JWKSUnavailableError("offline")
            if force_refresh:
                return {"kid": "kid", "kty": "EC", "alg": "ES256", "crv": "P-256"}
            return {"kid": "kid", "kty": "RSA", "alg": "RS256"}

    cache = RotatingCache()
    authenticator = _authenticator(cache)
    monkeypatch.setattr(
        oidc.jwt, "get_unverified_header", lambda _token: {"kid": "kid", "alg": "RS256"}
    )
    monkeypatch.setattr(authenticator, "_prepare_verification_key", lambda *_args: object())
    monkeypatch.setattr(
        oidc.jwt,
        "decode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(InvalidSignatureError("bad")),
    )
    with pytest.raises(oidc.OIDCUnavailableError):
        authenticator.decode("token")
    cache.fail = False
    with pytest.raises(PyJWTError, match="refreshed signing key"):
        authenticator.decode("token")


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ({"kty": "RSA", "alg": "RS256", "key_ops": ["verify"] * 17}, "oversized"),
        ({"kty": "RSA", "alg": "RS256", "key_ops": ["verify", 1]}, "list"),
        ({"kty": "EC", "crv": "unknown"}, "Unsupported"),
        ({"kty": "RSA", "alg": 1}, "Unsupported"),
    ],
)
def test_oidc_key_algorithm_policy_closes_remaining_edges(
    key: Mapping[str, Any], message: str
) -> None:
    with pytest.raises(PyJWTError, match=message):
        _authenticator()._resolve_key_algorithm(key)


def test_oidc_key_preparation_rejects_invalid_and_weak_keys() -> None:
    with pytest.raises(PyJWTError, match="Invalid signing JWK"):
        _authenticator()._prepare_verification_key({"kty": "RSA"}, "RS256")

    weak_private = rsa.generate_private_key(public_exponent=65_537, key_size=1_024)
    weak_jwk = json.loads(RSAAlgorithm.to_jwk(weak_private.public_key()))
    with pytest.raises(PyJWTError, match="at least 2048"):
        _authenticator()._prepare_verification_key(weak_jwk, "RS256")


@pytest.mark.parametrize(
    ("secrets", "exception", "message"),
    [
        ((), ValueError, "at least one"),
        ((b"x" * 32,) * 9, ValueError, "at most eight"),
        ((1,), TypeError, "bytes or strings"),
        ((b"short",), ValueError, "between 32 and 4096"),
        ((b"x" * 4_097,), ValueError, "between 32 and 4096"),
        ((b"x" * 32, b"x" * 32), ValueError, "unique"),
    ],
)
def test_development_authenticator_validates_rotation_secrets(
    secrets: tuple[object, ...],
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        oidc.DevelopmentJWTAuthenticator(
            secrets=secrets,  # type: ignore[arg-type]
            issuer="https://issuer.test",
            audience="audience",
        )


def test_development_authenticator_accepts_explicit_string_secret() -> None:
    authenticator = oidc.DevelopmentJWTAuthenticator(
        secrets=("x" * 32,),  # type: ignore[arg-type]
        issuer="https://issuer.test",
        audience="audience",
    )
    assert authenticator._secrets == (b"x" * 32,)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ({"alg": "none"}, "HS256"),
        ({"alg": "HS256", "crit": ["exp"]}, "critical"),
        ({"alg": "HS256", "b64": False}, "unencoded"),
        ({"alg": "HS256", "kid": "x" * 257}, "kid"),
    ],
)
def test_development_decode_rejects_unsafe_headers(
    monkeypatch: pytest.MonkeyPatch,
    header: Mapping[str, Any],
    message: str,
) -> None:
    authenticator = oidc.DevelopmentJWTAuthenticator(
        secrets=(b"x" * 32,),
        issuer="https://issuer.test",
        audience="audience",
    )
    monkeypatch.setattr(oidc.jwt, "get_unverified_header", lambda _token: header)
    with pytest.raises(PyJWTError, match=message):
        authenticator.decode("token")


def test_development_decode_reports_bad_signature_and_missing_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = int(time.time())
    token = jwt.encode(
        {
            "sub": "user",
            "iss": "https://issuer.test",
            "aud": "audience",
            "iat": now,
            "nbf": now,
            "exp": now + 60,
        },
        b"wrong-secret-value-that-is-long-enough!!",
        algorithm="HS256",
    )
    authenticator = oidc.DevelopmentJWTAuthenticator(
        secrets=(b"x" * 32,),
        issuer="https://issuer.test",
        audience="audience",
    )
    with pytest.raises(oidc.DevelopmentSignatureError, match="signature"):
        authenticator.decode(token)

    monkeypatch.setattr(
        oidc.jwt, "get_unverified_header", lambda _token: {"alg": "HS256", "kid": "dev"}
    )
    monkeypatch.setattr(oidc.jwt, "decode", lambda *_args, **_kwargs: {})
    with pytest.raises(PyJWTError, match="sub"):
        authenticator.decode("token")


@pytest.mark.parametrize(
    ("claims", "expected"),
    [
        ({"scope": "one two one"}, frozenset({"one", "two"})),
        ({"scope": ["one", "two"]}, frozenset({"one", "two"})),
        ({"scp": "one two"}, frozenset({"one", "two"})),
        ({"scp": ("one", "two")}, frozenset({"one", "two"})),
        ({}, frozenset()),
    ],
)
def test_scope_extraction_accepts_standard_claim_shapes(
    claims: Mapping[str, Any], expected: frozenset[str]
) -> None:
    assert oidc._extract_scopes(claims) == expected


@pytest.mark.parametrize(
    "claims",
    [
        {"scope": ["scope"] * 129},
        {"scope": [1]},
        {"scope": ["bad scope"]},
        {"scope": ['bad"scope']},
        {"scp": ["x" * 257]},
    ],
)
def test_scope_extraction_rejects_untrusted_values(claims: Mapping[str, Any]) -> None:
    with pytest.raises(PyJWTError, match="scope"):
        oidc._extract_scopes(claims)
