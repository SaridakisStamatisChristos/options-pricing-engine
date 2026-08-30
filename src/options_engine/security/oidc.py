"""OpenID Connect helpers for verifying JWT access tokens."""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

import httpx
import jwt
from jwt.exceptions import (
    InvalidSignatureError,
    PyJWTError,
)

if TYPE_CHECKING:
    from jwt.types import Options

CLOCK_SKEW_SECONDS = 60
_REFRESH_INTERVAL_SECONDS = 300
_GRACE_REFRESH_SECONDS = 60
_MAX_STALE_SECONDS = 900
_MAX_CLOCK_SKEW_SECONDS = 300
_MAX_TOKEN_BYTES = 16_384
_MAX_JWKS_BYTES = 1_048_576
_MAX_JWKS_KEYS = 128
_MAX_KID_LENGTH = 256
_MAX_SUBJECT_LENGTH = 512
_MAX_AUDIENCE_LENGTH = 512
_SCOPE_TOKEN_PATTERN = re.compile(r"[\x21\x23-\x5B\x5D-\x7E]{1,256}")

logger = logging.getLogger(__name__)

_JWKSFetcher = Callable[[str], Mapping[str, Any]]


class JWKSUnavailableError(RuntimeError):
    """Raised when signing keys cannot be fetched and none are cached."""


class OIDCUnavailableError(RuntimeError):
    """Raised when OIDC verification cannot be performed due to key outages."""


class DevelopmentSignatureError(PyJWTError):
    """Raised when development JWT signatures cannot be verified."""


def _validate_https_url(url: str, *, label: str, allow_query: bool = True) -> None:
    if not isinstance(url, str) or not url or url != url.strip():
        raise ValueError(f"{label} must be a non-empty URL without surrounding whitespace")
    if len(url) > 2048 or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in url
    ):
        raise ValueError(f"{label} must be a valid URL no longer than 2048 characters")
    parsed = urlparse(url)
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} must include a hostname and must not contain credentials")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} contains an invalid port") from exc
    if parsed.fragment:
        raise ValueError(f"{label} must not contain a URL fragment")
    if not allow_query and (parsed.params or parsed.query):
        raise ValueError(f"{label} must not contain URL parameters or a query")
    local_host = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (parsed.scheme == "http" and local_host):
        raise ValueError(f"{label} must use HTTPS (HTTP is allowed only for localhost)")


def _validate_jwks_url(url: str) -> None:
    _validate_https_url(url, label="JWKS URL")


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalised = int(value)
    if not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return normalised


def _validate_token(token: object) -> str:
    if not isinstance(token, str) or not token:
        raise PyJWTError("Token must be a non-empty string")
    if len(token.encode("utf-8")) > _MAX_TOKEN_BYTES:
        raise PyJWTError(f"Token exceeds the {_MAX_TOKEN_BYTES}-byte limit")
    return token


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"OIDC JWKS response contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _fetch_jwks(url: str) -> Mapping[str, Any]:
    """Fetch the JWKS document from the configured authority."""
    _validate_jwks_url(url)
    with (
        httpx.Client(timeout=httpx.Timeout(5.0), follow_redirects=False) as client,
        client.stream("GET", url, headers={"Accept": "application/json"}) as response,
    ):
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
        if content_type != "application/json" and not content_type.endswith("+json"):
            raise RuntimeError("OIDC JWKS response must use a JSON content type")
        content_length = response.headers.get("content-length")
        if content_length is not None:
            if not content_length.isascii() or not content_length.isdecimal():
                raise RuntimeError("OIDC JWKS response has an invalid Content-Length")
            if len(content_length) > 20 or int(content_length) > _MAX_JWKS_BYTES:
                raise RuntimeError("OIDC JWKS response exceeds the size limit")
        payload = bytearray()
        for chunk in response.iter_bytes():
            if len(chunk) > _MAX_JWKS_BYTES - len(payload):
                raise RuntimeError("OIDC JWKS response exceeds the size limit")
            payload.extend(chunk)
    try:
        data = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise RuntimeError("OIDC JWKS response is not valid JSON") from exc
    if not isinstance(data, Mapping) or "keys" not in data:
        raise RuntimeError("OIDC JWKS document missing 'keys' field")
    return data


@dataclass(frozen=True, slots=True)
class OIDCClaims:
    """Subset of token claims relevant for downstream consumers."""

    subject: str
    scopes: frozenset[str]
    claims: Mapping[str, Any]
    kid: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.subject, str)
            or not self.subject
            or len(self.subject) > _MAX_SUBJECT_LENGTH
        ):
            raise ValueError("subject must contain between 1 and 512 characters")
        if not isinstance(self.kid, str) or not self.kid or len(self.kid) > _MAX_KID_LENGTH:
            raise ValueError("kid must contain between 1 and 256 characters")
        if not isinstance(self.scopes, frozenset) or any(
            not isinstance(scope, str) or _SCOPE_TOKEN_PATTERN.fullmatch(scope) is None
            for scope in self.scopes
        ):
            raise ValueError("scopes must be a frozenset of valid OAuth scope tokens")
        if not isinstance(self.claims, Mapping):
            raise TypeError("claims must be a mapping")
        object.__setattr__(
            self,
            "claims",
            MappingProxyType(copy.deepcopy(dict(self.claims))),
        )


class JWKSCache:
    """Thread-safe cache of signing keys supporting rotation."""

    def __init__(
        self,
        jwks_url: str,
        *,
        refresh_interval_seconds: int = _REFRESH_INTERVAL_SECONDS,
        max_stale_seconds: int = _MAX_STALE_SECONDS,
        fetcher: _JWKSFetcher | None = None,
    ) -> None:
        _validate_jwks_url(jwks_url)
        self._jwks_url = jwks_url
        self._refresh_interval_seconds = _bounded_integer(
            "refresh_interval_seconds",
            refresh_interval_seconds,
            minimum=1,
            maximum=86_400,
        )
        self._max_stale_seconds = _bounded_integer(
            "max_stale_seconds",
            max_stale_seconds,
            minimum=self._refresh_interval_seconds,
            maximum=86_400,
        )
        self._fetcher = fetcher or _fetch_jwks
        self._lock = threading.RLock()
        self._current_keys: dict[str, Mapping[str, Any]] = {}
        self._previous_keys: dict[str, Mapping[str, Any]] = {}
        self._next_refresh: float = 0.0
        self._next_forced_refresh: float = 0.0
        self._last_successful_refresh: float | None = None

    def reset(self) -> None:
        """Force the cache to reload keys on next access."""
        with self._lock:
            self._next_refresh = 0.0
            self._next_forced_refresh = 0.0
            self._last_successful_refresh = None
            self._current_keys = {}
            self._previous_keys = {}

    def _refresh_locked(self) -> None:
        now = time.monotonic()
        try:
            payload = self._fetcher(self._jwks_url)
            keys = payload.get("keys", [])
            if not isinstance(keys, list):
                raise ValueError("OIDC JWKS payload 'keys' must be a list")
            if len(keys) > _MAX_JWKS_KEYS:
                raise ValueError(f"OIDC JWKS payload exceeds the {_MAX_JWKS_KEYS}-key limit")

            parsed: dict[str, Mapping[str, Any]] = {}
            for entry in keys:
                if not isinstance(entry, MutableMapping):
                    continue
                kid = entry.get("kid")
                if isinstance(kid, str) and 0 < len(kid) <= _MAX_KID_LENGTH:
                    if kid in parsed:
                        raise ValueError(f"OIDC JWKS payload contains duplicate kid: {kid!r}")
                    parsed[kid] = copy.deepcopy(dict(entry))
            if not parsed:
                raise ValueError("OIDC JWKS payload did not contain signing keys")
        except Exception as exc:  # pragma: no cover - network failures are environment specific
            grace = min(self._refresh_interval_seconds, _GRACE_REFRESH_SECONDS)
            self._next_refresh = now + grace
            self._next_forced_refresh = now + grace
            if (
                not self._current_keys
                or self._last_successful_refresh is None
                or now - self._last_successful_refresh > self._max_stale_seconds
            ):
                raise JWKSUnavailableError("JWKS signing keys are unavailable") from exc
            logger.warning(
                "Failed to refresh JWKS keys; continuing with cached keys",
                exc_info=exc,
            )
            return

        self._previous_keys = self._current_keys
        self._current_keys = parsed
        self._last_successful_refresh = now
        self._next_refresh = now + self._refresh_interval_seconds
        self._next_forced_refresh = now + min(self._refresh_interval_seconds, 30)

    def _ensure_keys_locked(self) -> None:
        now = time.monotonic()
        if now >= self._next_refresh or not self._current_keys:
            self._refresh_locked()
        if (
            self._last_successful_refresh is None
            or now - self._last_successful_refresh > self._max_stale_seconds
        ):
            raise JWKSUnavailableError("Cached JWKS signing keys have expired")

    def get_key(self, kid: str, *, force_refresh: bool = False) -> Mapping[str, Any]:
        """Return the signing key for the supplied key identifier."""
        if not isinstance(kid, str) or not kid:
            raise KeyError("kid must be provided")
        if len(kid) > _MAX_KID_LENGTH:
            raise KeyError("kid exceeds the supported length")
        if not isinstance(force_refresh, bool):
            raise TypeError("force_refresh must be a boolean")

        with self._lock:
            self._ensure_keys_locked()
            key = self._current_keys.get(kid)
            if key is not None and not force_refresh:
                return copy.deepcopy(dict(key))
            if key is None and not force_refresh:
                previous_key = self._previous_keys.get(kid)
                if previous_key is not None:
                    return copy.deepcopy(dict(previous_key))

            # Unknown kids and invalid signatures may indicate rotation. Bound
            # forced refreshes so attacker-controlled tokens cannot turn every
            # authentication attempt into an outbound JWKS request.
            now = time.monotonic()
            if now >= self._next_forced_refresh:
                self._refresh_locked()
            key = self._current_keys.get(kid)
            if key is not None:
                return copy.deepcopy(dict(key))

            # During rollouts, some providers serve both old/new sets; try PREVIOUS as a grace.
            key = self._previous_keys.get(kid)
            if key is not None:
                return copy.deepcopy(dict(key))

        raise KeyError(f"Unknown signing key: {kid}")


class OIDCAuthenticator:
    """Validate JWT access tokens issued by an OpenID Connect provider."""

    # Conservative default allow-list; narrow to your IdP if possible (e.g., {"RS256"}).
    _ALLOWED_ALGS = frozenset(
        {"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512"}
    )

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_cache: JWKSCache,
        clock_skew_seconds: int = CLOCK_SKEW_SECONDS,
    ) -> None:
        _validate_https_url(issuer, label="issuer", allow_query=False)
        if (
            not isinstance(audience, str)
            or not audience
            or audience != audience.strip()
            or len(audience) > _MAX_AUDIENCE_LENGTH
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in audience
            )
        ):
            raise ValueError("audience must contain between 1 and 512 characters")
        self._issuer = issuer
        self._audience = audience
        self._jwks_cache = jwks_cache
        self._clock_skew_seconds = _bounded_integer(
            "clock_skew_seconds",
            clock_skew_seconds,
            minimum=0,
            maximum=_MAX_CLOCK_SKEW_SECONDS,
        )

    def decode(self, token: str) -> OIDCClaims:
        """Decode and validate the supplied JWT access token."""
        token = _validate_token(token)

        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid or len(kid) > _MAX_KID_LENGTH:
            raise PyJWTError("Token missing 'kid' header")
        critical = header.get("crit")
        if critical not in (None, []):
            raise PyJWTError("Token uses unsupported critical JOSE headers")
        if "b64" in header and header["b64"] is not True:
            raise PyJWTError("Token uses unsupported unencoded payload semantics")

        # Fetch key by kid, then decide algorithm from the KEY (not untrusted header).
        try:
            key = self._jwks_cache.get_key(kid)
        except JWKSUnavailableError as exc:
            raise OIDCUnavailableError("JWKS signing keys are unavailable") from exc
        key_alg = self._resolve_key_algorithm(key)
        header_alg = header.get("alg")
        if header_alg != key_alg:
            raise PyJWTError("Token algorithm does not match the signing key")

        options: Options = {
            "enforce_minimum_key_length": True,
            "verify_signature": True,
            "verify_aud": True,
            "verify_exp": True,
            "verify_nbf": True,
            "verify_iat": True,
            "verify_iss": True,
            "verify_sub": True,
            "require": ["aud", "exp", "iat", "iss", "nbf", "sub"],
        }

        def _decode_with(signing_key: Mapping[str, Any], algorithm: str) -> Mapping[str, Any]:
            verification_key = self._prepare_verification_key(signing_key, algorithm)
            return cast(
                Mapping[str, Any],
                jwt.decode(
                    token,
                    verification_key,
                    algorithms=[algorithm],
                    audience=self._audience,
                    issuer=self._issuer,
                    leeway=self._clock_skew_seconds,
                    options=options,
                ),
            )

        try:
            claims = _decode_with(key, key_alg)
        except InvalidSignatureError as initial_error:
            # Attempt a rate-limited, non-destructive refresh in case the
            # provider rotated a key while retaining the same kid.
            try:
                refreshed_key = self._jwks_cache.get_key(kid, force_refresh=True)
            except JWKSUnavailableError as exc:  # pragma: no cover - integration path
                raise OIDCUnavailableError("JWKS signing keys are unavailable") from exc
            refreshed_alg = self._resolve_key_algorithm(refreshed_key)
            if header_alg != refreshed_alg:
                raise PyJWTError(
                    "Token algorithm does not match the refreshed signing key"
                ) from initial_error
            claims = _decode_with(refreshed_key, refreshed_alg)

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject or len(subject) > _MAX_SUBJECT_LENGTH:
            raise PyJWTError("Token missing 'sub' claim")

        scopes = _extract_scopes(claims)
        return OIDCClaims(subject=subject, scopes=scopes, claims=claims, kid=kid)

    def _resolve_key_algorithm(self, key: Mapping[str, Any]) -> str:
        use = key.get("use")
        if use not in (None, "sig"):
            raise PyJWTError("JWK is not authorized for signatures")
        key_ops = key.get("key_ops")
        if key_ops is not None:
            if not isinstance(key_ops, list) or not all(
                isinstance(operation, str) for operation in key_ops
            ):
                raise PyJWTError("JWK key_ops must be a list of strings")
            if len(key_ops) > 16 or len(set(key_ops)) != len(key_ops):
                raise PyJWTError("JWK key_ops is oversized or contains duplicates")
            if "verify" not in key_ops:
                raise PyJWTError("JWK is not authorized for verification")

        key_alg = key.get("alg")
        kty = key.get("kty")
        if key_alg is None:
            # Some JWKS omit 'alg' per key; default by kty conservatively.
            if kty == "RSA":
                key_alg = "RS256"
            elif kty == "EC":
                curve_algorithms = {"P-256": "ES256", "P-384": "ES384", "P-521": "ES512"}
                key_alg = curve_algorithms.get(str(key.get("crv")))
            else:
                raise PyJWTError("Unsupported or unknown key algorithm")
        elif not isinstance(key_alg, str) or key_alg not in self._ALLOWED_ALGS:
            raise PyJWTError("Unsupported or unknown key algorithm")
        if not isinstance(key_alg, str):
            raise PyJWTError("Unsupported or unknown key algorithm")
        if (key_alg.startswith(("RS", "PS")) and kty != "RSA") or (
            key_alg.startswith("ES") and kty != "EC"
        ):
            raise PyJWTError("JWK key type does not match its algorithm")
        expected_curve = {"ES256": "P-256", "ES384": "P-384", "ES512": "P-521"}.get(key_alg)
        if expected_curve is not None and key.get("crv") != expected_curve:
            raise PyJWTError("JWK elliptic curve does not match its algorithm")
        return key_alg

    @staticmethod
    def _prepare_verification_key(key: Mapping[str, Any], algorithm: str) -> Any:
        try:
            verification_key = jwt.PyJWK.from_dict(dict(key), algorithm=algorithm).key
        except (PyJWTError, TypeError, ValueError) as exc:
            raise PyJWTError("Invalid signing JWK") from exc
        key_size = getattr(verification_key, "key_size", None)
        if algorithm.startswith(("RS", "PS")) and (
            not isinstance(key_size, int) or key_size < 2_048
        ):
            raise PyJWTError("RSA signing keys must be at least 2048 bits")
        return verification_key


class DevelopmentJWTAuthenticator:
    """Validate HMAC-signed JWTs for non-production development flows."""

    _ALLOWED_ALGS = ("HS256",)

    def __init__(
        self,
        *,
        secrets: tuple[bytes, ...],
        issuer: str,
        audience: str,
        clock_skew_seconds: int = CLOCK_SKEW_SECONDS,
    ) -> None:
        if not secrets:
            raise ValueError("at least one development secret must be provided")
        if len(secrets) > 8:
            raise ValueError("at most eight development secrets are supported")
        _validate_https_url(issuer, label="issuer", allow_query=False)
        if (
            not isinstance(audience, str)
            or not audience
            or audience != audience.strip()
            or len(audience) > _MAX_AUDIENCE_LENGTH
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in audience
            )
        ):
            raise ValueError("audience must contain between 1 and 512 characters")

        normalised_secrets: list[bytes] = []
        for secret in secrets:
            if isinstance(secret, bytes):
                normalised_secrets.append(secret)
            elif isinstance(secret, str):
                normalised_secrets.append(secret.encode("utf-8"))
            else:
                raise TypeError("development JWT secrets must be bytes or strings")
        normalized = tuple(normalised_secrets)
        for index, secret in enumerate(normalized):
            if not 32 <= len(secret) <= 4096:
                raise ValueError(
                    "development JWT secrets must contain between 32 and 4096 bytes "
                    f"(secret #{index + 1})"
                )
        if len(set(normalized)) != len(normalized):
            raise ValueError("development JWT secrets must be unique")
        self._secrets = normalized
        self._issuer = issuer
        self._audience = audience
        self._clock_skew_seconds = _bounded_integer(
            "clock_skew_seconds",
            clock_skew_seconds,
            minimum=0,
            maximum=_MAX_CLOCK_SKEW_SECONDS,
        )

    def decode(self, token: str) -> OIDCClaims:
        token = _validate_token(token)

        header = jwt.get_unverified_header(token)
        if header.get("alg") != "HS256":
            raise PyJWTError("Development token must use HS256")
        critical = header.get("crit")
        if critical not in (None, []):
            raise PyJWTError("Token uses unsupported critical JOSE headers")
        if "b64" in header and header["b64"] is not True:
            raise PyJWTError("Token uses unsupported unencoded payload semantics")
        kid_raw = header.get("kid")
        if isinstance(kid_raw, str) and len(kid_raw) > _MAX_KID_LENGTH:
            raise PyJWTError("Development token kid exceeds the supported length")
        kid = kid_raw if isinstance(kid_raw, str) and kid_raw else "development"

        options: Options = {
            "enforce_minimum_key_length": True,
            "verify_signature": True,
            "verify_exp": True,
            "verify_nbf": True,
            "verify_iat": True,
            "verify_aud": True,
            "verify_iss": True,
            "verify_sub": True,
            "require": ["aud", "exp", "iat", "iss", "nbf", "sub"],
        }

        kwargs: dict[str, Any] = {
            "options": options,
            "algorithms": list(self._ALLOWED_ALGS),
            "audience": self._audience,
            "issuer": self._issuer,
            "leeway": self._clock_skew_seconds,
        }

        last_error: PyJWTError | None = None
        for secret in self._secrets:
            try:
                claims = jwt.decode(token, secret, **kwargs)
            except InvalidSignatureError as exc:
                last_error = exc
                continue
            except PyJWTError:
                # A claims error is raised only after this secret verified the
                # signature; preserve the precise result instead of masking it
                # by trying the remaining rotation keys.
                raise
            subject = claims.get("sub")
            if not isinstance(subject, str) or not subject or len(subject) > _MAX_SUBJECT_LENGTH:
                raise PyJWTError("Token missing 'sub' claim")
            scopes = _extract_scopes(claims)
            return OIDCClaims(subject=subject, scopes=scopes, claims=claims, kid=kid)

        if last_error is not None:
            raise DevelopmentSignatureError(
                "Development token signature verification failed"
            ) from last_error
        raise DevelopmentSignatureError("Token could not be verified with development secrets")


def _extract_scopes(claims: Mapping[str, Any]) -> frozenset[str]:
    raw_scope = claims.get("scope")
    if isinstance(raw_scope, str):
        parts = raw_scope.split()
    elif isinstance(raw_scope, (list, tuple, set, frozenset)):
        parts = list(raw_scope)
    else:
        raw_scope = claims.get("scp")
        if isinstance(raw_scope, str):
            parts = raw_scope.split()
        elif isinstance(raw_scope, (list, tuple, set, frozenset)):
            parts = list(raw_scope)
        else:
            parts = []
    if len(parts) > 128 or any(
        not isinstance(scope, str) or _SCOPE_TOKEN_PATTERN.fullmatch(scope) is None
        for scope in parts
    ):
        raise PyJWTError("Invalid scope claim")
    return frozenset(parts)


__all__ = [
    "CLOCK_SKEW_SECONDS",
    "DevelopmentJWTAuthenticator",
    "DevelopmentSignatureError",
    "JWKSCache",
    "JWKSUnavailableError",
    "OIDCAuthenticator",
    "OIDCClaims",
    "OIDCUnavailableError",
]
