"""OpenID Connect helpers for verifying JWT access tokens."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

try:  # pragma: no cover - httpx optional in offline tests
    import httpx
except ModuleNotFoundError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    import json
    from urllib.request import urlopen
from jose import JWTError, jwt

CLOCK_SKEW_SECONDS = 60
_REFRESH_INTERVAL_SECONDS = 300

_JWKSFetcher = Callable[[str], Mapping[str, Any]]


def _fetch_jwks(url: str) -> Mapping[str, Any]:
    """Fetch the JWKS document from the configured authority."""

    if httpx is not None:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
    else:  # pragma: no cover - fallback for environments without httpx
        with urlopen(url, timeout=5.0) as response:
            data = json.loads(response.read().decode())
    if not isinstance(data, Mapping) or "keys" not in data:
        raise RuntimeError("OIDC JWKS document missing 'keys' field")
    return data


@dataclass(slots=True)
class OIDCClaims:
    """Subset of token claims relevant for downstream consumers."""

    subject: str
    scopes: frozenset[str]
    claims: Mapping[str, Any]
    kid: str


class JWKSCache:
    """Thread-safe cache of signing keys supporting rotation."""

    def __init__(
        self,
        jwks_url: str,
        *,
        refresh_interval_seconds: int = _REFRESH_INTERVAL_SECONDS,
        fetcher: _JWKSFetcher | None = None,
    ) -> None:
        self._jwks_url = jwks_url
        self._refresh_interval_seconds = max(1, refresh_interval_seconds)
        self._fetcher = fetcher or _fetch_jwks
        self._lock = threading.RLock()
        self._current_keys: Dict[str, Mapping[str, Any]] = {}
        self._previous_keys: Dict[str, Mapping[str, Any]] = {}
        self._next_refresh: float = 0.0

    def reset(self) -> None:
        """Force the cache to reload keys on next access."""

        with self._lock:
            self._next_refresh = 0.0
            self._current_keys = {}
            self._previous_keys = {}

    def _refresh_locked(self) -> None:
        payload = self._fetcher(self._jwks_url)
        keys = payload.get("keys", [])
        if not isinstance(keys, Iterable):
            raise RuntimeError("OIDC JWKS payload invalid")

        parsed: Dict[str, Mapping[str, Any]] = {}
        for entry in keys:
            if not isinstance(entry, MutableMapping):
                continue
            kid = entry.get("kid")
            if isinstance(kid, str) and kid:
                parsed[kid] = dict(entry)

        if not parsed:
            raise RuntimeError("OIDC JWKS payload did not contain any signing keys")

        self._previous_keys = self._current_keys
        self._current_keys = parsed
        self._next_refresh = time.monotonic() + self._refresh_interval_seconds

    def _ensure_keys_locked(self) -> None:
        now = time.monotonic()
        if now >= self._next_refresh or not self._current_keys:
            self._refresh_locked()

    def _get_from_previous_locked(self, kid: str) -> Optional[Mapping[str, Any]]:
        value = self._previous_keys.get(kid)
        if value is not None:
            return value
        # Force a refresh in case a new key was introduced after rotation
        self._refresh_locked()
        return self._previous_keys.get(kid)

    def get_key(self, kid: str) -> Mapping[str, Any]:
        """Return the signing key for the supplied key identifier."""

        if not kid:
            raise KeyError("kid must be provided")

        with self._lock:
            self._ensure_keys_locked()
            key = self._current_keys.get(kid)
            if key is not None:
                return key
            previous = self._get_from_previous_locked(kid)
            if previous is not None:
                return previous
        raise KeyError(f"Unknown signing key: {kid}")


class OIDCAuthenticator:
    """Validate JWT access tokens issued by an OpenID Connect provider."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_cache: JWKSCache,
        clock_skew_seconds: int = CLOCK_SKEW_SECONDS,
    ) -> None:
        if not issuer:
            raise ValueError("issuer must be provided")
        if not audience:
            raise ValueError("audience must be provided")
        self._issuer = issuer
        self._audience = audience
        self._jwks_cache = jwks_cache
        self._clock_skew_seconds = max(0, clock_skew_seconds)

    def decode(self, token: str) -> OIDCClaims:
        """Decode and validate the supplied JWT access token."""

        if not token:
            raise JWTError("Token must not be empty")

        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise JWTError("Token missing 'kid' header")
        algorithm = header.get("alg")
        if not isinstance(algorithm, str) or not algorithm:
            raise JWTError("Token missing 'alg' header")

        key = self._jwks_cache.get_key(kid)
        options = {
            "require": ["exp", "iat", "nbf", "iss", "aud", "sub"],
            "leeway": self._clock_skew_seconds,
        }

        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options=options,
            )
        except JWTError:
            # Attempt a forced refresh in case the JWKS rotated between requests.
            self._jwks_cache.reset()
            key = self._jwks_cache.get_key(kid)
            claims = jwt.decode(
                token,
                key,
                algorithms=[algorithm],
                audience=self._audience,
                issuer=self._issuer,
                options=options,
            )

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise JWTError("Token missing 'sub' claim")

        scopes = _extract_scopes(claims)
        return OIDCClaims(subject=subject, scopes=scopes, claims=claims, kid=kid)


def _extract_scopes(claims: Mapping[str, Any]) -> frozenset[str]:
    raw_scope = claims.get("scope")
    if isinstance(raw_scope, str):
        parts = raw_scope.split()
    elif isinstance(raw_scope, Iterable):
        parts = [str(item) for item in raw_scope]
    else:
        raw_scope = claims.get("scp")
        if isinstance(raw_scope, str):
            parts = raw_scope.split()
        elif isinstance(raw_scope, Iterable):
            parts = [str(item) for item in raw_scope]
        else:
            parts = []
    return frozenset(scope for scope in parts if scope)


__all__ = ["CLOCK_SKEW_SECONDS", "OIDCAuthenticator", "OIDCClaims", "JWKSCache"]
