"""Centralised application configuration derived from environment variables."""

from __future__ import annotations

import base64
import binascii
import ipaddress
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse

DEFAULT_OIDC_CLOCK_SKEW_SECONDS = 60
DEFAULT_OIDC_JWKS_CACHE_TTL_SECONDS = 300
DEFAULT_OIDC_JWKS_MAX_STALE_SECONDS = 900


def _get_env(name: str, *, default: str | None = None, required: bool = False) -> str | None:
    """Return a trimmed environment variable value.

    Parameters
    ----------
    name:
        Name of the environment variable to read.
    default:
        Optional default returned when the variable is not set.
    required:
        When ``True`` a ``RuntimeError`` is raised if the variable is missing
        or blank.
    """

    value = os.getenv(name)
    if value is None:
        if required:
            raise RuntimeError(f"Environment variable {name} is required")
        return default

    trimmed = value.strip()
    if not trimmed:
        if required:
            raise RuntimeError(f"Environment variable {name} must not be blank")
        return default
    return trimmed


def _get_env_alias(*names: str, default: str | None = None, required: bool = False) -> str | None:
    """Return a non-empty value from equivalent environment aliases.

    Multiple aliases may be present during a migration, but silently choosing
    between conflicting security settings is unsafe.
    """

    configured = [(name, value) for name in names if (value := _get_env(name)) is not None]
    if len({value for _, value in configured}) > 1:
        raise RuntimeError(f"Environment aliases {' / '.join(names)} must not conflict")
    if configured:
        return configured[0][1]

    if required:
        joined = " / ".join(names)
        raise RuntimeError(f"Environment variable {joined} is required")

    return default


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _normalise_dev_secret(name: str, value: str) -> bytes:
    """Decode an explicitly encoded secret, or treat it as UTF-8 plaintext.

    Prefixes avoid guessing whether an otherwise valid-looking plaintext key
    was meant to be base64url or hexadecimal. Existing unprefixed plaintext
    configuration therefore remains stable.
    """

    trimmed = value.strip()
    if len(trimmed) > 4096:
        raise RuntimeError(f"{name} must not exceed 4096 characters")

    try:
        if trimmed.startswith("base64:"):
            encoded = trimmed.removeprefix("base64:")
            padded = encoded + "=" * ((4 - len(encoded) % 4) % 4)
            candidate = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        elif trimmed.startswith("hex:"):
            encoded = trimmed.removeprefix("hex:")
            if not re.fullmatch(r"[0-9A-Fa-f]+", encoded) or len(encoded) % 2:
                raise ValueError
            candidate = bytes.fromhex(encoded)
        else:
            candidate = trimmed.encode("utf-8")
    except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
        raise RuntimeError(f"{name} is not valid explicit base64url or hexadecimal data") from exc

    if not 32 <= len(candidate) <= 4096:
        raise RuntimeError(f"{name} must contain at least 32 bytes and at most 4096 decoded bytes")
    return candidate


def _validate_url(
    name: str,
    value: str,
    *,
    require_https: bool,
    allow_query: bool = True,
) -> None:
    if len(value) > 2048 or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value
    ):
        raise RuntimeError(f"{name} must be a valid URL no longer than 2048 characters")
    parsed = urlparse(value)
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise RuntimeError(f"{name} must include a hostname and must not contain credentials")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise RuntimeError(f"{name} contains an invalid port") from exc
    if parsed.fragment:
        raise RuntimeError(f"{name} must not contain a URL fragment")
    if not allow_query and (parsed.params or parsed.query):
        raise RuntimeError(f"{name} must not contain URL parameters or a query")
    local_host = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (
        not require_https and parsed.scheme == "http" and local_host
    ):
        qualifier = "" if require_https else " (HTTP is allowed only for localhost)"
        raise RuntimeError(f"{name} must use HTTPS{qualifier}")


def _validate_cors_origin(origin: str) -> None:
    if len(origin) > 2048 or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127 for character in origin
    ):
        raise RuntimeError(f"CORS origin is invalid or oversized: {origin!r}")
    if origin == "*":
        return
    parsed = urlparse(origin)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise RuntimeError(f"CORS origin contains an invalid port: {origin!r}") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError(f"CORS origin must be an exact HTTP(S) origin: {origin!r}")


def _validate_allowed_host(host: str) -> None:
    """Validate a Starlette trusted-host pattern without accepting URLs."""

    if host == "*":
        return
    candidate = host.removeprefix("*.")
    if "*" in candidate or (host.startswith("*") and not host.startswith("*.")):
        raise RuntimeError(f"Invalid ALLOWED_HOSTS pattern: {host!r}")
    if (
        not candidate
        or len(candidate) > 253
        or candidate.endswith(".")
        or any(ord(character) < 33 or ord(character) > 126 for character in candidate)
        or any(character in candidate for character in "/@?#[]")
    ):
        raise RuntimeError(f"Invalid ALLOWED_HOSTS entry: {host!r}")

    try:
        ipaddress.ip_address(candidate)
        return
    except ValueError:
        pass

    label_pattern = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")
    if not all(label_pattern.fullmatch(label) for label in candidate.split(".")):
        raise RuntimeError(f"Invalid ALLOWED_HOSTS entry: {host!r}")


def _as_int(
    name: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = _get_env(name)
    if raw is None:
        if default is None:
            raise RuntimeError(f"Environment variable {name} is required")
        return default
    try:
        value = int(raw)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc
    if minimum is not None and value < minimum:
        raise RuntimeError(f"Environment variable {name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"Environment variable {name} must be <= {maximum}")
    return value


def _as_float(
    name: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = _get_env(name)
    if raw is None:
        if default is None:
            raise RuntimeError(f"Environment variable {name} is required")
        return default
    try:
        value = float(raw)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Environment variable {name} must be a number") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"Environment variable {name} must be finite")
    if minimum is not None and value < minimum:
        raise RuntimeError(f"Environment variable {name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"Environment variable {name} must be <= {maximum}")
    return value


def _as_bool(name: str, *, default: bool) -> bool:
    raw = _get_env(name)
    if raw is None:
        return default
    normalised = raw.strip().lower()
    if normalised in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalised in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise RuntimeError(f"Environment variable {name} must be a boolean")


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable view over application configuration."""

    environment: str
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    cors_allow_credentials: bool
    threadpool_workers: int
    threadpool_queue_size: int
    threadpool_queue_timeout_seconds: float
    threadpool_task_timeout_seconds: float
    max_pricing_contracts: int
    max_risk_contracts: int
    monte_carlo_seed: int | None
    rate_limit_default: str
    max_body_bytes: int
    oidc_issuer: str | None
    oidc_audience: str | None
    oidc_jwks_url: str | None
    dev_jwt_secrets: tuple[bytes, ...]
    oidc_clock_skew_seconds: int
    oidc_jwks_cache_ttl_seconds: int
    oidc_jwks_max_stale_seconds: int

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "staging"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from the current process environment."""

    def normalise_environment(value: str) -> str:
        raw = value.lower()
        if raw in {"prod", "production"}:
            return "production"
        if raw in {"stage", "staging"}:
            return "staging"
        if raw in {"dev", "development"}:
            return "development"
        if raw in {"test", "testing"}:
            return "test"
        raise RuntimeError("ENV/OPE_ENVIRONMENT must be development, test, staging, or production")

    environment_candidates = [
        (name, normalise_environment(value))
        for name in ("ENV", "OPE_ENVIRONMENT")
        if (value := _get_env(name)) is not None
    ]
    if len({value for _, value in environment_candidates}) > 1:
        raise RuntimeError("ENV and OPE_ENVIRONMENT must not specify conflicting environments")
    environment = environment_candidates[0][1] if environment_candidates else "development"
    secure_environment = environment in {"production", "staging"}

    allowed_hosts = _split_csv(_get_env_alias("ALLOWED_HOSTS", "OPE_ALLOWED_HOSTS"))
    if not allowed_hosts:
        if secure_environment:
            raise RuntimeError("ALLOWED_HOSTS must be provided when ENV/OPE_ENVIRONMENT=production")
        allowed_hosts = ("localhost", "127.0.0.1")
    if len(allowed_hosts) > 100 or any(len(host) > 253 for host in allowed_hosts):
        raise RuntimeError("ALLOWED_HOSTS contains too many or oversized entries")
    for host in allowed_hosts:
        _validate_allowed_host(host)
    if len({host.casefold() for host in allowed_hosts}) != len(allowed_hosts):
        raise RuntimeError("ALLOWED_HOSTS must not contain duplicate entries")
    if secure_environment and any("*" in host for host in allowed_hosts):
        raise RuntimeError("Wildcard ALLOWED_HOSTS entries are forbidden in staging/production")

    allowed_origins = _split_csv(_get_env_alias("CORS_ALLOWED_ORIGINS", "OPE_ALLOWED_ORIGINS"))
    if not allowed_origins and not secure_environment:
        allowed_origins = (
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8000",
        )
    if len(allowed_origins) > 100:
        raise RuntimeError("CORS_ALLOWED_ORIGINS contains too many entries")
    for origin in allowed_origins:
        _validate_cors_origin(origin)
    if len({origin.casefold().rstrip("/") for origin in allowed_origins}) != len(allowed_origins):
        raise RuntimeError("CORS_ALLOWED_ORIGINS must not contain duplicate entries")
    cors_allow_credentials = _as_bool("OPE_CORS_ALLOW_CREDENTIALS", default=True)
    if cors_allow_credentials and "*" in allowed_origins:
        raise RuntimeError("Wildcard CORS origins cannot be used with credentials")

    dev_primary_candidates = [
        (name, value) for name in ("DEV_JWT_SECRET", "OPE_JWT_SECRET") if (value := _get_env(name))
    ]
    if len(dev_primary_candidates) > 1:
        raise RuntimeError("Only one of DEV_JWT_SECRET or OPE_JWT_SECRET may be set")
    dev_primary_secret = dev_primary_candidates[0] if dev_primary_candidates else None

    additional_candidates: list[tuple[str, str]] = []
    additional_aliases = [
        (env_name, raw)
        for env_name in ("DEV_JWT_ADDITIONAL_SECRETS", "OPE_JWT_ADDITIONAL_SECRETS")
        if (raw := _get_env(env_name)) is not None
    ]
    if len(additional_aliases) > 1:
        raise RuntimeError(
            "Only one of DEV_JWT_ADDITIONAL_SECRETS or OPE_JWT_ADDITIONAL_SECRETS may be set"
        )
    for env_name, raw in additional_aliases:
        additional_candidates.extend((env_name, item) for item in _split_csv(raw))
    if len(additional_candidates) > 7:
        raise RuntimeError("At most seven additional development JWT secrets are supported")
    if additional_candidates and dev_primary_secret is None:
        raise RuntimeError("Additional development JWT secrets require a primary secret")

    dev_secret_envs: set[str] = set()
    dev_secret_bytes: list[bytes] = []
    if dev_primary_secret is not None:
        env_name, raw_value = dev_primary_secret
        dev_secret_envs.add(env_name)
        dev_secret_bytes.append(_normalise_dev_secret(env_name, raw_value))

    for index, (env_name, raw_value) in enumerate(additional_candidates):
        dev_secret_envs.add(env_name)
        label = f"{env_name}[{index}]" if len(additional_candidates) > 1 else env_name
        dev_secret_bytes.append(_normalise_dev_secret(label, raw_value))

    dev_jwt_secrets = tuple(dev_secret_bytes)
    if len(set(dev_jwt_secrets)) != len(dev_jwt_secrets):
        raise RuntimeError("Development JWT secrets must be unique")

    oidc_issuer = _get_env_alias("OIDC_ISSUER")
    oidc_audience = _get_env_alias("OIDC_AUDIENCE")
    oidc_jwks_url = _get_env_alias("OIDC_JWKS_URL")

    if oidc_issuer:
        _validate_url(
            "OIDC_ISSUER",
            oidc_issuer,
            require_https=secure_environment,
            allow_query=False,
        )
    if oidc_jwks_url:
        _validate_url("OIDC_JWKS_URL", oidc_jwks_url, require_https=secure_environment)
    if oidc_audience and (
        len(oidc_audience) > 512
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in oidc_audience
        )
    ):
        raise RuntimeError("OIDC_AUDIENCE must contain 1-512 non-whitespace characters")

    if secure_environment:
        missing = [
            name
            for name, value in (
                ("OIDC_ISSUER", oidc_issuer),
                ("OIDC_AUDIENCE", oidc_audience),
                ("OIDC_JWKS_URL", oidc_jwks_url),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Production deployment requires OIDC configuration: " + ", ".join(sorted(missing))
            )
        if dev_secret_envs:
            names = ", ".join(sorted(dev_secret_envs))
            raise RuntimeError("Development JWT secrets are forbidden in production: " + names)

    threadpool_workers = _as_int("OPE_THREADS", default=8, minimum=1, maximum=256)
    threadpool_queue_size = _as_int("OPE_THREAD_QUEUE_MAX", default=32, minimum=0, maximum=100_000)
    threadpool_queue_timeout_seconds = _as_float(
        "OPE_THREAD_QUEUE_TIMEOUT_SECONDS",
        default=0.5,
        minimum=0.0,
        maximum=86_400.0,
    )
    threadpool_task_timeout_seconds = _as_float(
        "OPE_THREAD_TASK_TIMEOUT_SECONDS",
        default=30.0,
        minimum=0.0,
        maximum=86_400.0,
    )

    max_contracts = _as_int("OPE_MAX_CONTRACTS", default=1000, minimum=1, maximum=100_000)
    max_risk_contracts = _as_int(
        "OPE_MAX_RISK_CONTRACTS", default=max_contracts, minimum=1, maximum=100_000
    )

    monte_carlo_seed_raw = _get_env("OPE_MONTE_CARLO_SEED")
    if monte_carlo_seed_raw is None:
        monte_carlo_seed = None
    else:
        try:
            monte_carlo_seed = int(monte_carlo_seed_raw)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("OPE_MONTE_CARLO_SEED must be an integer") from exc
        if not 0 <= monte_carlo_seed <= 2**128 - 1:
            raise RuntimeError(f"OPE_MONTE_CARLO_SEED must be within [0, {2**128 - 1}]")

    rate_limit_default = _get_env("RATE_LIMIT_DEFAULT", default="60/minute") or "60/minute"
    if len(rate_limit_default) > 64:
        raise RuntimeError("RATE_LIMIT_DEFAULT must not exceed 64 characters")
    max_body_bytes = _as_int(
        "MAX_BODY_BYTES", default=1_048_576, minimum=1_024, maximum=104_857_600
    )

    oidc_clock_skew_seconds = _as_int(
        "OIDC_CLOCK_SKEW_S",
        default=DEFAULT_OIDC_CLOCK_SKEW_SECONDS,
        minimum=0,
        maximum=300,
    )
    oidc_jwks_cache_ttl_seconds = _as_int(
        "OIDC_JWKS_CACHE_TTL_S",
        default=DEFAULT_OIDC_JWKS_CACHE_TTL_SECONDS,
        minimum=60,
        maximum=86_400,
    )
    oidc_jwks_max_stale_seconds = _as_int(
        "OIDC_JWKS_MAX_STALE_S",
        default=max(DEFAULT_OIDC_JWKS_MAX_STALE_SECONDS, oidc_jwks_cache_ttl_seconds),
        minimum=oidc_jwks_cache_ttl_seconds,
        maximum=86_400,
    )

    if dev_jwt_secrets and (not oidc_issuer or not oidc_audience):
        raise RuntimeError(
            "DEV_JWT_SECRET requires OIDC_ISSUER and OIDC_AUDIENCE to validate dev tokens"
        )

    return Settings(
        environment=environment,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        cors_allow_credentials=cors_allow_credentials,
        threadpool_workers=threadpool_workers,
        threadpool_queue_size=threadpool_queue_size,
        threadpool_queue_timeout_seconds=threadpool_queue_timeout_seconds,
        threadpool_task_timeout_seconds=threadpool_task_timeout_seconds,
        max_pricing_contracts=max_contracts,
        max_risk_contracts=max_risk_contracts,
        monte_carlo_seed=monte_carlo_seed,
        rate_limit_default=rate_limit_default,
        max_body_bytes=max_body_bytes,
        oidc_issuer=oidc_issuer,
        oidc_audience=oidc_audience,
        oidc_jwks_url=oidc_jwks_url,
        dev_jwt_secrets=dev_jwt_secrets,
        oidc_clock_skew_seconds=oidc_clock_skew_seconds,
        oidc_jwks_cache_ttl_seconds=oidc_jwks_cache_ttl_seconds,
        oidc_jwks_max_stale_seconds=oidc_jwks_max_stale_seconds,
    )
