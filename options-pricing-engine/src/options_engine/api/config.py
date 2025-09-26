"""Centralised application configuration derived from environment variables."""

from __future__ import annotations

import base64
import binascii
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


DEFAULT_OIDC_CLOCK_SKEW_SECONDS = 60
DEFAULT_OIDC_JWKS_CACHE_TTL_SECONDS = 300


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


def _get_env_alias(
    *names: str, default: str | None = None, required: bool = False
) -> str | None:
    """Return the first non-empty value from the supplied environment aliases."""

    for name in names:
        value = _get_env(name)
        if value is not None:
            return value

    if required:
        joined = " / ".join(names)
        raise RuntimeError(f"Environment variable {joined} is required")

    return default


def _split_csv(value: str | None) -> Tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _normalise_dev_secret(name: str, value: str) -> bytes:
    trimmed = value.strip()
    candidates: list[bytes] = []
    try:
        padded = trimmed + "=" * ((4 - len(trimmed) % 4) % 4)
        candidates.append(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, binascii.Error, UnicodeEncodeError):
        pass

    try:
        candidates.append(bytes.fromhex(trimmed))
    except ValueError:
        pass

    candidates.append(trimmed.encode("utf-8"))

    for candidate in candidates:
        if len(candidate) >= 32:
            return candidate

    raise RuntimeError(
        f"{name} must decode to at least 32 bytes; supply a base64url or hex encoded secret"
    )


def _as_int(name: str, *, default: int | None = None, minimum: int | None = None) -> int:
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
    return value


def _as_float(
    name: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
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
    if minimum is not None and value < minimum:
        raise RuntimeError(f"Environment variable {name} must be >= {minimum}")
    return value


def _as_bool(name: str, *, default: bool) -> bool:
    raw = _get_env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable view over application configuration."""

    environment: str
    allowed_hosts: Tuple[str, ...]
    allowed_origins: Tuple[str, ...]
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
    dev_jwt_secrets: Tuple[bytes, ...]
    oidc_clock_skew_seconds: int
    oidc_jwks_cache_ttl_seconds: int

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from the current process environment."""

    environment_raw = (
        _get_env_alias("ENV", "OPE_ENVIRONMENT", default="development") or "development"
    ).lower()
    if environment_raw in {"prod", "production"}:
        environment = "production"
    elif environment_raw in {"dev", "development"}:
        environment = "development"
    else:
        environment = environment_raw

    allowed_hosts = _split_csv(_get_env_alias("ALLOWED_HOSTS", "OPE_ALLOWED_HOSTS"))
    if not allowed_hosts:
        if environment == "production":
            raise RuntimeError(
                "ALLOWED_HOSTS must be provided when ENV/OPE_ENVIRONMENT=production"
            )
        allowed_hosts = ("localhost", "127.0.0.1")

    allowed_origins = _split_csv(_get_env_alias("CORS_ALLOWED_ORIGINS", "OPE_ALLOWED_ORIGINS"))
    if not allowed_origins and environment != "production":
        allowed_origins = (
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8000",
        )

    dev_primary_candidates = [
        (name, value)
        for name in ("DEV_JWT_SECRET", "OPE_JWT_SECRET")
        if (value := _get_env(name))
    ]
    if len(dev_primary_candidates) > 1:
        raise RuntimeError("Only one of DEV_JWT_SECRET or OPE_JWT_SECRET may be set")
    dev_primary_secret = dev_primary_candidates[0] if dev_primary_candidates else None

    additional_candidates: list[tuple[str, str]] = []
    for env_name in ("DEV_JWT_ADDITIONAL_SECRETS", "OPE_JWT_ADDITIONAL_SECRETS"):
        additional_candidates.extend((env_name, item) for item in _split_csv(_get_env(env_name)))

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

    oidc_issuer = _get_env_alias("OIDC_ISSUER")
    oidc_audience = _get_env_alias("OIDC_AUDIENCE")
    oidc_jwks_url = _get_env_alias("OIDC_JWKS_URL")

    if environment == "production":
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
                "Production deployment requires OIDC configuration: "
                + ", ".join(sorted(missing))
            )
        if dev_secret_envs:
            names = ", ".join(sorted(dev_secret_envs))
            raise RuntimeError(
                "Development JWT secrets are forbidden in production: " + names
            )

    threadpool_workers = _as_int("OPE_THREADS", default=8, minimum=1)
    threadpool_queue_size = _as_int("OPE_THREAD_QUEUE_MAX", default=32, minimum=0)
    threadpool_queue_timeout_seconds = _as_float(
        "OPE_THREAD_QUEUE_TIMEOUT_SECONDS", default=0.5, minimum=0.0
    )
    threadpool_task_timeout_seconds = _as_float(
        "OPE_THREAD_TASK_TIMEOUT_SECONDS", default=30.0, minimum=0.0
    )

    max_contracts = _as_int("OPE_MAX_CONTRACTS", default=1000, minimum=1)
    max_risk_contracts = _as_int("OPE_MAX_RISK_CONTRACTS", default=max_contracts, minimum=1)

    monte_carlo_seed_raw = _get_env("OPE_MONTE_CARLO_SEED")
    if monte_carlo_seed_raw is None:
        monte_carlo_seed = None
    else:
        try:
            monte_carlo_seed = int(monte_carlo_seed_raw)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("OPE_MONTE_CARLO_SEED must be an integer") from exc

    rate_limit_default = _get_env("RATE_LIMIT_DEFAULT", default="60/minute") or "60/minute"
    max_body_bytes = _as_int("MAX_BODY_BYTES", default=1_048_576, minimum=1_024)

    clock_skew_raw = _get_env_alias("OIDC_CLOCK_SKEW_S")
    if clock_skew_raw is None:
        oidc_clock_skew_seconds = DEFAULT_OIDC_CLOCK_SKEW_SECONDS
    else:
        try:
            oidc_clock_skew_seconds = int(clock_skew_raw)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("OIDC_CLOCK_SKEW_S must be an integer") from exc
        if oidc_clock_skew_seconds < 0:
            raise RuntimeError("OIDC_CLOCK_SKEW_S must be >= 0")

    jwks_ttl_raw = _get_env_alias("OIDC_JWKS_CACHE_TTL_S")
    if jwks_ttl_raw is None:
        oidc_jwks_cache_ttl_seconds = DEFAULT_OIDC_JWKS_CACHE_TTL_SECONDS
    else:
        try:
            oidc_jwks_cache_ttl_seconds = int(jwks_ttl_raw)
        except ValueError as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("OIDC_JWKS_CACHE_TTL_S must be an integer") from exc
        if oidc_jwks_cache_ttl_seconds < 60:
            raise RuntimeError("OIDC_JWKS_CACHE_TTL_S must be >= 60 seconds")

    if dev_jwt_secrets and (not oidc_issuer or not oidc_audience):
        raise RuntimeError(
            "DEV_JWT_SECRET requires OIDC_ISSUER and OIDC_AUDIENCE to validate dev tokens"
        )

    return Settings(
        environment=environment,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        cors_allow_credentials=_as_bool("OPE_CORS_ALLOW_CREDENTIALS", default=True),
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
    )

