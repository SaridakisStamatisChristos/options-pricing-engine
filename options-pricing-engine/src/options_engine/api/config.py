"""Centralised application configuration derived from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


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


def _split_csv(value: str | None) -> Tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


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
    dev_jwt_secrets: Tuple[str, ...]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings from the current process environment."""

    environment = (_get_env("OPE_ENVIRONMENT", default="development") or "development").lower()

    allowed_hosts = _split_csv(_get_env("OPE_ALLOWED_HOSTS"))
    if not allowed_hosts:
        if environment == "production":
            raise RuntimeError(
                "OPE_ALLOWED_HOSTS must be provided when OPE_ENVIRONMENT=production"
            )
        allowed_hosts = ("localhost", "127.0.0.1")

    allowed_origins = _split_csv(_get_env("OPE_ALLOWED_ORIGINS"))
    if not allowed_origins and environment != "production":
        allowed_origins = (
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8000",
        )

    dev_primary_secret = _get_env("OPE_JWT_SECRET")
    rotation_secrets = _split_csv(_get_env("OPE_JWT_ADDITIONAL_SECRETS"))
    dev_jwt_secrets = tuple(secret for secret in (dev_primary_secret, *rotation_secrets) if secret)

    oidc_issuer = _get_env("OIDC_ISSUER")
    oidc_audience = _get_env("OIDC_AUDIENCE")
    oidc_jwks_url = _get_env("OIDC_JWKS_URL")

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
        if dev_primary_secret:
            raise RuntimeError("OPE_JWT_SECRET must not be set when OPE_ENVIRONMENT=production")

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
    )

