"""Authentication and authorization helpers backed by OIDC."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Mapping, Union

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from ..observability.metrics import AUTH_FAILURES
from ..security import (
    DevelopmentJWTAuthenticator,
    OIDCAuthenticator,
    OIDCClaims,
    JWKSCache,
)
from .config import get_settings

security = HTTPBearer(auto_error=False)


Authenticator = Union[OIDCAuthenticator, DevelopmentJWTAuthenticator]


@lru_cache(maxsize=1)
def _get_authenticator() -> Authenticator:
    settings = get_settings()
    if settings.oidc_issuer and settings.oidc_audience and settings.oidc_jwks_url:
        cache = JWKSCache(settings.oidc_jwks_url)
        return OIDCAuthenticator(
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
            jwks_cache=cache,
        )

    if settings.dev_jwt_secrets:
        return DevelopmentJWTAuthenticator(
            secrets=settings.dev_jwt_secrets,
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
        )

    raise RuntimeError(
        "Authentication requires OIDC_ISSUER/OIDC_AUDIENCE/OIDC_JWKS_URL or OPE_JWT_SECRET"
    )


@dataclass(slots=True)
class User:
    """Representation of the authenticated principal."""

    subject: str
    scopes: frozenset[str]
    claims: Mapping[str, object]


def _decode_token(raw_token: str) -> OIDCClaims:
    try:
        return _get_authenticator().decode(raw_token)
    except KeyError as exc:  # pragma: no cover - defensive guard
        AUTH_FAILURES.labels(reason="unknown_kid").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    except JWTError as exc:
        AUTH_FAILURES.labels(reason="jwt_error").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> User:
    if credentials is None:
        AUTH_FAILURES.labels(reason="missing_token").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    claims = _decode_token(credentials.credentials)
    request.state.user_sub = claims.subject
    return User(subject=claims.subject, scopes=claims.scopes, claims=dict(claims.claims))


def require_permission(permission: str) -> Callable[[User], User]:
    def dependency(user: User = Depends(get_current_user)) -> User:
        if permission not in user.scopes:
            AUTH_FAILURES.labels(reason="insufficient_scope").inc()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return user

    return dependency


__all__ = ["User", "get_current_user", "require_permission"]
