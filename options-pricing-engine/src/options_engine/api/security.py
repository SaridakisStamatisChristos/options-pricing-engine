"""Authentication and authorization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import get_settings

ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

settings = get_settings()

if not settings.jwt_secrets:
    raise RuntimeError("JWT configuration requires at least one signing secret")

PRIMARY_SECRET = settings.jwt_secrets[0]
ROTATION_SECRETS = settings.jwt_secrets[1:]


_USER_STORE: Dict[str, Dict[str, object]] = {
    "quant_trader": {
        "username": "quant_trader",
        "hashed_password": "$2b$12$fq8u5uTv8/bZLa4sWHofLeKPZnQgdNRu6mEL2BzfC/8xINwmt6wb6",
        "permissions": [
            "pricing:read",
            "pricing:write",
            "risk:read",
            "market-data:read",
            "market-data:write",
        ],
    }
}


@dataclass(slots=True)
class User:
    username: str
    permissions: List[str]


def authenticate_user(username: str, password: str) -> Optional[Dict[str, object]]:
    record = _USER_STORE.get(username)
    if not record:
        return None
    if not pwd_context.verify(password, record["hashed_password"]):
        return None
    return record


def _build_claims(data: Dict[str, object], expires_delta: Optional[timedelta]) -> Dict[str, object]:
    now = datetime.now(UTC)
    payload = dict(data)
    expire_at = now + (expires_delta or timedelta(minutes=settings.jwt_expire_minutes))

    payload.setdefault("jti", uuid4().hex)
    payload["iat"] = now
    payload["nbf"] = now
    payload["exp"] = expire_at
    payload.setdefault("iss", settings.jwt_issuer)
    payload.setdefault("aud", settings.jwt_audience)
    return payload


def create_access_token(data: Dict[str, object], expires_delta: Optional[timedelta] = None) -> str:
    payload = _build_claims(data, expires_delta)
    if "sub" not in payload:
        raise ValueError("JWT payload must include a 'sub' claim")
    return jwt.encode(payload, PRIMARY_SECRET, algorithm=ALGORITHM)


def _decode_token(token: str) -> Dict[str, object]:
    candidates = (PRIMARY_SECRET, *ROTATION_SECRETS)
    last_error: Optional[JWTError] = None
    for secret in candidates:
        try:
            return jwt.decode(
                token,
                secret,
                algorithms=[ALGORITHM],
                audience=settings.jwt_audience,
                issuer=settings.jwt_issuer,
                options={"require": ["exp", "iat", "nbf", "sub"]},
                leeway=settings.jwt_leeway_seconds,
            )
        except JWTError as exc:
            last_error = exc
    raise last_error or JWTError("Invalid token")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = _decode_token(credentials.credentials)
        subject = payload.get("sub")
        if subject is None:
            raise unauthorized
    except JWTError as exc:  # pragma: no cover - defensive programming
        raise unauthorized from exc

    record = _USER_STORE.get(subject)
    if not record:
        raise unauthorized

    return User(username=subject, permissions=list(record.get("permissions", [])))


def require_permission(permission: str) -> Callable[[User], User]:
    def dependency(user: User = Depends(get_current_user)) -> User:
        if permission not in user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required",
            )
        return user

    return dependency
