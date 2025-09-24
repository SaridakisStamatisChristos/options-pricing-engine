"""Authentication and authorization helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable, Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = os.getenv("OPE_JWT_SECRET", "change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

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


def create_access_token(data: Dict[str, object], expires_delta: Optional[timedelta] = None) -> str:
    payload = dict(data)
    expire_at = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload.update({"exp": expire_at})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
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
