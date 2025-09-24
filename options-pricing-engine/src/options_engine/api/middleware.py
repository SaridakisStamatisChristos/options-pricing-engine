"""Custom ASGI middleware used by the FastAPI application."""

from __future__ import annotations

from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach basic security headers to every HTTP response."""

    def __init__(self, app, *, hsts_max_age: int = 31536000, include_subdomains: bool = True) -> None:
        super().__init__(app)
        self._hsts_value = "max-age={}".format(hsts_max_age)
        if include_subdomains:
            self._hsts_value += "; includeSubDomains"

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Strict-Transport-Security", self._hsts_value)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response

