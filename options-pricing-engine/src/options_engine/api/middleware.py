"""Custom ASGI middleware used by the FastAPI application."""

from __future__ import annotations

import json
import logging
import time
from typing import Awaitable, Callable
from uuid import uuid4

from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..observability.metrics import PAYLOAD_TOO_LARGE

LOGGER = logging.getLogger("options_engine.request")


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


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests exceeding the configured payload limit."""

    def __init__(self, app, *, max_body_bytes: int) -> None:
        super().__init__(app)
        self._max_body_bytes = max(1, max_body_bytes)
        self._string_threshold = max(128, self._max_body_bytes // 32)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        header_value = request.headers.get("content-length")
        if header_value is not None:
            try:
                size = int(header_value)
            except ValueError:
                size = None
            if size is not None and size > self._max_body_bytes:
                return self._reject(request)

        body = await request.body()
        if body and len(body) > self._max_body_bytes:
            return self._reject(request)

        if body and self._string_threshold:
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if payload is not None and self._contains_large_string(payload):
                return self._reject(request)

        return await call_next(request)

    def _reject(self, request: Request) -> JSONResponse:
        route = request.url.path
        PAYLOAD_TOO_LARGE.labels(route=route).inc()
        response = JSONResponse(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            content={"detail": "Payload too large"},
        )
        response.headers.setdefault("X-Request-ID", ensure_request_id(request))
        return response

    def _contains_large_string(self, payload: object) -> bool:
        threshold = self._string_threshold
        if isinstance(payload, str):
            return len(payload.encode()) > threshold
        if isinstance(payload, dict):
            return any(self._contains_large_string(value) for value in payload.values())
        if isinstance(payload, (list, tuple, set)):
            return any(self._contains_large_string(item) for item in payload)
        return False


def log_request_completion(
    *,
    request: Request,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Emit structured JSON logs for completed requests."""

    payload = {
        "event": "request.complete",
        "request_id": getattr(request.state, "request_id", None),
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "latency_ms": round(duration_seconds * 1000.0, 3),
    }
    user_sub = getattr(request.state, "user_sub", None)
    if user_sub:
        payload["user"] = user_sub
    LOGGER.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def ensure_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id:
        return request_id
    request_id = request.headers.get("x-request-id") or uuid4().hex
    request.state.request_id = request_id
    return request_id


def track_request_duration(request: Request) -> Callable[[int], None]:
    start = time.perf_counter()

    def complete(status_code: int) -> None:
        duration = time.perf_counter() - start
        log_request_completion(request=request, status_code=status_code, duration_seconds=duration)

    return complete

