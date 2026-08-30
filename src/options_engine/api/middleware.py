"""Custom ASGI middleware used by the FastAPI application."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable, Collection
from uuid import uuid4

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_429_TOO_MANY_REQUESTS,
)
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..observability.metrics import PAYLOAD_TOO_LARGE, RATE_LIMIT_REJECTIONS

LOGGER = logging.getLogger("options_engine.request")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_HTTP_413_CONTENT_TOO_LARGE = 413


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach basic security headers to every HTTP response."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        hsts_max_age: int = 63_072_000,
        include_subdomains: bool = True,
        preload: bool = True,
    ) -> None:
        super().__init__(app)
        if isinstance(hsts_max_age, bool) or not isinstance(hsts_max_age, int):
            raise TypeError("hsts_max_age must be an integer")
        if not 0 <= hsts_max_age <= 63_072_000:
            raise ValueError("hsts_max_age must be within [0, 63072000]")
        if not isinstance(include_subdomains, bool) or not isinstance(preload, bool):
            raise TypeError("include_subdomains and preload must be booleans")
        parts = [f"max-age={hsts_max_age}"]
        if include_subdomains:
            parts.append("includeSubDomains")
        if preload:
            parts.append("preload")
        self._hsts_value = "; ".join(parts)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Strict-Transport-Security", self._hsts_value)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), geolocation=(), microphone=()",
        )
        return response


class BodySizeLimitMiddleware:
    """Bound request memory while consuming chunked or fixed-length bodies."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        if isinstance(max_body_bytes, bool) or not isinstance(max_body_bytes, int):
            raise TypeError("max_body_bytes must be an integer")
        if not 1 <= max_body_bytes <= 104_857_600:
            raise ValueError("max_body_bytes must be within [1, 104857600]")
        self._max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        content_lengths = [
            value.decode("latin-1") for key, value in headers if key.lower() == b"content-length"
        ]
        transfer_encodings = [
            value.decode("latin-1") for key, value in headers if key.lower() == b"transfer-encoding"
        ]
        if len(content_lengths) > 1 or len(transfer_encodings) > 1:
            await self._reject_invalid_framing(scope, receive, send)
            return
        if content_lengths and transfer_encodings:
            # RFC 9112 requires Transfer-Encoding to override Content-Length,
            # but rejecting the ambiguous combination closes request-smuggling
            # differences between a proxy and the ASGI server.
            await self._reject_invalid_framing(scope, receive, send)
            return
        if transfer_encodings and transfer_encodings[0].strip().lower() != "chunked":
            await self._reject_invalid_framing(scope, receive, send)
            return

        declared_size: int | None = None
        if content_lengths:
            raw_size = content_lengths[0]
            if len(raw_size) > 20 or re.fullmatch(r"[0-9]+", raw_size) is None:
                await self._reject_invalid_framing(scope, receive, send)
                return
            declared_size = int(raw_size)
            if declared_size > self._max_body_bytes:
                await self._reject(scope, receive, send)
                return

        # Buffer no more than the configured maximum and stop consuming as soon
        # as a chunk crosses it. This also protects requests without Content-Length.
        body = bytearray()
        disconnected = False
        total = 0
        while True:
            message = await receive()
            message_type = message.get("type")
            if message_type == "http.disconnect":
                disconnected = True
                break
            if message_type != "http.request":
                await self._reject_invalid_framing(scope, receive, send)
                return
            chunk = message.get("body", b"")
            if not isinstance(chunk, bytes):
                await self._reject_invalid_framing(scope, receive, send)
                return
            total += len(chunk)
            if total > self._max_body_bytes:
                await self._reject(scope, receive, send)
                return
            if declared_size is not None and total > declared_size:
                await self._reject_invalid_framing(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        if not disconnected and declared_size is not None and total != declared_size:
            await self._reject_invalid_framing(scope, receive, send)
            return

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                if disconnected:
                    return {"type": "http.disconnect"}
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _reject(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        # Routing has not run yet, so the raw path is attacker-controlled.
        # Keep this metric bounded and use trusted ingress logs for attribution.
        PAYLOAD_TOO_LARGE.labels(route="all").inc()
        response = JSONResponse(
            status_code=_HTTP_413_CONTENT_TOO_LARGE,
            content={"detail": "Payload too large"},
        )
        request = Request(scope, receive=receive)
        response.headers.setdefault("X-Request-ID", ensure_request_id(request))
        await response(scope, receive, send)

    async def _reject_invalid_framing(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        response = JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content={"detail": "Invalid request framing"},
        )
        request = Request(scope, receive=receive)
        response.headers.setdefault("X-Request-ID", ensure_request_id(request))
        await response(scope, receive, send)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply one exact moving-window limit per client across protected routes.

    The backing store is deliberately process-local. Deployments that need a
    fleet-wide quota enforce it at the trusted ingress, while this middleware
    remains a deterministic final line of defence for every process. Sharing
    the quota across routes prevents callers multiplying their allowance by
    distributing requests over multiple endpoint paths.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        rate_limit: str,
        excluded_paths: Collection[str] = (),
    ) -> None:
        super().__init__(app)
        try:
            self._rate_limit = parse(rate_limit)
        except ValueError as exc:
            raise RuntimeError(f"Invalid RATE_LIMIT_DEFAULT value: {rate_limit!r}") from exc
        if self._rate_limit.amount < 1:
            raise RuntimeError("RATE_LIMIT_DEFAULT must allow at least one request")
        if self._rate_limit.amount > 1_000_000 or self._rate_limit.get_expiry() > 86_400:
            raise RuntimeError("RATE_LIMIT_DEFAULT exceeds the supported quota or window")
        self._limiter = MovingWindowRateLimiter(MemoryStorage())
        self._excluded_paths = frozenset(excluded_paths)

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "OPTIONS" or request.url.path in self._excluded_paths:
            return await call_next(request)

        client_host = request.client.host if request.client is not None else "unknown"
        identity = f"client:{client_host[:255]}"
        allowed = self._limiter.hit(self._rate_limit, identity)
        window = self._limiter.get_window_stats(self._rate_limit, identity)
        reset_seconds = max(1, math.ceil(window.reset_time - time.time()))

        response: Response
        if not allowed:
            RATE_LIMIT_REJECTIONS.labels(route="all").inc()
            response = JSONResponse(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
            )
            response.headers["Retry-After"] = str(reset_seconds)
            response.headers["X-Request-ID"] = ensure_request_id(request)
        else:
            response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(self._rate_limit.amount)
        response.headers["X-RateLimit-Remaining"] = str(window.remaining)
        response.headers["X-RateLimit-Reset"] = str(math.ceil(window.reset_time))
        return response


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
    if isinstance(request_id, str) and request_id:
        return request_id
    candidate = request.headers.get("x-request-id")
    request_id = (
        candidate if candidate and _REQUEST_ID_PATTERN.fullmatch(candidate) else uuid4().hex
    )
    request.state.request_id = request_id
    return request_id


def track_request_duration(request: Request) -> Callable[[int], None]:
    start = time.perf_counter()

    def complete(status_code: int) -> None:
        duration = time.perf_counter() - start
        log_request_completion(request=request, status_code=status_code, duration_seconds=duration)

    return complete
