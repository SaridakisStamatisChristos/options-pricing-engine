"""FastAPI application exposing the pricing engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Iterable, Mapping, cast

try:  # pragma: no cover - psutil may not be installed in minimal environments
    import psutil
except Exception:  # pragma: no cover - best effort fallback
    psutil = None  # type: ignore[assignment]

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..observability.metrics import (
    RATE_LIMIT_REJECTIONS,
    REQUEST_COUNT,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
)
from .config import get_settings
from .middleware import BodySizeLimitMiddleware, SecurityHeadersMiddleware, ensure_request_id, track_request_duration
from .routes import market_data, pricing, risk

try:  # pragma: no cover - optional dependency for offline tests
    from slowapi import Limiter as SlowAPILimiter
    from slowapi.errors import RateLimitExceeded as SlowAPIRateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address
    HAS_SLOWAPI = True
except ModuleNotFoundError:  # pragma: no cover - fallback when slowapi is unavailable
    HAS_SLOWAPI = False

    @dataclass(frozen=True)
    class _RateLimit:
        amount: int
        period: float

    class _FallbackRateLimitExceeded(Exception):
        def __init__(self, detail: Mapping[str, Any] | None = None, status_code: int = 429) -> None:
            super().__init__("Rate limit exceeded")
            self.status_code = status_code
            self.detail = detail or {"detail": "Rate limit exceeded"}

    class _FallbackLimiter:  # type: ignore[override]
        def __init__(self, key_func: Callable[[Request], str], default_limits: Iterable[str] | None = None, **_: object) -> None:
            self._key_func = key_func
            self._default_limits = [self._parse(limit) for limit in (default_limits or [])]
            self._overrides: dict[Callable[..., Any], _RateLimit | None] = {}
            self._storage: dict[tuple[str, str], tuple[float, int]] = {}

        def _parse(self, value: str) -> _RateLimit:
            amount_str, _, unit = value.partition("/")
            amount = int(amount_str)
            unit = unit or "minute"
            period = 60.0 if unit.startswith("min") else 1.0
            if unit.startswith("hour"):
                period = 3600.0
            elif unit.startswith("day"):
                period = 86400.0
            return _RateLimit(amount=amount, period=period)

        def limit(self, limit: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            parsed = self._parse(limit)

            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self._overrides[func] = parsed
                return func

            return decorator

        def exempt(self, func: Callable[..., Any]) -> Callable[..., Any]:
            self._overrides[func] = None
            return func

        def get_limit(self, endpoint: Callable[..., Any] | None) -> _RateLimit | None:
            if endpoint in self._overrides:
                return self._overrides[endpoint]
            return self._default_limits[0] if self._default_limits else None

        def hit(self, request: Request, limit: _RateLimit | None) -> bool:
            if limit is None:
                return True
            key = (self._key_func(request), request.url.path)
            now = time.time()
            window_start, count = self._storage.get(key, (now, 0))
            if now - window_start >= limit.period:
                window_start, count = now, 0
            if count >= limit.amount:
                return False
            self._storage[key] = (window_start, count + 1)
            return True

    class _FallbackRateLimitMiddleware(BaseHTTPMiddleware):
        def __init__(self, app: ASGIApp, *, limiter: _FallbackLimiter) -> None:
            super().__init__(app)
            self._limiter = limiter

        async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
            limit = self._limiter.get_limit(request.scope.get("endpoint"))
            if limit is not None and not self._limiter.hit(request, limit):
                route = getattr(request.scope.get("route"), "path", request.url.path)
                RATE_LIMIT_REJECTIONS.labels(route=route).inc()
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"},
                )
                response.headers.setdefault("X-Request-ID", ensure_request_id(request))
                return response
            return await call_next(request)

    def get_remote_address(request: Request) -> str:
        client = request.client
        return client.host if client else "127.0.0.1"

    SlowAPIMiddleware = None  # type: ignore[assignment]
    RateLimitMiddleware: type[BaseHTTPMiddleware] | None = _FallbackRateLimitMiddleware
    SlowAPILimiter = _FallbackLimiter
    SlowAPIRateLimitExceeded = _FallbackRateLimitExceeded
else:
    RateLimitMiddleware = None

Limiter = SlowAPILimiter
RateLimitExceeded = SlowAPIRateLimitExceeded

LOGGER = logging.getLogger(__name__)
START_TIME = time.time()


def _create_rate_limit_handler(
    app: FastAPI,
) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        route = getattr(request.scope.get("route"), "path", request.url.path)
        RATE_LIMIT_REJECTIONS.labels(route=route).inc()
        request_id = ensure_request_id(request)
        response = JSONResponse(
            status_code=getattr(exc, "status_code", 429),
            content={"detail": "Rate limit exceeded"},
        )
        response.headers.setdefault("X-Request-ID", request_id)
        return response

    return handler


def create_app() -> FastAPI:
    settings = get_settings()
    limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit_default])

    app = FastAPI(
        title="UCTG/1 Options Pricing Engine",
        version="1.0.1",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    app.state.settings = settings
    app.state.limiter = limiter

    app.add_exception_handler(RateLimitExceeded, _create_rate_limit_handler(app))
    if HAS_SLOWAPI and SlowAPIMiddleware is not None:
        app.add_middleware(SlowAPIMiddleware)
    elif RateLimitMiddleware is not None:
        app.add_middleware(cast(Any, RateLimitMiddleware), limiter=limiter)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.max_body_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "Accept-Language", "X-Request-ID"],
    )

    app.include_router(pricing.router, prefix="/api/v1")
    app.include_router(risk.router, prefix="/api/v1")
    app.include_router(market_data.router, prefix="/api/v1")

    @app.middleware("http")
    async def _request_metrics(request: Request, call_next):
        request_id = ensure_request_id(request)
        method = request.method
        route_template = request.url.path
        recorder = track_request_duration(request)
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start
            route = getattr(request.scope.get("route"), "path", route_template)
            REQUEST_LATENCY.labels(method=method, route=route).observe(duration)
            REQUEST_ERRORS.labels(method=method, route=route, status_code="500").inc()
            REQUEST_COUNT.labels(method=method, route=route, status_code="500").inc()
            recorder(500)
            raise

        duration = time.perf_counter() - start
        route = getattr(request.scope.get("route"), "path", route_template)
        status_code = str(response.status_code)
        REQUEST_LATENCY.labels(method=method, route=route).observe(duration)
        REQUEST_COUNT.labels(method=method, route=route, status_code=status_code).inc()
        if response.status_code >= 500:
            REQUEST_ERRORS.labels(method=method, route=route, status_code=status_code).inc()
        response.headers.setdefault("X-Request-ID", request_id)
        recorder(response.status_code)
        return response

    @app.get("/metrics", tags=["monitoring"], include_in_schema=False)
    @limiter.exempt
    async def metrics() -> Response:
        """Expose Prometheus metrics for scraping."""

        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    def _health_payload(cpu: float | None, memory: float | None) -> dict[str, object]:
        uptime = max(0.0, time.time() - START_TIME)
        return {
            "status": "ok",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": app.version,
            "environment": settings.environment,
            "uptime_seconds": round(uptime, 3),
            "system": {
                "cpu_percent": cpu,
                "memory_percent": memory,
            },
        }

    @app.get("/healthz", tags=["monitoring"])
    @limiter.exempt
    async def healthz() -> dict[str, object]:
        """Expose the readiness of the service."""

        if psutil is None:
            return _health_payload(cpu=None, memory=None)

        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
        except (psutil.Error, PermissionError):  # pragma: no cover - defensive guard
            return _health_payload(cpu=None, memory=None)

        return _health_payload(cpu=cpu_usage, memory=memory_usage)

    @app.get("/health", tags=["monitoring"], include_in_schema=False)
    @limiter.exempt
    async def health() -> dict[str, object]:
        return await healthz()

    @app.exception_handler(Exception)
    async def global_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled exception: %s", exc)
        response = JSONResponse(status_code=500, content={"error": "Internal server error"})
        response.headers.setdefault("X-Request-ID", ensure_request_id(request))
        return response

    return app


app = create_app()
