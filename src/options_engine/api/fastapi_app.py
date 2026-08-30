"""FastAPI application exposing the pricing engine."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import psutil
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .. import __version__
from ..observability.metrics import (
    REQUEST_COUNT,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
)
from .config import get_settings
from .middleware import (
    BodySizeLimitMiddleware,
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    ensure_request_id,
    track_request_duration,
)
from .routes import market_data, pricing, register_routes, risk

LOGGER = logging.getLogger(__name__)
START_TIME = time.time()


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) and path else "__unmatched__"


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Options Pricing Engine",
        version=__version__,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
    )

    app.state.settings = settings

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        rate_limit=settings.rate_limit_default,
        excluded_paths={
            "/health",
            "/healthz",
            "/metrics",
            "/docs",
            "/docs/oauth2-redirect",
            "/openapi.json",
            "/redoc",
        },
    )
    app.add_middleware(BodySizeLimitMiddleware, max_body_bytes=settings.max_body_bytes)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Accept",
            "Accept-Language",
            "X-Request-ID",
        ],
    )

    app.include_router(pricing.router, prefix="/api/v1")
    app.include_router(risk.router, prefix="/api/v1")
    app.include_router(market_data.router, prefix="/api/v1")

    @app.middleware("http")
    async def _request_metrics(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = ensure_request_id(request)
        method = request.method
        recorder = track_request_duration(request)
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - start
            route = _route_label(request)
            REQUEST_LATENCY.labels(method=method, route=route).observe(duration)
            REQUEST_ERRORS.labels(method=method, route=route, status_code="500").inc()
            REQUEST_COUNT.labels(method=method, route=route, status_code="500").inc()
            recorder(500)
            raise

        duration = time.perf_counter() - start
        route = _route_label(request)
        status_code = str(response.status_code)
        REQUEST_LATENCY.labels(method=method, route=route).observe(duration)
        REQUEST_COUNT.labels(method=method, route=route, status_code=status_code).inc()
        if response.status_code >= 500:
            REQUEST_ERRORS.labels(method=method, route=route, status_code=status_code).inc()
        response.headers.setdefault("X-Request-ID", request_id)
        recorder(response.status_code)
        return response

    @app.get("/metrics", tags=["monitoring"], include_in_schema=False)
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
    async def healthz() -> dict[str, object]:
        """Expose the readiness of the service."""

        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
        except (psutil.Error, PermissionError):  # pragma: no cover - defensive guard
            return _health_payload(cpu=None, memory=None)

        return _health_payload(cpu=cpu_usage, memory=memory_usage)

    @app.get("/health", tags=["monitoring"], include_in_schema=False)
    async def health() -> dict[str, object]:
        return await healthz()

    @app.exception_handler(Exception)
    async def global_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled exception: %s", exc)
        response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        response.headers.setdefault("X-Request-ID", ensure_request_id(request))
        return response

    # Stable compatibility endpoints (/quote, /batch, /greeks, /replay) share
    # this app, middleware, authentication, and lifecycle with /api/v1.
    register_routes(app)

    return app


app = create_app()
