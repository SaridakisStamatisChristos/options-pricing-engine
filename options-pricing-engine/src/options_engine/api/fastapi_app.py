"""FastAPI application exposing the pricing engine."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..observability.metrics import (
    REQUEST_COUNT,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
)
from .config import get_settings
from .middleware import SecurityHeadersMiddleware
from .routes import market_data, pricing, risk

LOGGER = logging.getLogger(__name__)
SETTINGS = get_settings()

app = FastAPI(
    title="UCTG/1 Options Pricing Engine",
    version="1.0.1",
    docs_url="/docs" if not SETTINGS.is_production else None,
    redoc_url="/redoc" if not SETTINGS.is_production else None,
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(SETTINGS.allowed_hosts))
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.allowed_origins),
    allow_credentials=SETTINGS.cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Accept-Language"],
)
app.include_router(pricing.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")
app.include_router(market_data.router, prefix="/api/v1")


@app.middleware("http")
async def _request_metrics(request: Request, call_next):
    method = request.method
    route_template = request.url.path
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start
        route = getattr(request.scope.get("route"), "path", route_template)
        REQUEST_LATENCY.labels(method=method, route=route).observe(duration)
        REQUEST_ERRORS.labels(method=method, route=route, status_code="500").inc()
        REQUEST_COUNT.labels(method=method, route=route, status_code="500").inc()
        raise

    duration = time.perf_counter() - start
    route = getattr(request.scope.get("route"), "path", route_template)
    status_code = str(response.status_code)
    REQUEST_LATENCY.labels(method=method, route=route).observe(duration)
    REQUEST_COUNT.labels(method=method, route=route, status_code=status_code).inc()
    if response.status_code >= 500:
        REQUEST_ERRORS.labels(method=method, route=route, status_code=status_code).inc()
    return response


@app.get("/metrics", tags=["monitoring"], include_in_schema=False)
async def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", tags=["monitoring"])
async def health() -> dict[str, object]:
    """Expose the readiness of the service."""

    try:
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage("/").percent
        return {
            "status": "healthy",
            "timestamp": datetime.now(UTC).isoformat(),
            "version": app.version,
            "environment": SETTINGS.environment,
            "system": {
                "memory_usage_percent": memory_usage,
                "disk_usage_percent": disk_usage,
            },
        }
    except Exception as exc:  # pragma: no cover - defensive programming
        LOGGER.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Service unhealthy") from exc


@app.exception_handler(Exception)
async def global_error(_: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})
