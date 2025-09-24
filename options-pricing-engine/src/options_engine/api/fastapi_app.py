"""FastAPI application exposing the pricing engine."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .routes import market_data, pricing, risk

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="UCTG/1 Options Pricing Engine",
    version="1.0.1",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(pricing.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")
app.include_router(market_data.router, prefix="/api/v1")


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
