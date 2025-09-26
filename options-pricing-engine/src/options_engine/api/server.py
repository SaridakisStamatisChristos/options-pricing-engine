"""Minimal FastAPI application exposing the core pricing API."""

from __future__ import annotations

import sys
from importlib import util
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .config import get_settings
from .middleware import (
    BodySizeLimitMiddleware,
    RateLimitResponseMiddleware,
    SecurityHeadersMiddleware,
    ensure_request_id,
    track_request_duration,
)


def _load_routes_module():
    module_path = Path(__file__).with_name("routes.py")
    spec = util.spec_from_file_location("options_engine.api._minimal_routes", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError("Failed to load routes module")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_app() -> FastAPI:
    """Instantiate and configure the FastAPI application."""

    routes_module = _load_routes_module()
    register_routes = getattr(routes_module, "register_routes")
    app = FastAPI(title="Options Pricing Engine", version="minimal-core")

    settings = get_settings()
    app.state.settings = settings

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitResponseMiddleware)
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

    register_routes(app)

    @app.middleware("http")
    async def _request_context(request: Request, call_next):  # type: ignore[override]
        ensure_request_id(request)
        recorder = track_request_duration(request)
        try:
            response = await call_next(request)
        except Exception:
            recorder(500)
            raise

        response.headers.setdefault("X-Request-ID", ensure_request_id(request))
        recorder(response.status_code)
        return response

    return app


def build_app() -> FastAPI:
    """Compatibility wrapper mirroring the public API used in tests."""

    return create_app()


app = build_app()

