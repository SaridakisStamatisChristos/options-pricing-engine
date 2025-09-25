"""Minimal FastAPI application exposing the core pricing API."""

from __future__ import annotations

import sys
from importlib import util
from pathlib import Path

from fastapi import FastAPI


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
    register_routes(app)
    return app


app = create_app()

