"""Compatibility entry point for the single canonical FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI

from .fastapi_app import app as app
from .fastapi_app import create_app


def build_app() -> FastAPI:
    """Return the canonical application (kept for <=1.x compatibility)."""

    return create_app()
