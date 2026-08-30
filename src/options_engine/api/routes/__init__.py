"""Canonical API routers and compatibility endpoint registration."""

from __future__ import annotations

from ..legacy_routes import BUILD_ID, register_routes
from . import market_data, pricing, risk

__all__ = ["BUILD_ID", "market_data", "pricing", "register_routes", "risk"]
