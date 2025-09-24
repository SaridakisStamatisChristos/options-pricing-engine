"""Shared dependencies for FastAPI routes."""

from __future__ import annotations

import atexit
from functools import lru_cache

from ..core.pricing_engine import OptionsEngine
from ..core.volatility_surface import VolatilitySurface


@lru_cache(maxsize=1)
def get_vol_surface() -> VolatilitySurface:
    """Return the global volatility surface instance."""

    return VolatilitySurface()


@lru_cache(maxsize=1)
def get_engine() -> OptionsEngine:
    """Return a shared options engine instance."""

    surface = get_vol_surface()
    engine = OptionsEngine(num_threads=8, volatility_surface=surface)
    atexit.register(engine.shutdown, wait=False)
    return engine
