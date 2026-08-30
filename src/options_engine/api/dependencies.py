"""Shared dependencies for FastAPI routes."""

from __future__ import annotations

import atexit
from functools import lru_cache

from ..core.pricing_engine import OptionsEngine
from ..core.volatility_surface import VolatilitySurface
from .config import get_settings


@lru_cache(maxsize=1)
def get_vol_surface() -> VolatilitySurface:
    """Return the global volatility surface instance."""

    return VolatilitySurface()


@lru_cache(maxsize=1)
def get_engine() -> OptionsEngine:
    """Return a shared options engine instance."""

    surface = get_vol_surface()
    settings = get_settings()
    engine = OptionsEngine(
        num_threads=settings.threadpool_workers,
        queue_size=settings.threadpool_queue_size,
        queue_timeout_seconds=settings.threadpool_queue_timeout_seconds,
        task_timeout_seconds=settings.threadpool_task_timeout_seconds,
        volatility_surface=surface,
        monte_carlo_seed=settings.monte_carlo_seed,
    )
    atexit.register(engine.shutdown, wait=False)
    return engine
