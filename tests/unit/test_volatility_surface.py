from __future__ import annotations

import math

import numpy as np
import pytest

from options_engine.core import volatility_surface as surface_module
from options_engine.core.volatility_surface import VolatilitySurface


def test_surface_validates_updates_and_replaces_existing_point() -> None:
    surface = VolatilitySurface()

    with pytest.raises(ValueError, match="strike"):
        surface.update_volatility(0.0, 1.0, 0.2)
    with pytest.raises(ValueError, match="volatility"):
        surface.update_volatility(100.0, 1.0, 8.0)

    surface.update_volatility(100.0, 1.0, 0.2, source="first")
    surface.update_volatility(100.0, 1.0, 0.3, source="replacement")

    assert len(surface.points) == 1
    assert surface.points[0].volatility == 0.3
    assert surface.points[0].source == "replacement"


def test_regular_grid_interpolation_and_cache_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    timeline = {"now": 10.0}
    monkeypatch.setattr(surface_module.time, "time", lambda: timeline["now"])
    surface = VolatilitySurface(cache_ttl=2.0)
    for strike, maturity, volatility in (
        (90.0, 0.5, 0.20),
        (90.0, 1.0, 0.22),
        (110.0, 0.5, 0.24),
        (110.0, 1.0, 0.26),
    ):
        surface.update_volatility(strike, maturity, volatility)

    first = surface.get_volatility(100.0, 0.75, 100.0)
    assert first == pytest.approx(0.23, abs=1e-12)

    # The first repeat uses the cached value. After TTL expiry the interpolator
    # is evaluated again and reproduces the same deterministic value.
    assert surface.get_volatility(100.0, 0.75, 100.0) == first
    timeline["now"] = 13.0
    assert surface.get_volatility(100.0, 0.75, 100.0) == first


def test_scattered_surface_uses_griddata_and_nearest_fill() -> None:
    surface = VolatilitySurface()
    for strike, maturity, volatility in (
        (80.0, 0.5, 0.18),
        (100.0, 0.5, 0.20),
        (90.0, 1.0, 0.23),
        (120.0, 1.5, 0.28),
    ):
        surface.update_volatility(strike, maturity, volatility)

    inside = surface.get_volatility(95.0, 0.75, 100.0)
    outside = surface.get_volatility(150.0, 3.0, 100.0)
    assert math.isfinite(inside)
    assert math.isfinite(outside)
    assert 0.01 <= inside <= 5.0
    assert 0.01 <= outside <= 5.0


def test_interpolation_failure_and_invalid_value_use_nearest_point() -> None:
    surface = VolatilitySurface()
    for strike, maturity, volatility in (
        (90.0, 0.5, 0.18),
        (100.0, 0.5, 0.20),
        (90.0, 1.0, 0.22),
        (100.0, 1.0, 0.24),
    ):
        surface.update_volatility(strike, maturity, volatility)

    surface._interpolator = lambda _: np.array([math.nan])
    assert surface.get_volatility(100.0, 1.0, 100.0) == pytest.approx(0.24, abs=1e-5)

    def fail(_: object) -> np.ndarray:
        raise RuntimeError("interpolator unavailable")

    surface._cache.clear()
    surface._interpolator = fail
    assert surface.get_volatility(90.0, 0.5, 100.0) == pytest.approx(0.18, abs=1e-5)


def test_empty_or_small_surface_uses_documented_default() -> None:
    surface = VolatilitySurface()
    assert surface.get_volatility(100.0, 1.0, 100.0) == 0.20
    surface.update_volatility(100.0, 1.0, 0.35)
    assert surface.get_volatility(110.0, 2.0, 100.0) == 0.20


def test_surface_rejects_invalid_configuration_and_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="interpolation_method"):
        VolatilitySurface(interpolation_method="cubic")
    with pytest.raises(ValueError, match="cache_ttl"):
        VolatilitySurface(cache_ttl=math.nan)
    with pytest.raises(ValueError, match="cache_size"):
        VolatilitySurface(cache_size=0)

    surface = VolatilitySurface()
    with pytest.raises(ValueError, match="volatility"):
        surface.update_volatility(100.0, 1.0, math.nan)
    with pytest.raises(ValueError, match="maturity"):
        surface.get_volatility(100.0, math.inf, 100.0)


def test_surface_cache_is_exact_and_lru_bounded() -> None:
    surface = VolatilitySurface(cache_size=2)

    surface.get_volatility(100.000000001, 1.0, 100.0)
    surface.get_volatility(100.000000002, 1.0, 100.0)
    assert len(surface._cache) == 2

    surface.get_volatility(100.000000003, 1.0, 100.0)
    assert len(surface._cache) == 2
    assert (100.000000001, 1.0, 100.0) not in surface._cache
