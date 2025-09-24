from __future__ import annotations

import pytest

from options_engine.core.volatility_surface import VolatilitySurface


def test_surface_returns_observed_vol_with_sparse_data() -> None:
    surface = VolatilitySurface()
    surface.update_volatility(strike=100.0, maturity=1.0, volatility=0.35)

    estimated = surface.get_volatility(strike=100.0, maturity=1.0, spot=95.0)

    assert estimated == pytest.approx(0.35, rel=1e-6)


def test_surface_cache_includes_spot_dimension() -> None:
    surface = VolatilitySurface(cache_ttl=60.0)
    surface.update_volatility(strike=80.0, maturity=0.5, volatility=0.25)
    surface.update_volatility(strike=120.0, maturity=0.5, volatility=0.35)
    surface.update_volatility(strike=150.0, maturity=2.0, volatility=0.5)

    low_spot = surface.get_volatility(strike=100.0, maturity=1.0, spot=50.0)
    high_spot = surface.get_volatility(strike=100.0, maturity=1.0, spot=200.0)

    assert abs(low_spot - high_spot) > 1e-6
    assert len(surface._cache) == 2  # type: ignore[attr-defined]
