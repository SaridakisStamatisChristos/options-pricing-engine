"""Volatility surface utilities."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class VolatilityPoint:
    """A single observation on the implied volatility surface."""

    strike: float
    maturity: float
    volatility: float
    timestamp: float
    source: str


class VolatilitySurface:
    """Maintains an interpolated implied volatility surface."""

    def __init__(self, interpolation_method: str = "linear", cache_ttl: float = 60.0) -> None:
        self.interpolation_method = interpolation_method
        self._points: List[VolatilityPoint] = []
        self._interpolator: Optional[Callable[[Sequence[Sequence[float]]], np.ndarray]] = None
        self._cache: Dict[Tuple[float, float], Tuple[float, float]] = {}
        self._cache_ttl = max(0.0, cache_ttl)
        self._lock = threading.RLock()

    @property
    def points(self) -> Tuple[VolatilityPoint, ...]:
        with self._lock:
            return tuple(self._points)

    def update_volatility(
        self,
        strike: float,
        maturity: float,
        volatility: float,
        *,
        source: str = "market",
    ) -> None:
        if strike <= 0 or maturity <= 0:
            raise ValueError("strike and maturity must be positive")
        if not 0.01 <= volatility <= 5.0:
            raise ValueError("volatility must be within [0.01, 5.0]")

        timestamp = time.time()
        with self._lock:
            for index, point in enumerate(self._points):
                if math.isclose(point.strike, strike, abs_tol=1e-9) and math.isclose(
                    point.maturity, maturity, abs_tol=1e-9
                ):
                    self._points[index] = VolatilityPoint(
                        strike, maturity, volatility, timestamp, source
                    )
                    break
            else:
                self._points.append(
                    VolatilityPoint(strike, maturity, volatility, timestamp, source)
                )

            self._cache.clear()
            self._interpolator = self._build_interpolator()

    def _build_interpolator(self) -> Optional[Callable[[Sequence[Sequence[float]]], np.ndarray]]:
        if len(self._points) < 4:
            return None

        strikes = sorted({point.strike for point in self._points})
        maturities = sorted({point.maturity for point in self._points})
        grid = np.full((len(strikes), len(maturities)), np.nan, dtype=float)
        point_map = {(point.strike, point.maturity): point.volatility for point in self._points}

        for strike_index, strike in enumerate(strikes):
            for maturity_index, maturity in enumerate(maturities):
                grid[strike_index, maturity_index] = point_map.get((strike, maturity), np.nan)

        if np.isnan(grid).any():
            points = np.array(
                [[point.strike, point.maturity] for point in self._points], dtype=float
            )
            values = np.array([point.volatility for point in self._points], dtype=float)

            def interpolator(query: Sequence[Sequence[float]]) -> np.ndarray:
                query_array = np.asarray(query, dtype=float)
                result = griddata(
                    points, values, query_array, method=self.interpolation_method, fill_value=np.nan
                )
                if np.isnan(result).any():
                    nearest = griddata(points, values, query_array, method="nearest")
                    result = np.where(np.isnan(result), nearest, result)
                return result

            return interpolator

        return RegularGridInterpolator(
            (np.array(strikes, dtype=float), np.array(maturities, dtype=float)),
            grid,
            method=self.interpolation_method,
            bounds_error=False,
            fill_value=None,
        )

    def get_volatility(self, strike: float, maturity: float, spot: float) -> float:
        key = (round(strike, 4), round(maturity, 4))
        now = time.time()

        with self._lock:
            cached = self._cache.get(key)
            if cached and (not self._cache_ttl or now - cached[1] <= self._cache_ttl):
                return cached[0]

            interpolator = self._interpolator
            point_count = len(self._points)

        if interpolator is None or point_count < 4:
            volatility = 0.20
        else:
            try:
                interpolated = interpolator([[strike, maturity]])  # type: ignore[misc]
                volatility = float(interpolated[0])
            except Exception as exc:  # pragma: no cover - defensive programming
                LOGGER.warning("Volatility interpolation failed: %s", exc)
                volatility = self._fallback(strike, maturity, spot)
            else:
                if not 0.01 <= volatility <= 3.0 or math.isnan(volatility):
                    volatility = self._fallback(strike, maturity, spot)

        with self._lock:
            self._cache[key] = (volatility, now)
        return volatility

    def _fallback(self, strike: float, maturity: float, spot: float) -> float:
        with self._lock:
            if not self._points:
                return 0.20

            def distance(point: VolatilityPoint) -> float:
                strike_scale = max(strike, spot, 1e-6)
                maturity_scale = max(maturity, 1e-6)
                return ((point.strike - strike) / strike_scale) ** 2 + (
                    (point.maturity - maturity) / maturity_scale
                ) ** 2

            sorted_points = sorted(self._points, key=distance)
            nearest = sorted_points[: min(len(sorted_points), 5)]

        weights = np.array([1.0 / (distance(point) + 1e-6) for point in nearest], dtype=float)
        vols = np.array([point.volatility for point in nearest], dtype=float)
        return float(np.average(vols, weights=weights))
