"""Volatility surface utilities."""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata

LOGGER = logging.getLogger(__name__)


def _bounded_real(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalised = float(value)
    if not math.isfinite(normalised) or not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum:g}, {maximum:g}]")
    return normalised


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalised = int(value)
    if not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return normalised


@dataclass(frozen=True, slots=True)
class VolatilityPoint:
    """A single observation on the implied volatility surface."""

    strike: float
    maturity: float
    volatility: float
    timestamp: float
    source: str


class VolatilitySurface:
    """Maintains an interpolated implied volatility surface."""

    MAX_CACHE_SIZE = 100_000
    MAX_POINTS = 10_000
    MAX_REGULAR_GRID_CELLS = 1_000_000

    def __init__(
        self,
        interpolation_method: str = "linear",
        cache_ttl: float = 60.0,
        cache_size: int = 10_000,
        max_points: int = MAX_POINTS,
    ) -> None:
        if not isinstance(interpolation_method, str):
            raise TypeError("interpolation_method must be a string")
        self.interpolation_method = interpolation_method.strip().lower()
        if self.interpolation_method not in {"linear", "nearest"}:
            raise ValueError("interpolation_method must be 'linear' or 'nearest'")
        self._points: list[VolatilityPoint] = []
        self._interpolator: Callable[[Sequence[Sequence[float]]], np.ndarray] | None = None
        self._cache: OrderedDict[tuple[float, float, float], tuple[float, float]] = OrderedDict()
        self._cache_ttl = _bounded_real(
            "cache_ttl",
            cache_ttl,
            minimum=0.0,
            maximum=86_400.0,
        )
        self._cache_size = _bounded_integer(
            "cache_size",
            cache_size,
            minimum=1,
            maximum=self.MAX_CACHE_SIZE,
        )
        self._max_points = _bounded_integer(
            "max_points",
            max_points,
            minimum=4,
            maximum=self.MAX_POINTS,
        )
        self._dirty = False
        self._generation = 0
        self._lock = threading.RLock()

    @property
    def points(self) -> tuple[VolatilityPoint, ...]:
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
        strike = _bounded_real("strike", strike, minimum=1e-12, maximum=1e12)
        maturity = _bounded_real("maturity", maturity, minimum=1e-12, maximum=100.0)
        volatility = _bounded_real("volatility", volatility, minimum=0.01, maximum=5.0)
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        source = source.strip()
        if (
            not source
            or len(source) > 128
            or any(ord(character) < 32 or ord(character) == 127 for character in source)
        ):
            raise ValueError("source must contain between 1 and 128 characters")

        timestamp = time.time()
        with self._lock:
            for index, point in enumerate(self._points):
                if math.isclose(point.strike, strike, rel_tol=0.0, abs_tol=1e-9) and math.isclose(
                    point.maturity, maturity, rel_tol=0.0, abs_tol=1e-9
                ):
                    self._points[index] = VolatilityPoint(
                        strike, maturity, volatility, timestamp, source
                    )
                    break
            else:
                if len(self._points) >= self._max_points:
                    raise ValueError(
                        f"volatility surface exceeds the {self._max_points}-point limit"
                    )
                self._points.append(
                    VolatilityPoint(strike, maturity, volatility, timestamp, source)
                )

            self._cache.clear()
            self._interpolator = None
            self._dirty = True
            self._generation += 1

    def _build_interpolator(self) -> Callable[[Sequence[Sequence[float]]], np.ndarray] | None:
        if len(self._points) < 4:
            return None

        strikes = sorted({point.strike for point in self._points})
        maturities = sorted({point.maturity for point in self._points})
        grid_cells = len(strikes) * len(maturities)
        if grid_cells != len(self._points) or grid_cells > self.MAX_REGULAR_GRID_CELLS:
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
                return np.asarray(result, dtype=float)

            return interpolator

        point_map = {(point.strike, point.maturity): point.volatility for point in self._points}
        grid = np.empty((len(strikes), len(maturities)), dtype=float)
        for strike_index, strike in enumerate(strikes):
            for maturity_index, maturity in enumerate(maturities):
                grid[strike_index, maturity_index] = point_map[(strike, maturity)]

        regular_grid = RegularGridInterpolator(
            (np.array(strikes, dtype=float), np.array(maturities, dtype=float)),
            grid,
            method=self.interpolation_method,
            bounds_error=False,
            # Extrapolated smiles can become arbitrageable very quickly. A
            # query outside the observed rectangle is handled by the bounded
            # nearest-point fallback instead.
            fill_value=np.nan,
        )

        def regular_interpolator(query: Sequence[Sequence[float]]) -> np.ndarray:
            return np.asarray(regular_grid(query), dtype=float)

        return regular_interpolator

    def get_volatility(self, strike: float, maturity: float, spot: float) -> float:
        strike = _bounded_real("strike", strike, minimum=1e-12, maximum=1e12)
        maturity = _bounded_real("maturity", maturity, minimum=1e-12, maximum=100.0)
        spot = _bounded_real("spot", spot, minimum=1e-12, maximum=1e12)
        key = (strike, maturity, spot)

        # An update can arrive while SciPy evaluates an interpolation outside
        # the lock. Retry once rather than writing a stale value back into a
        # cache that the update just cleared.
        for attempt in range(2):
            now = time.monotonic()
            with self._lock:
                cached = self._cache.get(key)
                if cached and (not self._cache_ttl or now - cached[1] <= self._cache_ttl):
                    self._cache.move_to_end(key)
                    return cached[0]
                if cached:
                    self._cache.pop(key, None)

                if self._dirty:
                    # A manually supplied interpolator remains useful for testing or
                    # specialized callers; otherwise rebuild lazily after update bursts.
                    if self._interpolator is None:
                        self._interpolator = self._build_interpolator()
                    self._dirty = False
                interpolator = self._interpolator
                point_count = len(self._points)
                generation = self._generation

            if interpolator is None or point_count < 4:
                volatility = 0.20
            else:
                try:
                    interpolated = interpolator([[strike, maturity]])
                    volatility = float(interpolated[0])
                except Exception as exc:  # pragma: no cover - defensive programming
                    LOGGER.warning("Volatility interpolation failed: %s", exc)
                    volatility = self._fallback(strike, maturity, spot)
                else:
                    if not 0.01 <= volatility <= 5.0 or not math.isfinite(volatility):
                        volatility = self._fallback(strike, maturity, spot)

            with self._lock:
                if generation != self._generation:
                    if attempt == 0:
                        continue
                    # Under sustained writes, prefer the latest bounded local
                    # fallback and avoid caching an older interpolation.
                    return self._fallback(strike, maturity, spot)
                self._cache[key] = (volatility, time.monotonic())
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
                return volatility

        return self._fallback(strike, maturity, spot)

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
