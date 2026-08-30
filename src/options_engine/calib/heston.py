"""Heston stochastic-volatility pricing and calibration.

The implementation uses the original semi-closed-form characteristic function,
Gauss-Laguerre quadrature, and Black implied-volatility inversion. It is a real
Heston model; no polynomial smile proxy is presented under the Heston name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr

from .boards import CleanBoard

_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.laguerre.laggauss(64)
_QUADRATURE_FACTORS = _QUADRATURE_WEIGHTS * np.exp(_QUADRATURE_NODES)


@dataclass(frozen=True, slots=True)
class HestonConfig:
    """Controls deterministic multi-start Heston calibration."""

    seeds: tuple[int, ...] = (0, 1, 2)
    tolerance: float = 1e-8
    max_iterations: int = 300
    tenors: Sequence[float] | None = None
    min_strikes: int = 7

    def __post_init__(self) -> None:
        if not self.seeds or len(self.seeds) > 32:
            raise ValueError("seeds must contain between 1 and 32 entries")
        if any(
            isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**128 - 1
            for seed in self.seeds
        ):
            raise ValueError("calibration seeds must be integers within [0, 2**128 - 1]")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("calibration seeds must not contain duplicates")
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, Real)
            or not math.isfinite(self.tolerance)
        ):
            raise TypeError("tolerance must be a finite real number")
        if not 1e-14 <= self.tolerance <= 1e-2:
            raise ValueError("tolerance must be within [1e-14, 1e-2]")
        object.__setattr__(self, "tolerance", float(self.tolerance))
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, Integral)
            or not 1 <= self.max_iterations <= 100_000
        ):
            raise ValueError("max_iterations must be an integer within [1, 100000]")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        if (
            isinstance(self.min_strikes, bool)
            or not isinstance(self.min_strikes, Integral)
            or not 7 <= self.min_strikes <= 20_000
        ):
            raise ValueError("min_strikes must be an integer within [7, 20000]")
        object.__setattr__(self, "min_strikes", int(self.min_strikes))
        if self.tenors is not None:
            if not 1 <= len(self.tenors) <= 512:
                raise ValueError("tenors must contain between 1 and 512 entries")
            if any(isinstance(tenor, bool) or not isinstance(tenor, Real) for tenor in self.tenors):
                raise TypeError("tenors must contain real numbers")
            normalized = tuple(float(tenor) for tenor in self.tenors)
            if any(not math.isfinite(tenor) or not 0.0 < tenor <= 100.0 for tenor in normalized):
                raise ValueError("tenors must be finite and within (0, 100]")
            if len(set(normalized)) != len(normalized):
                raise ValueError("tenors must not contain duplicates")
            object.__setattr__(self, "tenors", tuple(sorted(normalized)))


@dataclass(slots=True)
class HestonTenorResult:
    tenor: float
    params: dict[str, float]
    rmse: float
    strikes: NDArray[np.float64]
    market_vols: NDArray[np.float64]
    model_vols: NDArray[np.float64]
    parameter_count: int = 5


def _heston_characteristic_function(
    u: NDArray[np.complex128],
    *,
    forward: float,
    tenor: float,
    v0: float,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho: float,
) -> NDArray[np.complex128]:
    """Characteristic function of log spot under zero carry."""

    imaginary = 1j
    b = kappa - rho * vol_of_vol * imaginary * u
    discriminant = b * b + vol_of_vol**2 * (u * u + imaginary * u)
    d = np.sqrt(discriminant)
    d = np.where(np.real(d) < 0.0, -d, d)
    denominator = b + d
    denominator = np.where(np.abs(denominator) < 1e-14, 1e-14 + 0j, denominator)
    g = (b - d) / denominator
    exp_minus_dt = np.exp(-d * tenor)
    ratio = (1.0 - g * exp_minus_dt) / (1.0 - g)

    c = (kappa * theta / vol_of_vol**2) * ((b - d) * tenor - 2.0 * np.log(ratio))
    d_term = ((b - d) / vol_of_vol**2) * ((1.0 - exp_minus_dt) / (1.0 - g * exp_minus_dt))
    return np.asarray(
        np.exp(imaginary * u * math.log(forward) + c + d_term * v0),
        dtype=np.complex128,
    )


def heston_call_prices(
    forward: float,
    strikes: ArrayLike,
    tenor: float,
    *,
    v0: float,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho: float,
) -> NDArray[np.float64]:
    """Return undiscounted European call prices under the Heston model."""

    strikes_input = np.asarray(strikes)
    if strikes_input.dtype.kind not in "iuf" or strikes_input.dtype.kind == "b":
        raise TypeError("strikes must contain real numbers")
    strikes_array = np.atleast_1d(np.asarray(strikes_input, dtype=float))
    if strikes_array.ndim != 1:
        raise ValueError("strikes must be a one-dimensional array")
    if strikes_array.size == 0:
        raise ValueError("strikes must not be empty")
    if strikes_array.size > 20_000:
        raise ValueError("strikes exceeds the 20000-value limit")
    scalars = {
        "forward": forward,
        "tenor": tenor,
        "v0": v0,
        "theta": theta,
        "kappa": kappa,
        "vol_of_vol": vol_of_vol,
        "rho": rho,
    }
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in scalars.values()):
        raise TypeError("Heston scalar parameters must be real numbers")
    if not all(math.isfinite(float(value)) for value in scalars.values()):
        raise ValueError("Heston scalar parameters must be finite")
    if (
        not 0.0 < forward <= 1e12
        or not 0.0 < tenor <= 100.0
        or not np.isfinite(strikes_array).all()
        or np.any(strikes_array <= 0.0)
        or np.any(strikes_array > 1e12)
    ):
        raise ValueError("forward, strikes, and tenor are outside the supported domain")
    if (
        not 0.0 < v0 <= 25.0
        or not 0.0 < theta <= 25.0
        or not 0.0 < kappa <= 100.0
        or not 0.0 < vol_of_vol <= 20.0
        or not -1.0 < rho < 1.0
    ):
        raise ValueError("invalid Heston parameters")

    u = _QUADRATURE_NODES.astype(np.complex128)
    common = {
        "forward": forward,
        "tenor": tenor,
        "v0": v0,
        "theta": theta,
        "kappa": kappa,
        "vol_of_vol": vol_of_vol,
        "rho": rho,
    }
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        phi_u = _heston_characteristic_function(u, **common)
        phi_u_minus_i = _heston_characteristic_function(u - 1j, **common)
        phi_minus_i = _heston_characteristic_function(
            np.array([-1j], dtype=np.complex128), **common
        )[0]
    if (
        not np.isfinite(phi_u).all()
        or not np.isfinite(phi_u_minus_i).all()
        or not np.isfinite(phi_minus_i)
        or abs(phi_minus_i) < 1e-14
    ):
        raise ValueError("Heston characteristic function is numerically unstable")

    oscillation = np.exp(-1j * np.outer(np.log(strikes_array), u))
    denominator = 1j * u
    integrand_p2 = np.real(oscillation * (phi_u / denominator))
    integrand_p1 = np.real(oscillation * (phi_u_minus_i / (denominator * phi_minus_i)))
    p1 = 0.5 + (integrand_p1 @ _QUADRATURE_FACTORS) / math.pi
    p2 = 0.5 + (integrand_p2 @ _QUADRATURE_FACTORS) / math.pi
    raw_prices = forward * p1 - strikes_array * p2
    if not np.isfinite(raw_prices).all():
        raise ValueError("Heston parameters produced non-finite option prices")
    intrinsic = np.maximum(forward - strikes_array, 0.0)
    price_tolerance = 1e-7 * max(1.0, forward)
    if np.any(raw_prices < intrinsic - price_tolerance) or np.any(
        raw_prices > forward + price_tolerance
    ):
        raise ValueError("Heston quadrature violated model-independent price bounds")

    order = np.argsort(strikes_array)
    sorted_strikes = strikes_array[order]
    sorted_prices = raw_prices[order]
    unique_strikes, unique_indices = np.unique(sorted_strikes, return_index=True)
    unique_prices = sorted_prices[unique_indices]
    if unique_strikes.size >= 2:
        slopes = np.diff(unique_prices) / np.diff(unique_strikes)
        slope_tolerance = 1e-7
        if np.any(slopes < -1.0 - slope_tolerance) or np.any(slopes > slope_tolerance):
            raise ValueError("Heston quadrature violated call-spread bounds")
        if unique_strikes.size >= 3 and np.any(np.diff(slopes) < -slope_tolerance):
            raise ValueError("Heston quadrature violated call-price convexity")

    return np.asarray(np.clip(raw_prices, intrinsic, forward), dtype=np.float64)


def _black_call_price(forward: float, strike: float, tenor: float, sigma: float) -> float:
    vol_sqrt_t = sigma * math.sqrt(tenor)
    if vol_sqrt_t <= 1e-14:
        return max(forward - strike, 0.0)
    d1 = (math.log(forward / strike) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(forward * ndtr(d1) - strike * ndtr(d2))


def _implied_volatility(forward: float, strike: float, tenor: float, price: float) -> float:
    intrinsic = max(forward - strike, 0.0)
    if price <= intrinsic + 1e-12:
        return 1e-6
    if price >= forward - 1e-12:
        return 5.0

    def objective(sigma: float) -> float:
        return _black_call_price(forward, strike, tenor, sigma) - price

    try:
        return float(brentq(objective, 1e-6, 5.0, xtol=1e-10, rtol=1e-10))
    except ValueError:
        return float("nan")


def heston_implied_volatilities(
    forward: float,
    strikes: ArrayLike,
    tenor: float,
    *,
    v0: float,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho: float,
) -> NDArray[np.float64]:
    """Return Black implied volatilities generated by a Heston parameter set."""

    strikes_array = np.atleast_1d(np.asarray(strikes, dtype=float))
    prices = heston_call_prices(
        forward,
        strikes_array,
        tenor,
        v0=v0,
        theta=theta,
        kappa=kappa,
        vol_of_vol=vol_of_vol,
        rho=rho,
    )
    return np.array(
        [
            _implied_volatility(forward, float(strike), tenor, float(price))
            for strike, price in zip(strikes_array, prices, strict=True)
        ],
        dtype=float,
    )


class HestonCalibrator:
    """Calibrate the full five-parameter Heston smile per tenor."""

    def __init__(self, config: HestonConfig | None = None) -> None:
        if config is not None and not isinstance(config, HestonConfig):
            raise TypeError("config must be a HestonConfig")
        self._config = config if config is not None else HestonConfig()

    def calibrate(
        self,
        clean_board: CleanBoard,
        *,
        forward_curve: Mapping[float, float] | None = None,
    ) -> list[HestonTenorResult]:
        if not isinstance(clean_board, CleanBoard):
            raise TypeError("clean_board must be a CleanBoard")
        data = clean_board.quotes
        if data.empty:
            return []
        if len(data) > 100_000:
            raise ValueError("clean board exceeds the 100000-row calibration limit")
        if forward_curve is not None and not isinstance(forward_curve, Mapping):
            raise TypeError("forward_curve must be a mapping or None")

        required = {"tenor", "strike", "mid_iv", "forward"}
        missing = required - set(data.columns)
        if missing:
            raise KeyError(f"clean board is missing required columns: {sorted(missing)}")
        numeric = data[list(required)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("clean board contains non-finite calibration values")
        if (
            (data["tenor"] <= 0.0).any()
            or (data["tenor"] > 100.0).any()
            or (data["strike"] <= 0.0).any()
            or (data["strike"] > 1e12).any()
            or (data["forward"] <= 0.0).any()
            or (data["forward"] > 1e12).any()
            or (~data["mid_iv"].between(1e-4, 5.0, inclusive="both")).any()
        ):
            raise ValueError("clean board contains values outside the Heston calibration domain")

        results: list[HestonTenorResult] = []
        allowed_tenors = self._config.tenors
        for tenor, group in data.groupby("tenor", sort=True):
            tenor_value = float(tenor)
            if allowed_tenors is not None and not any(
                math.isclose(allowed, tenor_value, rel_tol=0.0, abs_tol=1e-9)
                for allowed in allowed_tenors
            ):
                continue
            collapsed = group.groupby("strike", as_index=False)["mid_iv"].mean()
            if len(collapsed) > 20_000:
                raise ValueError("Heston tenor exceeds the 20000-strike limit")
            if len(collapsed) < self._config.min_strikes:
                continue
            forward = self._resolve_forward(tenor_value, group, forward_curve)
            results.append(self._calibrate_single_tenor(forward, tenor_value, collapsed))
        return results

    @staticmethod
    def _resolve_forward(
        tenor: float,
        group: pd.DataFrame,
        forward_curve: Mapping[float, float] | None,
    ) -> float:
        forward: float
        if forward_curve is not None:
            matches = [
                float(value)
                for key, value in forward_curve.items()
                if math.isclose(float(key), tenor, rel_tol=0.0, abs_tol=1e-9)
            ]
            if not matches:
                raise KeyError(f"tenor {tenor} not found in forward_curve")
            if len(matches) > 1:
                raise ValueError(f"forward_curve contains ambiguous keys for tenor {tenor}")
            forward = matches[0]
        else:
            forwards = group["forward"].to_numpy(dtype=float)
            if not np.allclose(forwards, forwards[0], rtol=1e-10, atol=1e-12):
                raise ValueError(f"inconsistent forwards for tenor {tenor}")
            forward = float(forwards.mean())
        if not math.isfinite(forward) or not 0.0 < forward <= 1e12:
            raise ValueError(f"forward for tenor {tenor} must be finite and within (0, 1e12]")
        return forward

    def _calibrate_single_tenor(
        self,
        forward: float,
        tenor: float,
        group: pd.DataFrame,
    ) -> HestonTenorResult:
        strikes = group["strike"].to_numpy(dtype=float)
        market_vols = group["mid_iv"].to_numpy(dtype=float)
        atm_vol = float(market_vols[np.argmin(np.abs(strikes - forward))])
        variance = max(atm_vol**2, 1e-4)
        initial = np.array(
            [math.log(variance), math.log(variance), math.log(1.5), math.log(0.5), -0.3],
            dtype=float,
        )
        lower = np.array([math.log(1e-4), math.log(1e-4), math.log(1e-2), math.log(1e-2), -3.8])
        upper = np.array([math.log(4.0), math.log(4.0), math.log(20.0), math.log(5.0), 3.8])

        def objective(raw: NDArray[np.float64]) -> NDArray[np.float64]:
            params = self._unpack(raw)
            try:
                model_vols = self._model_vols(forward, strikes, tenor, params)
            except ValueError:
                return np.full_like(market_vols, 10.0)
            residuals = model_vols - market_vols
            return np.where(np.isfinite(residuals), residuals, 10.0)

        best_raw: NDArray[np.float64] | None = None
        best_rmse = float("inf")
        for seed in self._config.seeds or (0,):
            start = initial.copy()
            if seed:
                rng = np.random.default_rng(seed)
                start += rng.normal(0.0, 0.15, size=start.size)
            start = np.clip(start, lower + 1e-8, upper - 1e-8)
            fit = least_squares(
                objective,
                start,
                bounds=(lower, upper),
                xtol=self._config.tolerance,
                ftol=self._config.tolerance,
                gtol=self._config.tolerance,
                max_nfev=self._config.max_iterations,
            )
            if not fit.success or not np.all(np.isfinite(fit.fun)):
                continue
            rmse = float(np.sqrt(np.mean(fit.fun**2)))
            if not math.isfinite(rmse) or rmse >= 5.0:
                continue
            if rmse < best_rmse:
                best_rmse = rmse
                best_raw = np.asarray(fit.x, dtype=float)

        if best_raw is None:
            raise RuntimeError(f"Heston calibration failed for tenor {tenor}")
        params = self._unpack(best_raw)
        model_vols = self._model_vols(forward, strikes, tenor, params)
        if not np.isfinite(model_vols).all():
            raise RuntimeError(f"Heston calibration produced invalid values for tenor {tenor}")
        return HestonTenorResult(
            tenor=tenor,
            params=params,
            rmse=best_rmse,
            strikes=strikes.copy(),
            market_vols=market_vols.copy(),
            model_vols=model_vols.copy(),
        )

    @staticmethod
    def _unpack(raw: ArrayLike) -> dict[str, float]:
        values = np.asarray(raw, dtype=float)
        return {
            "v0": float(np.exp(values[0])),
            "theta": float(np.exp(values[1])),
            "kappa": float(np.exp(values[2])),
            "vol_of_vol": float(np.exp(values[3])),
            "rho": float(np.tanh(values[4])),
        }

    @staticmethod
    def _model_vols(
        forward: float,
        strikes: NDArray[np.float64],
        tenor: float,
        params: Mapping[str, float],
    ) -> NDArray[np.float64]:
        return heston_implied_volatilities(
            forward,
            strikes,
            tenor,
            v0=params["v0"],
            theta=params["theta"],
            kappa=params["kappa"],
            vol_of_vol=params["vol_of_vol"],
            rho=params["rho"],
        )


# Compatibility alias for callers of versions <=1.0.1.
HestonQECalibrator = HestonCalibrator


__all__ = [
    "HestonCalibrator",
    "HestonConfig",
    "HestonQECalibrator",
    "HestonTenorResult",
    "heston_call_prices",
    "heston_implied_volatilities",
]
