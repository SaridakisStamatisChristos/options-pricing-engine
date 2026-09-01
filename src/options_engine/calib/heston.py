"""Heston stochastic-volatility pricing and calibration.

The reference pricing family in this module uses the original semi-closed-form
characteristic function, Gauss-Laguerre quadrature, and Black implied-volatility
inversion.  The independent Fang-Oosterlee COS family lives in
``options_engine.calib.heston_cos`` and deliberately does not replace this
implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Integral, Real

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import OptimizeResult, brentq, least_squares
from scipy.special import ndtr

from .boards import CleanBoard
from .validation import (
    CalibrationError,
    CalibrationFailureReason,
    ConditioningDiagnostics,
    FitQuality,
    HoldoutPolicy,
    InitializationSensitivity,
    OptimizerAttempt,
    ResidualObservation,
    ResidualSummary,
    WeightDiagnostics,
    analyze_initialization_sensitivity,
    conditioning_from_jacobian,
    deterministic_holdout_mask,
    residual_diagnostics,
    serializable,
)

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
    weighting: str = "auto"
    spread_floor: float = 1e-4
    holdout_fraction: float = 0.2
    holdout_policy: HoldoutPolicy | str = HoldoutPolicy.FRACTIONAL
    feller_penalty: float = 0.0
    calibration_mode: str = "per_tenor"
    pricing_method: str = "gauss_laguerre"
    cos_terms: int = 256
    cos_truncation: float = 12.0
    global_tenor_weighting: str = "equal"
    holdout_tenors: Sequence[float] | None = None

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
        if self.weighting not in {"auto", "uniform", "vega", "bid_ask", "hybrid"}:
            raise ValueError("weighting must be 'auto', 'uniform', 'vega', 'bid_ask', or 'hybrid'")
        try:
            object.__setattr__(self, "holdout_policy", HoldoutPolicy(self.holdout_policy))
        except ValueError as exc:
            raise ValueError(f"unsupported holdout policy: {self.holdout_policy!r}") from exc
        if self.calibration_mode not in {"per_tenor", "global"}:
            raise ValueError("calibration_mode must be 'per_tenor' or 'global'")
        if self.pricing_method not in {"gauss_laguerre", "cos"}:
            raise ValueError("pricing_method must be 'gauss_laguerre' or 'cos'")
        if self.global_tenor_weighting not in {"equal", "observations"}:
            raise ValueError("global_tenor_weighting must be 'equal' or 'observations'")
        if (
            isinstance(self.cos_terms, bool)
            or not isinstance(self.cos_terms, Integral)
            or not 32 <= self.cos_terms <= 4096
        ):
            raise ValueError("cos_terms must be an integer within [32, 4096]")
        object.__setattr__(self, "cos_terms", int(self.cos_terms))
        for name, value, low, high in (
            ("spread_floor", self.spread_floor, 1e-8, 1.0),
            ("holdout_fraction", self.holdout_fraction, 0.0, 0.5),
            ("feller_penalty", self.feller_penalty, 0.0, 1e6),
            ("cos_truncation", self.cos_truncation, 4.0, 40.0),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
            normalised = float(value)
            if not math.isfinite(normalised) or not low <= normalised <= high:
                raise ValueError(f"{name} must be within [{low:g}, {high:g}]")
            object.__setattr__(self, name, normalised)
        for name in ("tenors", "holdout_tenors"):
            values = getattr(self, name)
            if values is None:
                continue
            if not 1 <= len(values) <= 512:
                raise ValueError(f"{name} must contain between 1 and 512 entries")
            if any(isinstance(tenor, bool) or not isinstance(tenor, Real) for tenor in values):
                raise TypeError(f"{name} must contain real numbers")
            normalized = tuple(float(tenor) for tenor in values)
            if any(not math.isfinite(tenor) or not 0.0 < tenor <= 100.0 for tenor in normalized):
                raise ValueError(f"{name} must be finite and within (0, 100]")
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, tuple(sorted(normalized)))
        if self.holdout_tenors is not None and self.calibration_mode != "global":
            raise ValueError("holdout_tenors are supported only in global calibration mode")
        if self.tenors is not None and self.holdout_tenors is not None:
            selected = set(self.tenors)
            if any(tenor not in selected for tenor in self.holdout_tenors):
                raise ValueError("holdout_tenors must be contained in tenors when tenors is set")


@dataclass(frozen=True, slots=True)
class HestonOptimizerDiagnostics:
    """Serializable SciPy optimizer evidence for one calibrated parameter set."""

    success: bool
    status: int
    message: str
    evaluations: int
    jacobian_evaluations: int | None
    cost: float
    optimality: float
    active_mask: tuple[int, ...]
    seeds_attempted: int
    seeds_succeeded: int
    best_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "evaluations": self.evaluations,
            "jacobian_evaluations": self.jacobian_evaluations,
            "cost": self.cost,
            "optimality": self.optimality,
            "active_mask": self.active_mask,
            "seeds_attempted": self.seeds_attempted,
            "seeds_succeeded": self.seeds_succeeded,
            "best_seed": self.best_seed,
        }


@dataclass(slots=True)
class HestonTenorResult:
    tenor: float
    params: dict[str, float]
    rmse: float
    strikes: NDArray[np.float64]
    market_vols: NDArray[np.float64]
    model_vols: NDArray[np.float64]
    parameter_count: int = 5
    weighted_rmse: float = math.nan
    holdout_rmse: float | None = None
    feller_ratio: float = math.nan
    feller_satisfied: bool = False
    weighting: str = "uniform"
    calibration_observations: int = 0
    holdout_observations: int = 0
    parameter_change_l2: float | None = None
    calibration_mode: str = "per_tenor"
    in_sample_weighted_rmse: float = math.nan
    parameter_bound_proximity: dict[str, float] = field(default_factory=dict)
    optimizer_diagnostics: HestonOptimizerDiagnostics | None = None
    is_holdout_tenor: bool = False
    residuals: tuple[ResidualObservation, ...] = ()
    residual_summary: ResidualSummary | None = None
    weight_diagnostics: WeightDiagnostics | None = None
    initialization_sensitivity: InitializationSensitivity | None = None
    conditioning: ConditioningDiagnostics | None = None
    transformed_parameter_bound_proximity: dict[str, float] = field(default_factory=dict)
    economic_parameter_bound_proximity: dict[str, float] = field(default_factory=dict)
    fit_quality: str = FitQuality.INVALID.value
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class HestonCalibrationResult:
    """Detailed calibration result while ``calibrate`` retains its list API."""

    mode: str
    tenor_results: list[HestonTenorResult]
    shared_params: dict[str, float] | None
    in_sample_weighted_rmse: float
    holdout_rmse: float | None
    feller_ratio: float | None
    parameter_bound_proximity: dict[str, float]
    optimizer_diagnostics: tuple[HestonOptimizerDiagnostics, ...]
    calibration_observations: int
    holdout_observations: int
    pricing_method: str
    strike_weighting: str
    tenor_weighting: str | None
    residuals: tuple[ResidualObservation, ...] = ()
    residual_summary: ResidualSummary | None = None
    weight_diagnostics: WeightDiagnostics | None = None
    initialization_sensitivity: tuple[InitializationSensitivity, ...] = ()
    conditioning: tuple[ConditioningDiagnostics, ...] = ()
    strike_holdout_rmse: float | None = None
    tenor_holdout_rmse: float | None = None
    fit_quality: str = FitQuality.INVALID.value
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        result = serializable(
            {
                "mode": self.mode,
                "shared_params": self.shared_params,
                "in_sample_weighted_rmse": self.in_sample_weighted_rmse,
                "holdout_rmse": self.holdout_rmse,
                "feller_ratio": self.feller_ratio,
                "parameter_bound_proximity": self.parameter_bound_proximity,
                "optimizer_diagnostics": [item.to_dict() for item in self.optimizer_diagnostics],
                "calibration_observations": self.calibration_observations,
                "holdout_observations": self.holdout_observations,
                "pricing_method": self.pricing_method,
                "strike_weighting": self.strike_weighting,
                "tenor_weighting": self.tenor_weighting,
                "residuals": self.residuals,
                "residual_summary": self.residual_summary,
                "weight_diagnostics": self.weight_diagnostics,
                "initialization_sensitivity": self.initialization_sensitivity,
                "conditioning": self.conditioning,
                "strike_holdout_rmse": self.strike_holdout_rmse,
                "tenor_holdout_rmse": self.tenor_holdout_rmse,
                "fit_quality": self.fit_quality,
                "warnings": self.warnings,
                "tenors": [
                    {
                        "tenor": result.tenor,
                        "rmse": result.rmse,
                        "in_sample_weighted_rmse": result.in_sample_weighted_rmse,
                        "holdout_rmse": result.holdout_rmse,
                        "feller_ratio": result.feller_ratio,
                        "parameter_bound_proximity": result.parameter_bound_proximity,
                        "is_holdout_tenor": result.is_holdout_tenor,
                        "residual_summary": result.residual_summary,
                        "weight_diagnostics": result.weight_diagnostics,
                        "initialization_sensitivity": result.initialization_sensitivity,
                        "conditioning": result.conditioning,
                        "fit_quality": result.fit_quality,
                        "warnings": result.warnings,
                    }
                    for result in self.tenor_results
                ],
            }
        )
        if not isinstance(result, dict):  # pragma: no cover - structural invariant
            raise TypeError("Heston calibration serialization did not produce a mapping")
        return result


@dataclass(frozen=True, slots=True)
class HestonCalibrationComparison:
    """Side-by-side evidence; deliberately does not declare a winner."""

    per_tenor: HestonCalibrationResult
    global_fit: HestonCalibrationResult

    def to_dict(self) -> dict[str, object]:
        return {
            "per_tenor": self.per_tenor.to_dict(),
            "global": self.global_fit.to_dict(),
            "global_minus_per_tenor": {
                "in_sample_weighted_rmse": (
                    self.global_fit.in_sample_weighted_rmse - self.per_tenor.in_sample_weighted_rmse
                ),
                "holdout_rmse": (
                    None
                    if self.global_fit.holdout_rmse is None or self.per_tenor.holdout_rmse is None
                    else self.global_fit.holdout_rmse - self.per_tenor.holdout_rmse
                ),
            },
        }


@dataclass(slots=True)
class _HestonSlice:
    tenor: float
    forward: float
    group: pd.DataFrame
    weights: NDArray[np.float64]
    weighting: str
    holdout: NDArray[np.bool_]


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


def _validated_heston_inputs(
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
    """Validate the domain shared by all deterministic Heston pricers."""

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
    return np.asarray(strikes_array, dtype=np.float64)


def _validated_heston_call_prices(
    forward: float,
    strikes: NDArray[np.float64],
    raw_prices: ArrayLike,
    *,
    method: str,
) -> NDArray[np.float64]:
    """Enforce model-independent bounds and strike-shape constraints."""

    prices = np.asarray(raw_prices, dtype=float)
    if prices.shape != strikes.shape or not np.isfinite(prices).all():
        raise ValueError(f"Heston {method} produced non-finite option prices")
    intrinsic = np.maximum(forward - strikes, 0.0)
    price_tolerance = 1e-7 * max(1.0, forward)
    if np.any(prices < intrinsic - price_tolerance) or np.any(prices > forward + price_tolerance):
        raise ValueError(f"Heston {method} violated model-independent price bounds")

    order = np.argsort(strikes)
    sorted_strikes = strikes[order]
    sorted_prices = prices[order]
    unique_strikes, unique_indices = np.unique(sorted_strikes, return_index=True)
    unique_prices = sorted_prices[unique_indices]
    if unique_strikes.size >= 2:
        slopes = np.diff(unique_prices) / np.diff(unique_strikes)
        slope_tolerance = 1e-7
        if np.any(slopes < -1.0 - slope_tolerance) or np.any(slopes > slope_tolerance):
            raise ValueError(f"Heston {method} violated call-spread bounds")
        if unique_strikes.size >= 3 and np.any(np.diff(slopes) < -slope_tolerance):
            raise ValueError(f"Heston {method} violated call-price convexity")

    return np.asarray(np.clip(prices, intrinsic, forward), dtype=np.float64)


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
    """Return undiscounted calls using 64-point Gauss-Laguerre inversion."""

    strikes_array = _validated_heston_inputs(
        forward,
        strikes,
        tenor,
        v0=v0,
        theta=theta,
        kappa=kappa,
        vol_of_vol=vol_of_vol,
        rho=rho,
    )

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
    return _validated_heston_call_prices(
        forward,
        strikes_array,
        raw_prices,
        method="Gauss-Laguerre quadrature",
    )


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
    """Calibrate Heston per tenor or as one coherent multi-tenor process."""

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
        """Return tenor results, preserving the pre-global public contract."""

        return self.calibrate_detailed(clean_board, forward_curve=forward_curve).tenor_results

    def calibrate_detailed(
        self,
        clean_board: CleanBoard,
        *,
        forward_curve: Mapping[float, float] | None = None,
    ) -> HestonCalibrationResult:
        """Calibrate with aggregate fit, holdout, bound, and optimizer evidence."""

        if not isinstance(clean_board, CleanBoard):
            raise TypeError("clean_board must be a CleanBoard")
        data = clean_board.quotes
        if data.empty:
            return self._empty_detailed_result()
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

        slices: list[_HestonSlice] = []
        allowed_tenors = self._config.tenors
        for tenor, group in data.groupby("tenor", sort=True):
            tenor_value = float(tenor)
            if allowed_tenors is not None and not any(
                math.isclose(allowed, tenor_value, rel_tol=0.0, abs_tol=1e-9)
                for allowed in allowed_tenors
            ):
                continue
            aggregation: dict[str, str] = {"mid_iv": "mean"}
            if {"bid_iv", "ask_iv"}.issubset(group.columns):
                aggregation.update({"bid_iv": "mean", "ask_iv": "mean"})
            collapsed = group.groupby("strike", as_index=False).agg(aggregation)
            if len(collapsed) > 20_000:
                raise ValueError("Heston tenor exceeds the 20000-strike limit")
            if len(collapsed) < self._config.min_strikes:
                continue
            forward = self._resolve_forward(tenor_value, group, forward_curve)
            weights, weighting_method = self._weights(forward, tenor_value, collapsed)
            full_holdout = self._config.holdout_tenors is not None and any(
                math.isclose(held, tenor_value, rel_tol=0.0, abs_tol=1e-9)
                for held in self._config.holdout_tenors
            )
            holdout = (
                np.ones(len(collapsed), dtype=bool)
                if full_holdout
                else self._holdout_mask(collapsed["strike"].to_numpy(dtype=float), centre=forward)
            )
            slices.append(
                _HestonSlice(
                    tenor=tenor_value,
                    forward=forward,
                    group=collapsed,
                    weights=weights,
                    weighting=weighting_method,
                    holdout=holdout,
                )
            )

        if not slices:
            return self._empty_detailed_result()
        if self._config.holdout_tenors is not None:
            missing_holdouts = [
                held
                for held in self._config.holdout_tenors
                if not any(
                    math.isclose(held, current.tenor, rel_tol=0.0, abs_tol=1e-9)
                    for current in slices
                )
            ]
            if missing_holdouts:
                raise ValueError(
                    "holdout_tenors are absent from the retained calibration board: "
                    f"{missing_holdouts}"
                )
        if self._config.calibration_mode == "global":
            return self._calibrate_global(slices)

        results: list[HestonTenorResult] = []
        for current_slice in slices:
            result = self._calibrate_single_tenor(
                current_slice.forward,
                current_slice.tenor,
                current_slice.group,
            )
            if results:
                result.parameter_change_l2 = self._parameter_change_l2(
                    results[-1].params,
                    result.params,
                )
            results.append(result)
        return self._summarise_per_tenor(results, slices)

    def compare_modes(
        self,
        clean_board: CleanBoard,
        *,
        forward_curve: Mapping[float, float] | None = None,
    ) -> HestonCalibrationComparison:
        """Fit both architectures and report them side by side without ranking."""

        per_config = replace(
            self._config,
            calibration_mode="per_tenor",
            holdout_tenors=None,
        )
        # Whole-tenor exclusion has no per-tenor analogue: an independent
        # smile cannot be fitted with every quote removed. Clear it for both
        # sides so the comparison uses identical deterministic strike
        # holdouts. Whole-tenor extrapolation remains available through a
        # direct global ``calibrate_detailed`` call.
        global_config = replace(
            self._config,
            calibration_mode="global",
            holdout_tenors=None,
        )
        per_tenor = HestonCalibrator(per_config).calibrate_detailed(
            clean_board,
            forward_curve=forward_curve,
        )
        global_fit = HestonCalibrator(global_config).calibrate_detailed(
            clean_board,
            forward_curve=forward_curve,
        )
        return HestonCalibrationComparison(per_tenor=per_tenor, global_fit=global_fit)

    def _empty_detailed_result(self) -> HestonCalibrationResult:
        return HestonCalibrationResult(
            mode=self._config.calibration_mode,
            tenor_results=[],
            shared_params=None,
            in_sample_weighted_rmse=math.nan,
            holdout_rmse=None,
            feller_ratio=None,
            parameter_bound_proximity={},
            optimizer_diagnostics=(),
            calibration_observations=0,
            holdout_observations=0,
            pricing_method=self._config.pricing_method,
            strike_weighting=self._config.weighting,
            tenor_weighting=(
                self._config.global_tenor_weighting
                if self._config.calibration_mode == "global"
                else None
            ),
        )

    def _summarise_per_tenor(
        self,
        results: list[HestonTenorResult],
        slices: Sequence[_HestonSlice],
    ) -> HestonCalibrationResult:
        calibration_count = sum(result.calibration_observations for result in results)
        holdout_count = sum(result.holdout_observations for result in results)
        weighted_squared_error = 0.0
        calibration_weight_sum = 0.0
        for result, current in zip(results, slices, strict=True):
            calibration = ~current.holdout
            errors = result.model_vols - result.market_vols
            weighted_squared_error += float(
                np.sum(current.weights[calibration] * errors[calibration] ** 2)
            )
            calibration_weight_sum += float(np.sum(current.weights[calibration]))
        in_sample = math.sqrt(weighted_squared_error / calibration_weight_sum)
        holdout = (
            math.sqrt(
                sum(
                    (result.holdout_rmse or 0.0) ** 2 * result.holdout_observations
                    for result in results
                )
                / holdout_count
            )
            if holdout_count
            else None
        )
        methods = {result.weighting for result in results}
        residuals = tuple(row for result in results for row in result.residuals)
        _, residual_summary, weight_diagnostics = residual_diagnostics(
            tenors=np.array([row.tenor for row in residuals]),
            strikes=np.array([row.strike for row in residuals]),
            forwards=np.array([row.strike / math.exp(row.log_moneyness) for row in residuals]),
            market=np.array([row.market_volatility for row in residuals]),
            fitted=np.array([row.fitted_volatility for row in residuals]),
            weights=np.concatenate([current.weights for current in slices]),
            holdout=np.array([row.is_holdout for row in residuals]),
        )
        warnings = tuple(
            dict.fromkeys(warning for result in results for warning in result.warnings)
        )
        return HestonCalibrationResult(
            mode="per_tenor",
            tenor_results=results,
            shared_params=None,
            in_sample_weighted_rmse=in_sample,
            holdout_rmse=holdout,
            feller_ratio=None,
            parameter_bound_proximity={},
            optimizer_diagnostics=tuple(
                result.optimizer_diagnostics
                for result in results
                if result.optimizer_diagnostics is not None
            ),
            calibration_observations=calibration_count,
            holdout_observations=holdout_count,
            pricing_method=self._config.pricing_method,
            strike_weighting=next(iter(methods)) if len(methods) == 1 else "mixed",
            tenor_weighting=None,
            residuals=residuals,
            residual_summary=residual_summary,
            weight_diagnostics=weight_diagnostics,
            initialization_sensitivity=tuple(
                result.initialization_sensitivity
                for result in results
                if result.initialization_sensitivity is not None
            ),
            conditioning=tuple(
                result.conditioning for result in results if result.conditioning is not None
            ),
            strike_holdout_rmse=holdout,
            tenor_holdout_rmse=None,
            fit_quality=self._fit_quality(warnings, residual_summary),
            warnings=warnings,
        )

    @staticmethod
    def _risk_warnings(
        feller_ratio: float,
        proximity: Mapping[str, float],
        sensitivity: InitializationSensitivity,
        conditioning: ConditioningDiagnostics,
        residuals: ResidualSummary,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        if feller_ratio < 1.0:
            warnings.append("feller_condition_not_satisfied")
        if any(value >= 0.98 for value in proximity.values()):
            warnings.append("parameter_near_transformed_bound")
        if sensitivity.classification != "stable":
            warnings.append(f"initialization_{sensitivity.classification.replace('/', '_')}")
        if conditioning.weakly_identified:
            warnings.append("locally_weak_parameter_identification")
        if residuals.holdout_rmse is not None and residuals.holdout_rmse > max(
            0.02, 2.0 * residuals.weighted_rmse
        ):
            warnings.append("poor_strike_holdout_generalization")
        if residuals.maximum_absolute_residual > 0.05:
            warnings.append("large_local_residual")
        return tuple(warnings)

    @staticmethod
    def _fit_quality(warnings: Sequence[str], residuals: ResidualSummary) -> str:
        severe = {
            "parameter_near_transformed_bound",
            "locally_weak_parameter_identification",
        }
        if any(item.startswith("initialization_multimodal") for item in warnings):
            return str(FitQuality.UNSTABLE)
        if severe.intersection(warnings):
            return str(FitQuality.UNSTABLE)
        if "poor_strike_holdout_generalization" in warnings or residuals.weighted_rmse > 0.03:
            return str(FitQuality.POOR)
        if warnings or residuals.weighted_rmse > 0.01:
            return str(FitQuality.ACCEPTABLE)
        return str(FitQuality.GOOD)

    @staticmethod
    def _raw_bounds() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.array(
            [math.log(1e-4), math.log(1e-4), math.log(1e-2), math.log(1e-2), -3.8],
            dtype=float,
        )
        upper = np.array(
            [math.log(4.0), math.log(4.0), math.log(20.0), math.log(5.0), 3.8],
            dtype=float,
        )
        return lower, upper

    @staticmethod
    def _parameter_bound_proximity(
        raw: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
    ) -> dict[str, float]:
        """Return zero at the transformed-bound centre and one at a bound."""

        unit = np.clip((raw - lower) / (upper - lower), 0.0, 1.0)
        proximity = np.abs(2.0 * unit - 1.0)
        names = ("v0", "theta", "kappa", "vol_of_vol", "rho")
        return {name: float(value) for name, value in zip(names, proximity, strict=True)}

    @staticmethod
    def _economic_bound_proximity(params: Mapping[str, float]) -> dict[str, float]:
        bounds = {
            "v0": (1e-4, 4.0),
            "theta": (1e-4, 4.0),
            "kappa": (1e-2, 20.0),
            "vol_of_vol": (1e-2, 5.0),
            "rho": (math.tanh(-3.8), math.tanh(3.8)),
        }
        return {
            name: float(abs(2.0 * np.clip((params[name] - low) / (high - low), 0, 1) - 1.0))
            for name, (low, high) in bounds.items()
        }

    def _optimise(
        self,
        objective: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        initial: NDArray[np.float64],
        lower: NDArray[np.float64],
        upper: NDArray[np.float64],
        *,
        residual_count: int,
        failure_label: str,
    ) -> tuple[
        NDArray[np.float64],
        HestonOptimizerDiagnostics,
        InitializationSensitivity,
        ConditioningDiagnostics,
    ]:
        best_raw: NDArray[np.float64] | None = None
        best_fit: OptimizeResult | None = None
        best_seed = 0
        best_rmse = float("inf")
        succeeded = 0
        attempts: list[OptimizerAttempt] = []
        for seed in self._config.seeds:
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
            fit_values = np.asarray(getattr(fit, "fun", np.array([math.nan])), dtype=float)
            valid = (
                bool(getattr(fit, "success", False))
                and fit_values.size >= residual_count
                and np.all(np.isfinite(fit_values))
            )
            candidate_raw = np.asarray(getattr(fit, "x", start), dtype=float)
            candidate_params = self._unpack(candidate_raw)
            attempt_objective = float(np.mean(fit_values[:residual_count] ** 2)) if valid else None
            attempts.append(
                OptimizerAttempt(
                    seed=int(seed),
                    initial_parameters=tuple(self._unpack(start).values()),
                    success=bool(valid),
                    status=int(getattr(fit, "status", 0)),
                    message=str(getattr(fit, "message", "optimizer did not return a message")),
                    objective=attempt_objective,
                    parameters=tuple(candidate_params.values()) if valid else None,
                    evaluations=int(getattr(fit, "nfev", 0)),
                )
            )
            if not valid:
                continue
            succeeded += 1
            residuals = fit_values[:residual_count]
            rmse = float(np.sqrt(np.mean(residuals**2)))
            if not math.isfinite(rmse) or rmse >= 5.0:
                continue
            if rmse < best_rmse:
                best_rmse = rmse
                best_raw = np.asarray(fit.x, dtype=float)
                best_fit = fit
                best_seed = seed

        if best_raw is None or best_fit is None:
            raise CalibrationError(
                CalibrationFailureReason.OPTIMIZER_FAILED,
                f"Heston calibration failed for {failure_label}",
            )
        active = np.asarray(
            getattr(best_fit, "active_mask", np.zeros(best_raw.size, dtype=int)),
            dtype=int,
        )
        jacobian_evaluations = getattr(best_fit, "njev", None)
        fit_cost = getattr(best_fit, "cost", None)
        if fit_cost is None:
            fit_cost = 0.5 * float(np.sum(np.asarray(best_fit.fun, dtype=float) ** 2))
        diagnostics = HestonOptimizerDiagnostics(
            success=True,
            status=int(getattr(best_fit, "status", 0)),
            message=str(getattr(best_fit, "message", "optimizer reported success")),
            evaluations=int(getattr(best_fit, "nfev", 0)),
            jacobian_evaluations=(
                None if jacobian_evaluations is None else int(jacobian_evaluations)
            ),
            cost=float(fit_cost),
            optimality=float(getattr(best_fit, "optimality", math.nan)),
            active_mask=tuple(int(value) for value in active),
            seeds_attempted=len(self._config.seeds),
            seeds_succeeded=succeeded,
            best_seed=best_seed,
        )
        jacobian = getattr(best_fit, "jac", None)
        conditioning = (
            conditioning_from_jacobian(np.asarray(jacobian, dtype=float)[:residual_count, :])
            if jacobian is not None
            else ConditioningDiagnostics((), 0, None, True, None)
        )
        return best_raw, diagnostics, analyze_initialization_sensitivity(attempts), conditioning

    def _global_weights(self, slices: Sequence[_HestonSlice]) -> list[NDArray[np.float64]]:
        adjusted = [current.weights.copy() for current in slices]
        calibration_slices = [
            index for index, current in enumerate(slices) if np.any(~current.holdout)
        ]
        if len(calibration_slices) < 2:
            raise ValueError("global Heston calibration requires at least two non-holdout tenors")
        total_observations = sum(
            int(np.count_nonzero(~slices[index].holdout)) for index in calibration_slices
        )
        if total_observations < 7:
            raise ValueError("global Heston calibration has fewer than seven observations")
        for index in calibration_slices:
            calibration = ~slices[index].holdout
            target_sum = (
                total_observations / len(calibration_slices)
                if self._config.global_tenor_weighting == "equal"
                else int(np.count_nonzero(calibration))
            )
            current_sum = float(np.sum(adjusted[index][calibration]))
            adjusted[index] *= target_sum / current_sum
        return adjusted

    def _calibrate_global(self, slices: Sequence[_HestonSlice]) -> HestonCalibrationResult:
        adjusted_weights = self._global_weights(slices)
        calibration_count = sum(int(np.count_nonzero(~current.holdout)) for current in slices)
        objective_size = calibration_count + int(self._config.feller_penalty > 0.0)
        calibration_slices = [current for current in slices if np.any(~current.holdout)]
        shortest = calibration_slices[0]
        longest = calibration_slices[-1]

        def atm_variance(current: _HestonSlice) -> float:
            strikes = current.group["strike"].to_numpy(dtype=float)
            market_vols = current.group["mid_iv"].to_numpy(dtype=float)
            atm = float(market_vols[np.argmin(np.abs(strikes - current.forward))])
            return max(atm**2, 1e-4)

        initial = np.array(
            [
                math.log(atm_variance(shortest)),
                math.log(atm_variance(longest)),
                math.log(1.5),
                math.log(0.5),
                -0.3,
            ],
            dtype=float,
        )
        lower, upper = self._raw_bounds()

        def objective(raw: NDArray[np.float64]) -> NDArray[np.float64]:
            params = self._unpack(raw)
            parts: list[NDArray[np.float64]] = []
            try:
                for current, weights in zip(slices, adjusted_weights, strict=True):
                    calibration = ~current.holdout
                    if not np.any(calibration):
                        continue
                    strikes = current.group["strike"].to_numpy(dtype=float)
                    market = current.group["mid_iv"].to_numpy(dtype=float)
                    model = self._model_vols(current.forward, strikes, current.tenor, params)
                    parts.append(
                        np.sqrt(weights[calibration]) * (model[calibration] - market[calibration])
                    )
            except ValueError:
                return np.full(objective_size, 10.0, dtype=float)
            residuals = np.concatenate(parts)
            residuals = np.where(np.isfinite(residuals), residuals, 10.0)
            if self._config.feller_penalty > 0.0:
                feller_ratio = 2.0 * params["kappa"] * params["theta"] / params["vol_of_vol"] ** 2
                residuals = np.append(
                    residuals,
                    math.sqrt(self._config.feller_penalty) * max(0.0, 1.0 - feller_ratio),
                )
            return np.asarray(residuals, dtype=float)

        best_raw, optimizer, sensitivity, conditioning = self._optimise(
            objective,
            initial,
            lower,
            upper,
            residual_count=calibration_count,
            failure_label="global multi-tenor fit",
        )
        params = self._unpack(best_raw)
        bound_proximity = self._parameter_bound_proximity(best_raw, lower, upper)
        economic_proximity = self._economic_bound_proximity(params)
        feller_ratio = 2.0 * params["kappa"] * params["theta"] / params["vol_of_vol"] ** 2
        results: list[HestonTenorResult] = []
        weighted_squared_error = 0.0
        weight_sum = 0.0
        holdout_squared_error = 0.0
        holdout_count = 0
        for index, (current, weights) in enumerate(zip(slices, adjusted_weights, strict=True)):
            strikes = current.group["strike"].to_numpy(dtype=float)
            market = current.group["mid_iv"].to_numpy(dtype=float)
            model = self._model_vols(current.forward, strikes, current.tenor, params)
            if not np.isfinite(model).all():
                raise RuntimeError(
                    f"Heston global calibration produced invalid values for tenor {current.tenor}"
                )
            errors = model - market
            residuals, residual_summary, weight_diagnostics = residual_diagnostics(
                tenors=np.full(strikes.size, current.tenor),
                strikes=strikes,
                forwards=np.full(strikes.size, current.forward),
                market=market,
                fitted=model,
                weights=current.weights,
                holdout=(
                    np.zeros_like(current.holdout) if np.all(current.holdout) else current.holdout
                ),
            )
            if np.all(current.holdout):
                residuals = tuple(replace(row, is_holdout=True) for row in residuals)
                residual_summary = replace(
                    residual_summary,
                    calibration_observations=0,
                    holdout_observations=len(errors),
                    rmse=math.nan,
                    weighted_rmse=math.nan,
                    holdout_rmse=float(np.sqrt(np.mean(errors**2))),
                )
            calibration = ~current.holdout
            if np.any(calibration):
                weighted_squared_error += float(
                    np.sum(weights[calibration] * errors[calibration] ** 2)
                )
                weight_sum += float(np.sum(weights[calibration]))
                in_sample = float(
                    np.sqrt(
                        np.average(errors[calibration] ** 2, weights=current.weights[calibration])
                    )
                )
            else:
                in_sample = math.nan
            if np.any(current.holdout):
                current_holdout_rmse = float(np.sqrt(np.mean(errors[current.holdout] ** 2)))
                holdout_squared_error += float(np.sum(errors[current.holdout] ** 2))
                holdout_count += int(np.count_nonzero(current.holdout))
            else:
                current_holdout_rmse = None
            warnings = self._risk_warnings(
                feller_ratio, bound_proximity, sensitivity, conditioning, residual_summary
            )
            results.append(
                HestonTenorResult(
                    tenor=current.tenor,
                    params=params.copy(),
                    rmse=float(np.sqrt(np.mean(errors**2))),
                    strikes=strikes.copy(),
                    market_vols=market.copy(),
                    model_vols=model.copy(),
                    weighted_rmse=float(np.sqrt(np.average(errors**2, weights=current.weights))),
                    holdout_rmse=current_holdout_rmse,
                    feller_ratio=feller_ratio,
                    feller_satisfied=feller_ratio >= 1.0,
                    weighting=current.weighting,
                    calibration_observations=int(np.count_nonzero(calibration)),
                    holdout_observations=int(np.count_nonzero(current.holdout)),
                    parameter_change_l2=None if index == 0 else 0.0,
                    calibration_mode="global",
                    in_sample_weighted_rmse=in_sample,
                    parameter_bound_proximity=bound_proximity.copy(),
                    optimizer_diagnostics=optimizer,
                    is_holdout_tenor=bool(np.all(current.holdout)),
                    residuals=residuals,
                    residual_summary=residual_summary,
                    weight_diagnostics=weight_diagnostics,
                    initialization_sensitivity=sensitivity,
                    conditioning=conditioning,
                    transformed_parameter_bound_proximity=bound_proximity.copy(),
                    economic_parameter_bound_proximity=economic_proximity.copy(),
                    fit_quality=self._fit_quality(warnings, residual_summary),
                    warnings=warnings,
                )
            )
        methods = {current.weighting for current in slices}
        all_residuals = tuple(row for result in results for row in result.residuals)
        all_market = np.array([row.market_volatility for row in all_residuals])
        all_fitted = np.array([row.fitted_volatility for row in all_residuals])
        all_holdout = np.array([row.is_holdout for row in all_residuals])
        all_weights = np.concatenate(adjusted_weights)
        _, aggregate_summary, aggregate_weights = residual_diagnostics(
            tenors=np.array([row.tenor for row in all_residuals]),
            strikes=np.array([row.strike for row in all_residuals]),
            forwards=np.array([row.strike / math.exp(row.log_moneyness) for row in all_residuals]),
            market=all_market,
            fitted=all_fitted,
            weights=all_weights,
            holdout=all_holdout,
        )
        strike_errors = [
            row.residual
            for row, current in zip(
                all_residuals,
                np.concatenate([np.full(len(s.group), np.all(s.holdout)) for s in slices]),
                strict=True,
            )
            if row.is_holdout and not current
        ]
        tenor_errors = [
            row.residual
            for row, current in zip(
                all_residuals,
                np.concatenate([np.full(len(s.group), np.all(s.holdout)) for s in slices]),
                strict=True,
            )
            if row.is_holdout and current
        ]
        aggregate_warnings = tuple(
            dict.fromkeys(warning for result in results for warning in result.warnings)
        )
        return HestonCalibrationResult(
            mode="global",
            tenor_results=results,
            shared_params=params,
            in_sample_weighted_rmse=math.sqrt(weighted_squared_error / weight_sum),
            holdout_rmse=(
                math.sqrt(holdout_squared_error / holdout_count) if holdout_count else None
            ),
            feller_ratio=feller_ratio,
            parameter_bound_proximity=bound_proximity,
            optimizer_diagnostics=(optimizer,),
            calibration_observations=calibration_count,
            holdout_observations=holdout_count,
            pricing_method=self._config.pricing_method,
            strike_weighting=next(iter(methods)) if len(methods) == 1 else "mixed",
            tenor_weighting=self._config.global_tenor_weighting,
            residuals=all_residuals,
            residual_summary=aggregate_summary,
            weight_diagnostics=aggregate_weights,
            initialization_sensitivity=(sensitivity,),
            conditioning=(conditioning,),
            strike_holdout_rmse=float(np.sqrt(np.mean(np.square(strike_errors))))
            if strike_errors
            else None,
            tenor_holdout_rmse=float(np.sqrt(np.mean(np.square(tenor_errors))))
            if tenor_errors
            else None,
            fit_quality=self._fit_quality(aggregate_warnings, aggregate_summary),
            warnings=aggregate_warnings,
        )

    @staticmethod
    def _parameter_change_l2(
        previous: Mapping[str, float],
        current: Mapping[str, float],
    ) -> float:
        positive_names = ("v0", "theta", "kappa", "vol_of_vol")
        changes = [math.log(current[name] / previous[name]) for name in positive_names]
        changes.append(current["rho"] - previous["rho"])
        return float(np.linalg.norm(changes) / math.sqrt(len(changes)))

    def _weights(
        self,
        forward: float,
        tenor: float,
        group: pd.DataFrame,
    ) -> tuple[np.ndarray, str]:
        strikes = group["strike"].to_numpy(dtype=float)
        market_vols = group["mid_iv"].to_numpy(dtype=float)
        sqrt_t = math.sqrt(tenor)
        vol_sqrt_t = np.maximum(market_vols * sqrt_t, 1e-12)
        d1 = (np.log(forward / strikes) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
        vegas = forward * np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi) * sqrt_t
        vega_weights = np.maximum(vegas, 1e-12) ** 2

        has_spreads = {"bid_iv", "ask_iv"}.issubset(group.columns)
        requested = self._config.weighting
        method = ("hybrid" if has_spreads else "vega") if requested == "auto" else requested
        if method in {"bid_ask", "hybrid"} and not has_spreads:
            raise ValueError(f"Heston {method} weighting requires bid_iv and ask_iv")

        if method == "uniform":
            weights = np.ones_like(strikes)
        elif method == "vega":
            weights = vega_weights
        else:
            spreads = group["ask_iv"].to_numpy(dtype=float) - group["bid_iv"].to_numpy(dtype=float)
            if (
                not np.isfinite(spreads).all()
                or np.any(spreads < 0.0)
                or np.any(group["bid_iv"].to_numpy(dtype=float) < 0.0)
            ):
                raise ValueError("Heston calibration contains invalid bid/ask spreads")
            spread_weights = 1.0 / np.maximum(spreads, self._config.spread_floor) ** 2
            weights = spread_weights if method == "bid_ask" else spread_weights * vega_weights

        median = float(np.median(weights))
        if not math.isfinite(median) or median <= 0.0:
            raise ValueError("Heston calibration weights are degenerate")
        weights = np.clip(weights / median, 1e-3, 1e3)
        weights /= float(np.mean(weights))
        return np.asarray(weights, dtype=float), method

    def _holdout_mask(
        self, observations: int | ArrayLike, *, centre: float | None = None
    ) -> np.ndarray:
        values = (
            np.arange(observations, dtype=float)
            if isinstance(observations, Integral) and not isinstance(observations, bool)
            else np.asarray(observations, dtype=float)
        )
        if values.ndim != 1:
            raise ValueError("Heston holdout values must be one-dimensional")
        count_observations = values.size
        mask = np.zeros(count_observations, dtype=bool)
        maximum_holdout = max(0, count_observations - self._config.min_strikes)
        if maximum_holdout == 0 or self._config.holdout_fraction == 0.0:
            return mask
        return deterministic_holdout_mask(
            values,
            self._config.holdout_policy,
            fraction=self._config.holdout_fraction,
            minimum_training=self._config.min_strikes,
            centre=centre,
        )

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
        weights, weighting_method = self._weights(forward, tenor, group)
        holdout = self._holdout_mask(strikes, centre=forward)
        calibration = ~holdout
        atm_vol = float(market_vols[np.argmin(np.abs(strikes - forward))])
        variance = max(atm_vol**2, 1e-4)
        initial = np.array(
            [math.log(variance), math.log(variance), math.log(1.5), math.log(0.5), -0.3],
            dtype=float,
        )
        lower, upper = self._raw_bounds()
        objective_size = int(np.count_nonzero(calibration)) + int(self._config.feller_penalty > 0.0)

        def objective(raw: NDArray[np.float64]) -> NDArray[np.float64]:
            params = self._unpack(raw)
            try:
                model_vols = self._model_vols(forward, strikes, tenor, params)
            except ValueError:
                return np.full(objective_size, 10.0, dtype=float)
            residuals = np.sqrt(weights[calibration]) * (
                model_vols[calibration] - market_vols[calibration]
            )
            residuals = np.where(np.isfinite(residuals), residuals, 10.0)
            if self._config.feller_penalty > 0.0:
                feller_ratio = 2.0 * params["kappa"] * params["theta"] / params["vol_of_vol"] ** 2
                penalty = math.sqrt(self._config.feller_penalty) * max(
                    0.0,
                    1.0 - feller_ratio,
                )
                residuals = np.append(residuals, penalty)
            return np.asarray(residuals, dtype=float)

        best_raw, optimizer, sensitivity, conditioning = self._optimise(
            objective,
            initial,
            lower,
            upper,
            residual_count=int(np.count_nonzero(calibration)),
            failure_label=f"tenor {tenor}",
        )
        params = self._unpack(best_raw)
        model_vols = self._model_vols(forward, strikes, tenor, params)
        if not np.isfinite(model_vols).all():
            raise CalibrationError(
                CalibrationFailureReason.INVALID_MODEL_EVALUATION,
                f"Heston calibration produced invalid values for tenor {tenor}",
            )
        errors = model_vols - market_vols
        rmse = float(np.sqrt(np.mean(errors**2)))
        weighted_rmse = float(np.sqrt(np.average(errors**2, weights=weights)))
        in_sample_weighted_rmse = float(
            np.sqrt(np.average(errors[calibration] ** 2, weights=weights[calibration]))
        )
        holdout_rmse = float(np.sqrt(np.mean(errors[holdout] ** 2))) if np.any(holdout) else None
        feller_ratio = 2.0 * params["kappa"] * params["theta"] / params["vol_of_vol"] ** 2
        residuals, residual_summary, weight_diagnostics = residual_diagnostics(
            tenors=np.full(strikes.size, tenor),
            strikes=strikes,
            forwards=np.full(strikes.size, forward),
            market=market_vols,
            fitted=model_vols,
            weights=weights,
            holdout=holdout,
        )
        transformed_proximity = self._parameter_bound_proximity(best_raw, lower, upper)
        economic_proximity = self._economic_bound_proximity(params)
        warnings = self._risk_warnings(
            feller_ratio, transformed_proximity, sensitivity, conditioning, residual_summary
        )
        return HestonTenorResult(
            tenor=tenor,
            params=params,
            rmse=rmse,
            strikes=strikes.copy(),
            market_vols=market_vols.copy(),
            model_vols=model_vols.copy(),
            weighted_rmse=weighted_rmse,
            holdout_rmse=holdout_rmse,
            feller_ratio=feller_ratio,
            feller_satisfied=feller_ratio >= 1.0,
            weighting=weighting_method,
            calibration_observations=int(np.count_nonzero(calibration)),
            holdout_observations=int(np.count_nonzero(holdout)),
            calibration_mode="per_tenor",
            in_sample_weighted_rmse=in_sample_weighted_rmse,
            parameter_bound_proximity=transformed_proximity,
            optimizer_diagnostics=optimizer,
            residuals=residuals,
            residual_summary=residual_summary,
            weight_diagnostics=weight_diagnostics,
            initialization_sensitivity=sensitivity,
            conditioning=conditioning,
            transformed_parameter_bound_proximity=transformed_proximity.copy(),
            economic_parameter_bound_proximity=economic_proximity,
            fit_quality=self._fit_quality(warnings, residual_summary),
            warnings=warnings,
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

    def _model_vols(
        self,
        forward: float,
        strikes: NDArray[np.float64],
        tenor: float,
        params: Mapping[str, float],
    ) -> NDArray[np.float64]:
        arguments = {
            "v0": params["v0"],
            "theta": params["theta"],
            "kappa": params["kappa"],
            "vol_of_vol": params["vol_of_vol"],
            "rho": params["rho"],
        }
        if self._config.pricing_method == "cos":
            from .heston_cos import HestonCOSConfig, heston_cos_implied_volatilities

            return heston_cos_implied_volatilities(
                forward,
                strikes,
                tenor,
                config=HestonCOSConfig(
                    terms=self._config.cos_terms,
                    truncation=self._config.cos_truncation,
                    adaptive=False,
                    max_terms=self._config.cos_terms,
                    max_truncation=self._config.cos_truncation,
                ),
                **arguments,
            )
        return heston_implied_volatilities(forward, strikes, tenor, **arguments)


# Compatibility alias for callers of versions <=1.0.1.
HestonQECalibrator = HestonCalibrator


__all__ = [
    "HestonCalibrationComparison",
    "HestonCalibrationResult",
    "HestonCalibrator",
    "HestonConfig",
    "HestonOptimizerDiagnostics",
    "HestonQECalibrator",
    "HestonTenorResult",
    "heston_call_prices",
    "heston_implied_volatilities",
]
