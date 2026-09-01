"""SVI diagnostics and an arbitrage-aware SSVI volatility surface.

Raw SVI is useful for slice analysis but independent slice fits can introduce
calendar arbitrage. The production surface in this module therefore uses the
SSVI parameterisation with monotone ATM total variance, power-law ``phi``, Lee
wing constraints, the Gatheral-Jacquier density diagnostic, and an explicit
calendar check on a common log-moneyness grid.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

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


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalised = float(value)
    if not math.isfinite(normalised):
        raise ValueError(f"{name} must be finite")
    return normalised


def _one_dimensional_finite(name: str, values: ArrayLike, *, maximum: int = 100_000) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind not in "iuf" or source.dtype.kind == "b":
        raise TypeError(f"{name} must contain real numbers")
    array = np.asarray(source, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not 1 <= array.size <= maximum:
        raise ValueError(f"{name} must contain between 1 and {maximum} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True, slots=True)
class SVIParameters:
    """Raw SVI slice parameters with basic no-arbitrage bounds."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def __post_init__(self) -> None:
        values = {
            name: _finite_real(name, value)
            for name, value in {
                "a": self.a,
                "b": self.b,
                "rho": self.rho,
                "m": self.m,
                "sigma": self.sigma,
            }.items()
        }
        if values["b"] < 0.0:
            raise ValueError("b must be non-negative")
        if not -1.0 < values["rho"] < 1.0:
            raise ValueError("rho must be strictly within (-1, 1)")
        if values["sigma"] <= 0.0:
            raise ValueError("sigma must be strictly positive")
        minimum_variance = values["a"] + values["b"] * values["sigma"] * math.sqrt(
            1.0 - values["rho"] ** 2
        )
        if minimum_variance < 0.0:
            raise ValueError("raw SVI minimum total variance must be non-negative")
        if values["b"] * (1.0 + abs(values["rho"])) > 2.0:
            raise ValueError("raw SVI wing slopes violate Lee's moment bound")
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class SVIDiagnostics:
    """Static-arbitrage diagnostics for an SVI-style variance slice."""

    minimum_total_variance: float
    minimum_density_factor: float
    maximum_wing_slope: float
    butterfly_free: bool
    left_wing_slope: float = math.nan
    right_wing_slope: float = math.nan
    butterfly_violation_count: int = 0
    admissible: bool = False
    fit_quality: FitQuality = FitQuality.INVALID
    parameter_bound_proximity: dict[str, float] = field(default_factory=dict)


def raw_svi_total_variance(log_moneyness: ArrayLike, params: SVIParameters) -> np.ndarray:
    """Evaluate raw SVI total variance ``w(k)``."""

    if not isinstance(params, SVIParameters):
        raise TypeError("params must be SVIParameters")
    k = _one_dimensional_finite("log_moneyness", log_moneyness)
    shifted = k - params.m
    return np.asarray(
        params.a + params.b * (params.rho * shifted + np.sqrt(shifted**2 + params.sigma**2)),
        dtype=float,
    )


def svi_density_factor(
    log_moneyness: ArrayLike,
    total_variance: ArrayLike,
    first_derivative: ArrayLike,
    second_derivative: ArrayLike,
) -> np.ndarray:
    """Return the Gatheral-Jacquier butterfly-density factor ``g(k)``."""

    k = _one_dimensional_finite("log_moneyness", log_moneyness)
    variance = _one_dimensional_finite("total_variance", total_variance)
    first = _one_dimensional_finite("first_derivative", first_derivative)
    second = _one_dimensional_finite("second_derivative", second_derivative)
    if not (k.shape == variance.shape == first.shape == second.shape):
        raise ValueError("SVI density inputs must have identical shapes")
    if np.any(variance <= 0.0):
        raise ValueError("total_variance must be strictly positive")
    density = (
        (1.0 - k * first / (2.0 * variance)) ** 2
        - 0.25 * first**2 * (1.0 / variance + 0.25)
        + 0.5 * second
    )
    return np.asarray(density, dtype=float)


def validate_svi_slice(
    params: SVIParameters,
    *,
    log_moneyness: ArrayLike | None = None,
) -> SVIDiagnostics:
    """Validate a raw SVI slice on a dense, caller-overridable grid."""

    if not isinstance(params, SVIParameters):
        raise TypeError("params must be SVIParameters")
    k = (
        np.linspace(-5.0, 5.0, 2_001)
        if log_moneyness is None
        else _one_dimensional_finite("log_moneyness", log_moneyness)
    )
    shifted = k - params.m
    root = np.sqrt(shifted**2 + params.sigma**2)
    variance = raw_svi_total_variance(k, params)
    first = params.b * (params.rho + shifted / root)
    second = params.b * params.sigma**2 / root**3
    density = svi_density_factor(k, variance, first, second)
    minimum_variance = float(np.min(variance))
    minimum_density = float(np.min(density))
    wing_slope = params.b * (1.0 + abs(params.rho))
    return SVIDiagnostics(
        minimum_total_variance=minimum_variance,
        minimum_density_factor=minimum_density,
        maximum_wing_slope=wing_slope,
        butterfly_free=minimum_variance >= 0.0 and minimum_density >= -1e-10,
        left_wing_slope=params.b * (1.0 - params.rho),
        right_wing_slope=params.b * (1.0 + params.rho),
        butterfly_violation_count=int(np.sum(density < -1e-10)),
        admissible=minimum_variance >= 0.0 and minimum_density >= -1e-10,
        fit_quality=(
            FitQuality.GOOD
            if minimum_variance >= 0.0 and minimum_density >= -1e-10
            else FitQuality.INVALID
        ),
        parameter_bound_proximity={
            "rho": float(1.0 - abs(params.rho)),
            "lee_left": float(2.0 - params.b * (1.0 - params.rho)),
            "lee_right": float(2.0 - params.b * (1.0 + params.rho)),
            "sigma": float(params.sigma),
            "minimum_variance": minimum_variance,
        },
    )


@dataclass(frozen=True, slots=True)
class RawSVIConfig:
    """Deterministic audit configuration for a single raw-SVI slice."""

    seeds: tuple[int, ...] = (0, 1, 2, 3)
    holdout_policy: HoldoutPolicy | str = HoldoutPolicy.NONE
    holdout_fraction: float = 0.2
    max_iterations: int = 2_000
    tolerance: float = 1e-10
    validation_range: float = 5.0
    validation_points: int = 2_001

    def __post_init__(self) -> None:
        if (
            not self.seeds
            or len(self.seeds) > 32
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, Integral)
                or not 0 <= seed <= 2**128 - 1
                for seed in self.seeds
            )
        ):
            raise ValueError("seeds must contain between 1 and 32 non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "holdout_policy", HoldoutPolicy(self.holdout_policy))
        if (
            isinstance(self.holdout_fraction, bool)
            or not isinstance(self.holdout_fraction, Real)
            or not math.isfinite(self.holdout_fraction)
            or not 0.0 <= self.holdout_fraction <= 0.5
        ):
            raise ValueError("holdout_fraction must be in [0, 0.5]")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, Integral)
            or not 1 <= self.max_iterations <= 100_000
        ):
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if (
            self.validation_range <= 0.0
            or self.validation_points < 101
            or self.validation_points % 2 == 0
        ):
            raise ValueError("validation grid must have a positive range and an odd count >= 101")


@dataclass(frozen=True, slots=True)
class RawSVICalibrationResult:
    """Auditable result for an independently fitted raw-SVI slice."""

    parameters: SVIParameters
    diagnostics: SVIDiagnostics
    residuals: tuple[ResidualObservation, ...]
    residual_summary: ResidualSummary
    weight_diagnostics: WeightDiagnostics
    initialization_sensitivity: InitializationSensitivity
    conditioning: ConditioningDiagnostics
    parameter_bound_proximity: dict[str, float]
    fit_quality: FitQuality
    numerical_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], serializable(self))


class RawSVICalibrator:
    """Fit one raw-SVI total-variance slice and retain full audit evidence.

    Raw SVI is deliberately not presented as a globally arbitrage-safe surface.
    A dense-grid butterfly failure always classifies the result as ``invalid``.
    """

    def __init__(self, config: RawSVIConfig | None = None) -> None:
        self._config = RawSVIConfig() if config is None else config
        if not isinstance(self._config, RawSVIConfig):
            raise TypeError("config must be RawSVIConfig")

    def calibrate(
        self,
        log_moneyness: ArrayLike,
        total_variance: ArrayLike,
        *,
        tenor: float = 1.0,
        forward: float = 1.0,
        weights: ArrayLike | None = None,
    ) -> RawSVICalibrationResult:
        k = _one_dimensional_finite("log_moneyness", log_moneyness)
        market = _one_dimensional_finite("total_variance", total_variance)
        tenor_value = _finite_real("tenor", tenor)
        forward_value = _finite_real("forward", forward)
        if k.shape != market.shape or np.any(market <= 0.0):
            raise ValueError("log_moneyness and positive total_variance must have matching shapes")
        if tenor_value <= 0.0 or forward_value <= 0.0:
            raise ValueError("tenor and forward must be positive")
        if k.size < 6:
            raise ValueError("raw SVI requires at least six observations")
        weight = np.ones(k.size) if weights is None else _one_dimensional_finite("weights", weights)
        if weight.shape != k.shape or np.any(weight <= 0.0):
            raise ValueError("weights must be finite, positive, and match observations")
        weight = weight / np.mean(weight)
        held = deterministic_holdout_mask(
            k,
            self._config.holdout_policy,
            fraction=self._config.holdout_fraction,
            minimum_training=5,
            centre=0.0,
        )
        training = ~held
        lower = np.array([-2.0, 1e-8, -0.999, -5.0, 1e-5])
        upper = np.array([2.0, 2.0, 0.999, 5.0, 5.0])

        def values(x: np.ndarray, points: np.ndarray) -> np.ndarray:
            a, b, rho, m, sigma = x
            return np.asarray(
                a + b * (rho * (points - m) + np.sqrt((points - m) ** 2 + sigma**2)),
                dtype=float,
            )

        def objective(x: np.ndarray) -> np.ndarray:
            fitted = values(x, k[training])
            minimum = x[0] + x[1] * x[4] * math.sqrt(max(0.0, 1.0 - x[2] ** 2))
            lee = x[1] * (1.0 + abs(x[2]))
            penalty = np.array([max(0.0, -minimum) * 1e3, max(0.0, lee - 2.0) * 1e3])
            return np.concatenate(
                ((fitted - market[training]) * np.sqrt(weight[training]), penalty)
            )

        attempts: list[OptimizerAttempt] = []
        results = []
        base = np.array([max(float(np.min(market)) * 0.5, 1e-5), 0.1, -0.2, 0.0, 0.2])
        for seed in self._config.seeds:
            rng = np.random.default_rng(seed)
            initial = base.copy()
            if seed != self._config.seeds[0]:
                initial = np.clip(initial * rng.lognormal(0.0, 0.35, 5), lower + 1e-9, upper - 1e-9)
                initial[2] = float(np.clip(-0.2 + rng.normal(0.0, 0.35), -0.9, 0.9))
                initial[3] = float(np.clip(rng.normal(0.0, 0.25), -1.0, 1.0))
            result = least_squares(
                objective,
                initial,
                bounds=(lower, upper),
                max_nfev=self._config.max_iterations,
                xtol=self._config.tolerance,
                ftol=self._config.tolerance,
                gtol=self._config.tolerance,
            )
            params_tuple = tuple(float(v) for v in result.x)
            valid = result.success and objective(result.x)[-2:].max() <= 1e-8
            attempts.append(
                OptimizerAttempt(
                    seed,
                    tuple(float(v) for v in initial),
                    bool(valid),
                    int(result.status),
                    str(result.message),
                    float(np.mean(objective(result.x) ** 2)),
                    params_tuple,
                    int(result.nfev),
                )
            )
            if valid:
                results.append(result)
        if not results:
            raise CalibrationError(
                CalibrationFailureReason.ADMISSIBILITY_FAILED,
                "all raw SVI optimization starts failed admissibility",
            )
        best = min(results, key=lambda item: float(np.mean(objective(item.x) ** 2)))
        params = SVIParameters(*(float(v) for v in best.x))
        fitted_variance = raw_svi_total_variance(k, params)
        market_vol = np.sqrt(market / tenor_value)
        fitted_vol = np.sqrt(np.maximum(fitted_variance, 0.0) / tenor_value)
        strikes = forward_value * np.exp(k)
        rows, summary, weight_diag = residual_diagnostics(
            tenors=np.full(k.size, tenor_value),
            strikes=strikes,
            forwards=np.full(k.size, forward_value),
            market=market_vol,
            fitted=fitted_vol,
            weights=weight,
            holdout=held,
        )
        grid = np.linspace(
            -self._config.validation_range,
            self._config.validation_range,
            self._config.validation_points,
        )
        diagnostics = validate_svi_slice(params, log_moneyness=grid)
        sensitivity = analyze_initialization_sensitivity(attempts)
        conditioning = conditioning_from_jacobian(best.jac[:-2])
        proximity = dict(diagnostics.parameter_bound_proximity)
        proximity.update(
            {
                "a_lower": float(params.a - lower[0]),
                "a_upper": float(upper[0] - params.a),
                "b_lower": float(params.b - lower[1]),
                "b_upper": float(upper[1] - params.b),
                "m_lower": float(params.m - lower[3]),
                "m_upper": float(upper[3] - params.m),
                "sigma_upper": float(upper[4] - params.sigma),
            }
        )
        warnings: list[str] = []
        if not diagnostics.admissible:
            warnings.append("dense-grid butterfly arbitrage detected; raw SVI slice is invalid")
        if conditioning.weakly_identified:
            warnings.append("local Jacobian indicates weak parameter identification")
        if sensitivity.classification != "stable":
            warnings.append(f"initialization sensitivity is {sensitivity.classification}")
        quality = (
            FitQuality.INVALID
            if not diagnostics.admissible
            else FitQuality.UNSTABLE
            if conditioning.weakly_identified or sensitivity.classification != "stable"
            else FitQuality.POOR
            if summary.holdout_rmse is not None
            and summary.holdout_rmse > max(0.02, 2.0 * summary.rmse)
            else FitQuality.GOOD
            if summary.rmse < 0.005
            else FitQuality.ACCEPTABLE
        )
        return RawSVICalibrationResult(
            params,
            diagnostics,
            rows,
            summary,
            weight_diag,
            sensitivity,
            conditioning,
            proximity,
            quality,
            tuple(warnings),
        )


def ssvi_total_variance(
    log_moneyness: ArrayLike,
    theta: float,
    *,
    rho: float,
    eta: float,
    power: float,
) -> np.ndarray:
    """Evaluate power-law SSVI total variance for one ATM variance level."""

    k = _one_dimensional_finite("log_moneyness", log_moneyness)
    theta_value = _finite_real("theta", theta)
    rho_value = _finite_real("rho", rho)
    eta_value = _finite_real("eta", eta)
    power_value = _finite_real("power", power)
    if theta_value <= 0.0:
        raise ValueError("theta must be strictly positive")
    if not -1.0 < rho_value < 1.0:
        raise ValueError("rho must be strictly within (-1, 1)")
    if eta_value <= 0.0:
        raise ValueError("eta must be strictly positive")
    if not 0.0 <= power_value <= 0.5:
        raise ValueError("power must be within [0, 0.5]")

    phi = eta_value * theta_value ** (-power_value)
    x = phi * k + rho_value
    root = np.sqrt(x**2 + 1.0 - rho_value**2)
    return np.asarray(
        0.5 * theta_value * (1.0 + rho_value * phi * k + root),
        dtype=float,
    )


def _ssvi_derivatives(
    log_moneyness: np.ndarray,
    theta: float,
    rho: float,
    eta: float,
    power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = eta * theta ** (-power)
    x = phi * log_moneyness + rho
    root = np.sqrt(x**2 + 1.0 - rho**2)
    variance = 0.5 * theta * (1.0 + rho * phi * log_moneyness + root)
    first = 0.5 * theta * phi * (rho + x / root)
    second = 0.5 * theta * phi**2 * (1.0 - rho**2) / root**3
    return variance, first, second


def _isotonic_non_decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators projection."""

    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("isotonic inputs must be matching one-dimensional arrays")
    if values.size == 0 or not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("isotonic inputs must be non-empty and finite")
    if np.any(weights <= 0.0):
        raise ValueError("isotonic weights must be strictly positive")

    levels = [float(value) for value in values]
    block_weights = [float(value) for value in weights]
    starts = list(range(values.size))
    ends = list(range(values.size))
    index = 0
    while index < len(levels) - 1:
        if levels[index] <= levels[index + 1]:
            index += 1
            continue
        combined_weight = block_weights[index] + block_weights[index + 1]
        combined_level = (
            levels[index] * block_weights[index] + levels[index + 1] * block_weights[index + 1]
        ) / combined_weight
        levels[index : index + 2] = [combined_level]
        block_weights[index : index + 2] = [combined_weight]
        ends[index] = ends[index + 1]
        del starts[index + 1]
        del ends[index + 1]
        index = max(index - 1, 0)

    projected = np.empty_like(values, dtype=float)
    for level, start, end in zip(levels, starts, ends, strict=True):
        projected[start : end + 1] = level
    return projected


@dataclass(frozen=True, slots=True)
class SSVIConfig:
    """Deterministic global SSVI calibration controls."""

    seeds: tuple[int, ...] = (0, 1, 2)
    tolerance: float = 1e-9
    max_iterations: int = 1_000
    min_tenors: int = 2
    min_strikes_per_tenor: int = 5
    weighting: str = "auto"
    spread_floor: float = 1e-4
    validation_points: int = 1_001
    holdout_policy: HoldoutPolicy | str = HoldoutPolicy.NONE
    holdout_fraction: float = 0.2

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
        tolerance = _finite_real("tolerance", self.tolerance)
        if not 1e-14 <= tolerance <= 1e-2:
            raise ValueError("tolerance must be within [1e-14, 1e-2]")
        object.__setattr__(self, "tolerance", tolerance)
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, Integral)
            or not 1 <= self.max_iterations <= 100_000
        ):
            raise ValueError("max_iterations must be an integer within [1, 100000]")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        for name, value, low, high in (
            ("min_tenors", self.min_tenors, 2, 512),
            ("min_strikes_per_tenor", self.min_strikes_per_tenor, 3, 20_000),
            ("validation_points", self.validation_points, 101, 20_001),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be an integer within [{low}, {high}]")
            object.__setattr__(self, name, int(value))
        if self.validation_points % 2 == 0:
            raise ValueError("validation_points must be odd")
        if self.weighting not in {"auto", "uniform", "bid_ask"}:
            raise ValueError("weighting must be 'auto', 'uniform', or 'bid_ask'")
        spread_floor = _finite_real("spread_floor", self.spread_floor)
        if not 1e-8 <= spread_floor <= 1.0:
            raise ValueError("spread_floor must be within [1e-8, 1]")
        object.__setattr__(self, "spread_floor", spread_floor)
        try:
            object.__setattr__(self, "holdout_policy", HoldoutPolicy(self.holdout_policy))
        except ValueError as exc:
            raise ValueError(f"unsupported holdout policy: {self.holdout_policy!r}") from exc
        fraction = _finite_real("holdout_fraction", self.holdout_fraction)
        if not 0.0 <= fraction <= 0.5:
            raise ValueError("holdout_fraction must be within [0, 0.5]")
        object.__setattr__(self, "holdout_fraction", fraction)


@dataclass(frozen=True, slots=True)
class SSVISurface:
    """Interpolable, statically validated SSVI surface."""

    tenors: tuple[float, ...]
    atm_total_variances: tuple[float, ...]
    rho: float
    eta: float
    power: float

    def __post_init__(self) -> None:
        tenors = tuple(_finite_real("tenor", value) for value in self.tenors)
        theta = tuple(
            _finite_real("atm_total_variance", value) for value in self.atm_total_variances
        )
        rho = _finite_real("rho", self.rho)
        eta = _finite_real("eta", self.eta)
        power = _finite_real("power", self.power)
        if len(tenors) != len(theta) or len(tenors) < 2:
            raise ValueError("SSVI surface requires matching tenor/theta arrays of length >= 2")
        if not all(value > 0.0 for value in (*tenors, *theta)):
            raise ValueError("SSVI tenors and ATM variances must be finite and positive")
        if not all(left < right for left, right in itertools.pairwise(tenors)):
            raise ValueError("SSVI tenors must be strictly increasing")
        if not all(left <= right for left, right in itertools.pairwise(theta)):
            raise ValueError("SSVI ATM total variance must be non-decreasing")
        if not -1.0 < rho < 1.0 or eta <= 0.0 or not 0.0 <= power <= 0.5:
            raise ValueError("invalid SSVI shape parameters")
        wing, curvature = SSVICalibrator._shape_constraints(
            np.asarray(theta, dtype=float), rho, eta, power
        )
        if float(np.max(wing)) > 4.0 or float(np.max(curvature)) > 4.0:
            raise ValueError("SSVI shape parameters violate static-arbitrage constraints")
        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "atm_total_variances", theta)
        object.__setattr__(self, "rho", rho)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "power", power)

    def _theta(self, tenor: float) -> float:
        tenor_value = _finite_real("tenor", tenor)
        if not self.tenors[0] <= tenor_value <= self.tenors[-1]:
            raise ValueError("tenor is outside the calibrated SSVI range")
        return float(np.interp(tenor_value, self.tenors, self.atm_total_variances))

    def total_variance(self, log_moneyness: ArrayLike, tenor: float) -> np.ndarray:
        """Interpolate ATM variance and evaluate the SSVI smile."""

        return ssvi_total_variance(
            log_moneyness,
            self._theta(tenor),
            rho=self.rho,
            eta=self.eta,
            power=self.power,
        )

    def implied_volatility(
        self,
        strikes: ArrayLike,
        tenor: float,
        *,
        forward: float,
    ) -> np.ndarray:
        """Return Black implied volatility for positive strikes and forward."""

        strike_array = _one_dimensional_finite("strikes", strikes, maximum=20_000)
        forward_value = _finite_real("forward", forward)
        tenor_value = _finite_real("tenor", tenor)
        if forward_value <= 0.0 or np.any(strike_array <= 0.0) or tenor_value <= 0.0:
            raise ValueError("forward, strikes, and tenor must be strictly positive")
        variance = self.total_variance(np.log(strike_array / forward_value), tenor_value)
        return np.asarray(np.sqrt(np.maximum(variance, 0.0) / tenor_value), dtype=float)

    def to_dict(self) -> dict[str, object]:
        return {
            "tenors": list(self.tenors),
            "atm_total_variances": list(self.atm_total_variances),
            "rho": self.rho,
            "eta": self.eta,
            "power": self.power,
        }


@dataclass(frozen=True, slots=True)
class SSVICalibrationResult:
    surface: SSVISurface
    rmse: float
    weighted_rmse: float
    minimum_density_factor: float
    maximum_calendar_decrease: float
    maximum_wing_slope: float
    atm_projection_applied: bool
    observations: int
    calibration_observations: int = 0
    holdout_observations: int = 0
    holdout_rmse: float | None = None
    residuals: tuple[ResidualObservation, ...] = ()
    residual_summary: ResidualSummary | None = None
    weight_diagnostics: WeightDiagnostics | None = None
    initialization_sensitivity: InitializationSensitivity | None = None
    wing_constraint_slack: float = math.nan
    curvature_constraint_slack: float = math.nan
    parameter_bound_proximity: dict[str, float] = field(default_factory=dict)
    atm_projection_absolute_adjustment: float = 0.0
    atm_projection_relative_adjustment: float = 0.0
    largest_atm_projection_tenor: float | None = None
    total_weighted_projection_error: float = 0.0
    atm_projection_adjustments: tuple[tuple[float, float, float], ...] = ()
    interpolation_status: str = "unknown"
    fit_quality: FitQuality = FitQuality.ACCEPTABLE
    numerical_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], serializable(self))


class SSVICalibrator:
    """Fit one globally coherent power-law SSVI surface."""

    def __init__(self, config: SSVIConfig | None = None) -> None:
        if config is not None and not isinstance(config, SSVIConfig):
            raise TypeError("config must be an SSVIConfig")
        self._config = config if config is not None else SSVIConfig()

    @staticmethod
    def _resolve_forward(
        tenor: float,
        group: pd.DataFrame,
        forward_curve: Mapping[float, float] | None,
    ) -> float:
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
            forward = float(np.mean(forwards))
        if not math.isfinite(forward) or not 0.0 < forward <= 1e12:
            raise ValueError(f"forward for tenor {tenor} must be finite and within (0, 1e12]")
        return forward

    def _prepare(
        self,
        clean_board: CleanBoard,
        forward_curve: Mapping[float, float] | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        data = clean_board.quotes
        required = {"tenor", "strike", "mid_iv", "forward"}
        missing = required - set(data.columns)
        if missing:
            raise KeyError(f"clean board is missing required columns: {sorted(missing)}")
        if data.empty:
            raise ValueError("clean board is empty")
        if len(data) > 100_000:
            raise ValueError("clean board exceeds the 100000-row calibration limit")
        numeric = data[list(required)].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("clean board contains non-finite SSVI values")

        aggregate: dict[str, str] = {"mid_iv": "mean", "forward": "mean"}
        has_spreads = {"bid_iv", "ask_iv"}.issubset(data.columns)
        if has_spreads:
            aggregate.update({"bid_iv": "mean", "ask_iv": "mean"})
        collapsed = data.groupby(["tenor", "strike"], as_index=False).agg(aggregate)
        collapsed_numeric = collapsed[["tenor", "strike", "mid_iv", "forward"]].to_numpy(
            dtype=float
        )
        if (
            not np.isfinite(collapsed_numeric).all()
            or (collapsed["tenor"] <= 0.0).any()
            or (collapsed["tenor"] > 100.0).any()
            or (collapsed["strike"] <= 0.0).any()
            or (collapsed["strike"] > 1e12).any()
            or (collapsed["forward"] <= 0.0).any()
            or (collapsed["forward"] > 1e12).any()
            or (~collapsed["mid_iv"].between(1e-4, 5.0, inclusive="both")).any()
        ):
            raise ValueError("clean board contains values outside the SSVI domain")
        if has_spreads and (
            not np.isfinite(collapsed[["bid_iv", "ask_iv"]].to_numpy(dtype=float)).all()
            or (collapsed["bid_iv"] < 0.0).any()
            or (collapsed["ask_iv"] < collapsed["bid_iv"]).any()
        ):
            raise ValueError("clean board contains invalid SSVI bid/ask spreads")
        tenors = np.sort(collapsed["tenor"].unique().astype(float))
        if tenors.size < self._config.min_tenors:
            raise ValueError(f"SSVI requires at least {self._config.min_tenors} tenors")

        forward_by_tenor: dict[float, float] = {}
        theta_raw: list[float] = []
        theta_weights: list[float] = []
        for tenor in tenors:
            source_group = data[np.isclose(data["tenor"], tenor, rtol=0.0, atol=1e-12)]
            forward = self._resolve_forward(float(tenor), source_group, forward_curve)
            forward_by_tenor[float(tenor)] = forward
            group = collapsed[np.isclose(collapsed["tenor"], tenor, rtol=0.0, atol=1e-12)].copy()
            if len(group) < self._config.min_strikes_per_tenor:
                raise ValueError(
                    "each SSVI tenor must contain at least "
                    f"{self._config.min_strikes_per_tenor} distinct strikes"
                )
            group["log_moneyness"] = np.log(group["strike"].to_numpy(dtype=float) / forward)
            group = group.sort_values("log_moneyness")
            k = group["log_moneyness"].to_numpy(dtype=float)
            total_variance = group["mid_iv"].to_numpy(dtype=float) ** 2 * float(tenor)
            theta_raw.append(float(np.interp(0.0, k, total_variance)))
            theta_weights.append(float(len(group)))

        raw_theta = np.asarray(theta_raw, dtype=float)
        theta = _isotonic_non_decreasing(raw_theta, np.asarray(theta_weights, dtype=float))
        theta = np.maximum(theta, 1e-12)
        theta_lookup = dict(zip(tenors, theta, strict=True))

        row_tenors = collapsed["tenor"].to_numpy(dtype=float)
        strikes = collapsed["strike"].to_numpy(dtype=float)
        market_vols = collapsed["mid_iv"].to_numpy(dtype=float)
        if (
            np.any(row_tenors <= 0.0)
            or np.any(strikes <= 0.0)
            or np.any(market_vols <= 0.0)
            or np.any(market_vols > 5.0)
        ):
            raise ValueError("clean board contains values outside the SSVI domain")
        forwards = np.array([forward_by_tenor[float(tenor)] for tenor in row_tenors])
        log_moneyness = np.log(strikes / forwards)
        row_theta = np.array([theta_lookup[float(tenor)] for tenor in row_tenors])

        use_spreads = self._config.weighting == "bid_ask" or (
            self._config.weighting == "auto" and has_spreads
        )
        if self._config.weighting == "bid_ask" and not has_spreads:
            raise ValueError("SSVI bid_ask weighting requires bid_iv and ask_iv")
        if use_spreads:
            spreads = collapsed["ask_iv"].to_numpy(dtype=float) - collapsed["bid_iv"].to_numpy(
                dtype=float
            )
            inverse_variance = 1.0 / np.maximum(spreads, self._config.spread_floor) ** 2
            weights = np.clip(inverse_variance / np.median(inverse_variance), 1e-3, 1e3)
        else:
            weights = np.ones_like(market_vols)
        weights /= float(np.mean(weights))
        holdout = np.zeros(market_vols.size, dtype=bool)
        for tenor in tenors:
            indices = np.flatnonzero(np.isclose(row_tenors, tenor, rtol=0.0, atol=1e-12))
            holdout[indices] = deterministic_holdout_mask(
                log_moneyness[indices],
                self._config.holdout_policy,
                fraction=self._config.holdout_fraction,
                minimum_training=self._config.min_strikes_per_tenor,
                centre=0.0,
            )
        return (
            row_tenors,
            strikes,
            forwards,
            log_moneyness,
            market_vols,
            row_theta,
            weights,
            holdout,
            np.column_stack((raw_theta, theta, np.asarray(theta_weights, dtype=float))),
        )

    @staticmethod
    def _unpack(raw: NDArray[np.float64]) -> tuple[float, float, float]:
        rho = float(np.tanh(raw[0]))
        eta = float(np.exp(raw[1]))
        power = float(0.5 / (1.0 + np.exp(-raw[2])))
        return rho, eta, power

    @staticmethod
    def _shape_constraints(
        theta: np.ndarray,
        rho: float,
        eta: float,
        power: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        phi = eta * theta ** (-power)
        absolute_skew = 1.0 + abs(rho)
        return theta * phi * absolute_skew, theta * phi**2 * absolute_skew

    def calibrate(
        self,
        clean_board: CleanBoard,
        *,
        forward_curve: Mapping[float, float] | None = None,
    ) -> SSVICalibrationResult:
        if not isinstance(clean_board, CleanBoard):
            raise TypeError("clean_board must be a CleanBoard")
        if forward_curve is not None and not isinstance(forward_curve, Mapping):
            raise TypeError("forward_curve must be a mapping or None")
        (
            row_tenors,
            strikes,
            forwards,
            log_moneyness,
            market_vols,
            row_theta,
            weights,
            holdout,
            projection,
        ) = self._prepare(clean_board, forward_curve)
        projected = not np.allclose(projection[:, 0], projection[:, 1], rtol=0.0, atol=1e-14)
        unique_tenors, first_indices = np.unique(row_tenors, return_index=True)
        theta = row_theta[first_indices]

        def model_vols(raw: NDArray[np.float64]) -> np.ndarray:
            rho, eta, power = self._unpack(raw)
            phi = eta * row_theta ** (-power)
            x = phi * log_moneyness + rho
            variance = (
                0.5 * row_theta * (1.0 + rho * phi * log_moneyness + np.sqrt(x**2 + 1.0 - rho**2))
            )
            return np.asarray(
                np.sqrt(np.maximum(variance, 1e-16) / row_tenors),
                dtype=float,
            )

        def objective(raw: NDArray[np.float64]) -> NDArray[np.float64]:
            rho, eta, power = self._unpack(raw)
            training = ~holdout
            residuals = np.sqrt(weights[training]) * (
                model_vols(raw)[training] - market_vols[training]
            )
            wing, curvature = self._shape_constraints(theta, rho, eta, power)
            penalties = np.concatenate(
                (
                    np.maximum(wing - 3.999, 0.0),
                    np.maximum(curvature - 3.999, 0.0),
                )
            )
            return np.concatenate((residuals, 100.0 * penalties))

        initial = np.array([np.arctanh(-0.3), math.log(1.0), 0.0], dtype=float)
        lower = np.array([-3.8, math.log(1e-4), -8.0], dtype=float)
        upper = np.array([3.8, math.log(100.0), 8.0], dtype=float)
        best_raw: np.ndarray | None = None
        best_weighted_rmse = float("inf")
        attempts: list[OptimizerAttempt] = []
        for seed in self._config.seeds:
            start = initial.copy()
            if seed:
                start += np.random.default_rng(seed).normal(0.0, 0.2, size=start.size)
            start = np.clip(start, lower + 1e-10, upper - 1e-10)
            fit = least_squares(
                objective,
                start,
                bounds=(lower, upper),
                xtol=self._config.tolerance,
                ftol=self._config.tolerance,
                gtol=self._config.tolerance,
                max_nfev=self._config.max_iterations,
            )
            fitted_parameters = (
                self._unpack(np.asarray(fit.x, dtype=float)) if np.all(np.isfinite(fit.x)) else None
            )
            attempts.append(
                OptimizerAttempt(
                    seed=int(seed),
                    initial_parameters=tuple(float(value) for value in self._unpack(start)),
                    success=bool(fit.success and fitted_parameters is not None),
                    status=int(fit.status),
                    message=str(fit.message),
                    objective=(float(2.0 * fit.cost) if math.isfinite(float(fit.cost)) else None),
                    parameters=(
                        tuple(float(value) for value in fitted_parameters)
                        if fitted_parameters is not None
                        else None
                    ),
                    evaluations=int(fit.nfev),
                )
            )
            if not fit.success or not np.all(np.isfinite(fit.x)):
                continue
            fitted_rho, fitted_eta, fitted_power = self._unpack(np.asarray(fit.x, dtype=float))
            fitted_wing, fitted_curvature = self._shape_constraints(
                theta,
                fitted_rho,
                fitted_eta,
                fitted_power,
            )
            if float(np.max(fitted_wing)) > 4.0 or float(np.max(fitted_curvature)) > 4.0:
                continue
            fitted_vols = model_vols(np.asarray(fit.x, dtype=float))
            weighted_rmse = float(
                np.sqrt(
                    np.average(
                        (fitted_vols[~holdout] - market_vols[~holdout]) ** 2,
                        weights=weights[~holdout],
                    )
                )
            )
            if math.isfinite(weighted_rmse) and weighted_rmse < best_weighted_rmse:
                best_weighted_rmse = weighted_rmse
                best_raw = np.asarray(fit.x, dtype=float)
        if best_raw is None:
            raise CalibrationError(
                CalibrationFailureReason.OPTIMIZER_FAILED, "SSVI calibration failed"
            )

        rho, eta, power = self._unpack(best_raw)
        surface = SSVISurface(
            tenors=tuple(float(value) for value in unique_tenors),
            atm_total_variances=tuple(float(value) for value in theta),
            rho=rho,
            eta=eta,
            power=power,
        )
        fitted_vols = model_vols(best_raw)
        residuals, residual_summary, weight_diagnostics = residual_diagnostics(
            tenors=row_tenors,
            strikes=strikes,
            forwards=forwards,
            market=market_vols,
            fitted=fitted_vols,
            weights=weights,
            holdout=holdout,
        )
        rmse = residual_summary.rmse

        k_min = min(float(np.min(log_moneyness)) - 1.0, -2.0)
        k_max = max(float(np.max(log_moneyness)) + 1.0, 2.0)
        validation_grid = np.linspace(k_min, k_max, self._config.validation_points)
        validation_tenors = sorted(
            set(unique_tenors.tolist())
            | {0.5 * (left + right) for left, right in itertools.pairwise(unique_tenors)}
        )
        variance_rows: list[np.ndarray] = []
        minimum_density = float("inf")
        maximum_wing_slope = 0.0
        for tenor in validation_tenors:
            theta_value = surface._theta(float(tenor))
            variance, first, second = _ssvi_derivatives(
                validation_grid,
                theta_value,
                rho,
                eta,
                power,
            )
            density = svi_density_factor(validation_grid, variance, first, second)
            minimum_density = min(minimum_density, float(np.min(density)))
            phi = eta * theta_value ** (-power)
            maximum_wing_slope = max(
                maximum_wing_slope,
                0.5 * theta_value * phi * (1.0 + abs(rho)),
            )
            variance_rows.append(variance)
        calendar_differences = np.diff(np.vstack(variance_rows), axis=0)
        maximum_calendar_decrease = max(0.0, -float(np.min(calendar_differences)))
        wing_constraint, curvature_constraint = self._shape_constraints(theta, rho, eta, power)
        if (
            minimum_density < -1e-8
            or maximum_calendar_decrease > 1e-10
            or float(np.max(wing_constraint)) > 4.0 + 1e-8
            or float(np.max(curvature_constraint)) > 4.0 + 1e-8
        ):
            raise CalibrationError(
                CalibrationFailureReason.ARBITRAGE_VIOLATION,
                "SSVI calibration failed static-arbitrage validation",
            )

        sensitivity = analyze_initialization_sensitivity(attempts)
        wing_slack = float(4.0 - np.max(wing_constraint))
        curvature_slack = float(4.0 - np.max(curvature_constraint))
        adjustments = projection[:, 1] - projection[:, 0]
        absolute_adjustment = float(np.max(np.abs(adjustments)))
        relative_adjustment = float(
            np.max(np.abs(adjustments) / np.maximum(np.abs(projection[:, 0]), 1e-12))
        )
        largest_index = int(np.argmax(np.abs(adjustments)))
        weighted_projection_error = float(np.sum(projection[:, 2] * adjustments**2))
        proximity = {
            "rho": float(1.0 - abs(rho)),
            "eta_lower": float(eta - 1e-4),
            "eta_upper": float(100.0 - eta),
            "power_lower": float(power),
            "power_upper": float(0.5 - power),
            "wing_constraint": wing_slack,
            "curvature_constraint": curvature_slack,
        }
        warnings: list[str] = []
        if projected:
            warnings.append("observed ATM variance required monotone isotonic projection")
        if residual_summary.holdout_rmse is not None and residual_summary.holdout_rmse > max(
            0.02, 3.0 * residual_summary.rmse
        ):
            warnings.append("holdout error materially exceeds training error")
        if sensitivity.classification != "stable":
            warnings.append("SSVI parameters are sensitive to optimizer initialization")
        if min(wing_slack, curvature_slack) < 0.05:
            warnings.append("SSVI solution is close to a sufficient arbitrage constraint")
        if weight_diagnostics.effective_sample_size < 0.25 * market_vols.size:
            warnings.append("calibration weights are highly concentrated")
        poor_fit = (
            residual_summary.weighted_rmse > 0.03
            or residual_summary.maximum_absolute_residual > 0.075
        )
        if poor_fit:
            warnings.append("large residuals indicate economically poor quote fit")
        quality = FitQuality.GOOD
        if sensitivity.classification != "stable":
            quality = FitQuality.UNSTABLE
        if warnings and quality is FitQuality.GOOD:
            quality = FitQuality.ACCEPTABLE
        if residual_summary.holdout_rmse is not None and residual_summary.holdout_rmse > max(
            0.02, 3.0 * residual_summary.rmse
        ):
            quality = FitQuality.POOR
        if poor_fit and quality is not FitQuality.UNSTABLE:
            quality = FitQuality.POOR

        return SSVICalibrationResult(
            surface=surface,
            rmse=rmse,
            weighted_rmse=best_weighted_rmse,
            minimum_density_factor=minimum_density,
            maximum_calendar_decrease=maximum_calendar_decrease,
            maximum_wing_slope=maximum_wing_slope,
            atm_projection_applied=projected,
            observations=int(market_vols.size),
            calibration_observations=residual_summary.calibration_observations,
            holdout_observations=residual_summary.holdout_observations,
            holdout_rmse=residual_summary.holdout_rmse,
            residuals=residuals,
            residual_summary=residual_summary,
            weight_diagnostics=weight_diagnostics,
            initialization_sensitivity=sensitivity,
            wing_constraint_slack=wing_slack,
            curvature_constraint_slack=curvature_slack,
            parameter_bound_proximity=proximity,
            atm_projection_absolute_adjustment=absolute_adjustment,
            atm_projection_relative_adjustment=relative_adjustment,
            largest_atm_projection_tenor=(
                float(unique_tenors[largest_index]) if projected else None
            ),
            total_weighted_projection_error=weighted_projection_error,
            atm_projection_adjustments=tuple(
                (
                    float(tenor),
                    float(adjustment),
                    float(adjustment / max(abs(raw), 1e-12)),
                )
                for tenor, raw, adjustment in zip(
                    unique_tenors, projection[:, 0], adjustments, strict=True
                )
            ),
            interpolation_status="interpolation within calibrated tenor range; extrapolation rejected",
            fit_quality=quality,
            numerical_warnings=tuple(warnings),
        )


__all__ = [
    "SSVICalibrationResult",
    "SSVICalibrator",
    "SSVIConfig",
    "SSVISurface",
    "SVIDiagnostics",
    "SVIParameters",
    "raw_svi_total_variance",
    "ssvi_total_variance",
    "svi_density_factor",
    "validate_svi_slice",
]
