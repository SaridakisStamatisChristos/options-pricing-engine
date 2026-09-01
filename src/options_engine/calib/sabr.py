"""SABR calibration routines."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
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


@dataclass(frozen=True, slots=True)
class SABRConfig:
    beta: float = 0.5
    fit_beta: bool = False
    seeds: tuple[int, ...] = (0, 1, 2)
    tolerance: float = 1e-8
    max_iterations: int = 200
    holdout_policy: str = "none"
    holdout_fraction: float = 0.2
    weighting: str = "uniform"
    spread_floor: float = 1e-4

    def __post_init__(self) -> None:
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, Real)
            or not math.isfinite(self.beta)
        ):
            raise TypeError("beta must be a finite real number")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be within [0, 1]")
        if not isinstance(self.fit_beta, bool):
            raise TypeError("fit_beta must be a boolean")
        if self.fit_beta and not 0.0 < self.beta < 1.0:
            raise ValueError("a fitted beta initial value must be strictly within (0, 1)")
        if not self.seeds or len(self.seeds) > 32:
            raise ValueError("seeds must contain between 1 and 32 entries")
        if any(
            isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**128 - 1
            for seed in self.seeds
        ):
            raise ValueError("calibration seeds must be non-negative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("calibration seeds must not contain duplicates")
        object.__setattr__(self, "beta", float(self.beta))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, Real)
            or not math.isfinite(self.tolerance)
            or not 1e-14 <= self.tolerance <= 1e-2
        ):
            raise ValueError("tolerance must be within [1e-14, 1e-2]")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, Integral)
            or not 1 <= self.max_iterations <= 100_000
        ):
            raise ValueError("max_iterations must be an integer within [1, 100000]")
        object.__setattr__(self, "tolerance", float(self.tolerance))
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        try:
            HoldoutPolicy(self.holdout_policy)
        except ValueError as exc:
            raise ValueError("unsupported SABR holdout_policy") from exc
        if not isinstance(self.holdout_fraction, Real) or isinstance(self.holdout_fraction, bool):
            raise TypeError("holdout_fraction must be a real number")
        if not math.isfinite(self.holdout_fraction) or not 0.0 <= self.holdout_fraction <= 0.5:
            raise ValueError("holdout_fraction must be within [0, 0.5]")
        object.__setattr__(self, "holdout_fraction", float(self.holdout_fraction))
        if self.weighting not in {"uniform", "auto", "bid_ask"}:
            raise ValueError("weighting must be 'uniform', 'auto', or 'bid_ask'")
        if (
            isinstance(self.spread_floor, bool)
            or not isinstance(self.spread_floor, Real)
            or not math.isfinite(self.spread_floor)
            or not 1e-8 <= self.spread_floor <= 1.0
        ):
            raise ValueError("spread_floor must be within [1e-8, 1]")
        object.__setattr__(self, "spread_floor", float(self.spread_floor))


@dataclass(slots=True)
class SABRTenorResult:
    tenor: float
    params: dict[str, float]
    rmse: float
    strikes: np.ndarray
    market_vols: np.ndarray
    model_vols: np.ndarray
    parameter_count: int
    residuals: tuple[ResidualObservation, ...] = ()
    residual_summary: ResidualSummary | None = None
    weight_diagnostics: WeightDiagnostics | None = None
    initialization_sensitivity: InitializationSensitivity | None = None
    conditioning: ConditioningDiagnostics | None = None
    parameter_bound_proximity: dict[str, float] = field(default_factory=dict)
    fit_quality: str = FitQuality.ACCEPTABLE.value
    warnings: tuple[str, ...] = ()
    weighting: str = "uniform"
    calibrated_strike_range: tuple[float, float] | None = None
    admissible: bool = True
    classification_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], serializable(self))


@dataclass(slots=True)
class SABRCalibrationResult:
    """Detailed SABR audit result; ``calibrate`` retains its list contract."""

    tenor_results: list[SABRTenorResult]
    calibration_observations: int
    holdout_observations: int
    in_sample_weighted_rmse: float
    holdout_rmse: float | None
    fit_quality: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], serializable(self))


class SABRCalibrator:
    """Calibrate SABR parameters per tenor."""

    def __init__(self, config: SABRConfig | None = None) -> None:
        if config is not None and not isinstance(config, SABRConfig):
            raise TypeError("config must be a SABRConfig")
        self._config = config if config is not None else SABRConfig()

    def calibrate(
        self,
        clean_board: CleanBoard,
        *,
        forward_curve: Mapping[float, float] | None = None,
    ) -> list[SABRTenorResult]:
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
            raise ValueError("clean board contains values outside the SABR calibration domain")

        results: list[SABRTenorResult] = []
        warm_start: np.ndarray | None = None

        for tenor, group in data.groupby("tenor", sort=True):
            tenor_value = float(tenor)
            aggregation: dict[str, str] = {"mid_iv": "mean"}
            has_spreads = {"bid_iv", "ask_iv"}.issubset(group.columns)
            if has_spreads:
                aggregation.update({"bid_iv": "mean", "ask_iv": "mean"})
            smile = group.groupby("strike", as_index=False).agg(aggregation)
            strikes = smile["strike"].to_numpy(dtype=float)
            market_vols = smile["mid_iv"].to_numpy(dtype=float)
            weights = np.ones(strikes.size, dtype=float)
            weighting = "uniform"
            use_spreads = self._config.weighting == "bid_ask" or (
                self._config.weighting == "auto" and has_spreads
            )
            if self._config.weighting == "bid_ask" and not has_spreads:
                raise ValueError("SABR bid_ask weighting requires bid_iv and ask_iv")
            if use_spreads:
                bid = smile["bid_iv"].to_numpy(dtype=float)
                ask = smile["ask_iv"].to_numpy(dtype=float)
                if (
                    not np.isfinite(bid).all()
                    or not np.isfinite(ask).all()
                    or np.any(bid < 0.0)
                    or np.any(ask < bid)
                ):
                    raise ValueError("invalid SABR bid/ask spreads")
                inverse = 1.0 / np.maximum(ask - bid, self._config.spread_floor) ** 2
                weights = np.clip(inverse / np.median(inverse), 1e-3, 1e3)
                weighting = "bid_ask"
            weights /= float(np.mean(weights))
            minimum_observations = (4 if self._config.fit_beta else 3) + 2
            if strikes.size < minimum_observations:
                continue
            forward = self._resolve_forward(tenor_value, group, forward_curve)
            res = self._calibrate_single_tenor(
                forward,
                tenor_value,
                strikes,
                market_vols,
                warm_start=warm_start,
                weights=weights,
                weighting=weighting,
            )
            results.append(res)
            warm_start = self._pack_params(res.params)
        return results

    def calibrate_detailed(
        self,
        clean_board: CleanBoard,
        *,
        forward_curve: Mapping[float, float] | None = None,
    ) -> SABRCalibrationResult:
        """Return aggregate validation evidence without breaking ``calibrate``."""

        results = self.calibrate(clean_board, forward_curve=forward_curve)
        summaries = [item.residual_summary for item in results if item.residual_summary is not None]
        training_count = sum(item.calibration_observations for item in summaries)
        holdout_count = sum(item.holdout_observations for item in summaries)
        weighted_rmse = (
            math.sqrt(
                sum(item.weighted_rmse**2 * item.calibration_observations for item in summaries)
                / training_count
            )
            if training_count
            else math.nan
        )
        holdout_rmse = (
            math.sqrt(
                sum(
                    (item.holdout_rmse or 0.0) ** 2 * item.holdout_observations
                    for item in summaries
                )
                / holdout_count
            )
            if holdout_count
            else None
        )
        order = {"good": 0, "acceptable": 1, "poor": 2, "unstable": 3, "invalid": 4}
        quality = max(
            (item.fit_quality for item in results),
            key=lambda value: order[value],
            default=FitQuality.INVALID.value,
        )
        warnings = tuple(dict.fromkeys(warning for item in results for warning in item.warnings))
        return SABRCalibrationResult(
            results,
            training_count,
            holdout_count,
            weighted_rmse,
            holdout_rmse,
            quality,
            warnings,
        )

    def _resolve_forward(
        self,
        tenor: float,
        group: pd.DataFrame,
        forward_curve: Mapping[float, float] | None,
    ) -> float:
        forward: float
        if forward_curve is not None:
            if tenor in forward_curve:
                forward = float(forward_curve[tenor])
            else:
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

    def _calibrate_single_tenor(
        self,
        forward: float,
        tenor: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        *,
        warm_start: np.ndarray | None,
        weights: np.ndarray | None = None,
        weighting: str = "uniform",
    ) -> SABRTenorResult:
        cfg = self._config

        seeds: Sequence[int] = cfg.seeds or (0,)
        best_rmse = float("inf")
        best_result: tuple[np.ndarray, np.ndarray, Any] | None = None
        attempts: list[OptimizerAttempt] = []

        minimum_training = (4 if cfg.fit_beta else 3) + 1
        holdout = deterministic_holdout_mask(
            strikes,
            cfg.holdout_policy,
            fraction=cfg.holdout_fraction,
            minimum_training=minimum_training,
            centre=forward,
        )
        training = ~holdout
        observation_weights = (
            np.ones(strikes.size, dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        if (
            observation_weights.shape != strikes.shape
            or not np.isfinite(observation_weights).all()
            or np.any(observation_weights <= 0.0)
        ):
            raise ValueError("SABR weights must be finite, positive, and match strikes")

        def objective(theta: np.ndarray) -> np.ndarray:
            params = self._unpack_params(theta)
            try:
                model = hagan_implied_volatility(
                    forward,
                    strikes,
                    tenor,
                    alpha=params["alpha"],
                    beta=params["beta"],
                    rho=params["rho"],
                    nu=params["nu"],
                )
            except ValueError:
                return np.full_like(market_vols[training], 10.0)
            residual = np.asarray(model - market_vols, dtype=float)
            return np.asarray((np.sqrt(observation_weights) * residual)[training], dtype=float)

        initial = self._initial_guess(forward, strikes, market_vols, warm_start)
        lower = np.array([math.log(1e-8), -4.95, math.log(1e-8)], dtype=float)
        upper = np.array([math.log(100.0), 4.95, math.log(10.0)], dtype=float)
        if cfg.fit_beta:
            lower = np.append(lower, -9.21)
            upper = np.append(upper, 9.21)

        for seed in seeds:
            theta0 = initial.copy()
            if seed != 0:
                rng = np.random.default_rng(seed)
                theta0 = theta0 + rng.normal(scale=0.25, size=theta0.size)
            theta0 = np.clip(theta0, lower + 1e-10, upper - 1e-10)
            result = least_squares(
                objective,
                theta0,
                bounds=(lower, upper),
                xtol=cfg.tolerance,
                ftol=cfg.tolerance,
                gtol=cfg.tolerance,
                max_nfev=cfg.max_iterations,
            )
            attempt_params = (
                tuple(self._format_params(result.x).values()) if result.success else None
            )
            objective_value = (
                float(np.mean(result.fun**2))
                if result.success and np.isfinite(result.fun).all()
                else None
            )
            attempts.append(
                OptimizerAttempt(
                    seed=int(seed),
                    initial_parameters=tuple(float(x) for x in theta0),
                    success=bool(result.success),
                    status=int(getattr(result, "status", 0)),
                    message=str(getattr(result, "message", "optimizer supplied no message")),
                    objective=objective_value,
                    parameters=attempt_params,
                    evaluations=int(getattr(result, "nfev", 0)),
                )
            )
            if not result.success:
                continue
            params_for_model = self._unpack_params(result.x)
            try:
                model_vols = hagan_implied_volatility(forward, strikes, tenor, **params_for_model)
            except ValueError:
                continue
            rmse = float(np.sqrt(np.mean((model_vols[training] - market_vols[training]) ** 2)))
            if not math.isfinite(rmse) or rmse >= 5.0:
                continue
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = (result.x, model_vols, result)

        if best_result is None:
            raise CalibrationError(
                CalibrationFailureReason.OPTIMIZER_FAILED,
                f"SABR calibration failed for tenor {tenor}",
            )

        params = self._format_params(best_result[0])
        residuals, summary, weight_diagnostics = residual_diagnostics(
            tenors=np.full(strikes.size, tenor),
            strikes=strikes,
            forwards=np.full(strikes.size, forward),
            market=market_vols,
            fitted=best_result[1],
            weights=observation_weights,
            holdout=holdout,
        )
        sensitivity = analyze_initialization_sensitivity(attempts)
        lower_e, upper_e = (
            {"alpha": 1e-8, "rho": -math.tanh(4.95), "nu": 1e-8},
            {"alpha": 100.0, "rho": math.tanh(4.95), "nu": 10.0},
        )
        if cfg.fit_beta:
            lower_e["beta"], upper_e["beta"] = 0.0001, 0.9999
        proximity = {
            name: float(
                1.0
                - 2.0
                * min(
                    (value - lower_e[name]) / (upper_e[name] - lower_e[name]),
                    (upper_e[name] - value) / (upper_e[name] - lower_e[name]),
                )
            )
            for name, value in params.items()
            if name in lower_e
        }
        jacobian = getattr(best_result[2], "jac", None)
        conditioning = conditioning_from_jacobian(jacobian) if jacobian is not None else None
        warnings: list[str] = []
        if tenor < 1.0 / 52.0:
            warnings.append("short-tenor SABR parameters may be weakly identified")
        if abs(params["rho"]) > 0.95 or params["nu"] > 5.0:
            warnings.append("parameters are economically extreme")
        if conditioning is None:
            warnings.append("optimizer did not provide a Jacobian for conditioning analysis")
        elif conditioning.weakly_identified:
            warnings.append("local Jacobian indicates weak parameter identification")
        quality = (
            FitQuality.UNSTABLE
            if (
                sensitivity.classification != "stable"
                or (conditioning is not None and conditioning.weakly_identified)
            )
            else FitQuality.GOOD
        )
        if summary.holdout_rmse is not None and summary.holdout_rmse > max(
            0.02, 3.0 * summary.rmse
        ):
            quality = FitQuality.POOR
            warnings.append("holdout error materially exceeds training error")
        return SABRTenorResult(
            tenor=float(tenor),
            params=params,
            rmse=best_rmse,
            strikes=strikes.copy(),
            market_vols=market_vols.copy(),
            model_vols=best_result[1].copy(),
            parameter_count=4 if cfg.fit_beta else 3,
            residuals=residuals,
            residual_summary=summary,
            weight_diagnostics=weight_diagnostics,
            initialization_sensitivity=sensitivity,
            conditioning=conditioning,
            parameter_bound_proximity=proximity,
            fit_quality=quality.value,
            warnings=tuple(warnings),
            weighting=weighting,
            calibrated_strike_range=(
                float(np.min(strikes[training])),
                float(np.max(strikes[training])),
            ),
            admissible=True,
            classification_reasons=tuple(warnings),
        )

    def _initial_guess(
        self,
        forward: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        warm_start: np.ndarray | None,
    ) -> np.ndarray:
        cfg = self._config
        if warm_start is not None:
            return warm_start.copy()
        atm_index = int(np.argmin(np.abs(strikes - forward)))
        atm_vol = float(market_vols[atm_index])
        alpha = np.log(max(atm_vol * forward ** (1.0 - cfg.beta), 1e-4))
        rho = np.arctanh(np.clip(0.0, -0.95, 0.95))
        nu = np.log(0.5)
        if cfg.fit_beta:
            beta = np.log(cfg.beta / (1.0 - cfg.beta + 1e-12))
            return np.array([alpha, rho, nu, beta], dtype=float)
        return np.array([alpha, rho, nu], dtype=float)

    def _pack_params(self, params: Mapping[str, float]) -> np.ndarray:
        if self._config.fit_beta:
            beta = params.get("beta", self._config.beta)
            beta = np.clip(beta, 1e-6, 1.0 - 1e-6)
            beta_theta = np.log(beta / (1.0 - beta))
            return np.array(
                [
                    np.log(params["alpha"]),
                    np.arctanh(params["rho"]),
                    np.log(params["nu"]),
                    beta_theta,
                ],
                dtype=float,
            )
        return np.array(
            [
                np.log(params["alpha"]),
                np.arctanh(params["rho"]),
                np.log(params["nu"]),
            ],
            dtype=float,
        )

    def _unpack_params(self, theta: ArrayLike) -> dict[str, float]:
        theta = np.asarray(theta, dtype=float)
        alpha = float(np.exp(theta[0]))
        rho = float(np.tanh(theta[1]))
        nu = float(np.exp(theta[2]))
        beta = self._config.beta
        if self._config.fit_beta:
            beta = float(1.0 / (1.0 + np.exp(-theta[3])))
        beta = float(np.clip(beta, 0.0, 1.0))
        return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu}

    def _format_params(self, theta: ArrayLike) -> dict[str, float]:
        params = self._unpack_params(theta)
        return {key: float(value) for key, value in params.items()}


def hagan_implied_volatility(
    forward: float | np.ndarray,
    strike: float | np.ndarray,
    expiry: float,
    *,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    scalar_parameters = {
        "expiry": expiry,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "nu": nu,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for value in scalar_parameters.values()
    ):
        raise TypeError("SABR scalar parameters must be finite real numbers")
    if not 0.0 < expiry <= 100.0:
        raise ValueError("expiry must be within (0, 100]")
    if (
        not 0.0 < alpha <= 100.0
        or not 0.0 <= nu <= 10.0
        or not 0.0 <= beta <= 1.0
        or not -1.0 < rho < 1.0
    ):
        raise ValueError("invalid SABR parameters")

    forward_input = np.asarray(forward)
    strike_input = np.asarray(strike)
    if forward_input.dtype.kind not in "iuf" or forward_input.dtype.kind == "b":
        raise TypeError("forward values must be real numbers")
    if strike_input.dtype.kind not in "iuf" or strike_input.dtype.kind == "b":
        raise TypeError("strike values must be real numbers")
    F = np.asarray(forward_input, dtype=float)
    K = np.asarray(strike_input, dtype=float)
    if (
        not np.isfinite(F).all()
        or not np.isfinite(K).all()
        or np.any(F <= 0.0)
        or np.any(K <= 0.0)
        or np.any(F > 1e12)
        or np.any(K > 1e12)
    ):
        raise ValueError("forward and strike values must be finite and strictly positive")

    try:
        F, K = np.broadcast_arrays(F, K)
    except ValueError as exc:
        raise ValueError("forward and strike must be broadcastable") from exc
    if F.size > 100_000:
        raise ValueError("SABR broadcast input exceeds the 100000-value limit")
    with np.errstate(divide="ignore", invalid="ignore"):
        fk_beta = np.power(F * K, (1.0 - beta) / 2.0)
        fk_beta = np.where(fk_beta <= 0.0, 1e-12, fk_beta)
        log_fk = np.log(F / K)
        z = (nu / alpha) * fk_beta * log_fk
        sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z**2)
        numerator = sqrt_term + z - rho
        denominator = 1.0 - rho
        x_z = np.log(np.where(numerator <= 0.0, 1.0, numerator / denominator))
        small_z = np.abs(z) < 1e-7
        z_over_x = np.where(
            small_z,
            1.0 - 0.5 * rho * z + ((2.0 - 3.0 * rho**2) / 12.0) * z**2,
            z / x_z,
        )

        one_minus_beta = 1.0 - beta
        one_minus_beta_sq = one_minus_beta**2
        log_fk_sq = log_fk**2
        log_fk_quartic = log_fk_sq**2

        term1 = alpha / (
            fk_beta
            * (
                1.0
                + (one_minus_beta_sq / 24.0) * log_fk_sq
                + (one_minus_beta_sq**2 / 1920.0) * log_fk_quartic
            )
        )
        term2 = z_over_x
        term3 = (
            1.0
            + (
                ((one_minus_beta_sq / 24.0) * (alpha**2) / (fk_beta**2))
                + (0.25 * rho * beta * nu * alpha) / fk_beta
                + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
            )
            * expiry
        )

        implied = term1 * term2 * term3
    if not np.isfinite(implied).all() or np.any(implied <= 0.0):
        raise ValueError("SABR parameters produced non-finite or non-positive volatility")
    return np.asarray(implied, dtype=float)
