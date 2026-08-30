"""SABR calibration routines."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from .boards import CleanBoard


@dataclass(frozen=True, slots=True)
class SABRConfig:
    beta: float = 0.5
    fit_beta: bool = False
    seeds: tuple[int, ...] = (0, 1, 2)
    tolerance: float = 1e-8
    max_iterations: int = 200

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


@dataclass(slots=True)
class SABRTenorResult:
    tenor: float
    params: dict[str, float]
    rmse: float
    strikes: np.ndarray
    market_vols: np.ndarray
    model_vols: np.ndarray
    parameter_count: int


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
            smile = group.groupby("strike", as_index=False)["mid_iv"].mean()
            strikes = smile["strike"].to_numpy(dtype=float)
            market_vols = smile["mid_iv"].to_numpy(dtype=float)
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
            )
            results.append(res)
            warm_start = self._pack_params(res.params)
        return results

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
    ) -> SABRTenorResult:
        cfg = self._config

        seeds: Sequence[int] = cfg.seeds or (0,)
        best_rmse = float("inf")
        best_result: tuple[np.ndarray, np.ndarray] | None = None

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
                return np.full_like(market_vols, 10.0)
            return np.asarray(model - market_vols, dtype=float)

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
            if not result.success:
                continue
            model_vols = market_vols + result.fun
            rmse = float(np.sqrt(np.mean(result.fun**2)))
            if not math.isfinite(rmse) or rmse >= 5.0:
                continue
            if rmse < best_rmse:
                best_rmse = rmse
                best_result = (result.x, model_vols)

        if best_result is None:
            raise RuntimeError(f"SABR calibration failed for tenor {tenor}")

        params = self._format_params(best_result[0])
        return SABRTenorResult(
            tenor=float(tenor),
            params=params,
            rmse=best_rmse,
            strikes=strikes.copy(),
            market_vols=market_vols.copy(),
            model_vols=best_result[1].copy(),
            parameter_count=4 if cfg.fit_beta else 3,
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
