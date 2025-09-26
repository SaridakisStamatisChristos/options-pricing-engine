"""SABR calibration routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from .boards import CleanBoard


@dataclass(slots=True)
class SABRConfig:
    beta: float = 0.5
    fit_beta: bool = False
    seeds: Tuple[int, ...] = (0, 1, 2)
    tolerance: float = 1e-8
    max_iterations: int = 200


@dataclass(slots=True)
class SABRTenorResult:
    tenor: float
    params: Dict[str, float]
    rmse: float
    strikes: np.ndarray
    market_vols: np.ndarray
    model_vols: np.ndarray


class SABRCalibrator:
    """Calibrate SABR parameters per tenor."""

    def __init__(self, config: Optional[SABRConfig] = None) -> None:
        self._config = config or SABRConfig()

    def calibrate(
        self,
        clean_board: CleanBoard,
        *,
        forward_curve: Mapping[float, float] | None = None,
    ) -> List[SABRTenorResult]:
        data = clean_board.quotes
        if data.empty:
            return []

        results: List[SABRTenorResult] = []
        warm_start: Optional[np.ndarray] = None

        for tenor, group in data.groupby("tenor", sort=True):
            tenor_value = float(tenor)
            strikes = group["strike"].to_numpy(dtype=float)
            market_vols = group["mid_iv"].to_numpy(dtype=float)
            if strikes.size < 3:
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
        group,
        forward_curve: Mapping[float, float] | None,
    ) -> float:
        if forward_curve is not None:
            if tenor in forward_curve:
                return float(forward_curve[tenor])
            for key, value in forward_curve.items():
                if np.isclose(key, tenor, atol=1e-9):
                    return float(value)
        forwards = group["forward"].to_numpy(dtype=float)
        return float(np.mean(forwards))

    def _calibrate_single_tenor(
        self,
        forward: float,
        tenor: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        *,
        warm_start: Optional[np.ndarray],
    ) -> SABRTenorResult:
        cfg = self._config

        seeds: Sequence[int] = cfg.seeds or (0,)
        best_rmse = float("inf")
        best_result: Optional[Tuple[np.ndarray, np.ndarray]] = None

        def objective(theta: np.ndarray) -> np.ndarray:
            params = self._unpack_params(theta)
            model = hagan_implied_volatility(
                forward,
                strikes,
                tenor,
                alpha=params["alpha"],
                beta=params["beta"],
                rho=params["rho"],
                nu=params["nu"],
            )
            return model - market_vols

        initial = self._initial_guess(forward, strikes, market_vols, warm_start)

        for seed in seeds:
            theta0 = initial.copy()
            if seed != 0:
                rng = np.random.default_rng(seed)
                theta0 = theta0 + rng.normal(scale=0.25, size=theta0.size)
            result = least_squares(
                objective,
                theta0,
                xtol=cfg.tolerance,
                ftol=cfg.tolerance,
                gtol=cfg.tolerance,
                max_nfev=cfg.max_iterations,
            )
            if not result.success:
                continue
            model_vols = market_vols + result.fun
            rmse = float(np.sqrt(np.mean(result.fun**2)))
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
            model_vols=best_result[1],
        )

    def _initial_guess(
        self,
        forward: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        warm_start: Optional[np.ndarray],
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
            return np.array([
                np.log(params["alpha"]),
                np.arctanh(params["rho"]),
                np.log(params["nu"]),
                beta_theta,
            ], dtype=float)
        return np.array(
            [
                np.log(params["alpha"]),
                np.arctanh(params["rho"]),
                np.log(params["nu"]),
            ],
            dtype=float,
        )

    def _unpack_params(self, theta: ArrayLike) -> Dict[str, float]:
        theta = np.asarray(theta, dtype=float)
        alpha = float(np.exp(theta[0]))
        rho = float(np.tanh(theta[1]))
        nu = float(np.exp(theta[2]))
        beta = self._config.beta
        if self._config.fit_beta:
            beta = float(1.0 / (1.0 + np.exp(-theta[3])))
        beta = float(np.clip(beta, 0.0, 1.0))
        return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu}

    def _format_params(self, theta: ArrayLike) -> Dict[str, float]:
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
    if expiry <= 0.0:
        raise ValueError("expiry must be positive")

    F = np.asarray(forward, dtype=float)
    K = np.asarray(strike, dtype=float)
    beta = float(np.clip(beta, 0.0, 1.0))
    rho = float(np.clip(rho, -0.999, 0.999))
    alpha = float(max(alpha, 1e-12))
    nu = float(max(nu, 1e-12))

    F, K = np.broadcast_arrays(F, K)
    with np.errstate(divide="ignore", invalid="ignore"):
        fk_beta = np.power(F * K, (1.0 - beta) / 2.0)
        fk_beta = np.where(fk_beta <= 0.0, 1e-12, fk_beta)
        log_fk = np.log(np.where(K == 0.0, 1.0, F / K))
        z = (nu / alpha) * fk_beta * log_fk
        sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z**2)
        numerator = sqrt_term + z - rho
        denominator = 1.0 - rho
        x_z = np.log(np.where(numerator <= 0.0, 1.0, numerator / denominator))
        small_z = np.abs(z) < 1e-12
        z_over_x = np.where(small_z, 1.0, z / x_z)

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
        term3 = 1.0 + (
            ((one_minus_beta_sq / 24.0) * (alpha**2) / (fk_beta**2))
            + (0.25 * rho * beta * nu * alpha) / fk_beta
            + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
        ) * expiry

        implied = term1 * term2 * term3
        implied = np.where(np.isnan(implied), alpha / (F ** (1.0 - beta)), implied)
        implied = np.maximum(implied, 1e-4)
    return implied
