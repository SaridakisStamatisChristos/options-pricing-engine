"""Simplified Heston calibration using Andersen QE inspired objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import least_squares

from .boards import CleanBoard


@dataclass(slots=True)
class HestonConfig:
    seeds: tuple[int, ...] = (0, 1)
    tolerance: float = 1e-8
    max_iterations: int = 100
    tenors: Optional[List[float]] = None  # ``None`` => calibrate all


@dataclass(slots=True)
class HestonTenorResult:
    tenor: float
    params: Dict[str, float]
    rmse: float
    strikes: np.ndarray
    market_vols: np.ndarray
    model_vols: np.ndarray


class HestonQECalibrator:
    """Lightweight Heston calibration for cross-checking SABR."""

    def __init__(self, config: Optional[HestonConfig] = None) -> None:
        self._config = config or HestonConfig()

    def calibrate(
        self,
        clean_board: CleanBoard,
        *,
        forward_curve: Mapping[float, float] | None = None,
    ) -> List[HestonTenorResult]:
        data = clean_board.quotes
        if data.empty:
            return []

        results: List[HestonTenorResult] = []
        tenors = set(self._config.tenors) if self._config.tenors else None

        for tenor, group in data.groupby("tenor", sort=True):
            if tenors is not None and float(tenor) not in tenors:
                continue
            if len(group) < 3:
                continue
            forward = self._resolve_forward(float(tenor), group, forward_curve)
            result = self._calibrate_single_tenor(forward, float(tenor), group)
            results.append(result)
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
        return float(np.mean(group["forward"].to_numpy(dtype=float)))

    def _calibrate_single_tenor(self, forward: float, tenor: float, group) -> HestonTenorResult:
        strikes = group["strike"].to_numpy(dtype=float)
        market_vols = group["mid_iv"].to_numpy(dtype=float)
        log_m = np.log(np.clip(strikes / forward, 1e-12, None))

        seeds = self._config.seeds or (0,)
        best_rmse = float("inf")
        best_theta: Optional[np.ndarray] = None
        best_model = np.zeros_like(market_vols)

        def objective(theta: ArrayLike) -> np.ndarray:
            params = self._unpack(theta)
            model = self._heston_proxy(log_m, tenor, params)
            return model - market_vols

        initial = np.array([np.log(np.mean(market_vols) ** 2), 0.1, 0.5, 0.0], dtype=float)

        for seed in seeds:
            theta0 = initial.copy()
            if seed != 0:
                rng = np.random.default_rng(seed)
                theta0 += rng.normal(scale=0.25, size=theta0.size)
            res = least_squares(
                objective,
                theta0,
                xtol=self._config.tolerance,
                ftol=self._config.tolerance,
                gtol=self._config.tolerance,
                max_nfev=self._config.max_iterations,
            )
            if not res.success:
                continue
            rmse = float(np.sqrt(np.mean(res.fun**2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_theta = res.x.copy()
                best_model = market_vols + res.fun

        if best_theta is None:
            raise RuntimeError(f"Heston calibration failed for tenor {tenor}")

        params = self._format(best_theta)
        return HestonTenorResult(
            tenor=float(tenor),
            params=params,
            rmse=best_rmse,
            strikes=strikes,
            market_vols=market_vols,
            model_vols=best_model,
        )

    def _unpack(self, theta: ArrayLike) -> Dict[str, float]:
        theta = np.asarray(theta, dtype=float)
        v0 = float(np.exp(theta[0]))
        kappa = float(np.exp(theta[1]))
        sigma = float(np.exp(theta[2]))
        rho = float(np.tanh(theta[3]))
        return {"v0": v0, "kappa": kappa, "sigma": sigma, "rho": rho}

    def _format(self, theta: ArrayLike) -> Dict[str, float]:
        params = self._unpack(theta)
        return {key: float(value) for key, value in params.items()}

    def _heston_proxy(self, log_m: np.ndarray, tenor: float, params: Mapping[str, float]) -> np.ndarray:
        v0 = max(params["v0"], 1e-6)
        kappa = max(params["kappa"], 1e-6)
        sigma = max(params["sigma"], 1e-6)
        rho = np.clip(params["rho"], -0.999, 0.999)
        # Use a quadratic proxy in log-moneyness inspired by Andersen QE moments.
        a = np.sqrt(v0)
        b = sigma / (1.0 + kappa * tenor)
        c = rho * sigma * np.sqrt(tenor)
        vols = np.sqrt(np.maximum((a + b * log_m) ** 2 + c * log_m**2, 1e-8))
        return vols
