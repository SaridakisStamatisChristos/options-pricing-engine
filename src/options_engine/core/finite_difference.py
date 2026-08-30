"""Finite-difference valuation for European and American vanilla options.

The solver uses a uniform spot grid, Crank-Nicolson time stepping, two
Rannacher half-steps after the non-smooth terminal payoff, and projected SOR
for the American linear-complementarity problem. It is intentionally
independent of the analytic, tree, and Monte Carlo implementations so it can
serve as a numerical cross-check rather than another view of the same method.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar

import numpy as np

from ..utils.validation import validate_pricing_parameters
from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult

LOGGER = logging.getLogger(__name__)


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalised = int(value)
    if not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return normalised


def _bounded_float(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalised = float(value)
    if not math.isfinite(normalised) or not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum:g}, {maximum:g}]")
    return normalised


@dataclass(frozen=True, slots=True)
class FiniteDifferenceDiagnostics:
    """Convergence and grid metadata produced by the PDE solver."""

    space_steps: int
    time_steps: int
    s_max: float
    spot_step: float
    time_step: float
    rannacher_half_steps: int
    exercise_solver: str
    psor_converged: bool
    psor_max_iterations_used: int
    psor_total_iterations: int
    psor_max_update: float
    lower_bound: float
    upper_bound: float
    projection_applied: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "space_steps": self.space_steps,
            "time_steps": self.time_steps,
            "s_max": self.s_max,
            "spot_step": self.spot_step,
            "time_step": self.time_step,
            "rannacher_half_steps": self.rannacher_half_steps,
            "exercise_solver": self.exercise_solver,
            "psor_converged": self.psor_converged,
            "psor_max_iterations_used": self.psor_max_iterations_used,
            "psor_total_iterations": self.psor_total_iterations,
            "psor_max_update": self.psor_max_update,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "projection_applied": self.projection_applied,
        }


@dataclass(slots=True)
class FiniteDifferenceAnalysis:
    """Pricing result together with the final spatial solution."""

    pricing_result: PricingResult
    diagnostics: FiniteDifferenceDiagnostics
    spot_grid: np.ndarray
    value_grid: np.ndarray


@dataclass(slots=True)
class FiniteDifferenceModel:
    """Crank-Nicolson/Rannacher vanilla option model with PSOR exercise."""

    MAX_SPACE_STEPS: ClassVar[int] = 5_000
    MAX_TIME_STEPS: ClassVar[int] = 10_000
    MAX_WORK_ITEMS: ClassVar[int] = 5_000_000

    space_steps: int = 400
    time_steps: int = 400
    tail_standard_deviations: float = 3.5
    rannacher_smoothing: bool = True
    psor_omega: float = 1.2
    psor_tolerance: float = 1e-10
    psor_max_iterations: int = 10_000

    def __post_init__(self) -> None:
        self.space_steps = _bounded_integer(
            "space_steps", self.space_steps, minimum=20, maximum=self.MAX_SPACE_STEPS
        )
        self.time_steps = _bounded_integer(
            "time_steps", self.time_steps, minimum=10, maximum=self.MAX_TIME_STEPS
        )
        if self.space_steps * self.time_steps > self.MAX_WORK_ITEMS:
            raise ValueError(
                f"finite-difference workload exceeds {self.MAX_WORK_ITEMS} grid updates"
            )
        self.tail_standard_deviations = _bounded_float(
            "tail_standard_deviations",
            self.tail_standard_deviations,
            minimum=3.0,
            maximum=12.0,
        )
        if not isinstance(self.rannacher_smoothing, bool):
            raise TypeError("rannacher_smoothing must be a boolean")
        self.psor_omega = _bounded_float("psor_omega", self.psor_omega, minimum=1.0, maximum=1.99)
        self.psor_tolerance = _bounded_float(
            "psor_tolerance", self.psor_tolerance, minimum=1e-14, maximum=1e-2
        )
        self.psor_max_iterations = _bounded_integer(
            "psor_max_iterations",
            self.psor_max_iterations,
            minimum=10,
            maximum=1_000_000,
        )

    @staticmethod
    def _intrinsic(contract: OptionContract, spots: np.ndarray) -> np.ndarray:
        if contract.option_type is OptionType.CALL:
            return np.maximum(spots - contract.strike_price, 0.0)
        return np.maximum(contract.strike_price - spots, 0.0)

    @staticmethod
    def _boundaries(
        contract: OptionContract,
        market_data: MarketData,
        s_max: float,
        tau: float,
    ) -> tuple[float, float]:
        rate = market_data.risk_free_rate
        dividend = market_data.dividend_yield
        strike = contract.strike_price
        if contract.option_type is OptionType.CALL:
            lower = 0.0
            european_upper = max(
                s_max * math.exp(-dividend * tau) - strike * math.exp(-rate * tau),
                0.0,
            )
            upper = (
                max(s_max - strike, european_upper)
                if contract.exercise_style is ExerciseStyle.AMERICAN
                else european_upper
            )
            return lower, upper

        upper = 0.0
        european_lower = strike * math.exp(-rate * tau)
        lower = (
            max(strike, european_lower)
            if contract.exercise_style is ExerciseStyle.AMERICAN
            else european_lower
        )
        return lower, upper

    @staticmethod
    def _solve_tridiagonal(
        lower: np.ndarray,
        diagonal: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
    ) -> np.ndarray:
        """Solve a finite tridiagonal system using the Thomas algorithm."""

        size = diagonal.size
        if size == 0:
            return np.empty(0, dtype=float)
        modified_upper = np.empty(max(size - 1, 0), dtype=float)
        modified_rhs = np.empty(size, dtype=float)
        pivot = float(diagonal[0])
        if not math.isfinite(pivot) or abs(pivot) < 1e-14:
            raise ValueError("finite-difference system has a singular leading pivot")
        if size > 1:
            modified_upper[0] = upper[0] / pivot
        modified_rhs[0] = rhs[0] / pivot
        for index in range(1, size):
            pivot = float(diagonal[index] - lower[index - 1] * modified_upper[index - 1])
            if not math.isfinite(pivot) or abs(pivot) < 1e-14:
                raise ValueError("finite-difference system has a singular pivot")
            if index < size - 1:
                modified_upper[index] = upper[index] / pivot
            modified_rhs[index] = (rhs[index] - lower[index - 1] * modified_rhs[index - 1]) / pivot

        solution = np.empty(size, dtype=float)
        solution[-1] = modified_rhs[-1]
        for index in range(size - 2, -1, -1):
            solution[index] = modified_rhs[index] - modified_upper[index] * solution[index + 1]
        if not np.isfinite(solution).all():
            raise ValueError("finite-difference linear solve produced non-finite values")
        return solution

    def _solve_psor(
        self,
        lower: np.ndarray,
        diagonal: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
        obstacle: np.ndarray,
        initial: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        """Solve the American LCP using projected successive over-relaxation."""

        solution = np.maximum(np.asarray(initial, dtype=float), obstacle).copy()
        max_update = float("inf")
        for iteration in range(1, self.psor_max_iterations + 1):
            max_update = 0.0
            for index in range(solution.size):
                left = lower[index - 1] * solution[index - 1] if index else 0.0
                right = upper[index] * solution[index + 1] if index < solution.size - 1 else 0.0
                gauss_seidel = (rhs[index] - left - right) / diagonal[index]
                candidate = solution[index] + self.psor_omega * (gauss_seidel - solution[index])
                projected = max(float(obstacle[index]), float(candidate))
                max_update = max(max_update, float(abs(projected - solution[index])))
                solution[index] = projected
            scale = max(1.0, float(np.max(np.abs(solution))))
            if max_update <= self.psor_tolerance * scale:
                return solution, iteration, max_update
        raise RuntimeError(
            "PSOR failed to converge within "
            f"{self.psor_max_iterations} iterations (last update={max_update:.3e})"
        )

    def _advance(
        self,
        values: np.ndarray,
        intrinsic: np.ndarray,
        contract: OptionContract,
        market_data: MarketData,
        *,
        s_max: float,
        tau_old: float,
        step_size: float,
        theta: float,
        volatility: float,
    ) -> tuple[np.ndarray, int, float]:
        """Advance the solution by one theta-scheme time step."""

        interior_count = self.space_steps - 1
        indices = np.arange(1, self.space_steps, dtype=float)
        sigma = volatility
        carry = market_data.risk_free_rate - market_data.dividend_yield
        a = 0.5 * (sigma**2 * indices**2 - carry * indices)
        b = -(sigma**2 * indices**2 + market_data.risk_free_rate)
        c = 0.5 * (sigma**2 * indices**2 + carry * indices)

        diagonal = 1.0 - theta * step_size * b
        lower = -theta * step_size * a[1:]
        upper = -theta * step_size * c[:-1]

        rhs = (1.0 + (1.0 - theta) * step_size * b) * values[1:-1]
        rhs[1:] += (1.0 - theta) * step_size * a[1:] * values[1:-2]
        rhs[:-1] += (1.0 - theta) * step_size * c[:-1] * values[2:-1]

        old_lower, old_upper = self._boundaries(contract, market_data, s_max, tau_old)
        tau_new = tau_old + step_size
        new_lower, new_upper = self._boundaries(contract, market_data, s_max, tau_new)
        rhs[0] += (1.0 - theta) * step_size * a[0] * old_lower
        rhs[-1] += (1.0 - theta) * step_size * c[-1] * old_upper
        rhs[0] += theta * step_size * a[0] * new_lower
        rhs[-1] += theta * step_size * c[-1] * new_upper

        if not (
            diagonal.size == interior_count
            and lower.size == interior_count - 1
            and upper.size == interior_count - 1
        ):
            raise RuntimeError("finite-difference matrix shape invariant failed")

        if contract.exercise_style is ExerciseStyle.AMERICAN:
            interior, iterations, max_update = self._solve_psor(
                lower,
                diagonal,
                upper,
                rhs,
                intrinsic[1:-1],
                values[1:-1],
            )
        else:
            interior = self._solve_tridiagonal(lower, diagonal, upper, rhs)
            iterations = 0
            max_update = 0.0

        advanced = np.empty_like(values)
        advanced[0] = new_lower
        advanced[-1] = new_upper
        advanced[1:-1] = interior
        if contract.exercise_style is ExerciseStyle.AMERICAN:
            advanced = np.maximum(advanced, intrinsic)
        if not np.isfinite(advanced).all():
            raise ValueError("finite-difference step produced non-finite values")
        return advanced, iterations, max_update

    def price_with_diagnostics(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> FiniteDifferenceAnalysis:
        """Price a vanilla contract and return the full numerical audit trail."""

        validate_pricing_parameters(contract, market_data, volatility)
        start = time.perf_counter()

        maturity = contract.time_to_expiry
        drift_tail = max(
            (market_data.risk_free_rate - market_data.dividend_yield - 0.5 * volatility**2)
            * maturity,
            0.0,
        )
        log_tail = min(
            drift_tail + self.tail_standard_deviations * volatility * math.sqrt(maturity),
            250.0,
        )
        s_max = max(
            4.0 * contract.strike_price,
            2.0 * market_data.spot_price,
            market_data.spot_price * math.exp(log_tail),
        )
        spot_grid = np.linspace(0.0, s_max, self.space_steps + 1)
        spot_step = s_max / self.space_steps
        time_step = maturity / self.time_steps
        intrinsic = self._intrinsic(contract, spot_grid)
        values = intrinsic.copy()
        lower_zero, upper_zero = self._boundaries(contract, market_data, s_max, 0.0)
        values[0] = lower_zero
        values[-1] = upper_zero

        tau = 0.0
        psor_max_iterations_used = 0
        psor_total_iterations = 0
        psor_max_update = 0.0
        layer_before_final = values.copy()

        for time_index in range(self.time_steps):
            if time_index == self.time_steps - 1:
                layer_before_final = values.copy()
            substeps: tuple[tuple[float, float], ...]
            if time_index == 0 and self.rannacher_smoothing:
                substeps = ((0.5 * time_step, 1.0), (0.5 * time_step, 1.0))
            else:
                substeps = ((time_step, 0.5),)
            for step_size, theta in substeps:
                values, iterations, max_update = self._advance(
                    values,
                    intrinsic,
                    contract,
                    market_data,
                    s_max=s_max,
                    tau_old=tau,
                    step_size=step_size,
                    theta=theta,
                    volatility=volatility,
                )
                tau += step_size
                psor_max_iterations_used = max(psor_max_iterations_used, iterations)
                psor_total_iterations += iterations
                psor_max_update = max(psor_max_update, max_update)

        raw_price = float(np.interp(market_data.spot_price, spot_grid, values))
        intrinsic_now = float(self._intrinsic(contract, np.array([market_data.spot_price]))[0])
        lower_bound = intrinsic_now if contract.exercise_style is ExerciseStyle.AMERICAN else 0.0
        upper_bound = (
            market_data.spot_price * math.exp(max(-market_data.dividend_yield * maturity, 0.0))
            if contract.option_type is OptionType.CALL
            else contract.strike_price * math.exp(max(-market_data.risk_free_rate * maturity, 0.0))
        )
        price = min(max(raw_price, lower_bound), upper_bound)

        delta_grid = np.gradient(values, spot_step, edge_order=2)
        gamma_grid = np.gradient(delta_grid, spot_step, edge_order=2)
        delta = float(np.interp(market_data.spot_price, spot_grid, delta_grid))
        gamma = float(np.interp(market_data.spot_price, spot_grid, gamma_grid))
        previous_price = float(np.interp(market_data.spot_price, spot_grid, layer_before_final))
        theta = (previous_price - raw_price) / time_step / 365.0

        diagnostics = FiniteDifferenceDiagnostics(
            space_steps=self.space_steps,
            time_steps=self.time_steps,
            s_max=s_max,
            spot_step=spot_step,
            time_step=time_step,
            rannacher_half_steps=2 if self.rannacher_smoothing else 0,
            exercise_solver=(
                "psor" if contract.exercise_style is ExerciseStyle.AMERICAN else "thomas"
            ),
            psor_converged=True,
            psor_max_iterations_used=psor_max_iterations_used,
            psor_total_iterations=psor_total_iterations,
            psor_max_update=psor_max_update,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            projection_applied=price != raw_price,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        result = PricingResult(
            contract_id=contract.contract_id,
            theoretical_price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            implied_volatility=volatility,
            computation_time_ms=elapsed_ms,
            model_used=f"finite_difference_cn_{self.space_steps}x{self.time_steps}",
            numerical_diagnostics=diagnostics.to_dict(),
        )
        return FiniteDifferenceAnalysis(
            pricing_result=result,
            diagnostics=diagnostics,
            spot_grid=spot_grid,
            value_grid=values,
        )

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> PricingResult:
        """Return a :class:`PricingResult` for the finite-difference model."""

        try:
            return self.price_with_diagnostics(contract, market_data, volatility).pricing_result
        except Exception:  # pragma: no cover - preserve context for API error mapping
            LOGGER.exception("Finite-difference pricing failed")
            raise


__all__ = [
    "FiniteDifferenceAnalysis",
    "FiniteDifferenceDiagnostics",
    "FiniteDifferenceModel",
]
