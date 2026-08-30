"""Finite-difference valuation for European and American vanilla options.

The solver uses Crank--Nicolson time stepping with optional Rannacher
smoothing. Its default sinh-transformed spot mesh contains both spot and
strike exactly and concentrates points around them; a uniform mesh remains
available for reproducibility. American exercise can be solved with either
projected SOR or an independent active-set penalty method.

Refinement studies always reuse the same truncated domain. Richardson-style
error estimates are reported only when the formal order is justified for a
Rannacher-smoothed European solve, or when three grids provide a credible
observed order for an American/free-boundary solve. The extrapolated value is
diagnostic: the returned price is the directly computed finest-grid value.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar, Literal

import numpy as np

from ..utils.validation import validate_pricing_parameters
from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult

LOGGER = logging.getLogger(__name__)

GridType = Literal["sinh", "uniform"]
ExerciseSolver = Literal["psor", "penalty"]


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
    """Convergence, solver, grid, and boundary metadata for one PDE price."""

    # Existing fields are retained for consumers of the v2.1.0 diagnostics.
    space_steps: int
    time_steps: int
    s_max: float
    spot_step: float
    time_step: float
    rannacher_half_steps: int
    exercise_solver: str
    psor_converged: bool | None
    psor_max_iterations_used: int
    psor_total_iterations: int
    psor_max_update: float
    lower_bound: float
    upper_bound: float
    projection_applied: bool

    # Explicit numerical-error and solver diagnostics.
    grid_type: str = "uniform"
    min_spot_step: float = 0.0
    max_spot_step: float = 0.0
    spot_on_grid: bool = False
    strike_on_grid: bool = False
    configured_space_steps: int = 0
    configured_time_steps: int = 0
    refinement_levels: int = 1
    refinement_ratio: int = 2
    level_space_steps: tuple[int, ...] = ()
    level_time_steps: tuple[int, ...] = ()
    level_prices: tuple[float, ...] = ()
    level_differences: tuple[float, ...] = ()
    level_solver_max_iterations: tuple[int, ...] = ()
    level_solver_total_iterations: tuple[int, ...] = ()
    level_lcp_residuals: tuple[float, ...] = ()
    level_penalty_obstacle_violations: tuple[float, ...] = ()
    level_penalty_equation_residuals: tuple[float, ...] = ()
    level_projection_applied: tuple[bool, ...] = ()
    observed_order: float | None = None
    richardson_error_estimate: float | None = None
    richardson_extrapolated_price: float | None = None
    error_estimate_method: str = "not_requested"
    solver_converged: bool = True
    solver_max_iterations_used: int = 0
    solver_total_iterations: int = 0
    solver_max_update: float = 0.0
    lcp_residual: float = 0.0
    penalty_converged: bool | None = None
    penalty_max_iterations_used: int = 0
    penalty_total_iterations: int = 0
    penalty_max_update: float = 0.0
    penalty_parameter: float | None = None
    penalty_obstacle_violation: float = 0.0
    penalty_equation_residual: float = 0.0
    lower_boundary_delta_residual: float = 0.0
    upper_boundary_delta_residual: float = 0.0
    lognormal_upper_tail_probability: float = 0.0

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
            "grid_type": self.grid_type,
            "min_spot_step": self.min_spot_step,
            "max_spot_step": self.max_spot_step,
            "spot_on_grid": self.spot_on_grid,
            "strike_on_grid": self.strike_on_grid,
            "configured_space_steps": self.configured_space_steps,
            "configured_time_steps": self.configured_time_steps,
            "refinement_levels": self.refinement_levels,
            "refinement_ratio": self.refinement_ratio,
            "level_space_steps": self.level_space_steps,
            "level_time_steps": self.level_time_steps,
            "level_prices": self.level_prices,
            "level_differences": self.level_differences,
            "level_solver_max_iterations": self.level_solver_max_iterations,
            "level_solver_total_iterations": self.level_solver_total_iterations,
            "level_lcp_residuals": self.level_lcp_residuals,
            "level_penalty_obstacle_violations": (self.level_penalty_obstacle_violations),
            "level_penalty_equation_residuals": self.level_penalty_equation_residuals,
            "level_projection_applied": self.level_projection_applied,
            "observed_order": self.observed_order,
            "richardson_error_estimate": self.richardson_error_estimate,
            "richardson_extrapolated_price": self.richardson_extrapolated_price,
            "error_estimate_method": self.error_estimate_method,
            "solver_converged": self.solver_converged,
            "solver_max_iterations_used": self.solver_max_iterations_used,
            "solver_total_iterations": self.solver_total_iterations,
            "solver_max_update": self.solver_max_update,
            "lcp_residual": self.lcp_residual,
            "penalty_converged": self.penalty_converged,
            "penalty_max_iterations_used": self.penalty_max_iterations_used,
            "penalty_total_iterations": self.penalty_total_iterations,
            "penalty_max_update": self.penalty_max_update,
            "penalty_parameter": self.penalty_parameter,
            "penalty_obstacle_violation": self.penalty_obstacle_violation,
            "penalty_equation_residual": self.penalty_equation_residual,
            "lower_boundary_delta_residual": self.lower_boundary_delta_residual,
            "upper_boundary_delta_residual": self.upper_boundary_delta_residual,
            "lognormal_upper_tail_probability": self.lognormal_upper_tail_probability,
        }


@dataclass(slots=True)
class FiniteDifferenceAnalysis:
    """Pricing result together with the finest spatial solution."""

    pricing_result: PricingResult
    diagnostics: FiniteDifferenceDiagnostics
    spot_grid: np.ndarray
    value_grid: np.ndarray


@dataclass(slots=True)
class _StepStats:
    iterations: int = 0
    max_update: float = 0.0
    lcp_residual: float = 0.0
    obstacle_violation: float = 0.0
    equation_residual: float = 0.0


@dataclass(slots=True)
class _GridSolution:
    space_steps: int
    time_steps: int
    spot_grid: np.ndarray
    values: np.ndarray
    raw_price: float
    price: float
    delta: float
    gamma: float
    theta: float
    lower_bound: float
    upper_bound: float
    projection_applied: bool
    max_iterations_used: int
    total_iterations: int
    max_update: float
    lcp_residual: float
    obstacle_violation: float
    equation_residual: float
    lower_boundary_delta_residual: float
    upper_boundary_delta_residual: float


@dataclass(slots=True)
class FiniteDifferenceModel:
    """Crank--Nicolson/Rannacher vanilla option model with two LCP solvers."""

    MAX_SPACE_STEPS: ClassVar[int] = 5_000
    MAX_TIME_STEPS: ClassVar[int] = 10_000
    MAX_WORK_ITEMS: ClassVar[int] = 5_000_000

    space_steps: int = 400
    time_steps: int = 400
    tail_standard_deviations: float = 3.5
    rannacher_smoothing: bool = True
    # Keep the v2.1.0 positional argument order through psor_max_iterations.
    psor_omega: float = 1.2
    psor_tolerance: float = 1e-10
    psor_max_iterations: int = 10_000
    grid_type: GridType = "sinh"
    grid_concentration: float = 0.1
    s_max_override: float | None = None
    refinement_levels: int = 1
    refinement_ratio: int = 2
    exercise_solver: ExerciseSolver = "psor"
    penalty_parameter: float = 1e7
    penalty_tolerance: float = 1e-10
    penalty_max_iterations: int = 100

    def __post_init__(self) -> None:
        self.space_steps = _bounded_integer(
            "space_steps", self.space_steps, minimum=20, maximum=self.MAX_SPACE_STEPS
        )
        self.time_steps = _bounded_integer(
            "time_steps", self.time_steps, minimum=10, maximum=self.MAX_TIME_STEPS
        )
        self.refinement_levels = _bounded_integer(
            "refinement_levels", self.refinement_levels, minimum=1, maximum=4
        )
        self.refinement_ratio = _bounded_integer(
            "refinement_ratio", self.refinement_ratio, minimum=2, maximum=4
        )
        finest_factor = self.refinement_ratio ** (self.refinement_levels - 1)
        finest_space_steps = self.space_steps * finest_factor
        finest_time_steps = self.time_steps * finest_factor
        if finest_space_steps > self.MAX_SPACE_STEPS or finest_time_steps > self.MAX_TIME_STEPS:
            raise ValueError("refined finite-difference grid exceeds the configured step limits")
        requested_work_items = sum(
            self.space_steps * self.time_steps * self.refinement_ratio ** (2 * level)
            for level in range(self.refinement_levels)
        )
        if requested_work_items > self.MAX_WORK_ITEMS:
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
        if self.grid_type not in {"sinh", "uniform"}:
            raise ValueError("grid_type must be 'sinh' or 'uniform'")
        self.grid_concentration = _bounded_float(
            "grid_concentration", self.grid_concentration, minimum=0.01, maximum=2.0
        )
        if self.s_max_override is not None:
            self.s_max_override = _bounded_float(
                "s_max_override", self.s_max_override, minimum=1e-8, maximum=1e150
            )
        if self.exercise_solver not in {"psor", "penalty"}:
            raise ValueError("exercise_solver must be 'psor' or 'penalty'")
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
        self.penalty_parameter = _bounded_float(
            "penalty_parameter", self.penalty_parameter, minimum=1e3, maximum=1e10
        )
        self.penalty_tolerance = _bounded_float(
            "penalty_tolerance", self.penalty_tolerance, minimum=1e-14, maximum=1e-2
        )
        self.penalty_max_iterations = _bounded_integer(
            "penalty_max_iterations",
            self.penalty_max_iterations,
            minimum=2,
            maximum=10_000,
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
        """Return asymptotically exact Dirichlet values at zero and ``s_max``."""

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

    def _resolve_s_max(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> float:
        maturity = contract.time_to_expiry
        if self.s_max_override is not None:
            minimum_domain = max(contract.strike_price, market_data.spot_price)
            if self.s_max_override <= minimum_domain:
                raise ValueError("s_max_override must exceed both spot and strike")
            return self.s_max_override

        drift_tail = max(
            (market_data.risk_free_rate - market_data.dividend_yield - 0.5 * volatility**2)
            * maturity,
            0.0,
        )
        log_tail = min(
            drift_tail + self.tail_standard_deviations * volatility * math.sqrt(maturity),
            250.0,
        )
        return max(
            4.0 * contract.strike_price,
            2.0 * market_data.spot_price,
            market_data.spot_price * math.exp(log_tail),
        )

    @staticmethod
    def _allocate_interval_steps(spans: np.ndarray, total_steps: int) -> np.ndarray:
        """Allocate a fixed number of mesh intervals across transformed spans."""

        interval_count = int(spans.size)
        minimum_per_interval = 2
        if total_steps < minimum_per_interval * interval_count:
            raise ValueError("too few space steps for the anchored transformed grid")
        remaining = total_steps - minimum_per_interval * interval_count
        raw = spans / float(np.sum(spans)) * remaining
        extra = np.floor(raw).astype(int)
        leftover = remaining - int(np.sum(extra))
        if leftover:
            fractions = raw - extra
            order = np.argsort(-fractions, kind="stable")
            extra[order[:leftover]] += 1
        return extra + minimum_per_interval

    def _spot_grid(
        self,
        contract: OptionContract,
        market_data: MarketData,
        s_max: float,
        space_steps: int,
    ) -> np.ndarray:
        if self.grid_type == "uniform":
            return np.linspace(0.0, s_max, space_steps + 1)

        spot = market_data.spot_price
        strike = contract.strike_price
        focus = 0.5 * (spot + strike)
        scale = max(
            self.grid_concentration * max(spot, strike),
            0.25 * abs(spot - strike),
            np.finfo(float).eps * max(spot, strike),
        )

        # Exact anchors eliminate interpolation error at spot and align the
        # payoff kink with the mesh. Equal spot/strike values are deduplicated.
        anchors = np.array(sorted({0.0, float(spot), float(strike), float(s_max)}))
        transformed = np.arcsinh((anchors - focus) / scale)
        spans = np.diff(transformed)
        interval_steps = self._allocate_interval_steps(spans, space_steps)

        pieces: list[np.ndarray] = []
        for index, steps in enumerate(interval_steps):
            transformed_piece = np.linspace(
                transformed[index], transformed[index + 1], int(steps) + 1
            )
            piece = focus + scale * np.sinh(transformed_piece)
            piece[0] = anchors[index]
            piece[-1] = anchors[index + 1]
            pieces.append(piece if index == 0 else piece[1:])
        grid = np.concatenate(pieces)
        if grid.size != space_steps + 1 or not np.all(np.diff(grid) > 0.0):
            raise RuntimeError("finite-difference grid construction invariant failed")
        return grid

    @staticmethod
    def _operator_coefficients(
        spot_grid: np.ndarray,
        market_data: MarketData,
        volatility: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the Black--Scholes operator on an unequal-spacing mesh."""

        interior_spots = spot_grid[1:-1]
        left_steps = interior_spots - spot_grid[:-2]
        right_steps = spot_grid[2:] - interior_spots
        total_steps = left_steps + right_steps

        first_left = -right_steps / (left_steps * total_steps)
        first_diagonal = (right_steps - left_steps) / (left_steps * right_steps)
        first_right = left_steps / (right_steps * total_steps)
        second_left = 2.0 / (left_steps * total_steps)
        second_diagonal = -2.0 / (left_steps * right_steps)
        second_right = 2.0 / (right_steps * total_steps)

        diffusion = 0.5 * volatility**2 * interior_spots**2
        drift = (market_data.risk_free_rate - market_data.dividend_yield) * interior_spots
        operator_lower = diffusion * second_left + drift * first_left
        operator_diagonal = (
            diffusion * second_diagonal + drift * first_diagonal - market_data.risk_free_rate
        )
        operator_upper = diffusion * second_right + drift * first_right
        return operator_lower, operator_diagonal, operator_upper

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

    @staticmethod
    def _tridiagonal_product(
        lower: np.ndarray,
        diagonal: np.ndarray,
        upper: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        product = np.asarray(diagonal * values, dtype=float)
        product[1:] += lower * values[:-1]
        product[:-1] += upper * values[1:]
        return product

    @classmethod
    def _lcp_residual(
        cls,
        lower: np.ndarray,
        diagonal: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
        obstacle: np.ndarray,
        solution: np.ndarray,
    ) -> float:
        continuation_residual = cls._tridiagonal_product(lower, diagonal, upper, solution) - rhs
        exercise_gap = solution - obstacle
        scale = max(
            1.0,
            float(np.max(np.abs(rhs))),
            float(np.max(np.abs(solution))),
        )
        primal_violation = float(np.max(np.maximum(-exercise_gap, 0.0)))
        dual_violation = float(np.max(np.maximum(-continuation_residual, 0.0)))
        complementarity = float(np.max(np.abs(np.minimum(exercise_gap, continuation_residual))))
        return max(primal_violation, dual_violation, complementarity) / scale

    def _solve_psor(
        self,
        lower: np.ndarray,
        diagonal: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
        obstacle: np.ndarray,
        initial: np.ndarray,
    ) -> tuple[np.ndarray, _StepStats]:
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
                residual = self._lcp_residual(lower, diagonal, upper, rhs, obstacle, solution)
                if residual <= self.psor_tolerance:
                    return solution, _StepStats(
                        iterations=iteration,
                        max_update=max_update,
                        lcp_residual=residual,
                        equation_residual=residual,
                    )
        raise RuntimeError(
            "PSOR failed to converge within "
            f"{self.psor_max_iterations} iterations (last update={max_update:.3e})"
        )

    def _solve_penalty(
        self,
        lower: np.ndarray,
        diagonal: np.ndarray,
        upper: np.ndarray,
        rhs: np.ndarray,
        obstacle: np.ndarray,
    ) -> tuple[np.ndarray, _StepStats]:
        """Solve the American LCP with an active-set penalty iteration.

        This solves ``A v - b - lambda * max(phi - v, 0) = 0``. It uses only
        tridiagonal direct solves and is independent of PSOR. The final obstacle
        projection removes the bounded ``O(1/lambda)`` primal violation; the
        LCP residual reports the remaining penalty bias after that projection.
        """

        def penalised_residual(candidate: np.ndarray) -> np.ndarray:
            return np.asarray(
                self._tridiagonal_product(lower, diagonal, upper, candidate)
                - rhs
                - self.penalty_parameter * np.maximum(obstacle - candidate, 0.0),
                dtype=float,
            )

        solution = self._solve_tridiagonal(lower, diagonal, upper, rhs)
        max_update = float("inf")
        for iteration in range(1, self.penalty_max_iterations + 1):
            active = solution < obstacle
            penalised_diagonal = diagonal + self.penalty_parameter * active
            penalised_rhs = rhs + self.penalty_parameter * active * obstacle
            newton_candidate = self._solve_tridiagonal(
                lower, penalised_diagonal, upper, penalised_rhs
            )
            current_norm = float(np.max(np.abs(penalised_residual(solution))))
            direction = newton_candidate - solution

            # Semismooth Newton is usually finite-step for this piecewise-linear
            # equation. A backtracking merit search prevents an active-set
            # two-cycle in difficult short-dated/high-volatility regimes.
            step_length = 1.0
            updated = newton_candidate
            updated_norm = float(np.max(np.abs(penalised_residual(updated))))
            while (
                updated_norm > (1.0 - 1e-4 * step_length) * current_norm and step_length > 2.0**-24
            ):
                step_length *= 0.5
                updated = solution + step_length * direction
                updated_norm = float(np.max(np.abs(penalised_residual(updated))))

            max_update = float(np.max(np.abs(updated - solution)))
            scale = max(1.0, float(np.max(np.abs(updated))))
            updated_active = updated < obstacle
            active_stable = np.array_equal(updated_active, active)
            solution = updated
            residual_converged = updated_norm <= self.penalty_tolerance * scale
            update_converged = max_update <= self.penalty_tolerance * scale
            if residual_converged or (update_converged and active_stable):
                obstacle_violation = float(np.max(np.maximum(obstacle - solution, 0.0)))
                projected = np.maximum(solution, obstacle)
                residual = self._lcp_residual(lower, diagonal, upper, rhs, obstacle, projected)
                roundoff_lcp_limit = max(
                    100.0 * self.penalty_tolerance,
                    0.1 / self.penalty_parameter,
                )
                if residual_converged or residual <= roundoff_lcp_limit:
                    return projected, _StepStats(
                        iterations=iteration,
                        max_update=max_update,
                        lcp_residual=residual,
                        obstacle_violation=obstacle_violation,
                        equation_residual=updated_norm / scale,
                    )
        raise RuntimeError(
            "penalty iteration failed to converge within "
            f"{self.penalty_max_iterations} iterations (last update={max_update:.3e})"
        )

    def _advance(
        self,
        values: np.ndarray,
        intrinsic: np.ndarray,
        spot_grid: np.ndarray,
        contract: OptionContract,
        market_data: MarketData,
        *,
        s_max: float,
        tau_old: float,
        step_size: float,
        theta: float,
        volatility: float,
    ) -> tuple[np.ndarray, _StepStats]:
        """Advance the solution by one theta-scheme time step."""

        operator_lower, operator_diagonal, operator_upper = self._operator_coefficients(
            spot_grid, market_data, volatility
        )
        interior_count = spot_grid.size - 2
        diagonal = 1.0 - theta * step_size * operator_diagonal
        lower = -theta * step_size * operator_lower[1:]
        upper = -theta * step_size * operator_upper[:-1]

        rhs = (1.0 + (1.0 - theta) * step_size * operator_diagonal) * values[1:-1]
        rhs += (1.0 - theta) * step_size * operator_lower * values[:-2]
        rhs += (1.0 - theta) * step_size * operator_upper * values[2:]

        tau_new = tau_old + step_size
        new_lower, new_upper = self._boundaries(contract, market_data, s_max, tau_new)
        rhs[0] += theta * step_size * operator_lower[0] * new_lower
        rhs[-1] += theta * step_size * operator_upper[-1] * new_upper

        if not (
            diagonal.size == interior_count
            and lower.size == interior_count - 1
            and upper.size == interior_count - 1
        ):
            raise RuntimeError("finite-difference matrix shape invariant failed")

        if contract.exercise_style is ExerciseStyle.AMERICAN:
            if self.exercise_solver == "psor":
                interior, step_stats = self._solve_psor(
                    lower,
                    diagonal,
                    upper,
                    rhs,
                    intrinsic[1:-1],
                    values[1:-1],
                )
            else:
                interior, step_stats = self._solve_penalty(
                    lower,
                    diagonal,
                    upper,
                    rhs,
                    intrinsic[1:-1],
                )
        else:
            interior = self._solve_tridiagonal(lower, diagonal, upper, rhs)
            step_stats = _StepStats()

        advanced = np.empty_like(values)
        advanced[0] = new_lower
        advanced[-1] = new_upper
        advanced[1:-1] = interior
        if contract.exercise_style is ExerciseStyle.AMERICAN:
            advanced = np.maximum(advanced, intrinsic)
        if not np.isfinite(advanced).all():
            raise ValueError("finite-difference step produced non-finite values")
        return advanced, step_stats

    @staticmethod
    def _price_bounds(
        contract: OptionContract,
        market_data: MarketData,
    ) -> tuple[float, float]:
        maturity = contract.time_to_expiry
        spot = market_data.spot_price
        strike = contract.strike_price
        if contract.option_type is OptionType.CALL:
            intrinsic = max(spot - strike, 0.0)
            lower = intrinsic if contract.exercise_style is ExerciseStyle.AMERICAN else 0.0
            upper = spot * math.exp(max(-market_data.dividend_yield * maturity, 0.0))
        else:
            intrinsic = max(strike - spot, 0.0)
            lower = intrinsic if contract.exercise_style is ExerciseStyle.AMERICAN else 0.0
            upper = strike * math.exp(max(-market_data.risk_free_rate * maturity, 0.0))
        return lower, upper

    @staticmethod
    def _boundary_delta_residuals(
        contract: OptionContract,
        market_data: MarketData,
        spot_grid: np.ndarray,
        values: np.ndarray,
    ) -> tuple[float, float]:
        maturity = contract.time_to_expiry
        numerical_lower = float((values[1] - values[0]) / (spot_grid[1] - spot_grid[0]))
        numerical_upper = float((values[-1] - values[-2]) / (spot_grid[-1] - spot_grid[-2]))
        discounted_delta = math.exp(-market_data.dividend_yield * maturity)
        if contract.option_type is OptionType.CALL:
            expected_lower = 0.0
            european_upper = spot_grid[-1] * discounted_delta - contract.strike_price * math.exp(
                -market_data.risk_free_rate * maturity
            )
            intrinsic_upper = spot_grid[-1] - contract.strike_price
            comparison_tolerance = 1e-12 * max(1.0, abs(intrinsic_upper), abs(european_upper))
            if (
                contract.exercise_style is ExerciseStyle.AMERICAN
                and intrinsic_upper > european_upper + comparison_tolerance
            ):
                expected_upper = 1.0
            elif abs(intrinsic_upper - european_upper) <= comparison_tolerance:
                expected_upper = min(1.0, discounted_delta)
            else:
                expected_upper = discounted_delta
        else:
            expected_upper = 0.0
            european_lower = contract.strike_price * math.exp(
                -market_data.risk_free_rate * maturity
            )
            comparison_tolerance = 1e-12 * max(1.0, contract.strike_price, abs(european_lower))
            if (
                contract.exercise_style is ExerciseStyle.AMERICAN
                and contract.strike_price > european_lower + comparison_tolerance
            ):
                expected_lower = -1.0
            elif abs(contract.strike_price - european_lower) <= comparison_tolerance:
                expected_lower = -min(1.0, discounted_delta)
            else:
                expected_lower = -discounted_delta
        return (
            abs(numerical_lower - expected_lower),
            abs(numerical_upper - expected_upper),
        )

    @staticmethod
    def _local_spot_step(spot_grid: np.ndarray, spot: float) -> float:
        index = int(np.searchsorted(spot_grid, spot))
        index = min(max(index, 1), spot_grid.size - 1)
        if spot_grid[index] == spot and index < spot_grid.size - 1:
            return float(
                0.5
                * (
                    spot_grid[index]
                    - spot_grid[index - 1]
                    + spot_grid[index + 1]
                    - spot_grid[index]
                )
            )
        return float(spot_grid[index] - spot_grid[index - 1])

    def _solve_grid(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        s_max: float,
        space_steps: int,
        time_steps: int,
    ) -> _GridSolution:
        spot_grid = self._spot_grid(contract, market_data, s_max, space_steps)
        time_step = contract.time_to_expiry / time_steps
        intrinsic = self._intrinsic(contract, spot_grid)
        values = intrinsic.copy()
        values[0], values[-1] = self._boundaries(contract, market_data, s_max, 0.0)

        tau = 0.0
        max_iterations_used = 0
        total_iterations = 0
        max_update = 0.0
        lcp_residual = 0.0
        obstacle_violation = 0.0
        equation_residual = 0.0
        layer_before_final = values.copy()

        for time_index in range(time_steps):
            if time_index == time_steps - 1:
                layer_before_final = values.copy()
            substeps: tuple[tuple[float, float], ...]
            if time_index == 0 and self.rannacher_smoothing:
                substeps = ((0.5 * time_step, 1.0), (0.5 * time_step, 1.0))
            else:
                substeps = ((time_step, 0.5),)
            for step_size, theta in substeps:
                values, step_stats = self._advance(
                    values,
                    intrinsic,
                    spot_grid,
                    contract,
                    market_data,
                    s_max=s_max,
                    tau_old=tau,
                    step_size=step_size,
                    theta=theta,
                    volatility=volatility,
                )
                tau += step_size
                max_iterations_used = max(max_iterations_used, step_stats.iterations)
                total_iterations += step_stats.iterations
                max_update = max(max_update, step_stats.max_update)
                lcp_residual = max(lcp_residual, step_stats.lcp_residual)
                obstacle_violation = max(obstacle_violation, step_stats.obstacle_violation)
                equation_residual = max(equation_residual, step_stats.equation_residual)

        raw_price = float(np.interp(market_data.spot_price, spot_grid, values))
        lower_bound, upper_bound = self._price_bounds(contract, market_data)
        price = min(max(raw_price, lower_bound), upper_bound)

        delta_grid = np.gradient(values, spot_grid, edge_order=2)
        gamma_grid = np.gradient(delta_grid, spot_grid, edge_order=2)
        delta = float(np.interp(market_data.spot_price, spot_grid, delta_grid))
        gamma = float(np.interp(market_data.spot_price, spot_grid, gamma_grid))
        previous_price = float(np.interp(market_data.spot_price, spot_grid, layer_before_final))
        theta = (previous_price - raw_price) / time_step / 365.0
        lower_delta_residual, upper_delta_residual = self._boundary_delta_residuals(
            contract, market_data, spot_grid, values
        )

        return _GridSolution(
            space_steps=space_steps,
            time_steps=time_steps,
            spot_grid=spot_grid,
            values=values,
            raw_price=raw_price,
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            projection_applied=price != raw_price,
            max_iterations_used=max_iterations_used,
            total_iterations=total_iterations,
            max_update=max_update,
            lcp_residual=lcp_residual,
            obstacle_violation=obstacle_violation,
            equation_residual=equation_residual,
            lower_boundary_delta_residual=lower_delta_residual,
            upper_boundary_delta_residual=upper_delta_residual,
        )

    def _convergence_diagnostics(
        self,
        contract: OptionContract,
        prices: tuple[float, ...],
        projections: tuple[bool, ...],
    ) -> tuple[tuple[float, ...], float | None, float | None, float | None, str]:
        differences = tuple(
            abs(prices[index] - prices[index - 1]) for index in range(1, len(prices))
        )
        if len(prices) == 1:
            return differences, None, None, None, "not_requested"

        ratio = float(self.refinement_ratio)
        observed_order: float | None = None
        consecutive_changes = (
            (prices[-2] - prices[-3]) * (prices[-1] - prices[-2]) > 0.0
            if len(prices) >= 3
            else False
        )
        if (
            len(prices) >= 3
            and not any(projections)
            and consecutive_changes
            and differences[-1] > 1e-14
            and differences[-2] > 1e-14
        ):
            candidate = math.log(differences[-2] / differences[-1]) / math.log(ratio)
            if math.isfinite(candidate) and 0.25 <= candidate <= 4.0:
                observed_order = candidate

        formal_second_order = (
            contract.exercise_style is ExerciseStyle.EUROPEAN
            and self.rannacher_smoothing
            and not any(projections)
        )
        order = 2.0 if formal_second_order else observed_order
        if order is None:
            return differences, observed_order, None, None, "grid_difference_only"

        denominator = ratio**order - 1.0
        if denominator <= 0.0:
            return differences, observed_order, None, None, "grid_difference_only"
        signed_difference = prices[-1] - prices[-2]
        error = abs(signed_difference) / denominator
        extrapolated = prices[-1] + signed_difference / denominator
        method = "formal_second_order" if formal_second_order else "observed_order"
        return differences, observed_order, error, extrapolated, method

    @staticmethod
    def _upper_tail_probability(
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        s_max: float,
    ) -> float:
        maturity = contract.time_to_expiry
        standard_deviation = volatility * math.sqrt(maturity)
        z_score = (
            math.log(s_max / market_data.spot_price)
            - (market_data.risk_free_rate - market_data.dividend_yield - 0.5 * volatility**2)
            * maturity
        ) / standard_deviation
        return 0.5 * math.erfc(z_score / math.sqrt(2.0))

    def price_with_diagnostics(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> FiniteDifferenceAnalysis:
        """Price a vanilla contract and return the full numerical audit trail."""

        validate_pricing_parameters(contract, market_data, volatility)
        start = time.perf_counter()
        s_max = self._resolve_s_max(contract, market_data, volatility)

        solutions: list[_GridSolution] = []
        for level in range(self.refinement_levels):
            factor = self.refinement_ratio**level
            solutions.append(
                self._solve_grid(
                    contract,
                    market_data,
                    volatility,
                    s_max=s_max,
                    space_steps=self.space_steps * factor,
                    time_steps=self.time_steps * factor,
                )
            )
        finest = solutions[-1]
        level_prices = tuple(solution.price for solution in solutions)
        (
            level_differences,
            observed_order,
            error_estimate,
            extrapolated_price,
            error_method,
        ) = self._convergence_diagnostics(
            contract,
            level_prices,
            tuple(solution.projection_applied for solution in solutions),
        )

        spacing = np.diff(finest.spot_grid)
        spot_on_grid = bool(
            np.any(
                np.isclose(
                    finest.spot_grid,
                    market_data.spot_price,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        )
        strike_on_grid = bool(
            np.any(
                np.isclose(
                    finest.spot_grid,
                    contract.strike_price,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
        )
        is_american = contract.exercise_style is ExerciseStyle.AMERICAN
        uses_psor = is_american and self.exercise_solver == "psor"
        uses_penalty = is_american and self.exercise_solver == "penalty"
        diagnostics = FiniteDifferenceDiagnostics(
            space_steps=finest.space_steps,
            time_steps=finest.time_steps,
            s_max=s_max,
            spot_step=self._local_spot_step(finest.spot_grid, market_data.spot_price),
            time_step=contract.time_to_expiry / finest.time_steps,
            rannacher_half_steps=2 if self.rannacher_smoothing else 0,
            exercise_solver=self.exercise_solver if is_american else "thomas",
            psor_converged=None if uses_penalty else True,
            psor_max_iterations_used=finest.max_iterations_used if uses_psor else 0,
            psor_total_iterations=finest.total_iterations if uses_psor else 0,
            psor_max_update=finest.max_update if uses_psor else 0.0,
            lower_bound=finest.lower_bound,
            upper_bound=finest.upper_bound,
            projection_applied=finest.projection_applied,
            grid_type=self.grid_type,
            min_spot_step=float(np.min(spacing)),
            max_spot_step=float(np.max(spacing)),
            spot_on_grid=spot_on_grid,
            strike_on_grid=strike_on_grid,
            configured_space_steps=self.space_steps,
            configured_time_steps=self.time_steps,
            refinement_levels=self.refinement_levels,
            refinement_ratio=self.refinement_ratio,
            level_space_steps=tuple(solution.space_steps for solution in solutions),
            level_time_steps=tuple(solution.time_steps for solution in solutions),
            level_prices=level_prices,
            level_differences=level_differences,
            level_solver_max_iterations=tuple(
                solution.max_iterations_used for solution in solutions
            ),
            level_solver_total_iterations=tuple(
                solution.total_iterations for solution in solutions
            ),
            level_lcp_residuals=tuple(solution.lcp_residual for solution in solutions),
            level_penalty_obstacle_violations=tuple(
                solution.obstacle_violation for solution in solutions
            ),
            level_penalty_equation_residuals=tuple(
                solution.equation_residual for solution in solutions
            ),
            level_projection_applied=tuple(solution.projection_applied for solution in solutions),
            observed_order=observed_order,
            richardson_error_estimate=error_estimate,
            richardson_extrapolated_price=extrapolated_price,
            error_estimate_method=error_method,
            solver_converged=True,
            solver_max_iterations_used=finest.max_iterations_used,
            solver_total_iterations=finest.total_iterations,
            solver_max_update=finest.max_update,
            lcp_residual=finest.lcp_residual,
            penalty_converged=True if uses_penalty else None,
            penalty_max_iterations_used=finest.max_iterations_used if uses_penalty else 0,
            penalty_total_iterations=finest.total_iterations if uses_penalty else 0,
            penalty_max_update=finest.max_update if uses_penalty else 0.0,
            penalty_parameter=self.penalty_parameter if uses_penalty else None,
            penalty_obstacle_violation=finest.obstacle_violation if uses_penalty else 0.0,
            penalty_equation_residual=finest.equation_residual if uses_penalty else 0.0,
            lower_boundary_delta_residual=finest.lower_boundary_delta_residual,
            upper_boundary_delta_residual=finest.upper_boundary_delta_residual,
            lognormal_upper_tail_probability=self._upper_tail_probability(
                contract, market_data, volatility, s_max
            ),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        solver_suffix = (
            f"_{self.exercise_solver}" if is_american and self.exercise_solver != "psor" else ""
        )
        result = PricingResult(
            contract_id=contract.contract_id,
            theoretical_price=finest.price,
            delta=finest.delta,
            gamma=finest.gamma,
            theta=finest.theta,
            implied_volatility=volatility,
            computation_time_ms=elapsed_ms,
            model_used=(
                f"finite_difference_cn{solver_suffix}_{finest.space_steps}x{finest.time_steps}"
            ),
            numerical_diagnostics=diagnostics.to_dict(),
        )
        return FiniteDifferenceAnalysis(
            pricing_result=result,
            diagnostics=diagnostics,
            spot_grid=finest.spot_grid,
            value_grid=finest.values,
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
