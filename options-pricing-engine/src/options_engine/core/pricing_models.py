# mypy: ignore-errors
"""Pricing model implementations used by the engine."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator, SeedSequence
from scipy.stats import norm

from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from ..utils.validation import validate_pricing_parameters

LOGGER = logging.getLogger(__name__)

SQRT_TWO = math.sqrt(2.0)
INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)
TAU_MIN = 1e-8
SIGMA_MIN = 1e-6
LOG_MONEYNESS_CLAMP = (
    (1e-6, 8.0),
    (1e-4, 10.0),
    (1e-3, 12.0),
    (1e-2, 14.0),
    (1e-1, 16.0),
    (1.0, 18.0),
    (float("inf"), 20.0),
)


def _norm_pdf(value: float) -> float:
    return INV_SQRT_TWO_PI * math.exp(-0.5 * value * value)


def _norm_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / SQRT_TWO)


def _log_moneyness_threshold(time_to_expiry: float) -> float:
    for maturity, clamp in LOG_MONEYNESS_CLAMP:
        if time_to_expiry <= maturity:
            return clamp
    return LOG_MONEYNESS_CLAMP[-1][1]


def _black_scholes_payoff(
    contract: OptionContract,
    spot: float,
    strike: float,
) -> float:
    """Return the intrinsic value of an option contract."""
    intrinsic = max(0.0, spot - strike)
    if contract.option_type is OptionType.PUT:
        intrinsic = max(0.0, strike - spot)
    return intrinsic


def _black_scholes_greeks(
    contract: OptionContract,
    market_data: MarketData,
    volatility: float,
) -> Tuple[float, float, float, float, float, float]:
    """Calculate the Black-Scholes price and greeks."""
    spot = market_data.spot_price
    strike = contract.strike_price
    time_to_expiry = contract.time_to_expiry
    rate = market_data.risk_free_rate
    dividend = market_data.dividend_yield

    if time_to_expiry <= 0.0 or volatility <= 0.0:
        intrinsic = _black_scholes_payoff(contract, spot, strike)
        return intrinsic, 0.0, 0.0, 0.0, 0.0, 0.0

    time_to_expiry = max(time_to_expiry, TAU_MIN)
    volatility = max(volatility, SIGMA_MIN)

    sqrt_t = math.sqrt(time_to_expiry)
    discount_dividend = math.exp(-dividend * time_to_expiry)
    discount_rate = math.exp(-rate * time_to_expiry)
    parity_anchor = spot * discount_dividend - strike * discount_rate

    log_moneyness = math.log(spot / strike)
    numerator = log_moneyness + (rate - dividend + 0.5 * volatility**2) * time_to_expiry
    denominator = volatility * sqrt_t
    d1 = numerator / denominator
    d2 = d1 - volatility * sqrt_t

    pdf = _norm_pdf(d1)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)

    clamp_threshold = _log_moneyness_threshold(time_to_expiry)
    if log_moneyness > clamp_threshold:
        tail_call = strike * discount_rate * _norm_cdf(-d2) - spot * discount_dividend * _norm_cdf(-d1)
        call_price = parity_anchor + tail_call
    else:
        call_price = spot * discount_dividend * cdf_d1 - strike * discount_rate * cdf_d2

    call_delta = discount_dividend * cdf_d1
    call_gamma = discount_dividend * pdf / (spot * volatility * sqrt_t)
    call_theta = (
        -spot * discount_dividend * pdf * volatility / (2.0 * sqrt_t)
        - dividend * spot * discount_dividend * cdf_d1
        + rate * strike * discount_rate * cdf_d2
    ) / 365.0
    call_vega = spot * discount_dividend * pdf * sqrt_t / 100.0
    call_rho = strike * discount_rate * time_to_expiry * cdf_d2 / 100.0

    if contract.option_type is OptionType.CALL:
        price = call_price
        delta = call_delta
        gamma = call_gamma
        theta = call_theta
        vega = call_vega
        rho = call_rho
    else:
        price = call_price - parity_anchor
        delta = call_delta - discount_dividend
        gamma = call_gamma
        theta = call_theta + (dividend * spot * discount_dividend - rate * strike * discount_rate) / 365.0
        vega = call_vega
        rho = call_rho - time_to_expiry * strike * discount_rate / 100.0

    return price, delta, gamma, theta, vega, rho


@dataclass(slots=True)
class BlackScholesModel:
    """Deterministic Black-Scholes pricing model."""

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> PricingResult:
        start = time.perf_counter()
        try:
            validate_pricing_parameters(contract, market_data, volatility)
            price, delta, gamma, theta, vega, rho = _black_scholes_greeks(
                contract, market_data, volatility
            )
            price = max(0.0, price)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=price,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                implied_volatility=volatility,
                computation_time_ms=elapsed_ms,
                model_used="black_scholes",
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Black-Scholes pricing failed")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=0.0,
                computation_time_ms=elapsed_ms,
                model_used="black_scholes",
                error=str(exc),
            )


@dataclass(slots=True)
class BinomialModel:
    """Recombining binomial tree pricing model."""

    steps: int = 200

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> PricingResult:
        start = time.perf_counter()
        try:
            validate_pricing_parameters(contract, market_data, volatility)

            steps = max(1, self.steps)
            delta_t = contract.time_to_expiry / steps
            up = math.exp(volatility * math.sqrt(delta_t))
            down = 1.0 / up

            growth = math.exp((market_data.risk_free_rate - market_data.dividend_yield) * delta_t)
            probability = (growth - down) / (up - down)
            probability = min(1.0, max(0.0, probability))
            discount = math.exp(-market_data.risk_free_rate * delta_t)

            sqrt_dt = math.sqrt(delta_t)
            dup = up * sqrt_dt
            ddown = -down * sqrt_dt
            denom = up - down
            if denom == 0.0:
                raise ZeroDivisionError("up and down factors resulted in zero denominator")
            numerator = growth - down
            dnumerator = -ddown
            ddenom = dup - ddown
            dprobability = (dnumerator * denom - numerator * ddenom) / (denom**2)

            prices = np.array(
                [
                    market_data.spot_price * (up**j) * (down ** (steps - j))
                    for j in range(steps + 1)
                ],
                dtype=float,
            )

            price_vega = np.empty_like(prices)
            node_indices = np.arange(steps + 1)
            price_vega[:] = prices * ((2 * node_indices - steps) * sqrt_dt)

            if contract.option_type is OptionType.CALL:
                values = np.maximum(prices - contract.strike_price, 0.0)
                value_vega = np.where(values > 0.0, price_vega, 0.0)
            else:
                values = np.maximum(contract.strike_price - prices, 0.0)
                value_vega = np.where(values > 0.0, -price_vega, 0.0)

            first_step_values: Optional[np.ndarray] = None
            first_step_prices: Optional[np.ndarray] = None
            second_step_values: Optional[np.ndarray] = None
            second_step_prices: Optional[np.ndarray] = None

            for index in range(steps - 1, -1, -1):
                prev_values = values
                prev_value_vega = value_vega

                continuation_values = discount * (
                    probability * prev_values[1:] + (1.0 - probability) * prev_values[:-1]
                )
                continuation_vega = discount * (
                    probability * prev_value_vega[1:] + (1.0 - probability) * prev_value_vega[:-1]
                )
                continuation_vega += discount * dprobability * (
                    prev_values[1:] - prev_values[:-1]
                )

                prev_prices = prices
                prev_price_vega = price_vega
                prices = prev_prices[:-1] * up
                price_vega = prev_price_vega[:-1] * up + prev_prices[:-1] * dup

                values = continuation_values
                value_vega = continuation_vega

                if contract.exercise_style is ExerciseStyle.AMERICAN:
                    if contract.option_type is OptionType.CALL:
                        exercise_value = np.maximum(prices - contract.strike_price, 0.0)
                        exercise_vega = np.where(exercise_value > 0.0, price_vega, 0.0)
                    else:
                        exercise_value = np.maximum(contract.strike_price - prices, 0.0)
                        exercise_vega = np.where(exercise_value > 0.0, -price_vega, 0.0)

                    exercise_mask = exercise_value > values
                    if np.any(exercise_mask):
                        values = np.where(exercise_mask, exercise_value, values)
                        value_vega = np.where(exercise_mask, exercise_vega, value_vega)

                if index == 2:
                    second_step_values = values.copy()
                    second_step_prices = prices.copy()
                if index == 1:
                    first_step_values = values.copy()
                    first_step_prices = prices.copy()

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            delta: Optional[float] = None
            gamma: Optional[float] = None

            if first_step_values is not None and first_step_prices is not None:
                denom = first_step_prices[1] - first_step_prices[0]
                if abs(denom) > 0:
                    delta = float((first_step_values[1] - first_step_values[0]) / denom)

            if (
                delta is not None
                and second_step_values is not None
                and second_step_prices is not None
                and first_step_prices is not None
            ):
                down_denom = second_step_prices[1] - second_step_prices[0]
                up_denom = second_step_prices[2] - second_step_prices[1]
                root_denom = first_step_prices[1] - first_step_prices[0]
                if abs(down_denom) > 0 and abs(up_denom) > 0 and abs(root_denom) > 0:
                    delta_down = (second_step_values[1] - second_step_values[0]) / down_denom
                    delta_up = (second_step_values[2] - second_step_values[1]) / up_denom
                    gamma = float((delta_up - delta_down) / root_denom)

            vega = float(value_vega[0] / 100.0)

            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=float(values[0]),
                delta=delta,
                gamma=gamma,
                vega=vega,
                computation_time_ms=elapsed_ms,
                model_used=f"binomial_{steps}",
                implied_volatility=volatility,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Binomial pricing failed")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=0.0,
                computation_time_ms=elapsed_ms,
                model_used=f"binomial_{self.steps}",
                error=str(exc),
            )


_THREAD_LOCAL_RNG = threading.local()


def _thread_local_generator(seed_sequence: Optional[SeedSequence] = None) -> Generator:
    """Return a thread-local numpy Generator, creating one if needed."""

    generator: Optional[Generator] = getattr(_THREAD_LOCAL_RNG, "generator", None)
    current_seed = getattr(_THREAD_LOCAL_RNG, "seed_sequence", None)
    if generator is None or current_seed is not seed_sequence:
        generator = np.random.default_rng(seed_sequence)
        _THREAD_LOCAL_RNG.generator = generator
        _THREAD_LOCAL_RNG.seed_sequence = seed_sequence
    return generator


@dataclass(slots=True)
class MonteCarloModel:
    """Monte Carlo pricer supporting antithetic variates."""

    paths: int = 20_000
    antithetic: bool = True

    # Optional deterministic seed shared across threads
    seed_sequence: Optional[SeedSequence] = None

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        seed_sequence: Optional[SeedSequence] = None,
    ) -> PricingResult:
        start = time.perf_counter()
        try:
            validate_pricing_parameters(contract, market_data, volatility)

            # Ensure positive, integer number of simulation paths
            simulation_paths = int(max(1, self.paths))

            if contract.time_to_expiry <= 1e-12 or volatility <= 1e-12:
                (
                    price,
                    delta,
                    gamma,
                    _theta,
                    vega,
                    _rho,
                ) = _black_scholes_greeks(contract, market_data, max(volatility, 1e-12))
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return PricingResult(
                    contract_id=contract.contract_id,
                    theoretical_price=max(0.0, price),
                    delta=delta,
                    gamma=gamma,
                    vega=vega,
                    computation_time_ms=elapsed_ms,
                    model_used=f"monte_carlo_{simulation_paths}",
                    implied_volatility=volatility,
                )

            sequence = seed_sequence or self.seed_sequence
            rng = _thread_local_generator(sequence)

            if self.antithetic:
                # Make path count even and at least 2 so pairs exist
                simulation_paths = max(2, simulation_paths + (simulation_paths % 2))
                half_paths = simulation_paths // 2
                base_draws = rng.standard_normal(half_paths)
                draws = np.empty(simulation_paths, dtype=float)
                draws[:half_paths] = base_draws
                draws[half_paths:] = -base_draws
            else:
                draws = rng.standard_normal(simulation_paths)

            time_sqrt = math.sqrt(max(0.0, contract.time_to_expiry))
            drift = (
                market_data.risk_free_rate - market_data.dividend_yield - 0.5 * volatility**2
            ) * contract.time_to_expiry
            diffusion = volatility * time_sqrt * draws
            terminal_prices = market_data.spot_price * np.exp(drift + diffusion)

            if contract.option_type is OptionType.CALL:
                payoff = np.maximum(terminal_prices - contract.strike_price, 0.0)
            else:
                payoff = np.maximum(contract.strike_price - terminal_prices, 0.0)

            discount_factor = math.exp(-market_data.risk_free_rate * contract.time_to_expiry)
            discounted_payoffs = discount_factor * payoff
            theoretical_price = float(np.mean(discounted_payoffs))

            sqrt_time = time_sqrt
            if contract.option_type is OptionType.CALL:
                indicator = terminal_prices > contract.strike_price
                delta_path = discount_factor * np.where(
                    indicator,
                    terminal_prices / market_data.spot_price,
                    0.0,
                )
                sensitivity_term = terminal_prices * (
                    sqrt_time * draws - volatility * contract.time_to_expiry
                )
                vega_path = discount_factor * np.where(indicator, sensitivity_term, 0.0)
            else:
                indicator = terminal_prices < contract.strike_price
                delta_path = -discount_factor * np.where(
                    indicator,
                    terminal_prices / market_data.spot_price,
                    0.0,
                )
                sensitivity_term = -terminal_prices * (
                    sqrt_time * draws - volatility * contract.time_to_expiry
                )
                vega_path = discount_factor * np.where(indicator, sensitivity_term, 0.0)

            drift_term = (
                np.log(terminal_prices / market_data.spot_price)
                - (market_data.risk_free_rate - market_data.dividend_yield - 0.5 * volatility**2)
                * contract.time_to_expiry
            )
            spot = market_data.spot_price
            sigma_sq_t = max(volatility**2 * contract.time_to_expiry, 1e-18)
            inv_spot_sigma_sq_t = 1.0 / (spot * sigma_sq_t)
            inv_spot_sq_sigma_sq_t = 1.0 / (spot**2 * sigma_sq_t)
            likelihood_ratio = drift_term * inv_spot_sigma_sq_t
            gamma_path = discount_factor * payoff * (
                likelihood_ratio**2 - (1.0 + drift_term) * inv_spot_sq_sigma_sq_t
            )

            delta_estimate = float(np.mean(delta_path))
            gamma_estimate = float(np.mean(gamma_path))
            vega_estimate = float(np.mean(vega_path) / 100.0)

            standard_error: Optional[float] = None
            confidence_interval: Optional[Tuple[float, float]] = None
            if simulation_paths > 1:
                sample_std = float(np.std(discounted_payoffs, ddof=1))
                standard_error = sample_std / math.sqrt(simulation_paths)
                z_score = norm.ppf(0.975)  # 95% CI
                half_width = z_score * standard_error
                confidence_interval = (
                    theoretical_price - half_width,
                    theoretical_price + half_width,
                )

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=max(0.0, theoretical_price),
                delta=delta_estimate,
                gamma=gamma_estimate,
                vega=vega_estimate,
                computation_time_ms=elapsed_ms,
                model_used=f"monte_carlo_{simulation_paths}",
                implied_volatility=volatility,
                standard_error=standard_error,
                confidence_interval=confidence_interval,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Monte Carlo pricing failed")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=0.0,
                computation_time_ms=elapsed_ms,
                model_used=f"monte_carlo_{self.paths}",
                error=str(exc),
            )


def _default_basis_factories() -> Dict[str, Sequence[Callable[[np.ndarray], np.ndarray]]]:
    """Return a dictionary describing the default LSMC basis candidates."""

    def _constant(_: np.ndarray) -> np.ndarray:
        return np.ones_like(_, dtype=float)

    def _identity(x: np.ndarray) -> np.ndarray:
        return x

    def _square(x: np.ndarray) -> np.ndarray:
        return x**2

    def _cube(x: np.ndarray) -> np.ndarray:
        return x**3

    def _log(x: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(x, 1e-12))

    def _sqrt(x: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(x, 0.0))

    return {
        "polynomial_2": (_constant, _identity, _square),
        "polynomial_3": (_constant, _identity, _square, _cube),
        "log_linear": (_constant, _identity, _log),
        "sqrt_polynomial": (_constant, _sqrt, _identity, _square),
    }


def _build_design_matrix(
    basis: Sequence[Callable[[np.ndarray], np.ndarray]], values: np.ndarray
) -> np.ndarray:
    """Evaluate the provided basis functions and build the design matrix."""

    columns: List[np.ndarray] = []
    for function in basis:
        evaluated = function(values)
        if evaluated.ndim != 1:
            evaluated = np.asarray(evaluated, dtype=float).reshape(-1)
        columns.append(np.asarray(evaluated, dtype=float))
    return np.column_stack(columns)


def _information_criteria(rss: float, n: int, k: int) -> Tuple[float, float]:
    """Compute the AIC and BIC for a linear regression fit."""

    if n <= k or n == 0:
        return float("inf"), float("inf")

    variance = max(rss / n, 1e-16)
    log_likelihood = -0.5 * n * (math.log(2.0 * math.pi) + math.log(variance) + 1.0)
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = math.log(n) * k - 2.0 * log_likelihood
    return float(aic), float(bic)


def _kfold_indices(sample_size: int, folds: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Generate shuffled k-fold indices."""

    folds = max(2, min(sample_size, folds))
    indices = np.arange(sample_size)
    rng.shuffle(indices)
    return [fold for fold in np.array_split(indices, folds) if fold.size > 0]


def _fit_linear_regression(design: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit a linear regression returning coefficients and residual sum of squares."""

    coefficients, residuals, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if residuals.size:
        rss = float(residuals[0])
    else:
        predictions = design @ coefficients
        rss = float(np.sum((targets - predictions) ** 2))
    if rank < design.shape[1]:
        rss = float("inf")
    return coefficients, rss


@dataclass(slots=True)
class BasisMetrics:
    """Diagnostics describing a fitted basis at a single exercise date."""

    name: str
    coefficients: np.ndarray
    rss: float
    aic: float
    bic: float
    cv_rmse: float


@dataclass(slots=True)
class ExercisePolicyStep:
    """Summary of the extracted early exercise policy at a given time step."""

    time_index: int
    time: float
    basis: Optional[BasisMetrics]
    in_the_money: int
    exercised: int
    exercise_fraction: float
    exercise_spot_mean: Optional[float]


@dataclass(slots=True)
class LSMCAnalysis:
    """Container for diagnostics returned by the Longstaff-Schwartz model."""

    pricing_result: PricingResult
    policy: np.ndarray
    policy_steps: List[ExercisePolicyStep]
    basis_diagnostics: List[List[BasisMetrics]]
    reference_price: float
    reference_model_used: str
    price_diff_bps: float


@dataclass(slots=True)
class LongstaffSchwartzModel:
    """American option pricing using the Longstaff-Schwartz method with diagnostics."""

    paths: int = 80_000
    steps: int = 60
    cv_folds: int = 5
    antithetic: bool = True
    seed_sequence: Optional[SeedSequence] = None
    basis_factories: Optional[Dict[str, Sequence[Callable[[np.ndarray], np.ndarray]]]] = None
    reference_steps: int = 2_000

    def _prepare_paths(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        rng: Generator,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simulate price paths under the risk-neutral measure."""

        time_to_expiry = contract.time_to_expiry
        step_count = max(1, int(self.steps))
        dt = time_to_expiry / step_count
        sqrt_dt = math.sqrt(dt)

        path_count = max(1, int(self.paths))
        if self.antithetic:
            path_count = max(2, path_count + (path_count % 2))
            half = path_count // 2
            base_draws = rng.standard_normal((step_count, half))
            draws = np.concatenate([base_draws, -base_draws], axis=1)
        else:
            draws = rng.standard_normal((step_count, path_count))
            path_count = draws.shape[1]

        prices = np.empty((step_count + 1, path_count), dtype=float)
        prices[0, :] = market_data.spot_price

        drift = (market_data.risk_free_rate - market_data.dividend_yield - 0.5 * volatility**2) * dt
        diffusion = volatility * sqrt_dt

        for index in range(1, step_count + 1):
            shock = diffusion * draws[index - 1, :]
            prices[index, :] = prices[index - 1, :] * np.exp(drift + shock)

        times = np.linspace(0.0, time_to_expiry, step_count + 1)
        discount = math.exp(-market_data.risk_free_rate * dt)
        return prices, times, discount

    def _intrinsic_value(self, contract: OptionContract, prices: np.ndarray) -> np.ndarray:
        """Return intrinsic values for the provided price vector."""

        if contract.option_type is OptionType.CALL:
            return np.maximum(prices - contract.strike_price, 0.0)
        return np.maximum(contract.strike_price - prices, 0.0)

    def _evaluate_basis(
        self,
        basis_name: str,
        basis_functions: Sequence[Callable[[np.ndarray], np.ndarray]],
        features: np.ndarray,
        targets: np.ndarray,
        folds: int,
        rng: np.random.Generator,
    ) -> BasisMetrics:
        """Fit a regression basis computing AIC/BIC and cross-validation RMSE."""

        design = _build_design_matrix(basis_functions, features)
        coefficients, rss = _fit_linear_regression(design, targets)

        sample_size = features.size
        parameters = design.shape[1]
        aic, bic = _information_criteria(rss, sample_size, parameters)

        cv_indices = _kfold_indices(sample_size, folds, rng)
        sq_errors: List[float] = []
        for fold in cv_indices:
            train_mask = np.ones(sample_size, dtype=bool)
            train_mask[fold] = False
            if not train_mask.any():
                continue
            train_design = design[train_mask]
            train_targets = targets[train_mask]
            test_design = design[~train_mask]
            test_targets = targets[~train_mask]
            if train_design.size == 0 or test_design.size == 0:
                continue
            fold_coefficients, _ = _fit_linear_regression(train_design, train_targets)
            predictions = test_design @ fold_coefficients
            sq_errors.append(float(np.mean((test_targets - predictions) ** 2)))

        if sq_errors:
            cv_rmse = float(math.sqrt(max(0.0, float(np.mean(sq_errors)))))
        else:
            cv_rmse = float("inf")

        return BasisMetrics(
            name=basis_name,
            coefficients=coefficients,
            rss=rss,
            aic=aic,
            bic=bic,
            cv_rmse=cv_rmse,
        )

    def price_with_diagnostics(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        seed_sequence: Optional[SeedSequence] = None,
    ) -> LSMCAnalysis:
        """Run the Longstaff-Schwartz algorithm returning diagnostics."""

        if contract.exercise_style is not ExerciseStyle.AMERICAN:
            raise ValueError("Longstaff-Schwartz model requires an American option contract")

        validate_pricing_parameters(contract, market_data, volatility)

        start = time.perf_counter()

        basis_factories = self.basis_factories or _default_basis_factories()
        if not basis_factories:
            raise ValueError("At least one basis must be provided for LSMC")

        sequence = seed_sequence or self.seed_sequence
        rng = _thread_local_generator(sequence)
        diagnostic_rng = np.random.default_rng(42)

        prices, times, discount = self._prepare_paths(contract, market_data, volatility, rng)
        step_count, path_count = prices.shape[0] - 1, prices.shape[1]

        intrinsic_maturity = self._intrinsic_value(contract, prices[-1, :])
        cashflows = intrinsic_maturity.copy()

        policy = np.zeros((step_count + 1, path_count), dtype=bool)
        policy[-1, :] = intrinsic_maturity > 0.0

        basis_diagnostics: List[List[BasisMetrics]] = []
        policy_steps: List[ExercisePolicyStep] = []

        strike = contract.strike_price

        for step in range(step_count - 1, -1, -1):
            spot = prices[step, :]
            intrinsic = self._intrinsic_value(contract, spot)
            in_the_money = intrinsic > 0.0

            continuation = discount * cashflows
            evaluated_bases: List[BasisMetrics] = []
            selected: Optional[BasisMetrics] = None

            if np.any(in_the_money):
                features = spot[in_the_money] / strike
                targets = continuation[in_the_money]
                for name, basis_functions in basis_factories.items():
                    try:
                        metrics = self._evaluate_basis(
                            name,
                            basis_functions,
                            features,
                            targets,
                            self.cv_folds,
                            diagnostic_rng,
                        )
                    except np.linalg.LinAlgError:
                        continue
                    evaluated_bases.append(metrics)

                basis_diagnostics.append(evaluated_bases)

                if evaluated_bases:
                    valid_bases = [
                        metrics
                        for metrics in evaluated_bases
                        if math.isfinite(metrics.cv_rmse)
                        and math.isfinite(metrics.aic)
                        and math.isfinite(metrics.bic)
                    ]
                    if valid_bases:
                        valid_bases.sort(key=lambda m: (m.cv_rmse, m.bic))
                        selected = valid_bases[0]

            else:
                basis_diagnostics.append([])

            exercised_paths = np.zeros(path_count, dtype=bool)
            exercise_mean: Optional[float] = None

            if selected is not None:
                basis_functions = basis_factories[selected.name]
                design = _build_design_matrix(basis_functions, spot[in_the_money] / strike)
                predictions = design @ selected.coefficients
                exercise_region = intrinsic[in_the_money] >= predictions

                in_money_indices = np.flatnonzero(in_the_money)
                exercised_indices = in_money_indices[exercise_region]
                exercised_paths[exercised_indices] = True
                if exercised_indices.size:
                    exercise_mean = float(np.mean(spot[exercised_indices]))

            policy[step, exercised_paths] = True

            exercise_count = int(np.count_nonzero(exercised_paths))
            if exercise_count:
                cashflows[exercised_paths] = intrinsic[exercised_paths]
            cashflows[~exercised_paths] = continuation[~exercised_paths]

            policy_steps.append(
                ExercisePolicyStep(
                    time_index=step,
                    time=times[step],
                    basis=selected,
                    in_the_money=int(np.count_nonzero(in_the_money)),
                    exercised=exercise_count,
                    exercise_fraction=(exercise_count / path_count) if path_count else 0.0,
                    exercise_spot_mean=exercise_mean,
                )
            )

        price = float(np.mean(cashflows))
        if path_count > 1:
            std_err = float(np.std(cashflows, ddof=1) / math.sqrt(path_count))
        else:
            std_err = 0.0

        binomial_model = BinomialModel(steps=self.reference_steps)
        reference = binomial_model.calculate_price(contract, market_data, volatility)
        reference_price = reference.theoretical_price
        price_diff_bps = (price - reference_price) / max(reference_price, 1e-12) * 10_000.0

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        pricing_result = PricingResult(
            contract_id=contract.contract_id,
            theoretical_price=max(0.0, price),
            implied_volatility=volatility,
            computation_time_ms=elapsed_ms,
            model_used=f"lsmc_{path_count}x{step_count}",
            standard_error=std_err,
        )

        policy_steps.reverse()

        return LSMCAnalysis(
            pricing_result=pricing_result,
            policy=policy,
            policy_steps=policy_steps,
            basis_diagnostics=basis_diagnostics[::-1],
            reference_price=reference_price,
            reference_model_used=reference.model_used,
            price_diff_bps=float(price_diff_bps),
        )

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        seed_sequence: Optional[SeedSequence] = None,
    ) -> PricingResult:
        """Return a :class:`PricingResult` for the LSMC model."""

        analysis = self.price_with_diagnostics(
            contract,
            market_data,
            volatility,
            seed_sequence=seed_sequence,
        )
        return analysis.pricing_result
