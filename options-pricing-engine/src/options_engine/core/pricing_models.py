"""Pricing model implementations used by the engine."""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.stats import norm

from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from ..utils.validation import validate_pricing_parameters

LOGGER = logging.getLogger(__name__)


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

    if time_to_expiry <= 1e-12 or volatility <= 1e-12:
        intrinsic = _black_scholes_payoff(contract, spot, strike)
        return intrinsic, 0.0, 0.0, 0.0, 0.0, 0.0

    sqrt_t = math.sqrt(time_to_expiry)
    numerator = math.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * time_to_expiry
    denominator = volatility * sqrt_t
    d1 = numerator / denominator
    d2 = d1 - volatility * sqrt_t

    pdf = norm.pdf(d1)

    if contract.option_type is OptionType.CALL:
        price = spot * math.exp(-dividend * time_to_expiry) * norm.cdf(d1) - strike * math.exp(
            -rate * time_to_expiry
        ) * norm.cdf(d2)
        delta = math.exp(-dividend * time_to_expiry) * norm.cdf(d1)
        rho = strike * time_to_expiry * math.exp(-rate * time_to_expiry) * norm.cdf(d2) / 100.0
        theta = (
            -spot * pdf * volatility * math.exp(-dividend * time_to_expiry) / (2.0 * sqrt_t)
            - dividend * spot * math.exp(-dividend * time_to_expiry) * norm.cdf(d1)
            + rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(d2)
        ) / 365.0
    else:
        price = strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2) - spot * math.exp(
            -dividend * time_to_expiry
        ) * norm.cdf(-d1)
        delta = -math.exp(-dividend * time_to_expiry) * norm.cdf(-d1)
        rho = -strike * time_to_expiry * math.exp(-rate * time_to_expiry) * norm.cdf(-d2) / 100.0
        theta = (
            -spot * pdf * volatility * math.exp(-dividend * time_to_expiry) / (2.0 * sqrt_t)
            + dividend * spot * math.exp(-dividend * time_to_expiry) * norm.cdf(-d1)
            - rate * strike * math.exp(-rate * time_to_expiry) * norm.cdf(-d2)
        ) / 365.0

    gamma = math.exp(-dividend * time_to_expiry) * pdf / (spot * volatility * sqrt_t)
    vega = spot * math.exp(-dividend * time_to_expiry) * pdf * sqrt_t / 100.0

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

            prices = np.array(
                [
                    market_data.spot_price * (up**j) * (down ** (steps - j))
                    for j in range(steps + 1)
                ],
                dtype=float,
            )

            if contract.option_type is OptionType.CALL:
                values = np.maximum(prices - contract.strike_price, 0.0)
            else:
                values = np.maximum(contract.strike_price - prices, 0.0)

            for index in range(steps - 1, -1, -1):
                values = discount * (probability * values[1:] + (1.0 - probability) * values[:-1])
                prices = prices[:-1] / up

                if contract.exercise_style is ExerciseStyle.AMERICAN:
                    if contract.option_type is OptionType.CALL:
                        exercise_value = np.maximum(prices - contract.strike_price, 0.0)
                    else:
                        exercise_value = np.maximum(contract.strike_price - prices, 0.0)
                    values = np.maximum(values, exercise_value)

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=float(values[0]),
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


def _thread_local_generator() -> np.random.Generator:
    """Return a thread-local numpy Generator, creating one if needed."""
    generator = getattr(_THREAD_LOCAL_RNG, "generator", None)
    if generator is None:
        generator = np.random.default_rng()
        _THREAD_LOCAL_RNG.generator = generator
    return generator


@dataclass(slots=True)
class MonteCarloModel:
    """Monte Carlo pricer supporting antithetic variates."""

    paths: int = 20_000
    antithetic: bool = True

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> PricingResult:
        start = time.perf_counter()
        try:
            validate_pricing_parameters(contract, market_data, volatility)

            # Ensure positive, integer number of simulation paths
            simulation_paths = int(max(1, self.paths))

            rng = _thread_local_generator()

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
