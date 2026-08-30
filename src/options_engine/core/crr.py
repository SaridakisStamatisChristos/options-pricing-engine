"""Adaptive Cox-Ross-Rubinstein tree valuation."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..utils.validation import validate_pricing_parameters
from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from .pricing_common import (
    MAX_BINOMIAL_STEPS,
    PriceResult,
    _bounded_integer,
    _normalise_option_type,
    _require_boolean,
)

LOGGER = logging.getLogger(__name__)


def binomial_price(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    *,
    steps: int = 200,
) -> PriceResult:
    """Convenience wrapper around :class:`BinomialModel`."""

    opt = _normalise_option_type(option_type)
    contract = OptionContract(
        symbol="BINOM",
        strike_price=strike,
        time_to_expiry=tau,
        option_type=OptionType.CALL if opt == "call" else OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
    )
    market = MarketData(spot_price=spot, risk_free_rate=r, dividend_yield=q)
    model = BinomialModel(steps=steps)
    result = model.calculate_price(contract, market, sigma)
    meta: dict[str, object] = {
        "method": f"binomial_{steps}",
        "option_type": opt,
        "policy_flags": [],
    }
    return PriceResult(
        price=result.theoretical_price, ci_half_width=0.0, meta=meta, standard_error=0.0
    )


@dataclass(slots=True)
class BinomialModel:
    """Recombining binomial tree pricing model."""

    MAX_STEPS: ClassVar[int] = MAX_BINOMIAL_STEPS

    steps: int = 200
    _compute_rho: bool = True

    def __post_init__(self) -> None:
        self.steps = _bounded_integer(
            "steps",
            self.steps,
            minimum=2,
            maximum=self.MAX_STEPS,
        )
        self._compute_rho = _require_boolean("_compute_rho", self._compute_rho)

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> PricingResult:
        start = time.perf_counter()
        try:
            validate_pricing_parameters(contract, market_data, volatility)

            # The CRR risk-neutral probability is valid only when d < exp((r-q)dt) < u.
            # Increase the resolution until that condition holds; clamping p creates a
            # different, arbitrageable process and can even produce negative vega.
            steps = self.steps
            while True:
                delta_t = contract.time_to_expiry / steps
                log_up = volatility * math.sqrt(delta_t)
                up = math.exp(log_up)
                down = 1.0 / up
                growth = math.exp(
                    (market_data.risk_free_rate - market_data.dividend_yield) * delta_t
                )
                probability = (growth - down) / (up - down)
                if 0.0 < probability < 1.0:
                    break
                if steps >= self.MAX_STEPS:
                    raise ValueError(
                        "CRR tree cannot satisfy no-arbitrage probability; "
                        "increase volatility or use Black-Scholes"
                    )
                steps = min(self.MAX_STEPS, steps * 2)
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

            node_indices = np.arange(steps + 1)
            log_price_span = volatility * math.sqrt(contract.time_to_expiry * steps)
            max_terminal_log_price = math.log(market_data.spot_price) + log_price_span
            vega_scale = math.sqrt(contract.time_to_expiry * steps)
            safe_log_limit = math.log(np.finfo(float).max) - math.log1p(vega_scale)
            if max_terminal_log_price > safe_log_limit:
                raise ValueError(
                    "CRR tree exceeds the supported floating-point range; "
                    "reduce volatility, maturity, or steps"
                )
            log_prices = math.log(market_data.spot_price) + (2 * node_indices - steps) * log_up
            prices = np.exp(log_prices)

            price_vega = np.empty_like(prices)
            price_vega[:] = prices * ((2 * node_indices - steps) * sqrt_dt)

            if contract.option_type is OptionType.CALL:
                values = np.maximum(prices - contract.strike_price, 0.0)
                value_vega = np.where(values > 0.0, price_vega, 0.0)
            else:
                values = np.maximum(contract.strike_price - prices, 0.0)
                value_vega = np.where(values > 0.0, -price_vega, 0.0)

            first_step_values: np.ndarray | None = None
            first_step_prices: np.ndarray | None = None
            second_step_values: np.ndarray | None = None
            second_step_prices: np.ndarray | None = None

            for index in range(steps - 1, -1, -1):
                prev_values = values
                prev_value_vega = value_vega

                continuation_values = discount * (
                    probability * prev_values[1:] + (1.0 - probability) * prev_values[:-1]
                )
                continuation_vega = discount * (
                    probability * prev_value_vega[1:] + (1.0 - probability) * prev_value_vega[:-1]
                )
                continuation_vega += discount * dprobability * (prev_values[1:] - prev_values[:-1])

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
            delta: float | None = None
            gamma: float | None = None
            theta: float | None = None

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
                root_denom = 0.5 * (second_step_prices[2] - second_step_prices[0])
                if abs(down_denom) > 0 and abs(up_denom) > 0 and abs(root_denom) > 0:
                    delta_down = (second_step_values[1] - second_step_values[0]) / down_denom
                    delta_up = (second_step_values[2] - second_step_values[1]) / up_denom
                    gamma = float((delta_up - delta_down) / root_denom)

            if second_step_values is not None and second_step_values.size >= 2:
                theta = float((second_step_values[1] - values[0]) / (2.0 * delta_t) / 365.0)

            vega = float(value_vega[0] / 100.0)
            rho: float | None = None
            if self._compute_rho:
                bump = 1e-4
                rate_low = max(-1.0, market_data.risk_free_rate - bump)
                rate_high = min(1.0, market_data.risk_free_rate + bump)
                if rate_high > rate_low:
                    low_market = MarketData(
                        spot_price=market_data.spot_price,
                        risk_free_rate=rate_low,
                        dividend_yield=market_data.dividend_yield,
                    )
                    high_market = MarketData(
                        spot_price=market_data.spot_price,
                        risk_free_rate=rate_high,
                        dividend_yield=market_data.dividend_yield,
                    )
                    rho_model = BinomialModel(steps=steps, _compute_rho=False)
                    low_price = rho_model.calculate_price(
                        contract, low_market, volatility
                    ).theoretical_price
                    high_price = rho_model.calculate_price(
                        contract, high_market, volatility
                    ).theoretical_price
                    rho = float((high_price - low_price) / (rate_high - rate_low) / 100.0)

            numerical_outputs = [values[0], value_vega[0]]
            numerical_outputs.extend(
                value for value in (delta, gamma, theta, rho) if value is not None
            )
            if not all(math.isfinite(float(value)) for value in numerical_outputs):
                raise ValueError("CRR tree produced a non-finite price or Greek")

            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=float(values[0]),
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                computation_time_ms=elapsed_ms,
                model_used=f"binomial_{steps}",
                implied_volatility=volatility,
            )
        except Exception:  # pragma: no cover - preserve context for API error mapping
            LOGGER.exception("Binomial pricing failed")
            raise


__all__ = ["BinomialModel", "binomial_price"]
