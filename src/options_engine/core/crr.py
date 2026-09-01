"""Adaptive Cox-Ross-Rubinstein tree valuation."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..term_structure import DeterministicTermStructure
from ..utils.validation import validate_discrete_dividends, validate_pricing_parameters
from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from .pricing_common import (
    MAX_BINOMIAL_STEPS,
    PriceResult,
    _bounded_integer,
    _normalise_option_type,
    _require_boolean,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _CurveTreeSolution:
    price: float
    steps: int
    delta_t: float
    first_step_values: np.ndarray | None
    first_step_prices: np.ndarray | None
    second_step_values: np.ndarray | None
    second_step_prices: np.ndarray | None
    aligned_cash_times: tuple[float, ...]
    cash_alignment_error: float
    funding_rates: tuple[float, ...]
    carry_rates: tuple[float, ...]
    probabilities: tuple[float, ...]
    discounts: tuple[float, ...]


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
            validate_discrete_dividends(contract, market_data)

            if market_data.cash_dividends:
                return self._calculate_discrete_dividend_price(
                    contract,
                    market_data,
                    volatility,
                    start=start,
                )

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
                        timestamp=market_data.timestamp,
                        cash_dividends=market_data.cash_dividends,
                    )
                    high_market = MarketData(
                        spot_price=market_data.spot_price,
                        risk_free_rate=rate_high,
                        dividend_yield=market_data.dividend_yield,
                        timestamp=market_data.timestamp,
                        cash_dividends=market_data.cash_dividends,
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

    @staticmethod
    def _validate_term_structure(
        contract: OptionContract,
        market_data: MarketData,
        term_structure: DeterministicTermStructure,
    ) -> None:
        if not isinstance(term_structure, DeterministicTermStructure):
            raise TypeError("term_structure must be a DeterministicTermStructure")
        if not math.isclose(
            term_structure.maturity,
            contract.time_to_expiry,
            rel_tol=0.0,
            abs_tol=1e-14 * max(1.0, contract.time_to_expiry),
        ):
            raise ValueError("term-structure maturity must match the option contract")
        requested_cash_times = tuple(
            dividend.ex_time for dividend in market_data.cash_dividends.dividends
        )
        if len(requested_cash_times) != len(term_structure.cash_dividend_times) or any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-14)
            for left, right in zip(
                requested_cash_times,
                term_structure.cash_dividend_times,
                strict=True,
            )
        ):
            raise ValueError("term-structure cash-dividend anchors must match market_data exactly")

    def _curve_tree_parameters(
        self,
        contract: OptionContract,
        volatility: float,
        term_structure: DeterministicTermStructure,
        minimum_steps: int,
    ) -> tuple[
        int,
        float,
        float,
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        """Build exact per-interval CRR probabilities without clipping."""

        steps = minimum_steps
        while True:
            delta_t = contract.time_to_expiry / steps
            up = math.exp(volatility * math.sqrt(delta_t))
            down = 1.0 / up
            denominator = up - down
            funding_rates: list[float] = []
            carry_rates: list[float] = []
            probabilities: list[float] = []
            discounts: list[float] = []
            invalid_interval: int | None = None
            for index in range(steps):
                start_time = index * delta_t
                end_time = contract.time_to_expiry if index == steps - 1 else (index + 1) * delta_t
                funding_rate, carry_rate = term_structure.step_rates(start_time, end_time)
                discount = term_structure.discount_factor(start_time, end_time)
                growth = term_structure.growth_factor(start_time, end_time)
                probability = (growth - down) / denominator
                funding_rates.append(funding_rate)
                carry_rates.append(carry_rate)
                discounts.append(discount)
                probabilities.append(probability)
                if not 0.0 < probability < 1.0 and invalid_interval is None:
                    invalid_interval = index
            if invalid_interval is None:
                return (
                    steps,
                    delta_t,
                    up,
                    tuple(funding_rates),
                    tuple(carry_rates),
                    tuple(probabilities),
                    tuple(discounts),
                )
            if steps >= self.MAX_STEPS:
                probability = probabilities[invalid_interval]
                raise ValueError(
                    "curve-aware CRR tree cannot satisfy the no-arbitrage condition "
                    f"at interval {invalid_interval} with {steps} steps "
                    f"(p={probability:.17g}); maximum supported resolution reached"
                )
            steps = min(self.MAX_STEPS, steps * 2)

    def _curve_cash_boundary_value(
        self,
        contract: OptionContract,
        forward_time: float,
        term_structure: DeterministicTermStructure,
    ) -> float:
        if contract.option_type is OptionType.CALL:
            return 0.0
        european = contract.strike_price * term_structure.discount_factor(
            forward_time, contract.time_to_expiry
        )
        if contract.exercise_style is ExerciseStyle.AMERICAN:
            return max(contract.strike_price, european)
        return european

    def _curve_tree_solve(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        term_structure: DeterministicTermStructure,
        *,
        minimum_steps: int,
        capture_early_layers: bool,
    ) -> _CurveTreeSolution:
        (
            steps,
            delta_t,
            up,
            funding_rates,
            carry_rates,
            probabilities,
            discounts,
        ) = self._curve_tree_parameters(
            contract,
            volatility,
            term_structure,
            minimum_steps,
        )

        event_amounts: dict[int, float] = {}
        aligned_times: list[float] = []
        max_alignment_error = 0.0
        for dividend in market_data.cash_dividends.dividends:
            event_index = min(
                steps - 1,
                max(1, math.floor(dividend.ex_time / delta_t + 0.5)),
            )
            aligned_time = event_index * delta_t
            event_amounts[event_index] = event_amounts.get(event_index, 0.0) + dividend.amount
            aligned_times.append(aligned_time)
            max_alignment_error = max(
                max_alignment_error,
                abs(aligned_time - dividend.ex_time),
            )

        log_up = math.log(up)
        node_indices = np.arange(steps + 1)
        log_price_span = volatility * math.sqrt(contract.time_to_expiry * steps)
        max_terminal_log_price = math.log(market_data.spot_price) + log_price_span
        min_terminal_log_price = math.log(market_data.spot_price) - log_price_span
        if max_terminal_log_price > math.log(np.finfo(float).max):
            raise ValueError(
                "curve-aware CRR tree exceeds the supported floating-point range; "
                "reduce volatility, maturity, or steps"
            )
        if market_data.cash_dividends and min_terminal_log_price < math.log(np.finfo(float).tiny):
            raise ValueError(
                "cash-dividend curve-aware CRR tree enters the subnormal floating-point "
                "range; increase spot or reduce volatility, maturity, or steps"
            )
        prices = np.exp(math.log(market_data.spot_price) + (2 * node_indices - steps) * log_up)
        if contract.option_type is OptionType.CALL:
            values = np.maximum(prices - contract.strike_price, 0.0)
        else:
            values = np.maximum(contract.strike_price - prices, 0.0)

        first_step_values: np.ndarray | None = None
        first_step_prices: np.ndarray | None = None
        second_step_values: np.ndarray | None = None
        second_step_prices: np.ndarray | None = None
        for index in range(steps - 1, -1, -1):
            probability = probabilities[index]
            values = discounts[index] * (
                probability * values[1:] + (1.0 - probability) * values[:-1]
            )
            prices = prices[:-1] * up

            amount = event_amounts.get(index)
            if amount is not None:
                if contract.exercise_style is ExerciseStyle.AMERICAN:
                    if contract.option_type is OptionType.CALL:
                        post_event_exercise = np.maximum(prices - contract.strike_price, 0.0)
                    else:
                        post_event_exercise = np.maximum(contract.strike_price - prices, 0.0)
                    values = np.maximum(values, post_event_exercise)
                targets = np.maximum(prices - amount, 0.0)
                zero_value = self._curve_cash_boundary_value(
                    contract,
                    index * delta_t,
                    term_structure,
                )
                values = np.interp(
                    targets,
                    np.concatenate(([0.0], prices)),
                    np.concatenate(([zero_value], values)),
                )

            if contract.exercise_style is ExerciseStyle.AMERICAN:
                if contract.option_type is OptionType.CALL:
                    exercise_value = np.maximum(prices - contract.strike_price, 0.0)
                else:
                    exercise_value = np.maximum(contract.strike_price - prices, 0.0)
                values = np.maximum(values, exercise_value)

            if capture_early_layers and index == 2:
                second_step_values = values.copy()
                second_step_prices = prices.copy()
            if capture_early_layers and index == 1:
                first_step_values = values.copy()
                first_step_prices = prices.copy()

        if not np.isfinite(values).all():
            raise ValueError("curve-aware CRR tree produced non-finite values")
        return _CurveTreeSolution(
            price=float(values[0]),
            steps=steps,
            delta_t=delta_t,
            first_step_values=first_step_values,
            first_step_prices=first_step_prices,
            second_step_values=second_step_values,
            second_step_prices=second_step_prices,
            aligned_cash_times=tuple(aligned_times),
            cash_alignment_error=max_alignment_error,
            funding_rates=funding_rates,
            carry_rates=carry_rates,
            probabilities=probabilities,
            discounts=discounts,
        )

    def calculate_price_curve_aware(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        term_structure: DeterministicTermStructure,
    ) -> PricingResult:
        """Price with exact per-step deterministic funding/carry factors."""

        start = time.perf_counter()
        try:
            validate_pricing_parameters(contract, market_data, volatility)
            validate_discrete_dividends(contract, market_data)
            self._validate_term_structure(contract, market_data, term_structure)
            solution = self._curve_tree_solve(
                contract,
                market_data,
                volatility,
                term_structure,
                minimum_steps=self.steps,
                capture_early_layers=True,
            )

            delta: float | None = None
            gamma: float | None = None
            theta: float | None = None
            if solution.first_step_values is not None and solution.first_step_prices is not None:
                denominator = solution.first_step_prices[1] - solution.first_step_prices[0]
                if denominator != 0.0:
                    delta = float(
                        (solution.first_step_values[1] - solution.first_step_values[0])
                        / denominator
                    )
            if solution.second_step_values is not None and solution.second_step_prices is not None:
                down_denominator = solution.second_step_prices[1] - solution.second_step_prices[0]
                up_denominator = solution.second_step_prices[2] - solution.second_step_prices[1]
                root_denominator = 0.5 * (
                    solution.second_step_prices[2] - solution.second_step_prices[0]
                )
                if down_denominator and up_denominator and root_denominator:
                    delta_down = (
                        solution.second_step_values[1] - solution.second_step_values[0]
                    ) / down_denominator
                    delta_up = (
                        solution.second_step_values[2] - solution.second_step_values[1]
                    ) / up_denominator
                    gamma = float((delta_up - delta_down) / root_denominator)
                theta = float(
                    (solution.second_step_values[1] - solution.price)
                    / (2.0 * solution.delta_t)
                    / 365.0
                )

            volatility_bump = max(1e-4, volatility * 1e-4)
            volatility_low = max(1.000001e-6, volatility - volatility_bump)
            volatility_high = min(5.0, volatility + volatility_bump)
            low_price = self._curve_tree_solve(
                contract,
                market_data,
                volatility_low,
                term_structure,
                minimum_steps=solution.steps,
                capture_early_layers=False,
            ).price
            high_price = self._curve_tree_solve(
                contract,
                market_data,
                volatility_high,
                term_structure,
                minimum_steps=solution.steps,
                capture_early_layers=False,
            ).price
            vega = (high_price - low_price) / (volatility_high - volatility_low) / 100.0

            numerical_outputs = (solution.price, delta, gamma, theta, vega)
            if not all(value is None or math.isfinite(float(value)) for value in numerical_outputs):
                raise ValueError("curve-aware CRR tree produced a non-finite price or Greek")

            actual_time_mesh_list = [
                index * solution.delta_t for index in range(solution.steps + 1)
            ]
            actual_time_mesh_list[-1] = contract.time_to_expiry
            actual_time_mesh = tuple(actual_time_mesh_list)
            alignment_tolerance = 2e-13 * max(1.0, contract.time_to_expiry)
            curve_node_aligned = all(
                any(
                    math.isclose(node, time_value, rel_tol=0.0, abs_tol=alignment_tolerance)
                    for time_value in actual_time_mesh
                )
                for node in term_structure.curve_node_times
            )
            cash_aligned = solution.cash_alignment_error <= alignment_tolerance
            diagnostics = term_structure.diagnostics()
            diagnostics.update(
                {
                    "configured_steps": self.steps,
                    "effective_steps": solution.steps,
                    "time_step": solution.delta_t,
                    "actual_time_mesh": actual_time_mesh,
                    "effective_step_funding_rates": solution.funding_rates,
                    "effective_step_carry_rates": solution.carry_rates,
                    "min_local_funding_rate": min(solution.funding_rates),
                    "max_local_funding_rate": max(solution.funding_rates),
                    "min_local_carry_rate": min(solution.carry_rates),
                    "max_local_carry_rate": max(solution.carry_rates),
                    "risk_neutral_probabilities": solution.probabilities,
                    "min_risk_neutral_probability": min(solution.probabilities),
                    "max_risk_neutral_probability": max(solution.probabilities),
                    "probability_clipping": False,
                    "step_discount_factors": solution.discounts,
                    "curve_node_alignment_status": curve_node_aligned,
                    "curve_node_alignment_required": False,
                    "cash_dividend_count": len(market_data.cash_dividends),
                    "cash_dividend_effective_jump_count": len(set(solution.aligned_cash_times)),
                    "cash_dividend_jump_model": "limited_liability_spot_jump",
                    "cash_dividend_interpolation": (
                        "piecewise_linear" if market_data.cash_dividends else None
                    ),
                    "cash_dividend_interpolation_accuracy": (
                        "event interpolation can reduce global tree convergence to first order"
                        if market_data.cash_dividends
                        else None
                    ),
                    "cash_dividend_requested_ex_times": tuple(
                        dividend.ex_time for dividend in market_data.cash_dividends.dividends
                    ),
                    "cash_dividend_amounts": tuple(
                        dividend.amount for dividend in market_data.cash_dividends.dividends
                    ),
                    "cash_dividend_aligned_ex_times": solution.aligned_cash_times,
                    "cash_dividend_max_time_alignment_error": (solution.cash_alignment_error),
                    "cash_dividend_alignment_status": cash_aligned,
                    "cash_dividend_event_curve_factors": tuple(
                        {
                            "ex_time": dividend.ex_time,
                            "discount_to_expiry": term_structure.discount_factor(
                                dividend.ex_time, contract.time_to_expiry
                            ),
                            "carry_to_expiry": term_structure.carry_factor(
                                dividend.ex_time, contract.time_to_expiry
                            ),
                            "growth_to_expiry": term_structure.growth_factor(
                                dividend.ex_time, contract.time_to_expiry
                            ),
                        }
                        for dividend in market_data.cash_dividends.dividends
                    ),
                    "cash_dividend_schedule_id": (
                        market_data.cash_dividends.schedule_id
                        if market_data.cash_dividends
                        else None
                    ),
                    "vega_method": "central_finite_difference",
                    "rho_method": "not_reported_without_parallel_curve_bump",
                }
            )
            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=solution.price,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=float(vega),
                rho=None,
                computation_time_ms=(time.perf_counter() - start) * 1000.0,
                model_used=(
                    "binomial_curve_aware"
                    f"{'_cash_dividend' if market_data.cash_dividends else ''}"
                    f"_{solution.steps}"
                ),
                implied_volatility=volatility,
                numerical_diagnostics=diagnostics,
            )
        except Exception:  # pragma: no cover - preserve context for API error mapping
            LOGGER.exception("Curve-aware binomial pricing failed")
            raise

    @staticmethod
    def _cash_boundary_value(
        contract: OptionContract,
        market_data: MarketData,
        forward_time: float,
    ) -> float:
        """Value at zero spot, used when a cash jump reaches limited liability."""

        if contract.option_type is OptionType.CALL:
            return 0.0
        remaining = max(contract.time_to_expiry - forward_time, 0.0)
        european = contract.strike_price * math.exp(-market_data.risk_free_rate * remaining)
        if contract.exercise_style is ExerciseStyle.AMERICAN:
            return max(contract.strike_price, european)
        return european

    def _cash_tree_parameters(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        minimum_steps: int,
    ) -> tuple[int, float, float, float, float, float]:
        steps = minimum_steps
        while True:
            delta_t = contract.time_to_expiry / steps
            up = math.exp(volatility * math.sqrt(delta_t))
            down = 1.0 / up
            growth = math.exp((market_data.risk_free_rate - market_data.dividend_yield) * delta_t)
            probability = (growth - down) / (up - down)
            if 0.0 < probability < 1.0:
                return (
                    steps,
                    delta_t,
                    up,
                    down,
                    probability,
                    math.exp(-market_data.risk_free_rate * delta_t),
                )
            if steps >= self.MAX_STEPS:
                raise ValueError(
                    "CRR tree cannot satisfy no-arbitrage probability; "
                    "increase volatility or use the finite-difference model"
                )
            steps = min(self.MAX_STEPS, steps * 2)

    def _cash_tree_solve(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        minimum_steps: int,
        capture_early_layers: bool,
    ) -> tuple[
        float,
        int,
        float,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        tuple[float, ...],
        float,
    ]:
        """Solve a recombining interpolation tree with explicit cash jumps."""

        (
            steps,
            delta_t,
            up,
            _down,
            probability,
            discount,
        ) = self._cash_tree_parameters(contract, market_data, volatility, minimum_steps)

        event_amounts: dict[int, float] = {}
        aligned_times: list[float] = []
        max_alignment_error = 0.0
        for dividend in market_data.cash_dividends.dividends:
            event_index = min(
                steps - 1,
                max(1, math.floor(dividend.ex_time / delta_t + 0.5)),
            )
            aligned_time = event_index * delta_t
            event_amounts[event_index] = event_amounts.get(event_index, 0.0) + dividend.amount
            aligned_times.append(aligned_time)
            max_alignment_error = max(
                max_alignment_error,
                abs(aligned_time - dividend.ex_time),
            )

        log_up = math.log(up)
        node_indices = np.arange(steps + 1)
        log_price_span = volatility * math.sqrt(contract.time_to_expiry * steps)
        max_terminal_log_price = math.log(market_data.spot_price) + log_price_span
        min_terminal_log_price = math.log(market_data.spot_price) - log_price_span
        if max_terminal_log_price > math.log(np.finfo(float).max):
            raise ValueError(
                "CRR tree exceeds the supported floating-point range; "
                "reduce volatility, maturity, or steps"
            )
        if min_terminal_log_price < math.log(np.finfo(float).tiny):
            raise ValueError(
                "cash-dividend CRR tree enters the subnormal floating-point range; "
                "increase spot or reduce volatility, maturity, or steps"
            )
        prices = np.exp(math.log(market_data.spot_price) + (2 * node_indices - steps) * log_up)
        if contract.option_type is OptionType.CALL:
            values = np.maximum(prices - contract.strike_price, 0.0)
        else:
            values = np.maximum(contract.strike_price - prices, 0.0)

        first_step_values: np.ndarray | None = None
        first_step_prices: np.ndarray | None = None
        second_step_values: np.ndarray | None = None
        second_step_prices: np.ndarray | None = None

        for index in range(steps - 1, -1, -1):
            values = discount * (probability * values[1:] + (1.0 - probability) * values[:-1])
            prices = prices[:-1] * up

            # At an ex-time, continuation first arrives at the post-dividend
            # value function. Map the pre-dividend state S to max(S-D, 0).
            amount = event_amounts.get(index)
            if amount is not None:
                if contract.exercise_style is ExerciseStyle.AMERICAN:
                    if contract.option_type is OptionType.CALL:
                        post_event_exercise = np.maximum(prices - contract.strike_price, 0.0)
                    else:
                        post_event_exercise = np.maximum(contract.strike_price - prices, 0.0)
                    values = np.maximum(values, post_event_exercise)
                targets = np.maximum(prices - amount, 0.0)
                zero_value = self._cash_boundary_value(contract, market_data, index * delta_t)
                interpolation_spots = np.concatenate(([0.0], prices))
                interpolation_values = np.concatenate(([zero_value], values))
                values = np.interp(
                    targets,
                    interpolation_spots,
                    interpolation_values,
                )

            if contract.exercise_style is ExerciseStyle.AMERICAN:
                if contract.option_type is OptionType.CALL:
                    exercise_value = np.maximum(prices - contract.strike_price, 0.0)
                else:
                    exercise_value = np.maximum(contract.strike_price - prices, 0.0)
                values = np.maximum(values, exercise_value)

            if capture_early_layers and index == 2:
                second_step_values = values.copy()
                second_step_prices = prices.copy()
            if capture_early_layers and index == 1:
                first_step_values = values.copy()
                first_step_prices = prices.copy()

        if not np.isfinite(values).all():
            raise ValueError("cash-dividend CRR tree produced non-finite values")
        return (
            float(values[0]),
            steps,
            delta_t,
            first_step_values,
            first_step_prices,
            second_step_values,
            second_step_prices,
            tuple(aligned_times),
            max_alignment_error,
        )

    def _calculate_discrete_dividend_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        start: float,
    ) -> PricingResult:
        (
            price,
            steps,
            delta_t,
            first_values,
            first_prices,
            second_values,
            second_prices,
            aligned_times,
            alignment_error,
        ) = self._cash_tree_solve(
            contract,
            market_data,
            volatility,
            minimum_steps=self.steps,
            capture_early_layers=True,
        )

        delta: float | None = None
        gamma: float | None = None
        theta: float | None = None
        if first_values is not None and first_prices is not None:
            denominator = first_prices[1] - first_prices[0]
            if denominator != 0.0:
                delta = float((first_values[1] - first_values[0]) / denominator)
        if second_values is not None and second_prices is not None:
            down_denominator = second_prices[1] - second_prices[0]
            up_denominator = second_prices[2] - second_prices[1]
            root_denominator = 0.5 * (second_prices[2] - second_prices[0])
            if down_denominator and up_denominator and root_denominator:
                delta_down = (second_values[1] - second_values[0]) / down_denominator
                delta_up = (second_values[2] - second_values[1]) / up_denominator
                gamma = float((delta_up - delta_down) / root_denominator)
            theta = float((second_values[1] - price) / (2.0 * delta_t) / 365.0)

        volatility_bump = max(1e-4, volatility * 1e-4)
        volatility_low = max(1.000001e-6, volatility - volatility_bump)
        volatility_high = min(5.0, volatility + volatility_bump)
        low_price = self._cash_tree_solve(
            contract,
            market_data,
            volatility_low,
            minimum_steps=steps,
            capture_early_layers=False,
        )[0]
        high_price = self._cash_tree_solve(
            contract,
            market_data,
            volatility_high,
            minimum_steps=steps,
            capture_early_layers=False,
        )[0]
        vega = (high_price - low_price) / (volatility_high - volatility_low) / 100.0

        rho: float | None = None
        if self._compute_rho:
            rate_bump = 1e-4
            rate_low = max(-1.0, market_data.risk_free_rate - rate_bump)
            rate_high = min(1.0, market_data.risk_free_rate + rate_bump)
            if rate_high > rate_low:
                low_market = MarketData(
                    spot_price=market_data.spot_price,
                    risk_free_rate=rate_low,
                    dividend_yield=market_data.dividend_yield,
                    timestamp=market_data.timestamp,
                    cash_dividends=market_data.cash_dividends,
                )
                high_market = MarketData(
                    spot_price=market_data.spot_price,
                    risk_free_rate=rate_high,
                    dividend_yield=market_data.dividend_yield,
                    timestamp=market_data.timestamp,
                    cash_dividends=market_data.cash_dividends,
                )
                rate_low_price = self._cash_tree_solve(
                    contract,
                    low_market,
                    volatility,
                    minimum_steps=steps,
                    capture_early_layers=False,
                )[0]
                rate_high_price = self._cash_tree_solve(
                    contract,
                    high_market,
                    volatility,
                    minimum_steps=steps,
                    capture_early_layers=False,
                )[0]
                rho = (rate_high_price - rate_low_price) / (rate_high - rate_low) / 100.0

        numerical_outputs = (price, vega, delta, gamma, theta, rho)
        if not all(value is None or math.isfinite(float(value)) for value in numerical_outputs):
            raise ValueError("cash-dividend CRR tree produced a non-finite price or Greek")

        return PricingResult(
            contract_id=contract.contract_id,
            theoretical_price=price,
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=float(vega),
            rho=float(rho) if rho is not None else None,
            computation_time_ms=(time.perf_counter() - start) * 1000.0,
            model_used=f"binomial_cash_dividend_{steps}",
            implied_volatility=volatility,
            numerical_diagnostics={
                "cash_dividend_count": len(market_data.cash_dividends),
                "cash_dividend_effective_jump_count": len(set(aligned_times)),
                "cash_dividend_jump_model": "limited_liability_spot_jump",
                "cash_dividend_interpolation": "piecewise_linear",
                "cash_dividend_requested_ex_times": tuple(
                    dividend.ex_time for dividend in market_data.cash_dividends.dividends
                ),
                "cash_dividend_amounts": tuple(
                    dividend.amount for dividend in market_data.cash_dividends.dividends
                ),
                "cash_dividend_aligned_ex_times": aligned_times,
                "cash_dividend_max_time_alignment_error": alignment_error,
                "cash_dividend_schedule_id": market_data.cash_dividends.schedule_id,
                "configured_steps": self.steps,
                "effective_steps": steps,
                "time_step": delta_t,
                "vega_method": "central_finite_difference",
            },
        )


__all__ = ["BinomialModel", "binomial_price"]
