"""Black-Scholes valuation and analytic Greeks."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

from ..term_structure import DeterministicTermStructure
from ..utils.validation import reject_discrete_dividends, validate_pricing_parameters
from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from .pricing_common import (
    INV_SQRT_TWO_PI,
    LOG_MONEYNESS_CLAMP,
    SIGMA_MIN,
    SQRT_TWO,
    TAU_MIN,
    PriceResult,
    _normalise_option_type,
)

LOGGER = logging.getLogger(__name__)


def black_scholes_price(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
) -> PriceResult:
    """Convenience wrapper returning a :class:`PriceResult` for Black-Scholes."""

    opt = _normalise_option_type(option_type)
    contract = OptionContract(
        symbol="BS",
        strike_price=strike,
        time_to_expiry=tau,
        option_type=OptionType.CALL if opt == "call" else OptionType.PUT,
    )
    market = MarketData(spot_price=spot, risk_free_rate=r, dividend_yield=q)
    model = BlackScholesModel()
    result = model.calculate_price(contract, market, sigma)
    meta: dict[str, object] = {
        "method": "black_scholes",
        "option_type": opt,
        "policy_flags": [],
    }
    return PriceResult(
        price=result.theoretical_price, ci_half_width=0.0, meta=meta, standard_error=0.0
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
) -> tuple[float, float, float, float, float, float]:
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
        # Tail-safe formulation to reduce catastrophic cancellation
        tail_call = strike * discount_rate * _norm_cdf(-d2) - spot * discount_dividend * _norm_cdf(
            -d1
        )
        call_price = parity_anchor + tail_call
    else:
        call_price = spot * discount_dividend * cdf_d1 - strike * discount_rate * cdf_d2

    call_delta = discount_dividend * cdf_d1
    call_gamma = discount_dividend * pdf / (spot * volatility * sqrt_t)
    call_theta = (
        -spot * discount_dividend * pdf * volatility / (2.0 * sqrt_t)
        + dividend * spot * discount_dividend * cdf_d1
        - rate * strike * discount_rate * cdf_d2
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
        theta = (
            call_theta
            + (-dividend * spot * discount_dividend + rate * strike * discount_rate) / 365.0
        )
        vega = call_vega
        rho = call_rho - time_to_expiry * strike * discount_rate / 100.0

    return price, delta, gamma, theta, vega, rho


def _black_scholes_factor_price_and_greeks(
    contract: OptionContract,
    spot: float,
    volatility: float,
    discount_factor: float,
    carry_factor: float,
) -> tuple[float, float, float, float]:
    """Exact European Black--Scholes result from terminal curve factors."""

    maturity = contract.time_to_expiry
    sqrt_t = math.sqrt(maturity)
    volatility_sqrt_t = volatility * sqrt_t
    log_moneyness = math.log(spot / contract.strike_price)
    log_growth = math.log(carry_factor / discount_factor)
    d1 = (log_moneyness + log_growth + 0.5 * volatility**2 * maturity) / volatility_sqrt_t
    d2 = d1 - volatility_sqrt_t
    pdf = _norm_pdf(d1)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)
    parity_anchor = spot * carry_factor - contract.strike_price * discount_factor

    if log_moneyness > _log_moneyness_threshold(maturity):
        tail_call = contract.strike_price * discount_factor * _norm_cdf(
            -d2
        ) - spot * carry_factor * _norm_cdf(-d1)
        call_price = parity_anchor + tail_call
    else:
        call_price = spot * carry_factor * cdf_d1 - contract.strike_price * discount_factor * cdf_d2
    call_delta = carry_factor * cdf_d1
    gamma = carry_factor * pdf / (spot * volatility_sqrt_t)
    vega = spot * carry_factor * pdf * sqrt_t / 100.0
    if contract.option_type is OptionType.CALL:
        return call_price, call_delta, gamma, vega
    return call_price - parity_anchor, call_delta - carry_factor, gamma, vega


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
            if contract.exercise_style is not ExerciseStyle.EUROPEAN:
                raise ValueError("Black-Scholes supports European exercise only")
            validate_pricing_parameters(contract, market_data, volatility)
            reject_discrete_dividends(market_data, "Black-Scholes")
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
        except Exception:  # pragma: no cover - preserve context for API error mapping
            LOGGER.exception("Black-Scholes pricing failed")
            raise

    def calculate_price_curve_aware(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        term_structure: DeterministicTermStructure,
    ) -> PricingResult:
        """Price exactly from deterministic expiry discount/carry factors.

        Parallel-curve rho and calendar theta are intentionally not reported:
        neither Greek has a unique meaning without specifying how the dated
        curves roll or are bumped.  Delta, gamma, and vega remain exact under
        the deterministic-term-structure Black--Scholes assumptions.
        """

        start = time.perf_counter()
        try:
            if contract.exercise_style is not ExerciseStyle.EUROPEAN:
                raise ValueError("Black-Scholes supports European exercise only")
            validate_pricing_parameters(contract, market_data, volatility)
            reject_discrete_dividends(market_data, "curve-aware Black-Scholes")
            if not isinstance(term_structure, DeterministicTermStructure):
                raise TypeError("term_structure must be a DeterministicTermStructure")
            if not math.isclose(
                term_structure.maturity,
                contract.time_to_expiry,
                rel_tol=0.0,
                abs_tol=1e-14 * max(1.0, contract.time_to_expiry),
            ):
                raise ValueError("term-structure maturity must match the option contract")
            discount = term_structure.discount_factor(0.0, term_structure.maturity)
            carry = term_structure.carry_factor(0.0, term_structure.maturity)
            price, delta, gamma, vega = _black_scholes_factor_price_and_greeks(
                contract,
                market_data.spot_price,
                volatility,
                discount,
                carry,
            )
            diagnostics = term_structure.diagnostics()
            diagnostics.update(
                {
                    "analytic_curve_reduction": "exact_terminal_factors",
                    "curve_theta": "not_reported_without_curve_roll_convention",
                    "curve_rho": "not_reported_without_parallel_bump_convention",
                    "put_call_parity_anchor": (
                        market_data.spot_price * carry - contract.strike_price * discount
                    ),
                }
            )
            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=max(0.0, price),
                delta=delta,
                gamma=gamma,
                theta=None,
                vega=vega,
                rho=None,
                implied_volatility=volatility,
                computation_time_ms=(time.perf_counter() - start) * 1000.0,
                model_used="black_scholes_curve_aware",
                numerical_diagnostics=diagnostics,
            )
        except Exception:  # pragma: no cover - preserve context for API error mapping
            LOGGER.exception("Curve-aware Black-Scholes pricing failed")
            raise


__all__ = [
    "BlackScholesModel",
    "_black_scholes_factor_price_and_greeks",
    "_black_scholes_greeks",
    "black_scholes_price",
]
