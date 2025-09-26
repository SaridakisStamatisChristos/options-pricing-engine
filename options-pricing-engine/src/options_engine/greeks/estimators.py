"""Monte Carlo greek estimators and finite-difference fallbacks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from options_engine.core.models import MarketData, OptionContract, OptionType

from .stability import (
    CI_Z_VALUE,
    SIGMA_FLOOR,
    clamp_log_moneyness,
    clamp_sigma,
    clamp_tau,
    contributions_finite,
    guard_against_pathologies,
    half_width,
    safe_spot,
    standard_error,
)


@dataclass(slots=True)
class GreekSummary:
    """Container representing a Monte Carlo greek estimate."""

    value: float
    standard_error: float
    half_width_abs: float
    contributions: np.ndarray


def aggregate_statistics(contributions: np.ndarray) -> GreekSummary:
    """Aggregate an array of per-path contributions into summary statistics."""

    sample = np.asarray(contributions, dtype=float)
    if sample.size == 0:
        return GreekSummary(value=0.0, standard_error=math.inf, half_width_abs=math.inf, contributions=sample)
    estimate = float(np.mean(sample))
    se = standard_error(sample)
    hw = half_width(se, CI_Z_VALUE)
    return GreekSummary(value=estimate, standard_error=se, half_width_abs=hw, contributions=sample)


def _indicator(contract: OptionContract, terminal_prices: np.ndarray) -> np.ndarray:
    if contract.option_type is OptionType.CALL:
        return np.greater(terminal_prices, contract.strike_price)
    return np.less(terminal_prices, contract.strike_price)


def pathwise_delta(
    contract: OptionContract,
    market_data: MarketData,
    *,
    discount_factor: float,
    terminal_prices: np.ndarray,
) -> np.ndarray:
    """Return per-path delta contributions using the pathwise method."""

    indicator = _indicator(contract, terminal_prices)
    spot = safe_spot(market_data.spot_price)
    contributions = discount_factor * np.where(indicator, terminal_prices / spot, 0.0)
    if contract.option_type is OptionType.PUT:
        contributions = -contributions
    return contributions


def pathwise_gamma(
    contract: OptionContract,
    market_data: MarketData,
    *,
    discount_factor: float,
    terminal_prices: np.ndarray,
    volatility: float,
    time_to_expiry: float,
) -> np.ndarray:
    """Return per-path gamma contributions via mixed pathwise / LR estimator."""

    delta_paths = pathwise_delta(
        contract,
        market_data,
        discount_factor=discount_factor,
        terminal_prices=terminal_prices,
    )
    tau = clamp_tau(time_to_expiry)
    sigma = clamp_sigma(volatility)
    sigma_sq = sigma**2
    if tau <= 0.0 or sigma_sq <= 0.0:
        return np.zeros_like(delta_paths)
    log_ratio = np.log(np.maximum(terminal_prices, 1e-16) / safe_spot(market_data.spot_price))
    log_ratio = clamp_log_moneyness(log_ratio)
    score_s0 = (log_ratio - (market_data.risk_free_rate - market_data.dividend_yield + 0.5 * sigma_sq) * tau) / (
        sigma_sq * tau * safe_spot(market_data.spot_price)
    )
    contributions = delta_paths * score_s0
    return contributions


def pathwise_vega(
    contract: OptionContract,
    market_data: MarketData,
    *,
    discount_factor: float,
    terminal_prices: np.ndarray,
    volatility: float,
    time_to_expiry: float,
    draws: np.ndarray,
) -> np.ndarray:
    """Return per-path vega contributions using the pathwise method."""

    indicator = _indicator(contract, terminal_prices)
    tau = clamp_tau(time_to_expiry)
    sigma = clamp_sigma(volatility)
    sqrt_tau = math.sqrt(tau)
    sensitivity_term = terminal_prices * (sqrt_tau * draws - sigma * tau)
    contributions = discount_factor * np.where(indicator, sensitivity_term, 0.0)
    if contract.option_type is OptionType.PUT:
        contributions = -contributions
    return contributions / 100.0


def theta_likelihood_ratio(
    contract: OptionContract,
    market_data: MarketData,
    *,
    payoff: np.ndarray,
    discount_factor: float,
    terminal_prices: np.ndarray,
    volatility: float,
    time_to_expiry: float,
) -> np.ndarray:
    """Return per-path theta contributions using the LR/score-function method."""

    tau = clamp_tau(time_to_expiry)
    sigma = clamp_sigma(volatility)
    sigma_sq = sigma**2
    spot = safe_spot(market_data.spot_price)
    log_ratio = clamp_log_moneyness(np.log(np.maximum(terminal_prices, 1e-16) / spot))
    drift = (
        market_data.risk_free_rate
        - market_data.dividend_yield
        - 0.5 * sigma_sq
    )
    deviation = log_ratio - drift * tau
    score_tau = (
        -0.5 / tau
        + (deviation**2 + 2.0 * deviation * drift * tau) / (2.0 * sigma_sq * tau**2)
    )
    discount_derivative = -market_data.risk_free_rate * discount_factor
    contributions = payoff * (discount_factor * score_tau + discount_derivative)
    return contributions / 365.0


def rho_likelihood_ratio(
    contract: OptionContract,
    market_data: MarketData,
    *,
    payoff: np.ndarray,
    discount_factor: float,
    terminal_prices: np.ndarray,
    volatility: float,
    time_to_expiry: float,
) -> np.ndarray:
    """Return per-path rho contributions using the LR/score-function method."""

    tau = clamp_tau(time_to_expiry)
    sigma = clamp_sigma(volatility)
    sigma_sq = sigma**2
    spot = safe_spot(market_data.spot_price)
    log_ratio = clamp_log_moneyness(np.log(np.maximum(terminal_prices, 1e-16) / spot))
    drift = (
        market_data.risk_free_rate
        - market_data.dividend_yield
        - 0.5 * sigma_sq
    )
    deviation = log_ratio - drift * tau
    score_r = deviation / sigma_sq
    discount_derivative = -tau * discount_factor
    contributions = payoff * (discount_factor * score_r + discount_derivative)
    return contributions / 100.0


def simulate_terminal_prices(
    spot: float,
    market_data: MarketData,
    volatility: float,
    time_to_expiry: float,
    draws: np.ndarray,
) -> np.ndarray:
    """Simulate terminal prices for the supplied parameters using common draws."""

    tau = max(float(time_to_expiry), 0.0)
    if tau == 0.0:
        return np.full_like(draws, fill_value=spot, dtype=float)
    sigma = max(float(volatility), 0.0)
    sqrt_tau = math.sqrt(tau)
    drift = (market_data.risk_free_rate - market_data.dividend_yield - 0.5 * sigma**2) * tau
    diffusion = sigma * sqrt_tau * draws
    return float(spot) * np.exp(drift + diffusion)


def _payoff(contract: OptionContract, terminal_prices: np.ndarray) -> np.ndarray:
    if contract.option_type is OptionType.CALL:
        return np.maximum(terminal_prices - contract.strike_price, 0.0)
    return np.maximum(contract.strike_price - terminal_prices, 0.0)


def finite_difference_delta(
    contract: OptionContract,
    market_data: MarketData,
    *,
    volatility: float,
    time_to_expiry: float,
    draws: np.ndarray,
    discounted_payoffs: np.ndarray,
) -> GreekSummary:
    spot = market_data.spot_price
    bump = max(1e-4 * spot, 1e-6)
    up_spot = spot + bump
    down_spot = max(spot - bump, 1e-6)
    discount_factor = math.exp(-market_data.risk_free_rate * time_to_expiry)
    up_terminal = simulate_terminal_prices(up_spot, market_data, volatility, time_to_expiry, draws)
    down_terminal = simulate_terminal_prices(down_spot, market_data, volatility, time_to_expiry, draws)
    up_payoff = _payoff(contract, up_terminal)
    down_payoff = _payoff(contract, down_terminal)
    up_disc = discount_factor * up_payoff
    down_disc = discount_factor * down_payoff
    h_up = up_spot - spot
    h_down = spot - down_spot
    if h_up <= 0.0 and h_down <= 0.0:
        contributions = np.zeros_like(discounted_payoffs)
    elif h_down <= 0.0:
        contributions = (up_disc - discounted_payoffs) / h_up
    elif h_up <= 0.0:
        contributions = (discounted_payoffs - down_disc) / h_down
    else:
        contributions = (up_disc - down_disc) / (h_up + h_down)
    return aggregate_statistics(contributions)


def finite_difference_gamma(
    contract: OptionContract,
    market_data: MarketData,
    *,
    volatility: float,
    time_to_expiry: float,
    draws: np.ndarray,
    discounted_payoffs: np.ndarray,
) -> GreekSummary:
    spot = market_data.spot_price
    bump = max(1e-4 * spot, 1e-6)
    up_spot = spot + bump
    down_spot = max(spot - bump, 1e-6)
    discount_factor = math.exp(-market_data.risk_free_rate * time_to_expiry)
    up_terminal = simulate_terminal_prices(up_spot, market_data, volatility, time_to_expiry, draws)
    down_terminal = simulate_terminal_prices(down_spot, market_data, volatility, time_to_expiry, draws)
    up_payoff = _payoff(contract, up_terminal)
    down_payoff = _payoff(contract, down_terminal)
    up_disc = discount_factor * up_payoff
    down_disc = discount_factor * down_payoff
    h_up = up_spot - spot
    h_down = spot - down_spot
    if h_up <= 0.0 or h_down <= 0.0:
        contributions = np.zeros_like(discounted_payoffs)
    else:
        numerator = h_down * up_disc - (h_up + h_down) * discounted_payoffs + h_up * down_disc
        contributions = 2.0 * numerator / (h_up * h_down * (h_up + h_down))
    return aggregate_statistics(contributions)


def finite_difference_vega(
    contract: OptionContract,
    market_data: MarketData,
    *,
    volatility: float,
    time_to_expiry: float,
    draws: np.ndarray,
    discounted_payoffs: np.ndarray,
) -> GreekSummary:
    sigma = volatility
    bump = max(1e-4 * sigma, 1e-6)
    up_sigma = sigma + bump
    down_sigma = max(sigma - bump, SIGMA_FLOOR)
    discount_factor = math.exp(-market_data.risk_free_rate * time_to_expiry)
    up_terminal = simulate_terminal_prices(market_data.spot_price, market_data, up_sigma, time_to_expiry, draws)
    down_terminal = simulate_terminal_prices(market_data.spot_price, market_data, down_sigma, time_to_expiry, draws)
    up_disc = discount_factor * _payoff(contract, up_terminal)
    down_disc = discount_factor * _payoff(contract, down_terminal)
    h_up = up_sigma - sigma
    h_down = sigma - down_sigma
    if h_up <= 0.0 and h_down <= 0.0:
        contributions = np.zeros_like(discounted_payoffs)
    elif h_down <= 0.0:
        contributions = (up_disc - discounted_payoffs) / h_up
    elif h_up <= 0.0:
        contributions = (discounted_payoffs - down_disc) / h_down
    else:
        contributions = (up_disc - down_disc) / (h_up + h_down)
    return aggregate_statistics(contributions / 100.0)


def finite_difference_theta(
    contract: OptionContract,
    market_data: MarketData,
    *,
    volatility: float,
    time_to_expiry: float,
    draws: np.ndarray,
    discounted_payoffs: np.ndarray,
) -> GreekSummary:
    tau = time_to_expiry
    bump = max(1e-4 * tau, 1e-7)
    up_tau = clamp_tau(tau + bump)
    down_tau = clamp_tau(max(tau - bump, 0.0))
    up_discount = math.exp(-market_data.risk_free_rate * up_tau)
    down_discount = math.exp(-market_data.risk_free_rate * down_tau)
    up_terminal = simulate_terminal_prices(market_data.spot_price, market_data, volatility, up_tau, draws)
    down_terminal = simulate_terminal_prices(market_data.spot_price, market_data, volatility, down_tau, draws)
    up_disc = up_discount * _payoff(contract, up_terminal)
    down_disc = down_discount * _payoff(contract, down_terminal)
    h_up = up_tau - time_to_expiry
    h_down = time_to_expiry - down_tau
    denominator = h_up + h_down
    if denominator <= 0.0:
        contributions = np.zeros_like(discounted_payoffs)
    elif h_down <= 0.0:
        contributions = (up_disc - discounted_payoffs) / h_up
    elif h_up <= 0.0:
        contributions = (discounted_payoffs - down_disc) / h_down
    else:
        contributions = (up_disc - down_disc) / denominator
    return aggregate_statistics(contributions / 365.0)


def finite_difference_rho(
    contract: OptionContract,
    market_data: MarketData,
    *,
    volatility: float,
    time_to_expiry: float,
    draws: np.ndarray,
    discounted_payoffs: np.ndarray,
) -> GreekSummary:
    rate = market_data.risk_free_rate
    bump = 1e-6
    up_rate = rate + bump
    down_rate = rate - bump
    up_market = MarketData(
        spot_price=market_data.spot_price,
        risk_free_rate=up_rate,
        dividend_yield=market_data.dividend_yield,
    )
    down_market = MarketData(
        spot_price=market_data.spot_price,
        risk_free_rate=down_rate,
        dividend_yield=market_data.dividend_yield,
    )
    up_discount = math.exp(-up_rate * time_to_expiry)
    down_discount = math.exp(-down_rate * time_to_expiry)
    up_terminal = simulate_terminal_prices(market_data.spot_price, up_market, volatility, time_to_expiry, draws)
    down_terminal = simulate_terminal_prices(market_data.spot_price, down_market, volatility, time_to_expiry, draws)
    up_disc = up_discount * _payoff(contract, up_terminal)
    down_disc = down_discount * _payoff(contract, down_terminal)
    h_up = up_rate - rate
    h_down = rate - down_rate
    if h_up <= 0.0 and h_down <= 0.0:
        contributions = np.zeros_like(discounted_payoffs)
    elif h_down <= 0.0:
        contributions = (up_disc - discounted_payoffs) / h_up
    elif h_up <= 0.0:
        contributions = (discounted_payoffs - down_disc) / h_down
    else:
        contributions = (up_disc - down_disc) / (h_up + h_down)
    return aggregate_statistics(contributions / 100.0)


def ensure_fd_inputs(
    contract: OptionContract,
    market_data: MarketData,
    *,
    volatility: float,
    time_to_expiry: float,
    draws: np.ndarray,
    discounted_payoffs: np.ndarray,
) -> None:
    """Validate FD harness inputs to ensure deterministic behaviour."""

    guard_against_pathologies([draws, discounted_payoffs])
