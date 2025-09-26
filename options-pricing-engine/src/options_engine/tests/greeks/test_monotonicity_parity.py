"""Monotonicity, convexity, and parity checks for Monte Carlo pricing."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pytest
from numpy.random import SeedSequence

from options_engine.core.models import MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import MonteCarloModel
from options_engine.tests.performance.test_pricing_benchmarks import BenchmarkScenario, _golden_grid


@pytest.fixture(scope="module")
def golden_grid() -> List[BenchmarkScenario]:
    return _golden_grid()


def _sigma_bump(volatility: float) -> float:
    return max(1e-4 * volatility, 1e-6)


def _tau_bump(time_to_expiry: float) -> float:
    return max(1e-4 * time_to_expiry, 1e-7)


def _strike_bump(strike: float) -> float:
    return max(1e-4 * strike, 1e-4)


@pytest.mark.parametrize("paths", [80_000])
def test_monotonicity_convexity_parity(golden_grid: List[BenchmarkScenario], paths: int) -> None:
    model = MonteCarloModel(paths=paths, antithetic=True, use_control_variates=True)

    seed_root = SeedSequence(4096)
    seed_pool = iter(seed_root.spawn(len(golden_grid)))
    seed_lookup: Dict[Tuple[float, float, float], SeedSequence] = {}
    market_lookup: Dict[Tuple[float, float, float], MarketData] = {}
    base_results: Dict[Tuple[OptionType, float, float, float], float] = {}
    tolerance = 5e-4
    parity_tolerance = 2e-4

    for scenario in golden_grid:
        contract = scenario.contract
        market = scenario.market
        volatility = scenario.volatility
        key = (contract.strike_price, contract.time_to_expiry, volatility)
        seed = seed_lookup.get(key)
        if seed is None:
            seed = next(seed_pool)
            seed_lookup[key] = seed
        market_lookup[key] = market

        base_result = model.calculate_price(contract, market, volatility, seed_sequence=seed)
        base_results[(contract.option_type, *key)] = base_result.theoretical_price

        # Volatility monotonicity
        bump_sigma = _sigma_bump(volatility)
        price_up_sigma = model.calculate_price(contract, market, volatility + bump_sigma, seed_sequence=seed)
        price_down_sigma = model.calculate_price(
            contract, market, max(volatility - bump_sigma, 1e-4), seed_sequence=seed
        )
        assert price_up_sigma.theoretical_price >= base_result.theoretical_price - tolerance
        assert price_down_sigma.theoretical_price <= base_result.theoretical_price + tolerance

        # Time monotonicity
        bump_tau = _tau_bump(contract.time_to_expiry)
        tau_up = contract.time_to_expiry + bump_tau
        tau_down = max(contract.time_to_expiry - bump_tau, 1e-6)
        contract_up = OptionContract(
            symbol=f"{contract.symbol}_tau_up",
            strike_price=contract.strike_price,
            time_to_expiry=tau_up,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )
        contract_down = OptionContract(
            symbol=f"{contract.symbol}_tau_dn",
            strike_price=contract.strike_price,
            time_to_expiry=tau_down,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )
        price_up_tau = model.calculate_price(contract_up, market, volatility, seed_sequence=seed)
        price_down_tau = model.calculate_price(contract_down, market, volatility, seed_sequence=seed)
        assert price_up_tau.theoretical_price >= base_result.theoretical_price - tolerance
        assert price_down_tau.theoretical_price <= base_result.theoretical_price + tolerance

        # Strike convexity
        bump_strike = _strike_bump(contract.strike_price)
        strike_up = contract.strike_price + bump_strike
        strike_down = max(contract.strike_price - bump_strike, 1e-4)
        contract_up_k = OptionContract(
            symbol=f"{contract.symbol}_k_up",
            strike_price=strike_up,
            time_to_expiry=contract.time_to_expiry,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )
        contract_down_k = OptionContract(
            symbol=f"{contract.symbol}_k_dn",
            strike_price=strike_down,
            time_to_expiry=contract.time_to_expiry,
            option_type=contract.option_type,
            exercise_style=contract.exercise_style,
        )
        price_up_k = model.calculate_price(contract_up_k, market, volatility, seed_sequence=seed)
        price_down_k = model.calculate_price(contract_down_k, market, volatility, seed_sequence=seed)
        lhs = base_result.theoretical_price
        rhs = 0.5 * (price_up_k.theoretical_price + price_down_k.theoretical_price)
        assert lhs <= rhs + tolerance

    # Put-call parity on shared keys
    for (strike, tau, vol), market in market_lookup.items():
        call_price = base_results.get((OptionType.CALL, strike, tau, vol))
        put_price = base_results.get((OptionType.PUT, strike, tau, vol))
        if call_price is None or put_price is None:
            continue
        forward_anchor = (
            market.spot_price * math.exp(-market.dividend_yield * tau)
            - strike * math.exp(-market.risk_free_rate * tau)
        )
        parity_gap = call_price - put_price - forward_anchor
        assert abs(parity_gap) <= parity_tolerance
