"""Tests for the high level pricing engine."""

from __future__ import annotations

import pytest

from options_engine.core.models import MarketData, OptionContract, OptionType
from options_engine.core.pricing_engine import OptionsEngine


def _make_contract(symbol: str, strike: float) -> OptionContract:
    return OptionContract(
        symbol=symbol,
        strike_price=strike,
        time_to_expiry=1.0,
        option_type=OptionType.CALL,
    )


def test_portfolio_pricing_preserves_order_and_respects_override() -> None:
    engine = OptionsEngine(num_threads=2)
    contracts = [_make_contract("TEST1", 90.0), _make_contract("TEST2", 110.0)]
    market_data = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)

    default_results = engine.price_portfolio(contracts, market_data, model_name="black_scholes")
    override_results = engine.price_portfolio(
        contracts,
        market_data,
        model_name="black_scholes",
        override_volatility=0.35,
    )

    assert [result["contract_id"] for result in override_results] == [
        contract.contract_id for contract in contracts
    ]
    assert override_results[0]["theoretical_price"] != pytest.approx(
        default_results[0]["theoretical_price"]
    )


def test_calculate_portfolio_greeks_handles_empty_input() -> None:
    totals = OptionsEngine.calculate_portfolio_greeks([])
    assert totals["delta"] == 0.0
    assert totals["position_count"] == 0.0
