"""Tests covering the pricing API routes."""

from __future__ import annotations

import asyncio

from fastapi import BackgroundTasks

from options_engine.api.routes import pricing
from options_engine.api.schemas.request import (
    ExerciseStyle,
    MarketDataRequest,
    OptionContractRequest,
    OptionType,
    PricingModel,
    PricingRequest,
)


def test_single_route_respects_calculate_greeks_flag() -> None:
    """The single pricing endpoint should omit Greeks when not requested."""

    request = PricingRequest(
        contracts=[
            OptionContractRequest(
                symbol="TQA",
                strike_price=100.0,
                time_to_expiry=1.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN,
                quantity=5,
            )
        ],
        market_data=MarketDataRequest(
            spot_price=105.0,
            risk_free_rate=0.01,
            dividend_yield=0.0,
            volatility=0.2,
        ),
        model=PricingModel.BLACK_SCHOLES,
        calculate_greeks=False,
    )

    response = asyncio.run(pricing.single(request, BackgroundTasks()))

    assert response.portfolio_greeks is None
    assert len(response.results) == 1
