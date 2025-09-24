"""Helpers for converting API schemas into domain models."""

from __future__ import annotations

from ..core.models import (
    ExerciseStyle as DomainExerciseStyle,
    MarketData,
    OptionContract,
    OptionType as DomainOptionType,
)
from .schemas.request import OptionContractRequest, PricingRequest


def to_option_contract(contract: OptionContractRequest) -> OptionContract:
    """Convert an API contract request into the domain representation."""

    return OptionContract(
        symbol=contract.symbol,
        strike_price=contract.strike_price,
        time_to_expiry=contract.time_to_expiry,
        option_type=DomainOptionType(contract.option_type.value),
        exercise_style=DomainExerciseStyle(contract.exercise_style.value),
    )


def to_market_data(request: PricingRequest) -> MarketData:
    """Extract domain market data from the pricing request."""

    data = request.market_data
    return MarketData(
        spot_price=data.spot_price,
        risk_free_rate=data.risk_free_rate,
        dividend_yield=data.dividend_yield,
    )
