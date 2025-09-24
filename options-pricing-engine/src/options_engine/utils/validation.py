"""Validation helpers for pricing inputs."""

from __future__ import annotations

from typing import Final

from ..core.models import MarketData, OptionContract

MAX_VOLATILITY: Final[float] = 5.0
MIN_VOLATILITY: Final[float] = 1e-6
MIN_TIME_TO_EXPIRY: Final[float] = 1e-6


def validate_pricing_parameters(
    contract: OptionContract,
    market_data: MarketData,
    volatility: float,
) -> None:
    """Validate that inputs to a pricing model are well formed."""

    if volatility <= MIN_VOLATILITY or volatility > MAX_VOLATILITY:
        raise ValueError("volatility is outside the supported range")

    if contract.time_to_expiry <= MIN_TIME_TO_EXPIRY:
        raise ValueError("time_to_expiry is too small for stable pricing")

    if market_data.spot_price <= 0:
        raise ValueError("spot_price must be strictly positive")

    if contract.strike_price <= 0:
        raise ValueError("strike_price must be strictly positive")
