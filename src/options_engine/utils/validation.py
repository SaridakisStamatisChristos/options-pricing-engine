"""Validation helpers for pricing inputs."""

from __future__ import annotations

import math
from numbers import Real
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

    if not isinstance(contract, OptionContract):
        raise TypeError("contract must be an OptionContract")
    if not isinstance(market_data, MarketData):
        raise TypeError("market_data must be MarketData")
    if isinstance(volatility, bool) or not isinstance(volatility, Real):
        raise TypeError("volatility must be a real number")
    if not math.isfinite(volatility) or not MIN_VOLATILITY < volatility <= MAX_VOLATILITY:
        raise ValueError("volatility is outside the supported range")

    if contract.time_to_expiry <= MIN_TIME_TO_EXPIRY:
        raise ValueError("time_to_expiry is too small for stable pricing")

    if market_data.spot_price <= 0:
        raise ValueError("spot_price must be strictly positive")

    if contract.strike_price <= 0:
        raise ValueError("strike_price must be strictly positive")


def validate_discrete_dividends(
    contract: OptionContract,
    market_data: MarketData,
) -> None:
    """Validate event times against the contract's scalar maturity clock."""

    market_data.cash_dividends.validate_for_maturity(contract.time_to_expiry)


def reject_discrete_dividends(market_data: MarketData, model_name: str) -> None:
    """Fail explicitly when a model has no exact cash-jump implementation."""

    if market_data.cash_dividends:
        raise ValueError(
            f"{model_name} does not support deterministic discrete cash dividends; "
            "use BinomialModel or FiniteDifferenceModel"
        )
