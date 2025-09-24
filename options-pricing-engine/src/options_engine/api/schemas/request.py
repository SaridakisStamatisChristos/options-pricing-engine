"""Pydantic request schemas for the pricing API."""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    EUROPEAN = "european"
    AMERICAN = "american"


class PricingModel(str, Enum):
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial_200"
    MONTE_CARLO = "monte_carlo_20k"


class OptionContractRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    strike_price: float = Field(..., gt=0, le=1_000_000_000)
    time_to_expiry: float = Field(..., gt=0, le=50.0)
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    quantity: int = Field(1, ge=1, le=1_000_000)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        value = value.upper()
        if not re.fullmatch(r"[A-Z0-9]{1,20}", value):
            raise ValueError("symbol must be 1-20 uppercase alphanumerics")
        return value


class MarketDataRequest(BaseModel):
    spot_price: float = Field(..., gt=0, le=1_000_000_000)
    risk_free_rate: float = Field(..., ge=-1.0, le=1.0)
    dividend_yield: float = Field(0.0, ge=0.0, le=1.0)
    volatility: Optional[float] = Field(None, gt=0.0, le=5.0)


class PricingRequest(BaseModel):
    contracts: List[OptionContractRequest]
    market_data: MarketDataRequest
    model: PricingModel = PricingModel.BLACK_SCHOLES
    calculate_greeks: bool = True

