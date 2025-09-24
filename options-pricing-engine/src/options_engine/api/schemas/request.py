from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum
import re


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
    strike_price: float = Field(..., gt=0, le=1e9)
    time_to_expiry: float = Field(..., gt=0, le=50.0)
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    quantity: int = Field(1, ge=1, le=1_000_000)

    @field_validator("symbol")
    @classmethod
    def sym(cls, v: str) -> str:
        v = v.upper()
        if not re.fullmatch(r"[A-Z0-9]{1,20}", v):
            raise ValueError("symbol must be 1-20 uppercase alphanumerics")
        return v


class MarketDataRequest(BaseModel):
    spot_price: float = Field(..., gt=0, le=1e9)
    risk_free_rate: float = Field(..., ge=-1.0, le=1.0)
    dividend_yield: float = Field(0.0, ge=0, le=1.0)
    volatility: Optional[float] = Field(None, gt=0, le=5.0)


class PricingRequest(BaseModel):
    contracts: List[OptionContractRequest]
    market_data: MarketDataRequest
    model: PricingModel = PricingModel.BLACK_SCHOLES
    calculate_greeks: bool = True
