"""Pydantic request schemas exposed by the public API."""

from __future__ import annotations

import re
from enum import StrEnum
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OptionType(StrEnum):
    CALL = "call"
    PUT = "put"


class ExerciseStyle(StrEnum):
    EUROPEAN = "european"
    AMERICAN = "american"


class PricingModel(StrEnum):
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial_200"
    MONTE_CARLO = "monte_carlo_20k"
    LONGSTAFF_SCHWARTZ = "longstaff_schwartz_20k"
    FINITE_DIFFERENCE = "finite_difference_400"


class APIRequestModel(BaseModel):
    """Base request schema that rejects misspelled or unsupported fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class OptionContractRequest(APIRequestModel):
    symbol: str = Field(..., min_length=1, max_length=64)
    strike_price: float = Field(..., gt=0, le=1e12)
    time_to_expiry: float = Field(..., gt=1e-6, le=100.0)
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    quantity: int = Field(1, ge=1, le=1_000_000)

    @field_validator("symbol")
    @classmethod
    def sym(cls, v: str) -> str:
        v = v.strip().upper()
        if not re.fullmatch(r"[A-Z0-9][A-Z0-9._:-]{0,63}", v):
            raise ValueError("symbol contains unsupported characters or length")
        return v

    @field_validator("option_type", mode="before")
    @classmethod
    def normalise_option_type(cls, value: object) -> OptionType:
        if isinstance(value, OptionType):
            return value
        if isinstance(value, str):
            return OptionType(value.strip().lower())
        raise TypeError("option_type must be a string")

    @field_validator("exercise_style", mode="before")
    @classmethod
    def normalise_exercise_style(cls, value: object) -> ExerciseStyle:
        if isinstance(value, ExerciseStyle):
            return value
        if isinstance(value, str):
            return ExerciseStyle(value.strip().lower())
        raise TypeError("exercise_style must be a string")


class CashDividendRequest(APIRequestModel):
    """One deterministic cash dividend on the scalar model clock."""

    ex_time: float = Field(..., gt=0.0, le=100.0)
    amount: float = Field(..., gt=0.0, le=1e12)


class MarketDataRequest(APIRequestModel):
    spot_price: float = Field(..., gt=0, le=1e12)
    risk_free_rate: float = Field(..., ge=-1.0, le=1.0)
    dividend_yield: float = Field(0.0, ge=-1.0, le=1.0)
    volatility: float | None = Field(None, gt=1e-6, le=5.0)
    cash_dividends: list[CashDividendRequest] = Field(default_factory=list, max_length=64)

    @field_validator("cash_dividends")
    @classmethod
    def validate_cash_dividend_order(
        cls, values: list[CashDividendRequest]
    ) -> list[CashDividendRequest]:
        times = [value.ex_time for value in values]
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("cash-dividend ex-times must be strictly increasing and unique")
        return values


class PricingRequest(APIRequestModel):
    contracts: list[OptionContractRequest]
    market_data: MarketDataRequest
    model: PricingModel = PricingModel.BLACK_SCHOLES
    calculate_greeks: bool = True
    seed: int | None = Field(default=None, ge=0, le=2**128 - 1)

    @model_validator(mode="after")
    def validate_cash_dividends_before_every_expiry(self) -> PricingRequest:
        if self.contracts and self.market_data.cash_dividends:
            shortest_maturity = min(contract.time_to_expiry for contract in self.contracts)
            if self.market_data.cash_dividends[-1].ex_time >= shortest_maturity:
                raise ValueError(
                    "cash-dividend ex-times must be strictly before every option expiry"
                )
        return self

    @field_validator("model", mode="before")
    @classmethod
    def normalise_model(cls, value: object) -> PricingModel:
        if isinstance(value, PricingModel):
            return value
        if isinstance(value, str):
            return PricingModel(value.strip().lower())
        raise TypeError("model must be a string")


class VolatilityPointRequest(APIRequestModel):
    strike: float = Field(..., gt=0, le=1e12)
    maturity: float = Field(..., gt=0, le=100.0)
    volatility: float = Field(..., ge=0.01, le=5.0)
    source: str = Field("market", min_length=1, max_length=32)

    @field_validator("source")
    @classmethod
    def normalise_source(cls, value: str) -> str:
        source = value.strip()
        if not source or any(ord(character) < 32 for character in source):
            raise ValueError("source must be printable and non-empty")
        return source


class VolatilityEstimateRequest(APIRequestModel):
    strike: float = Field(..., gt=0, le=1e12)
    maturity: float = Field(..., gt=0, le=100.0)
    spot: float = Field(..., gt=0, le=1e12)
