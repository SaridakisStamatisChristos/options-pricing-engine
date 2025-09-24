"""Response schemas exposed by the API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class PricingResultResponse(BaseModel):
    contract_id: str
    theoretical_price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None
    computation_time_ms: Optional[float] = None
    model_used: str
    volatility_used: Optional[float] = None
    cached: Optional[bool] = None
    error: Optional[str] = None


class PortfolioGreeksResponse(BaseModel):
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    total_value: float
    total_vega_exposure: float
    position_count: float


class PricingBatchResponse(BaseModel):
    results: List[PricingResultResponse]
    total_computation_time_ms: float
    options_per_second: float
    portfolio_greeks: Optional[PortfolioGreeksResponse] = None
