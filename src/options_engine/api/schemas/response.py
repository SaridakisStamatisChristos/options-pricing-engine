"""Response schemas exposed by the API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class PricingResultResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    contract_id: str
    theoretical_price: float
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    implied_volatility: float | None = None
    computation_time_ms: float | None = None
    model_used: str
    volatility_used: float | None = None
    cached: bool | None = None
    error: str | None = None
    standard_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    estimate_diagnostics: dict[str, object] | None = None
    numerical_diagnostics: dict[str, object] | None = None
    quantity: float | None = None
    position_value: float | None = None
    position_delta: float | None = None
    position_gamma: float | None = None
    position_theta: float | None = None
    position_vega: float | None = None
    position_rho: float | None = None
    position_standard_error: float | None = None
    position_confidence_interval: tuple[float, float] | None = None


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
    results: list[PricingResultResponse]
    total_computation_time_ms: float
    options_per_second: float
    portfolio_greeks: PortfolioGreeksResponse | None = None


class VolatilityPointResponse(BaseModel):
    strike: float
    maturity: float
    volatility: float
    timestamp: float
    source: str


class VolatilitySurfaceResponse(BaseModel):
    points: list[VolatilityPointResponse]


class VolatilityEstimateResponse(BaseModel):
    volatility: float
