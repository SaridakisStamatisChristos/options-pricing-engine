from pydantic import BaseModel
from typing import List, Optional
class PricingResultResponse(BaseModel):
    contract_id: str; theoretical_price: float
    delta: float|None=None; gamma: float|None=None; theta: float|None=None; vega: float|None=None; rho: float|None=None
    implied_volatility: float|None=None; computation_time_ms: float|None=None
    model_used: str; volatility_used: float|None=None; cached: bool|None=None; error: str|None=None
class PortfolioGreeksResponse(BaseModel):
    delta: float; gamma: float; theta: float; vega: float; rho: float; total_value: float; total_vega_exposure: float; position_count: float
class PricingBatchResponse(BaseModel):
    results: List[PricingResultResponse]; total_computation_time_ms: float; options_per_second: float
    portfolio_greeks: PortfolioGreeksResponse | None = None
