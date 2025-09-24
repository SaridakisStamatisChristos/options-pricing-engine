"""Risk analytics endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_engine
from ..mappers import to_market_data, to_option_contract
from ..schemas.request import PricingRequest
from ..schemas.response import PortfolioGreeksResponse
from ..security import require_permission
from ..services import annotate_results_with_quantity

router = APIRouter(
    prefix="/risk", tags=["risk"], dependencies=[Depends(require_permission("risk:read"))]
)


@router.post("/greeks", response_model=PortfolioGreeksResponse)
async def aggregate_greeks(request: PricingRequest) -> PortfolioGreeksResponse:
    """Aggregate position-level Greeks for the requested contracts."""

    if not request.contracts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No contracts provided")

    engine = get_engine()
    contracts = [to_option_contract(contract) for contract in request.contracts]
    market_data = to_market_data(request)

    raw_results = await asyncio.to_thread(
        engine.price_portfolio,
        contracts,
        market_data,
        model_name=request.model.value,
        override_volatility=request.market_data.volatility,
    )
    enriched_results = annotate_results_with_quantity(
        raw_results, (contract.quantity for contract in request.contracts)
    )

    totals = engine.calculate_portfolio_greeks(enriched_results)
    return PortfolioGreeksResponse(**totals)
