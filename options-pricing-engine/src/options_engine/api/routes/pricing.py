"""Pricing endpoints."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ...core.models import ExerciseStyle as DomainExerciseStyle
from ...core.models import MarketData, OptionContract, OptionType as DomainOptionType
from ...core.pricing_engine import OptionsEngine
from ..schemas.request import OptionContractRequest, PricingRequest
from ..schemas.response import PricingBatchResponse
from ..security import require_permission

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/pricing", tags=["pricing"])
engine = OptionsEngine(num_threads=8)


def _to_option_contract(contract: OptionContractRequest) -> OptionContract:
    return OptionContract(
        symbol=contract.symbol,
        strike_price=contract.strike_price,
        time_to_expiry=contract.time_to_expiry,
        option_type=DomainOptionType(contract.option_type.value),
        exercise_style=DomainExerciseStyle(contract.exercise_style.value),
    )


def _to_market_data(request: PricingRequest) -> MarketData:
    data = request.market_data
    return MarketData(
        spot_price=data.spot_price,
        risk_free_rate=data.risk_free_rate,
        dividend_yield=data.dividend_yield,
    )


@router.post(
    "/single",
    response_model=PricingBatchResponse,
    dependencies=[Depends(require_permission("pricing:read"))],
)
async def single(
    request: PricingRequest, background_tasks: BackgroundTasks
) -> PricingBatchResponse:
    if not request.contracts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No contracts provided")

    domain_contract = _to_option_contract(request.contracts[0])
    market_data = _to_market_data(request)
    override_volatility = request.market_data.volatility

    start = time.perf_counter()
    result = engine.price_option(
        domain_contract,
        market_data,
        model_name=request.model.value,
        override_volatility=override_volatility,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0

    background_tasks.add_task(
        lambda contract_id=result["contract_id"]: LOGGER.info("Priced %s", contract_id)
    )

    options_per_second = 1000.0 / duration_ms if duration_ms > 0 else float("inf")
    return PricingBatchResponse(
        results=[result],
        total_computation_time_ms=duration_ms,
        options_per_second=options_per_second,
    )


@router.post(
    "/batch",
    response_model=PricingBatchResponse,
    dependencies=[Depends(require_permission("pricing:read"))],
)
async def batch(request: PricingRequest, background_tasks: BackgroundTasks) -> PricingBatchResponse:
    if not request.contracts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No contracts provided")

    contracts = [_to_option_contract(contract) for contract in request.contracts]
    market_data = _to_market_data(request)
    override_volatility = request.market_data.volatility

    start = time.perf_counter()
    results = engine.price_portfolio(
        contracts,
        market_data,
        model_name=request.model.value,
        override_volatility=override_volatility,
    )
    duration_ms = (time.perf_counter() - start) * 1000.0
    options_per_second = len(results) / (duration_ms / 1000.0) if duration_ms > 0 else float("inf")

    portfolio_greeks = (
        engine.calculate_portfolio_greeks(results) if request.calculate_greeks else None
    )
    background_tasks.add_task(lambda count=len(results): LOGGER.info("Priced %d contracts", count))

    return PricingBatchResponse(
        results=results,
        total_computation_time_ms=duration_ms,
        options_per_second=options_per_second,
        portfolio_greeks=portfolio_greeks,
    )
