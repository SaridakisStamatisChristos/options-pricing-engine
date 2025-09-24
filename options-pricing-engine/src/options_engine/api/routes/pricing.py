"""Pricing endpoints."""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ..dependencies import get_engine
from ..mappers import to_market_data, to_option_contract
from ..schemas.request import PricingRequest
from ..schemas.response import PricingBatchResponse
from ..security import require_permission
from ..services import annotate_results_with_quantity, enrich_pricing_result

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/pricing", tags=["pricing"])
engine = get_engine()


def _log_single_pricing(contract_id: str) -> None:
    LOGGER.info("Priced %s", contract_id)


def _log_batch_pricing(count: int) -> None:
    LOGGER.info("Priced %d contracts", count)


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
    if len(request.contracts) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Single pricing endpoint expects exactly one contract",
        )

    domain_contract = to_option_contract(request.contracts[0])
    market_data = to_market_data(request)
    override_volatility = request.market_data.volatility

    start = time.perf_counter()
    try:
        result = await asyncio.to_thread(
            engine.price_option,
            domain_contract,
            market_data,
            model_name=request.model.value,
            override_volatility=override_volatility,
        )
    except ValueError as exc:
        # bad inputs, unsupported model, etc.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        # engine unavailable or internal transient failure
        LOGGER.exception("Pricing engine unavailable", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pricing engine unavailable",
        ) from exc

    enriched_result = enrich_pricing_result(result, request.contracts[0].quantity)
    duration_ms = (time.perf_counter() - start) * 1000.0

    background_tasks.add_task(_log_single_pricing, enriched_result["contract_id"])

    options_per_second = 1000.0 / duration_ms if duration_ms > 0 else float("inf")
    portfolio_greeks = (
        engine.calculate_portfolio_greeks([enriched_result])
        if request.calculate_greeks
        else None
    )
    return PricingBatchResponse(
        results=[enriched_result],
        total_computation_time_ms=duration_ms,
        options_per_second=options_per_second,
        portfolio_greeks=portfolio_greeks,
    )


@router.post(
    "/batch",
    response_model=PricingBatchResponse,
    dependencies=[Depends(require_permission("pricing:read"))],
)
async def batch(request: PricingRequest, background_tasks: BackgroundTasks) -> PricingBatchResponse:
    if not request.contracts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No contracts provided")

    contracts = [to_option_contract(contract) for contract in request.contracts]
    market_data = to_market_data(request)
    override_volatility = request.market_data.volatility

    start = time.perf_counter()
    try:
        raw_results = await asyncio.to_thread(
            engine.price_portfolio,
            contracts,
            market_data,
            model_name=request.model.value,
            override_volatility=override_volatility,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Pricing engine unavailable", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pricing engine unavailable",
        ) from exc

    try:
        enriched_results = annotate_results_with_quantity(
            raw_results, (contract.quantity for contract in request.contracts)
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    duration_ms = (time.perf_counter() - start) * 1000.0
    options_per_second = (
        len(enriched_results) / (duration_ms / 1000.0) if duration_ms > 0 else float("inf")
    )

    portfolio_greeks = (
        engine.calculate_portfolio_greeks(enriched_results) if request.calculate_greeks else None
    )
    background_tasks.add_task(_log_batch_pricing, len(enriched_results))

    return PricingBatchResponse(
        results=list(enriched_results),
        total_computation_time_ms=duration_ms,
        options_per_second=options_per_second,
        portfolio_greeks=portfolio_greeks,
    )
