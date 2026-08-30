"""Pricing endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from ...core.pricing_engine import OptionsEngine
from ..config import get_settings
from ..dependencies import get_engine
from ..mappers import to_market_data, to_option_contract
from ..schemas.request import PricingRequest
from ..schemas.response import (
    PortfolioGreeksResponse,
    PricingBatchResponse,
    PricingResultResponse,
)
from ..security import require_permission
from ..services import annotate_results_with_quantity, enrich_pricing_result

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/pricing", tags=["pricing"])
_GREEK_FIELDS = frozenset({"delta", "gamma", "theta", "vega", "rho"})
_HTTP_413_CONTENT_TOO_LARGE = 413


def _log_single_pricing(contract_id: str) -> None:
    LOGGER.info("Priced %s", contract_id)


def _log_batch_pricing(count: int) -> None:
    LOGGER.info("Priced %d contracts", count)


def _respect_greek_selection(
    result: dict[str, object], *, calculate_greeks: bool
) -> dict[str, object]:
    if calculate_greeks:
        return dict(result)
    return {key: value for key, value in result.items() if key not in _GREEK_FIELDS}


@router.post(
    "/single",
    response_model=PricingBatchResponse,
    dependencies=[Depends(require_permission("pricing:read"))],
)
async def single(
    request: PricingRequest,
    background_tasks: BackgroundTasks,
    engine: Annotated[OptionsEngine, Depends(get_engine)],
) -> PricingBatchResponse:
    if not request.contracts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No contracts provided")
    max_contracts = get_settings().max_pricing_contracts
    if len(request.contracts) > max_contracts:
        raise HTTPException(
            status_code=_HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Too many contracts; limit is {max_contracts}",
        )
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
            seed=request.seed,
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

    selected_result = _respect_greek_selection(
        result,
        calculate_greeks=request.calculate_greeks,
    )
    enriched_result = enrich_pricing_result(selected_result, request.contracts[0].quantity)
    duration_ms = (time.perf_counter() - start) * 1000.0

    background_tasks.add_task(_log_single_pricing, enriched_result["contract_id"])

    options_per_second = 1000.0 / duration_ms if duration_ms > 0 else float("inf")
    portfolio_greeks = (
        engine.calculate_portfolio_greeks([enriched_result]) if request.calculate_greeks else None
    )
    return PricingBatchResponse(
        results=[PricingResultResponse.model_validate(enriched_result)],
        total_computation_time_ms=duration_ms,
        options_per_second=options_per_second,
        portfolio_greeks=(
            PortfolioGreeksResponse.model_validate(portfolio_greeks)
            if portfolio_greeks is not None
            else None
        ),
    )


@router.post(
    "/batch",
    response_model=PricingBatchResponse,
    dependencies=[Depends(require_permission("pricing:read"))],
)
async def batch(
    request: PricingRequest,
    background_tasks: BackgroundTasks,
    engine: Annotated[OptionsEngine, Depends(get_engine)],
) -> PricingBatchResponse:
    if not request.contracts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No contracts provided")
    max_contracts = get_settings().max_pricing_contracts
    if len(request.contracts) > max_contracts:
        raise HTTPException(
            status_code=_HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Too many contracts; limit is {max_contracts}",
        )

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
            seed=request.seed,
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
        selected_results = (
            _respect_greek_selection(
                result,
                calculate_greeks=request.calculate_greeks,
            )
            for result in raw_results
        )
        enriched_results = annotate_results_with_quantity(
            selected_results,
            (contract.quantity for contract in request.contracts),
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
        results=[PricingResultResponse.model_validate(item) for item in enriched_results],
        total_computation_time_ms=duration_ms,
        options_per_second=options_per_second,
        portfolio_greeks=(
            PortfolioGreeksResponse.model_validate(portfolio_greeks)
            if portfolio_greeks is not None
            else None
        ),
    )
