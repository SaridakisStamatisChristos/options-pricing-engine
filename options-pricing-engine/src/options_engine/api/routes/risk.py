"""Pricing endpoints."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

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


# ------------------------
# Validation helpers
# ------------------------
def _err(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


def _validate_numeric(
    name: str,
    value: Optional[float],
    *,
    gt: Optional[float] = None,
    ge: Optional[float] = None,
    lt: Optional[float] = None,
    le: Optional[float] = None,
) -> None:
    if value is None:
        raise _err(f"Missing required field: {name}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise _err(f"{name} must be a number")
    if gt is not None and not (x > gt):
        raise _err(f"{name} must be > {gt}")
    if ge is not None and not (x >= ge):
        raise _err(f"{name} must be ≥ {ge}")
    if lt is not None and not (x < lt):
        raise _err(f"{name} must be < {lt}")
    if le is not None and not (x <= le):
        raise _err(f"{name} must be ≤ {le}")


def _validate_request_common(req: PricingRequest) -> None:
    # Market data sanity checks
    md = getattr(req, "market_data", None)
    if md is None:
        raise _err("Missing market_data")

    # Conservative ranges; adjust if your domain allows wider inputs
    _validate_numeric("market_data.spot_price", getattr(md, "spot_price", None), gt=0.0)
    vol = getattr(md, "volatility", None)
    if vol is not None:
        # Allow 0 ≤ vol ≤ 5.0 (i.e., 0%–500% annualized) — wide but finite
        _validate_numeric("market_data.volatility", vol, ge=0.0, le=5.0)

    # Contracts list present?
    if not req.contracts:
        raise _err("No contracts provided")

    # Per-contract checks
    for i, c in enumerate(req.contracts):
        prefix = f"contracts[{i}]"
        _validate_numeric(f"{prefix}.strike", getattr(c, "strike", None), gt=0.0)

        qty = getattr(c, "quantity", None)
        if qty is None:
            raise _err(f"{prefix}.quantity is required")
        # Ensure integer-ish and ≥1
        try:
            q = int(qty)
        except (TypeError, ValueError):
            raise _err(f"{prefix}.quantity must be an integer ≥ 1")
        if q < 1:
            raise _err(f"{prefix}.quantity must be ≥ 1")


# ------------------------
# Logging helpers
# ------------------------
def _log_single_pricing(contract_id: str) -> None:
    LOGGER.info("Priced %s", contract_id)


def _log_batch_pricing(count: int) -> None:
    LOGGER.info("Priced %d contracts", count)


# ------------------------
# Endpoints
# ------------------------
@router.post(
    "/single",
    response_model=PricingBatchResponse,
    dependencies=[Depends(require_permission("pricing:read"))],
)
async def single(
    request: PricingRequest, background_tasks: BackgroundTasks
) -> PricingBatchResponse:
    _validate_request_common(request)
    if len(request.contracts) != 1:
        raise _err("Single pricing endpoint expects exactly one contract")

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
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        LOGGER.exception("Pricing engine unavailable", exc_info=exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pricing engine unavailable",
        ) from exc

    enriched_result = enrich_pricing_result(result, request.contracts[0].quantity)
    duration_ms = (time.perf_counter() - start) * 1000.0
    background_tasks.add_task(_log_single_pricing, enriched_result["contract_id"])

    options_per_second = 1000.0 / duration_ms if duration_ms > 0 else float("inf")
    try:
        portfolio_greeks = (
            engine.calculate_portfolio_greeks([enriched_result])
            if request.calculate_greeks
            else None
        )
    except Exception as exc:
        LOGGER.exception("Portfolio greeks calculation failed", exc_info=exc)
        # Non-fatal: proceed without portfolio greeks
        portfolio_greeks = None

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
    _validate_request_common(request)

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

    try:
        portfolio_greeks = (
            engine.calculate_portfolio_greeks(enriched_results) if request.calculate_greeks else None
        )
    except Exception as exc:
        LOGGER.exception("Portfolio greeks calculation failed", exc_info=exc)
        portfolio_greeks = None

    background_tasks.add_task(_log_batch_pricing, len(enriched_results))

    return PricingBatchResponse(
        results=list(enriched_results),
        total_computation_time_ms=duration_ms,
        options_per_second=options_per_second,
        portfolio_greeks=portfolio_greeks,
    )
