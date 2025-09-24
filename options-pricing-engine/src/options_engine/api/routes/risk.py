"""Risk analytics endpoints."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import get_engine
from ..mappers import to_market_data, to_option_contract
from ..schemas.request import PricingRequest
from ..schemas.response import PortfolioGreeksResponse
from ..security import require_permission
from ..services import annotate_results_with_quantity

LOGGER = logging.getLogger(__name__)
router = APIRouter(prefix="/risk", tags=["risk"])


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
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise _err(f"{name} must be a number") from exc
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

    _validate_numeric("market_data.spot_price", getattr(md, "spot_price", None), gt=0.0)
    vol = getattr(md, "volatility", None)
    if vol is not None:
        _validate_numeric("market_data.volatility", vol, ge=0.0, le=5.0)

    if not req.contracts:
        raise _err("No contracts provided")

    for i, contract in enumerate(req.contracts):
        prefix = f"contracts[{i}]"
        _validate_numeric(f"{prefix}.strike_price", getattr(contract, "strike_price", None), gt=0.0)
        _validate_numeric(
            f"{prefix}.time_to_expiry",
            getattr(contract, "time_to_expiry", None),
            gt=0.0,
        )

        quantity = getattr(contract, "quantity", None)
        if quantity is None:
            raise _err(f"{prefix}.quantity is required")
        try:
            q_int = int(quantity)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
            raise _err(f"{prefix}.quantity must be an integer ≥ 1") from exc
        if q_int < 1:
            raise _err(f"{prefix}.quantity must be ≥ 1")


# ------------------------
# Endpoints
# ------------------------
@router.post(
    "/aggregate-greeks",
    response_model=PortfolioGreeksResponse,
    dependencies=[Depends(require_permission("risk:read"))],
)
async def aggregate_greeks(request: PricingRequest) -> PortfolioGreeksResponse:
    """Aggregate portfolio Greeks for the provided pricing request."""

    _validate_request_common(request)

    contracts = [to_option_contract(contract) for contract in request.contracts]
    market_data = to_market_data(request)
    override_volatility = request.market_data.volatility

    engine = get_engine()

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

    totals = engine.calculate_portfolio_greeks(enriched_results)
    return PortfolioGreeksResponse(**totals)
