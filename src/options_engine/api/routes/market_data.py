"""Market data management endpoints."""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, status

from ...core.volatility_surface import VolatilityPoint
from ..dependencies import get_vol_surface
from ..schemas.request import VolatilityEstimateRequest, VolatilityPointRequest
from ..schemas.response import (
    VolatilityEstimateResponse,
    VolatilityPointResponse,
    VolatilitySurfaceResponse,
)
from ..security import require_permission

router = APIRouter(prefix="/market-data", tags=["market-data"])


def _to_response(point: VolatilityPoint) -> VolatilityPointResponse:
    return VolatilityPointResponse(
        strike=point.strike,
        maturity=point.maturity,
        volatility=point.volatility,
        timestamp=point.timestamp,
        source=point.source,
    )


@router.get(
    "/volatility",
    response_model=VolatilitySurfaceResponse,
    dependencies=[Depends(require_permission("market-data:read"))],
)
async def list_volatility_points() -> VolatilitySurfaceResponse:
    surface = get_vol_surface()
    points = [_to_response(point) for point in surface.points]
    return VolatilitySurfaceResponse(points=points)


@router.post(
    "/volatility",
    response_model=VolatilityPointResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_permission("market-data:write"))],
)
async def upsert_volatility_point(request: VolatilityPointRequest) -> VolatilityPointResponse:
    surface = get_vol_surface()
    try:
        surface.update_volatility(
            strike=request.strike,
            maturity=request.maturity,
            volatility=request.volatility,
            source=request.source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    for point in surface.points:
        if math.isclose(point.strike, request.strike, rel_tol=0.0, abs_tol=1e-9) and math.isclose(
            point.maturity, request.maturity, rel_tol=0.0, abs_tol=1e-9
        ):
            return _to_response(point)

    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Failed to persist volatility point",
    )


@router.post(
    "/volatility/estimate",
    response_model=VolatilityEstimateResponse,
    dependencies=[Depends(require_permission("market-data:read"))],
)
async def estimate_volatility(request: VolatilityEstimateRequest) -> VolatilityEstimateResponse:
    surface = get_vol_surface()
    volatility = surface.get_volatility(
        strike=request.strike,
        maturity=request.maturity,
        spot=request.spot,
    )
    return VolatilityEstimateResponse(volatility=volatility)
