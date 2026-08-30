from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from options_engine.api.routes import market_data, risk
from options_engine.api.schemas.request import (
    MarketDataRequest,
    OptionContractRequest,
    OptionType,
    PricingModel,
    PricingRequest,
    VolatilityEstimateRequest,
    VolatilityPointRequest,
)
from options_engine.core.volatility_surface import VolatilitySurface


def _pricing_request(*, quantity: int = 2) -> PricingRequest:
    return PricingRequest(
        contracts=[
            OptionContractRequest(
                symbol="RISK",
                strike_price=100.0,
                time_to_expiry=1.0,
                option_type=OptionType.CALL,
                quantity=quantity,
            )
        ],
        market_data=MarketDataRequest(
            spot_price=105.0,
            risk_free_rate=0.02,
            dividend_yield=0.0,
            volatility=0.2,
        ),
        model=PricingModel.BLACK_SCHOLES,
    )


def test_market_data_routes_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    surface = VolatilitySurface()
    monkeypatch.setattr(market_data, "get_vol_surface", lambda: surface)

    created = asyncio.run(
        market_data.upsert_volatility_point(
            VolatilityPointRequest(
                strike=100.0,
                maturity=1.0,
                volatility=0.27,
                source="unit-test",
            )
        )
    )
    listed = asyncio.run(market_data.list_volatility_points())
    estimate = asyncio.run(
        market_data.estimate_volatility(
            VolatilityEstimateRequest(strike=100.0, maturity=1.0, spot=100.0)
        )
    )

    assert created.source == "unit-test"
    assert listed.points == [created]
    assert estimate.volatility == 0.20  # fewer than four points uses the documented default


def test_market_data_route_maps_surface_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    class RejectingSurface:
        points: tuple[()] = ()

        def update_volatility(self, **_: object) -> None:
            raise ValueError("bad quote")

    monkeypatch.setattr(market_data, "get_vol_surface", RejectingSurface)
    request = VolatilityPointRequest(strike=100.0, maturity=1.0, volatility=0.2)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(market_data.upsert_volatility_point(request))
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "bad quote"


def test_market_data_route_detects_missing_persisted_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DiscardingSurface:
        points: tuple[()] = ()

        def update_volatility(self, **_: object) -> None:
            return None

    monkeypatch.setattr(market_data, "get_vol_surface", DiscardingSurface)
    request = VolatilityPointRequest(strike=100.0, maturity=1.0, volatility=0.2)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(market_data.upsert_volatility_point(request))
    assert exc_info.value.status_code == 500


def test_risk_numeric_validation_boundaries() -> None:
    risk._validate_numeric("x", 1.0, gt=0.0, ge=1.0, lt=2.0, le=1.0)

    for value, kwargs, message in (
        (None, {}, "Missing"),
        (0.0, {"gt": 0.0}, "must be >"),
        (0.0, {"ge": 1.0}, "must be ≥"),
        (2.0, {"lt": 2.0}, "must be <"),
        (2.0, {"le": 1.0}, "must be ≤"),
    ):
        with pytest.raises(HTTPException, match=message):
            risk._validate_numeric("x", value, **kwargs)


def test_risk_aggregate_success() -> None:
    class Engine:
        def price_portfolio(self, *_: object, **__: object) -> list[dict[str, object]]:
            return [
                {
                    "contract_id": "risk-1",
                    "theoretical_price": 4.0,
                    "delta": 0.5,
                    "gamma": 0.1,
                    "theta": -0.01,
                    "vega": 0.2,
                    "rho": 0.3,
                }
            ]

        def calculate_portfolio_greeks(self, _: object) -> dict[str, float]:
            return {
                "delta": 1.0,
                "gamma": 0.2,
                "theta": -0.02,
                "vega": 0.4,
                "rho": 0.6,
                "total_value": 8.0,
                "total_vega_exposure": 40.0,
                "position_count": 2.0,
            }

    response = asyncio.run(risk.aggregate_greeks(_pricing_request(), Engine()))
    assert response.total_value == 8.0
    assert response.delta == 1.0


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [(ValueError("bad model"), 400), (RuntimeError("busy"), 503)],
)
def test_risk_maps_pricing_failures(
    error: Exception,
    expected_status: int,
) -> None:
    class Engine:
        def price_portfolio(self, *_: object, **__: object) -> list[dict[str, object]]:
            raise error

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(risk.aggregate_greeks(_pricing_request(), Engine()))
    assert exc_info.value.status_code == expected_status
