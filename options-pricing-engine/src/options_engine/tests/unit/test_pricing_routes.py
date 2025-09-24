"""Tests covering the pricing API routes."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import BackgroundTasks, HTTPException, status

from options_engine.api.routes import pricing, risk
from options_engine.api.schemas.request import (
    ExerciseStyle,
    MarketDataRequest,
    OptionContractRequest,
    OptionType,
    PricingModel,
    PricingRequest,
)


def test_single_route_respects_calculate_greeks_flag() -> None:
    """The single pricing endpoint should omit Greeks when not requested."""
    request = PricingRequest(
        contracts=[
            OptionContractRequest(
                symbol="TQA",
                strike_price=100.0,
                time_to_expiry=1.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN,
                quantity=5,
            )
        ],
        market_data=MarketDataRequest(
            spot_price=105.0,
            risk_free_rate=0.01,
            dividend_yield=0.0,
            volatility=0.2,
        ),
        model=PricingModel.BLACK_SCHOLES,
        calculate_greeks=False,
    )

    response = asyncio.run(pricing.single(request, BackgroundTasks()))

    assert response.portfolio_greeks is None
    assert len(response.results) == 1


def test_single_route_rejects_multiple_contracts() -> None:
    request = PricingRequest(
        contracts=[
            OptionContractRequest(
                symbol="AAA",
                strike_price=100.0,
                time_to_expiry=1.0,
                option_type=OptionType.CALL,
                exercise_style=ExerciseStyle.EUROPEAN,
                quantity=1,
            ),
            OptionContractRequest(
                symbol="BBB",
                strike_price=105.0,
                time_to_expiry=1.0,
                option_type=OptionType.PUT,
                exercise_style=ExerciseStyle.EUROPEAN,
                quantity=1,
            ),
        ],
        market_data=MarketDataRequest(
            spot_price=105.0,
            risk_free_rate=0.01,
            dividend_yield=0.0,
            volatility=0.2,
        ),
        model=PricingModel.BLACK_SCHOLES,
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(pricing.single(request, BackgroundTasks()))

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "exactly one contract" in excinfo.value.detail


def test_single_route_translates_engine_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    request = PricingRequest(
        contracts=[
            OptionContractRequest(
                symbol="ERR",
                strike_price=100.0,
                time_to_expiry=1.0,
                option_type=OptionType.CALL,
                quantity=1,
            )
        ],
        market_data=MarketDataRequest(
            spot_price=105.0,
            risk_free_rate=0.01,
            dividend_yield=0.0,
            volatility=0.2,
        ),
        model=PricingModel.BLACK_SCHOLES,
    )

    def _raise(*_: object, **__: object) -> object:
        raise ValueError("invalid contract")

    monkeypatch.setattr(pricing.engine, "price_option", _raise)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(pricing.single(request, BackgroundTasks()))

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "invalid contract"


def test_batch_route_translates_annotation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    request = PricingRequest(
        contracts=[
            OptionContractRequest(
                symbol="B1",
                strike_price=100.0,
                time_to_expiry=1.0,
                option_type=OptionType.CALL,
                quantity=1,
            )
        ],
        market_data=MarketDataRequest(
            spot_price=105.0,
            risk_free_rate=0.01,
            dividend_yield=0.0,
            volatility=0.2,
        ),
        model=PricingModel.BLACK_SCHOLES,
    )

    def _annotate(*_: object, **__: object) -> object:
        raise ValueError("quantity mismatch")

    monkeypatch.setattr(pricing, "annotate_results_with_quantity", _annotate)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(pricing.batch(request, BackgroundTasks()))

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "quantity mismatch"


def test_risk_route_translates_engine_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    request = PricingRequest(
        contracts=[
            OptionContractRequest(
                symbol="R1",
                strike_price=100.0,
                time_to_expiry=1.0,
                option_type=OptionType.CALL,
                quantity=1,
            )
        ],
        market_data=MarketDataRequest(
            spot_price=105.0,
            risk_free_rate=0.01,
            dividend_yield=0.0,
            volatility=0.2,
        ),
        model=PricingModel.BLACK_SCHOLES,
    )

    class _FailingEngine:
        def price_portfolio(self, *args: object, **kwargs: object) -> object:
            raise ValueError("invalid risk request")

        def calculate_portfolio_greeks(self, _: object) -> dict[str, float]:
            return {}

    monkeypatch.setattr(risk, "get_engine", lambda: _FailingEngine())

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(risk.aggregate_greeks(request))

    assert excinfo.value.status_code == status.HTTP_400_BAD_REQUEST
    assert excinfo.value.detail == "invalid risk request"
