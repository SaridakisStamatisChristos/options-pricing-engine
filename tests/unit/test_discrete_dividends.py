from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from options_engine.api.mappers import to_market_data
from options_engine.api.schemas.request import PricingRequest
from options_engine.core.black_scholes import BlackScholesModel
from options_engine.core.crr import BinomialModel
from options_engine.core.dividends import CashDividend, CashDividendSchedule
from options_engine.core.lsmc import LongstaffSchwartzModel
from options_engine.core.models import (
    ExerciseStyle,
    MarketData,
    OptionContract,
    OptionType,
)
from options_engine.core.monte_carlo import MonteCarloModel
from options_engine.core.pricing_engine import OptionsEngine
from options_engine.greeks.estimators import pathwise_delta


def _schedule() -> CashDividendSchedule:
    return CashDividendSchedule((CashDividend(0.25, 1.0), CashDividend(0.75, 1.5)))


def test_cash_dividend_schedule_has_exact_stable_identity() -> None:
    first = _schedule()
    second = CashDividendSchedule((CashDividend(0.25, 1.0), CashDividend(0.75, 1.5)))

    assert first.schedule_id == second.schedule_id
    assert first.to_list() == [
        {"ex_time": 0.25, "amount": 1.0},
        {"ex_time": 0.75, "amount": 1.5},
    ]


@pytest.mark.parametrize(
    "dividends, error",
    [
        (
            (CashDividend(0.75, 1.0), CashDividend(0.25, 1.0)),
            "strictly increasing",
        ),
        ((CashDividend(0.25, 1.0), CashDividend(0.25, 2.0)), "unique"),
    ],
)
def test_cash_dividend_schedule_rejects_ambiguous_event_order(
    dividends: tuple[CashDividend, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        CashDividendSchedule(dividends)


def test_cash_dividend_must_be_strictly_before_expiry() -> None:
    contract = OptionContract("DIV", 100.0, 1.0, OptionType.CALL)
    market = MarketData(
        100.0,
        0.03,
        cash_dividends=CashDividendSchedule((CashDividend(1.0, 1.0),)),
    )

    with pytest.raises(ValueError, match="strictly before option expiry"):
        BinomialModel().calculate_price(contract, market, 0.2)


@pytest.mark.parametrize(
    ("model", "contract"),
    [
        (BlackScholesModel(), OptionContract("DIV", 100.0, 1.0, OptionType.CALL)),
        (MonteCarloModel(paths=32), OptionContract("DIV", 100.0, 1.0, OptionType.CALL)),
        (
            LongstaffSchwartzModel(paths=100, steps=10),
            OptionContract(
                "DIV",
                100.0,
                1.0,
                OptionType.PUT,
                ExerciseStyle.AMERICAN,
            ),
        ),
    ],
)
def test_unsupported_models_reject_cash_dividends(model: Any, contract: OptionContract) -> None:
    market = MarketData(100.0, 0.03, cash_dividends=_schedule())

    with pytest.raises(ValueError, match="does not support deterministic discrete cash"):
        model.calculate_price(contract, market, 0.2)


def test_engine_rejects_unsupported_schedule_before_dispatch() -> None:
    contract = OptionContract("DIV", 100.0, 1.0, OptionType.CALL)
    market = MarketData(100.0, 0.03, cash_dividends=_schedule())

    with (
        OptionsEngine(num_threads=1) as engine,
        pytest.raises(ValueError, match="use binomial_200 or finite_difference_400"),
    ):
        engine.price_option(
            contract,
            market,
            model_name="black_scholes",
            override_volatility=0.2,
        )


def test_terminal_log_normal_greek_estimators_reject_cash_schedule() -> None:
    contract = OptionContract("DIV", 100.0, 1.0, OptionType.CALL)
    market = MarketData(100.0, 0.03, cash_dividends=_schedule())

    with pytest.raises(ValueError, match="does not support deterministic discrete cash"):
        pathwise_delta(
            contract,
            market,
            discount_factor=1.0,
            terminal_prices=np.asarray([100.0]),
        )


def test_api_mapper_preserves_cash_schedule_without_yield_conversion() -> None:
    request = PricingRequest.model_validate(
        {
            "contracts": [
                {
                    "symbol": "DIV",
                    "strike_price": 100.0,
                    "time_to_expiry": 1.0,
                    "option_type": "call",
                }
            ],
            "market_data": {
                "spot_price": 100.0,
                "risk_free_rate": 0.03,
                "dividend_yield": 0.01,
                "volatility": 0.2,
                "cash_dividends": [{"ex_time": 0.25, "amount": 2.0}],
            },
            "model": "binomial_200",
        }
    )

    market = to_market_data(request)

    assert market.dividend_yield == 0.01
    assert market.cash_dividends.to_list() == [{"ex_time": 0.25, "amount": 2.0}]


@pytest.mark.parametrize(
    "cash_dividends",
    [
        [{"ex_time": 0.75, "amount": 1.0}, {"ex_time": 0.25, "amount": 1.0}],
        [{"ex_time": 1.0, "amount": 1.0}],
    ],
)
def test_api_schema_rejects_ambiguous_cash_event_ordering(
    cash_dividends: list[dict[str, float]],
) -> None:
    with pytest.raises(ValidationError, match="cash-dividend"):
        PricingRequest.model_validate(
            {
                "contracts": [
                    {
                        "symbol": "DIV",
                        "strike_price": 100.0,
                        "time_to_expiry": 1.0,
                        "option_type": "call",
                    }
                ],
                "market_data": {
                    "spot_price": 100.0,
                    "risk_free_rate": 0.03,
                    "cash_dividends": cash_dividends,
                },
                "model": "binomial_200",
            }
        )
