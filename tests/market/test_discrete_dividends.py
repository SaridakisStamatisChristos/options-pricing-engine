from __future__ import annotations

import math
from datetime import date

import pytest

from options_engine.core.models import OptionType
from options_engine.market import (
    ALL_DAYS_CALENDAR,
    DatedCashDividend,
    DatedCashDividendSchedule,
    DatedOptionContract,
    ExDividendDate,
    ExpiryDate,
    MarketEnvironment,
    ValuationDate,
)


def test_dated_cash_schedule_resolves_dates_and_forward_deduction() -> None:
    valuation = ValuationDate(date(2026, 1, 1))
    schedule = DatedCashDividendSchedule(
        (DatedCashDividend(ExDividendDate(date(2026, 4, 1)), 1.5),)
    )
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=valuation,
        risk_free_rate=0.03,
        dividend_yield=0.01,
        calendar=ALL_DAYS_CALENDAR,
        cash_dividends=schedule,
    )

    resolved = environment.resolve(
        DatedOptionContract(
            "DIV",
            100.0,
            ExpiryDate(date(2027, 1, 1)),
            OptionType.CALL,
        )
    )

    assert resolved.market_data.dividend_yield == pytest.approx(0.01)
    assert resolved.market_data.cash_dividends.to_list() == [
        {"ex_time": 90.0 / 365.0, "amount": 1.5}
    ]
    expected_future_deduction = 1.5 * math.exp(0.02 * 275.0 / 365.0)
    assert resolved.forward.cash_dividend_future_value == pytest.approx(expected_future_deduction)
    assert resolved.forward.forward_price == pytest.approx(
        resolved.forward.continuous_carry_forward_price - expected_future_deduction
    )
    assert resolved.diagnostics()["cash_dividend_forward_deduction_mismatch"] == pytest.approx(
        0.0, abs=1e-14
    )


def test_dated_schedule_filters_events_after_contract_expiry() -> None:
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=ValuationDate(date(2026, 1, 1)),
        risk_free_rate=0.03,
        calendar=ALL_DAYS_CALENDAR,
        cash_dividends=DatedCashDividendSchedule(
            (
                DatedCashDividend(ExDividendDate(date(2026, 3, 1)), 1.0),
                DatedCashDividend(ExDividendDate(date(2027, 3, 1)), 2.0),
            )
        ),
    )

    resolved = environment.resolve(
        DatedOptionContract(
            "DIV",
            100.0,
            ExpiryDate(date(2026, 7, 1)),
            OptionType.PUT,
        )
    )

    assert len(resolved.market_data.cash_dividends) == 1


def test_dated_schedule_rejects_expiry_event_ordering_ambiguity() -> None:
    expiry = ExpiryDate(date(2026, 7, 1))
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=ValuationDate(date(2026, 1, 1)),
        risk_free_rate=0.03,
        calendar=ALL_DAYS_CALENDAR,
        cash_dividends=DatedCashDividendSchedule(
            (DatedCashDividend(ExDividendDate(expiry.value), 1.0),)
        ),
    )

    with pytest.raises(ValueError, match="must not equal adjusted expiry"):
        environment.resolve(DatedOptionContract("DIV", 100.0, expiry, OptionType.CALL))
