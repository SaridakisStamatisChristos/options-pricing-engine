"""Day-count, calendar, and settlement convention tests."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from options_engine.market import (
    ALL_DAYS_CALENDAR,
    BusinessCalendar,
    BusinessDayConvention,
    DayCountConvention,
    ExpiryDate,
    SettlementLag,
    ValuationDate,
)


def test_typed_dates_reject_datetimes_and_preserve_civil_dates() -> None:
    valuation = ValuationDate(date(2026, 1, 2))
    expiry = ExpiryDate(date(2026, 12, 18))

    assert str(valuation) == "2026-01-02"
    assert str(expiry) == "2026-12-18"
    with pytest.raises(TypeError, match="not a datetime"):
        ValuationDate(datetime(2026, 1, 2))
    with pytest.raises(TypeError, match="not a datetime"):
        ExpiryDate("2026-12-18")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("convention", "expected"),
    [
        (DayCountConvention.ACTUAL_365_FIXED, 366.0 / 365.0),
        (DayCountConvention.ACTUAL_360, 366.0 / 360.0),
        (
            DayCountConvention.ACTUAL_ACTUAL_ISDA,
            184.0 / 365.0 + 182.0 / 366.0,
        ),
        (DayCountConvention.THIRTY_360_US, 1.0),
        (DayCountConvention.THIRTY_E_360, 1.0),
    ],
)
def test_day_counts_cover_a_leap_year(convention: DayCountConvention, expected: float) -> None:
    start = date(2023, 7, 1)
    end = date(2024, 7, 1)

    assert convention.year_fraction(start, end) == pytest.approx(expected)
    assert convention.year_fraction(end, start) == pytest.approx(-expected)
    assert convention.year_fraction(start, start) == 0.0


def test_actual_actual_isda_splits_calendar_year_denominators() -> None:
    start = date(2019, 7, 1)
    end = date(2020, 7, 1)

    expected = 184.0 / 365.0 + 182.0 / 366.0
    assert DayCountConvention.ACTUAL_ACTUAL_ISDA.year_fraction(start, end) == pytest.approx(
        expected
    )


def test_thirty_360_end_of_month_rules_are_explicit() -> None:
    assert DayCountConvention.THIRTY_360_US.year_fraction(
        date(2024, 2, 29), date(2024, 3, 31)
    ) == pytest.approx(1.0 / 12.0)
    assert DayCountConvention.THIRTY_E_360.year_fraction(
        date(2024, 1, 31), date(2024, 2, 29)
    ) == pytest.approx(29.0 / 360.0)
    with pytest.raises(TypeError, match="not a datetime"):
        DayCountConvention.ACTUAL_365_FIXED.year_fraction(datetime(2024, 1, 1), date(2025, 1, 1))


def test_business_calendar_adjusts_weekends_holidays_and_month_boundaries() -> None:
    calendar = BusinessCalendar(
        name="desk_calendar",
        holidays=frozenset({date(2026, 6, 1)}),
    )
    sunday = date(2026, 5, 31)

    assert calendar.adjust(sunday, BusinessDayConvention.UNADJUSTED) == sunday
    assert calendar.adjust(sunday, BusinessDayConvention.FOLLOWING) == date(2026, 6, 2)
    assert calendar.adjust(sunday, BusinessDayConvention.MODIFIED_FOLLOWING) == date(2026, 5, 29)
    assert calendar.adjust(sunday, BusinessDayConvention.PRECEDING) == date(2026, 5, 29)
    assert calendar.is_business_day(date(2026, 6, 1)) is False


def test_modified_preceding_stays_in_the_original_month() -> None:
    calendar = BusinessCalendar()
    sunday = date(2026, 2, 1)

    assert calendar.adjust(sunday, BusinessDayConvention.PRECEDING) == date(2026, 1, 30)
    assert calendar.adjust(sunday, BusinessDayConvention.MODIFIED_PRECEDING) == date(2026, 2, 2)


def test_settlement_lag_counts_business_days_excluding_trade_date() -> None:
    calendar = BusinessCalendar(
        name="settlement",
        holidays=frozenset({date(2025, 1, 6)}),
    )
    trade_date = date(2025, 1, 3)  # Friday

    assert SettlementLag(0).settlement_date(trade_date, calendar) == trade_date
    assert SettlementLag(2).settlement_date(trade_date, calendar) == date(2025, 1, 8)
    assert calendar.advance_business_days(date(2025, 1, 8), -2) == trade_date
    assert ALL_DAYS_CALENDAR.advance_business_days(trade_date, 2) == date(2025, 1, 5)


def test_calendar_identity_is_order_independent_and_sensitive_to_rules() -> None:
    first = BusinessCalendar(
        name="same",
        holidays=frozenset({date(2026, 12, 25), date(2026, 1, 1)}),
    )
    second = BusinessCalendar(
        name="same",
        holidays=frozenset({date(2026, 1, 1), date(2026, 12, 25)}),
    )
    different = BusinessCalendar(name="same", holidays=first.holidays, weekend_days={4, 5})

    assert first.calendar_id == second.calendar_id
    assert first.calendar_id != different.calendar_id


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    [
        (lambda: BusinessCalendar(name=" "), ValueError, "calendar name"),
        (lambda: BusinessCalendar(holidays={datetime(2026, 1, 1)}), TypeError, "holiday"),
        (lambda: BusinessCalendar(weekend_days={True}), TypeError, "weekday integers"),
        (lambda: BusinessCalendar(weekend_days={0, 1, 2, 3, 4, 5, 6}), ValueError, "at least"),
        (lambda: SettlementLag(True), TypeError, "integer"),
        (lambda: SettlementLag(31), ValueError, "within"),
    ],
)
def test_calendar_and_settlement_validation(
    factory: object, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        factory()  # type: ignore[operator]


def test_calendar_requires_typed_conventions_and_dates() -> None:
    calendar = BusinessCalendar()
    with pytest.raises(TypeError, match="BusinessDayConvention"):
        calendar.adjust(date(2026, 1, 1), "following")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not a datetime"):
        calendar.is_business_day(datetime(2026, 1, 1))
    with pytest.raises(TypeError, match="integer"):
        calendar.advance_business_days(date(2026, 1, 1), 1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"name": 1}, TypeError, "name must be a string"),
        ({"holidays": "2026-01-01"}, TypeError, "iterable"),
        ({"holidays": 1}, TypeError, "iterable"),
        ({"weekend_days": "56"}, TypeError, "iterable"),
        ({"weekend_days": 5}, TypeError, "iterable"),
        ({"weekend_days": {-1}}, ValueError, "within"),
        ({"weekend_days": {7}}, ValueError, "within"),
    ],
)
def test_calendar_collection_validation(
    kwargs: dict[str, object], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        BusinessCalendar(**kwargs)  # type: ignore[arg-type]


def test_calendar_modified_adjustments_cover_same_month_paths() -> None:
    calendar = BusinessCalendar()

    assert calendar.adjust(date(2026, 5, 24), BusinessDayConvention.MODIFIED_FOLLOWING) == date(
        2026, 5, 25
    )
    assert calendar.adjust(date(2026, 2, 8), BusinessDayConvention.MODIFIED_PRECEDING) == date(
        2026, 2, 6
    )


def test_calendar_guards_civil_date_overflow_and_excessive_advance() -> None:
    max_calendar = BusinessCalendar(weekend_days={date.max.weekday()})
    min_calendar = BusinessCalendar(weekend_days={date.min.weekday()})

    with pytest.raises(RuntimeError, match="civil-date range"):
        max_calendar.adjust(date.max, BusinessDayConvention.FOLLOWING)
    with pytest.raises(RuntimeError, match="civil-date range"):
        min_calendar.adjust(date.min, BusinessDayConvention.PRECEDING)
    with pytest.raises(RuntimeError, match="civil-date range"):
        BusinessCalendar().advance_business_days(date.max, 1)
    with pytest.raises(ValueError, match="business_days must be within"):
        BusinessCalendar().advance_business_days(date(2026, 1, 1), 3_661)
    with pytest.raises(TypeError, match="zero_lag_convention"):
        BusinessCalendar().advance_business_days(
            date(2026, 1, 1),
            0,
            zero_lag_convention="following",  # type: ignore[arg-type]
        )


def test_settlement_lag_requires_a_business_calendar() -> None:
    with pytest.raises(TypeError, match="BusinessCalendar"):
        SettlementLag(2).settlement_date(
            date(2026, 1, 1),
            "calendar",  # type: ignore[arg-type]
        )
