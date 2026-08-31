"""Typed civil dates and deterministic day-count conventions."""

from __future__ import annotations

import calendar as _calendar
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum


def _require_civil_date(name: str, value: object) -> date:
    """Return a date while rejecting datetimes and implicit time-zone loss."""

    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a datetime.date, not a datetime")
    return value


@dataclass(frozen=True, order=True, slots=True)
class ValuationDate:
    """Explicit civil date on which a market snapshot is valued."""

    value: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _require_civil_date("valuation_date", self.value))

    def __str__(self) -> str:
        return self.value.isoformat()


@dataclass(frozen=True, order=True, slots=True)
class ExpiryDate:
    """Explicit civil expiry date for a dated option contract."""

    value: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _require_civil_date("expiry_date", self.value))

    def __str__(self) -> str:
        return self.value.isoformat()


@dataclass(frozen=True, order=True, slots=True)
class ExDividendDate:
    """Civil ex-date for a deterministic cash-dividend entitlement event."""

    value: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _require_civil_date("ex_dividend_date", self.value))

    def __str__(self) -> str:
        return self.value.isoformat()


def _is_last_day_of_february(value: date) -> bool:
    return value.month == 2 and value.day == _calendar.monthrange(value.year, 2)[1]


def _actual_actual_isda(start: date, end: date) -> float:
    cursor = start
    fraction = 0.0
    while cursor < end:
        segment_end = end if cursor.year == end.year else date(cursor.year + 1, 1, 1)
        denominator = 366.0 if _calendar.isleap(cursor.year) else 365.0
        fraction += (segment_end - cursor).days / denominator
        cursor = segment_end
    return fraction


def _thirty_360_us(start: date, end: date) -> float:
    start_day = start.day
    end_day = end.day
    start_is_february_end = _is_last_day_of_february(start)
    end_is_february_end = _is_last_day_of_february(end)

    if start_is_february_end or start_day == 31:
        start_day = 30
    if (end_is_february_end and start_is_february_end) or (end_day == 31 and start_day >= 30):
        end_day = 30

    days = 360 * (end.year - start.year) + 30 * (end.month - start.month) + end_day - start_day
    return days / 360.0


def _thirty_e_360(start: date, end: date) -> float:
    days = (
        360 * (end.year - start.year)
        + 30 * (end.month - start.month)
        + min(end.day, 30)
        - min(start.day, 30)
    )
    return days / 360.0


class DayCountConvention(StrEnum):
    """Supported year-fraction rules.

    The calculation uses civil dates only. Reverse intervals are defined as
    the negative of the corresponding forward interval, which makes every
    convention exactly antisymmetric.
    """

    ACTUAL_365_FIXED = "actual_365_fixed"
    ACTUAL_360 = "actual_360"
    ACTUAL_ACTUAL_ISDA = "actual_actual_isda"
    THIRTY_360_US = "thirty_360_us"
    THIRTY_E_360 = "thirty_e_360"

    def year_fraction(self, start: date, end: date) -> float:
        """Calculate a signed year fraction between two civil dates."""

        start_date = _require_civil_date("start", start)
        end_date = _require_civil_date("end", end)
        if start_date == end_date:
            return 0.0
        if end_date < start_date:
            return -self.year_fraction(end_date, start_date)

        if self is DayCountConvention.ACTUAL_365_FIXED:
            return (end_date - start_date).days / 365.0
        if self is DayCountConvention.ACTUAL_360:
            return (end_date - start_date).days / 360.0
        if self is DayCountConvention.ACTUAL_ACTUAL_ISDA:
            return _actual_actual_isda(start_date, end_date)
        if self is DayCountConvention.THIRTY_360_US:
            return _thirty_360_us(start_date, end_date)
        if self is DayCountConvention.THIRTY_E_360:
            return _thirty_e_360(start_date, end_date)
        raise AssertionError(f"Unhandled day-count convention: {self}")


__all__ = ["DayCountConvention", "ExDividendDate", "ExpiryDate", "ValuationDate"]
