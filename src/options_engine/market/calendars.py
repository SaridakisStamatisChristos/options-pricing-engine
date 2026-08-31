"""Immutable business calendars, date adjustment, and settlement lags."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from numbers import Integral

MAX_CALENDAR_HOLIDAYS = 100_000
MAX_CALENDAR_SEARCH_DAYS = 3_660
MAX_SETTLEMENT_LAG = 30


def _civil_date(name: str, value: object) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a datetime.date, not a datetime")
    return value


class BusinessDayConvention(StrEnum):
    """Rules for moving a non-business date onto a business date."""

    UNADJUSTED = "unadjusted"
    FOLLOWING = "following"
    MODIFIED_FOLLOWING = "modified_following"
    PRECEDING = "preceding"
    MODIFIED_PRECEDING = "modified_preceding"


@dataclass(frozen=True, slots=True)
class BusinessCalendar:
    """User-supplied holiday calendar with an explicit weekend definition.

    The package deliberately does not download or infer exchange holidays.
    Callers own the holiday set, making historical valuation reproducible.
    Weekdays use :meth:`datetime.date.weekday` numbering (Monday is zero).
    """

    name: str = "weekend_only"
    holidays: frozenset[date] = field(default_factory=frozenset)
    weekend_days: frozenset[int] = field(default_factory=lambda: frozenset({5, 6}))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("calendar name must be a string")
        name = self.name.strip()
        if (
            not name
            or len(name) > 64
            or any(unicodedata.category(character).startswith("C") for character in name)
        ):
            raise ValueError("calendar name must contain 1 to 64 printable characters")

        if isinstance(self.holidays, (str, bytes)):
            raise TypeError("holidays must be an iterable of datetime.date values")
        try:
            raw_holidays = tuple(self.holidays)
        except TypeError as exc:
            raise TypeError("holidays must be an iterable of datetime.date values") from exc
        if len(raw_holidays) > MAX_CALENDAR_HOLIDAYS:
            raise ValueError(f"calendar cannot contain more than {MAX_CALENDAR_HOLIDAYS} holidays")
        holidays = frozenset(_civil_date("holiday", value) for value in raw_holidays)

        if isinstance(self.weekend_days, (str, bytes)):
            raise TypeError("weekend_days must be an iterable of weekday integers")
        try:
            raw_weekend = tuple(self.weekend_days)
        except TypeError as exc:
            raise TypeError("weekend_days must be an iterable of weekday integers") from exc
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw_weekend):
            raise TypeError("weekend_days must contain weekday integers")
        weekend_days = frozenset(int(value) for value in raw_weekend)
        if any(value < 0 or value > 6 for value in weekend_days):
            raise ValueError("weekend_days must be within [0, 6]")
        if len(weekend_days) == 7:
            raise ValueError("calendar must retain at least one potential business weekday")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "holidays", holidays)
        object.__setattr__(self, "weekend_days", weekend_days)

    @property
    def calendar_id(self) -> str:
        payload = {
            "holidays": sorted(value.isoformat() for value in self.holidays),
            "name": self.name,
            "weekend_days": sorted(self.weekend_days),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def is_business_day(self, value: date) -> bool:
        on_date = _civil_date("date", value)
        return on_date.weekday() not in self.weekend_days and on_date not in self.holidays

    def _seek(self, value: date, direction: int) -> date:
        candidate = value
        for _ in range(MAX_CALENDAR_SEARCH_DAYS + 1):
            if self.is_business_day(candidate):
                return candidate
            try:
                candidate += timedelta(days=direction)
            except OverflowError as exc:
                raise RuntimeError("calendar adjustment exceeded the civil-date range") from exc
        raise RuntimeError(
            f"calendar '{self.name}' has no reachable business day within "
            f"{MAX_CALENDAR_SEARCH_DAYS} days"
        )

    def adjust(
        self,
        value: date,
        convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
    ) -> date:
        """Adjust ``value`` according to an explicit business-day rule."""

        on_date = _civil_date("date", value)
        if not isinstance(convention, BusinessDayConvention):
            raise TypeError("convention must be a BusinessDayConvention")
        if convention is BusinessDayConvention.UNADJUSTED or self.is_business_day(on_date):
            return on_date
        if convention is BusinessDayConvention.FOLLOWING:
            return self._seek(on_date, 1)
        if convention is BusinessDayConvention.PRECEDING:
            return self._seek(on_date, -1)
        if convention is BusinessDayConvention.MODIFIED_FOLLOWING:
            following = self._seek(on_date, 1)
            return following if following.month == on_date.month else self._seek(on_date, -1)
        if convention is BusinessDayConvention.MODIFIED_PRECEDING:
            preceding = self._seek(on_date, -1)
            return preceding if preceding.month == on_date.month else self._seek(on_date, 1)
        raise AssertionError(f"Unhandled business-day convention: {convention}")

    def advance_business_days(
        self,
        value: date,
        business_days: int,
        *,
        zero_lag_convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
    ) -> date:
        """Advance by signed business days, excluding the starting date."""

        on_date = _civil_date("date", value)
        if isinstance(business_days, bool) or not isinstance(business_days, Integral):
            raise TypeError("business_days must be an integer")
        count = int(business_days)
        if abs(count) > MAX_CALENDAR_SEARCH_DAYS:
            raise ValueError(
                f"business_days must be within [-{MAX_CALENDAR_SEARCH_DAYS}, "
                f"{MAX_CALENDAR_SEARCH_DAYS}]"
            )
        if not isinstance(zero_lag_convention, BusinessDayConvention):
            raise TypeError("zero_lag_convention must be a BusinessDayConvention")
        if count == 0:
            return self.adjust(on_date, zero_lag_convention)

        direction = 1 if count > 0 else -1
        remaining = abs(count)
        candidate = on_date
        searched = 0
        while remaining:
            try:
                candidate += timedelta(days=direction)
            except OverflowError as exc:
                raise RuntimeError("business-day advance exceeded the civil-date range") from exc
            searched += 1
            if searched > MAX_CALENDAR_SEARCH_DAYS:
                raise RuntimeError(
                    f"calendar '{self.name}' cannot advance {business_days} business days "
                    f"within {MAX_CALENDAR_SEARCH_DAYS} calendar days"
                )
            if self.is_business_day(candidate):
                remaining -= 1
        return candidate


@dataclass(frozen=True, slots=True)
class SettlementLag:
    """A spot-settlement lag measured in business days."""

    business_days: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.business_days, bool) or not isinstance(self.business_days, Integral):
            raise TypeError("settlement business_days must be an integer")
        business_days = int(self.business_days)
        if not 0 <= business_days <= MAX_SETTLEMENT_LAG:
            raise ValueError(f"settlement business_days must be within [0, {MAX_SETTLEMENT_LAG}]")
        object.__setattr__(self, "business_days", business_days)

    def settlement_date(
        self,
        valuation_date: date,
        calendar: BusinessCalendar,
        convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
    ) -> date:
        if not isinstance(calendar, BusinessCalendar):
            raise TypeError("calendar must be a BusinessCalendar")
        return calendar.advance_business_days(
            _civil_date("valuation_date", valuation_date),
            self.business_days,
            zero_lag_convention=convention,
        )


ALL_DAYS_CALENDAR = BusinessCalendar(name="all_days", weekend_days=frozenset())
WEEKEND_CALENDAR = BusinessCalendar()
ZERO_SETTLEMENT_LAG = SettlementLag()


__all__ = [
    "ALL_DAYS_CALENDAR",
    "WEEKEND_CALENDAR",
    "ZERO_SETTLEMENT_LAG",
    "BusinessCalendar",
    "BusinessDayConvention",
    "SettlementLag",
]
