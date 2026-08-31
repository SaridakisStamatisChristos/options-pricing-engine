"""Aggregate market-date conventions used before numerical dispatch."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date

from .calendars import (
    WEEKEND_CALENDAR,
    ZERO_SETTLEMENT_LAG,
    BusinessCalendar,
    BusinessDayConvention,
    SettlementLag,
)
from .dates import DayCountConvention, ExpiryDate, ValuationDate


@dataclass(frozen=True, slots=True)
class MarketConventions:
    """Explicit date, calendar, and settlement rules for one market snapshot."""

    valuation_date: ValuationDate
    day_count: DayCountConvention = DayCountConvention.ACTUAL_365_FIXED
    calendar: BusinessCalendar = WEEKEND_CALENDAR
    settlement_lag: SettlementLag = ZERO_SETTLEMENT_LAG
    settlement_convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING
    expiry_convention: BusinessDayConvention = BusinessDayConvention.UNADJUSTED

    def __post_init__(self) -> None:
        if not isinstance(self.valuation_date, ValuationDate):
            raise TypeError("valuation_date must be a ValuationDate")
        if not isinstance(self.day_count, DayCountConvention):
            raise TypeError("day_count must be a DayCountConvention")
        if not isinstance(self.calendar, BusinessCalendar):
            raise TypeError("calendar must be a BusinessCalendar")
        if not isinstance(self.settlement_lag, SettlementLag):
            raise TypeError("settlement_lag must be a SettlementLag")
        if not isinstance(self.settlement_convention, BusinessDayConvention):
            raise TypeError("settlement_convention must be a BusinessDayConvention")
        if not isinstance(self.expiry_convention, BusinessDayConvention):
            raise TypeError("expiry_convention must be a BusinessDayConvention")

    @property
    def conventions_id(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def settlement_date(self) -> date:
        return self.settlement_lag.settlement_date(
            self.valuation_date.value,
            self.calendar,
            self.settlement_convention,
        )

    def adjusted_expiry(self, expiry: ExpiryDate) -> date:
        if not isinstance(expiry, ExpiryDate):
            raise TypeError("expiry must be an ExpiryDate")
        return self.calendar.adjust(expiry.value, self.expiry_convention)

    def year_fraction(self, expiry: ExpiryDate) -> float:
        return self.day_count.year_fraction(
            self.valuation_date.value,
            self.adjusted_expiry(expiry),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a canonical, JSON-safe description of the conventions."""

        return {
            "calendar_id": self.calendar.calendar_id,
            "calendar_name": self.calendar.name,
            "day_count": self.day_count.value,
            "expiry_convention": self.expiry_convention.value,
            "settlement_convention": self.settlement_convention.value,
            "settlement_lag_business_days": self.settlement_lag.business_days,
            "valuation_date": self.valuation_date.value.isoformat(),
        }


__all__ = ["MarketConventions"]
