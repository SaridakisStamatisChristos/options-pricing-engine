"""Dated deterministic cash-dividend schedules for market snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real

from ..core.dividends import MAX_CASH_DIVIDEND_AMOUNT, MAX_CASH_DIVIDENDS
from .dates import ExDividendDate


@dataclass(frozen=True, order=True, slots=True)
class DatedCashDividend:
    """One fixed same-currency cash amount on an unadjusted ex-date."""

    ex_date: ExDividendDate
    amount: float

    def __post_init__(self) -> None:
        if not isinstance(self.ex_date, ExDividendDate):
            raise TypeError("ex_date must be an ExDividendDate")
        if isinstance(self.amount, bool) or not isinstance(self.amount, Real):
            raise TypeError("amount must be a real number")
        amount = float(self.amount)
        if not math.isfinite(amount) or not 0.0 < amount <= MAX_CASH_DIVIDEND_AMOUNT:
            raise ValueError(f"amount must be within (0, {MAX_CASH_DIVIDEND_AMOUNT:g}]")
        object.__setattr__(self, "amount", amount)


@dataclass(frozen=True, slots=True)
class DatedCashDividendSchedule:
    """Strictly ordered immutable dated dividend schedule."""

    dividends: tuple[DatedCashDividend, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.dividends, tuple):
            raise TypeError("dividends must be a tuple of DatedCashDividend values")
        if len(self.dividends) > MAX_CASH_DIVIDENDS:
            raise ValueError(f"cash-dividend schedule exceeds {MAX_CASH_DIVIDENDS} events")
        if any(not isinstance(dividend, DatedCashDividend) for dividend in self.dividends):
            raise TypeError("every schedule entry must be a DatedCashDividend")
        dates = tuple(dividend.ex_date.value for dividend in self.dividends)
        if any(right <= left for left, right in pairwise(dates)):
            raise ValueError("cash-dividend ex-dates must be strictly increasing and unique")

    def __bool__(self) -> bool:
        return bool(self.dividends)

    def __len__(self) -> int:
        return len(self.dividends)

    @property
    def schedule_id(self) -> str:
        payload = [
            {"amount": dividend.amount.hex(), "ex_date": dividend.ex_date.value.isoformat()}
            for dividend in self.dividends
        ]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_list(self) -> list[dict[str, object]]:
        return [
            {"amount": dividend.amount, "ex_date": dividend.ex_date.value.isoformat()}
            for dividend in self.dividends
        ]


EMPTY_DATED_CASH_DIVIDEND_SCHEDULE = DatedCashDividendSchedule()


__all__ = [
    "EMPTY_DATED_CASH_DIVIDEND_SCHEDULE",
    "DatedCashDividend",
    "DatedCashDividendSchedule",
]
