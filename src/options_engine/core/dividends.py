"""Typed deterministic cash-dividend schedules for vanilla equity models."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real

MAX_CASH_DIVIDENDS = 64
MAX_CASH_DIVIDEND_AMOUNT = 1e12


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalised = float(value)
    if not math.isfinite(normalised):
        raise ValueError(f"{name} must be finite")
    return normalised


@dataclass(frozen=True, order=True, slots=True)
class CashDividend:
    """One known cash amount paid at a model ex-time.

    ``ex_time`` is a year fraction after valuation using the same clock as the
    option's ``time_to_expiry``. The event is intentionally not represented as
    a continuous yield.
    """

    ex_time: float
    amount: float

    def __post_init__(self) -> None:
        ex_time = _finite_real("ex_time", self.ex_time)
        amount = _finite_real("amount", self.amount)
        if ex_time <= 0.0:
            raise ValueError("ex_time must be strictly positive")
        if not 0.0 < amount <= MAX_CASH_DIVIDEND_AMOUNT:
            raise ValueError(f"amount must be within (0, {MAX_CASH_DIVIDEND_AMOUNT:g}]")
        object.__setattr__(self, "ex_time", ex_time)
        object.__setattr__(self, "amount", amount)

    def to_dict(self) -> dict[str, float]:
        return {"ex_time": self.ex_time, "amount": self.amount}

    def identity_payload(self) -> dict[str, str]:
        return {"amount": self.amount.hex(), "ex_time": self.ex_time.hex()}


@dataclass(frozen=True, slots=True)
class CashDividendSchedule:
    """Strictly ordered immutable collection of deterministic cash dividends."""

    dividends: tuple[CashDividend, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.dividends, tuple):
            raise TypeError("dividends must be a tuple of CashDividend values")
        if len(self.dividends) > MAX_CASH_DIVIDENDS:
            raise ValueError(f"cash-dividend schedule exceeds {MAX_CASH_DIVIDENDS} events")
        if any(not isinstance(dividend, CashDividend) for dividend in self.dividends):
            raise TypeError("every schedule entry must be a CashDividend")
        times = tuple(dividend.ex_time for dividend in self.dividends)
        if any(right <= left for left, right in pairwise(times)):
            raise ValueError("cash-dividend ex-times must be strictly increasing and unique")

    def __bool__(self) -> bool:
        return bool(self.dividends)

    def __len__(self) -> int:
        return len(self.dividends)

    @property
    def schedule_id(self) -> str:
        canonical = json.dumps(
            [dividend.identity_payload() for dividend in self.dividends],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def validate_for_maturity(self, maturity: float) -> None:
        maturity_value = _finite_real("maturity", maturity)
        if maturity_value <= 0.0:
            raise ValueError("maturity must be strictly positive")
        if self.dividends and self.dividends[-1].ex_time >= maturity_value:
            raise ValueError("cash-dividend ex-times must be strictly before option expiry")

    def to_list(self) -> list[dict[str, float]]:
        return [dividend.to_dict() for dividend in self.dividends]


EMPTY_CASH_DIVIDEND_SCHEDULE = CashDividendSchedule()


__all__ = [
    "EMPTY_CASH_DIVIDEND_SCHEDULE",
    "MAX_CASH_DIVIDENDS",
    "MAX_CASH_DIVIDEND_AMOUNT",
    "CashDividend",
    "CashDividendSchedule",
]
