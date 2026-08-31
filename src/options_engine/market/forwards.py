"""Forward construction from spot, funding, carry, and settlement conventions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from numbers import Real

from .conventions import MarketConventions
from .curves import CarryCurve, DiscountCurve
from .dates import ExpiryDate
from .dividends import (
    EMPTY_DATED_CASH_DIVIDEND_SCHEDULE,
    DatedCashDividendSchedule,
)

MAX_SPOT_PRICE = 1e12


def _spot(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("spot_price must be a real number")
    spot = float(value)
    if not math.isfinite(spot) or not 0.0 < spot <= MAX_SPOT_PRICE:
        raise ValueError(f"spot_price must be within (0, {MAX_SPOT_PRICE:g}]")
    return spot


def _factor(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    factor = float(value)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return factor


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """Auditable components of one spot-to-expiry forward construction."""

    spot_price: float
    valuation_date: date
    settlement_date: date
    unadjusted_expiry_date: date
    expiry_date: date
    discount_factor: float
    carry_factor: float
    settlement_discount_factor: float
    settlement_carry_factor: float
    forward_price: float
    continuous_carry_forward_price: float | None = None
    cash_dividend_future_value: float = 0.0

    @property
    def prepaid_forward(self) -> float:
        return self.discount_factor * self.forward_price

    def to_dict(self) -> dict[str, object]:
        return {
            "carry_factor": self.carry_factor,
            "discount_factor": self.discount_factor,
            "expiry_date": self.expiry_date.isoformat(),
            "forward_price": self.forward_price,
            "continuous_carry_forward_price": (
                self.continuous_carry_forward_price
                if self.continuous_carry_forward_price is not None
                else self.forward_price
            ),
            "cash_dividend_future_value": self.cash_dividend_future_value,
            "prepaid_forward": self.prepaid_forward,
            "settlement_carry_factor": self.settlement_carry_factor,
            "settlement_date": self.settlement_date.isoformat(),
            "settlement_discount_factor": self.settlement_discount_factor,
            "spot_price": self.spot_price,
            "unadjusted_expiry_date": self.unadjusted_expiry_date.isoformat(),
            "valuation_date": self.valuation_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ForwardBuilder:
    """Construct forwards without embedding curve logic in pricing models.

    Spot is interpreted as settling on the configured spot-settlement date.
    Therefore the forward accrues funding and carry from settlement to expiry,
    while option present values still use the valuation-to-expiry discount
    factor.
    """

    spot_price: float
    discount_curve: DiscountCurve
    carry_curve: CarryCurve
    conventions: MarketConventions
    cash_dividends: DatedCashDividendSchedule = EMPTY_DATED_CASH_DIVIDEND_SCHEDULE

    def __post_init__(self) -> None:
        object.__setattr__(self, "spot_price", _spot(self.spot_price))
        if not isinstance(self.discount_curve, DiscountCurve):
            raise TypeError("discount_curve must implement DiscountCurve")
        if not isinstance(self.carry_curve, CarryCurve):
            raise TypeError("carry_curve must implement CarryCurve")
        if not isinstance(self.conventions, MarketConventions):
            raise TypeError("conventions must be MarketConventions")
        if not isinstance(self.cash_dividends, DatedCashDividendSchedule):
            raise TypeError("cash_dividends must be a DatedCashDividendSchedule")
        reference_date = self.conventions.valuation_date
        if self.discount_curve.reference_date != reference_date:
            raise ValueError("discount_curve reference date must equal the valuation date")
        if self.carry_curve.reference_date != reference_date:
            raise ValueError("carry_curve reference date must equal the valuation date")

    def build(self, expiry: ExpiryDate) -> ForwardResult:
        if not isinstance(expiry, ExpiryDate):
            raise TypeError("expiry must be an ExpiryDate")

        valuation_date = self.conventions.valuation_date.value
        settlement_date = self.conventions.settlement_date
        adjusted_expiry = self.conventions.adjusted_expiry(expiry)
        if adjusted_expiry <= valuation_date:
            raise ValueError("adjusted expiry date must be after the valuation date")
        if adjusted_expiry <= settlement_date:
            raise ValueError("adjusted expiry date must be after the settlement date")

        discount_factor = _factor(
            "valuation discount factor",
            self.discount_curve.forward_discount_factor(valuation_date, adjusted_expiry),
        )
        carry_factor = _factor(
            "valuation carry factor",
            self.carry_curve.forward_carry_factor(valuation_date, adjusted_expiry),
        )
        settlement_discount_factor = _factor(
            "settlement discount factor",
            self.discount_curve.forward_discount_factor(settlement_date, adjusted_expiry),
        )
        settlement_carry_factor = _factor(
            "settlement carry factor",
            self.carry_curve.forward_carry_factor(settlement_date, adjusted_expiry),
        )
        continuous_forward = self.spot_price * (
            settlement_carry_factor / settlement_discount_factor
        )
        cash_dividend_future_value = 0.0
        for dividend in self.cash_dividends.dividends:
            ex_date = dividend.ex_date.value
            if ex_date <= valuation_date:
                raise ValueError("cash-dividend ex-dates must be after the valuation date")
            if ex_date == adjusted_expiry:
                raise ValueError("cash-dividend ex-date must not equal adjusted expiry")
            if ex_date > adjusted_expiry:
                continue
            if ex_date <= settlement_date:
                raise ValueError(
                    "cash-dividend ex-dates on or before spot settlement have ambiguous entitlement"
                )
            if not self.conventions.calendar.is_business_day(ex_date):
                raise ValueError(
                    "cash-dividend ex-dates must be business days in the configured calendar"
                )
            event_discount = _factor(
                "cash-dividend event discount factor",
                self.discount_curve.forward_discount_factor(ex_date, adjusted_expiry),
            )
            event_carry = _factor(
                "cash-dividend event carry factor",
                self.carry_curve.forward_carry_factor(ex_date, adjusted_expiry),
            )
            cash_dividend_future_value += dividend.amount * event_carry / event_discount

        forward_price = continuous_forward - cash_dividend_future_value
        if not math.isfinite(forward_price) or forward_price <= 0.0:
            raise ValueError("constructed forward price is outside floating-point range")
        prepaid_forward = discount_factor * forward_price
        if not math.isfinite(prepaid_forward) or prepaid_forward <= 0.0:
            raise ValueError("constructed prepaid forward is outside floating-point range")

        return ForwardResult(
            spot_price=self.spot_price,
            valuation_date=valuation_date,
            settlement_date=settlement_date,
            unadjusted_expiry_date=expiry.value,
            expiry_date=adjusted_expiry,
            discount_factor=discount_factor,
            carry_factor=carry_factor,
            settlement_discount_factor=settlement_discount_factor,
            settlement_carry_factor=settlement_carry_factor,
            forward_price=forward_price,
            continuous_carry_forward_price=continuous_forward,
            cash_dividend_future_value=cash_dividend_future_value,
        )


__all__ = ["ForwardBuilder", "ForwardResult"]
