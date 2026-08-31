"""Dated market snapshots resolved into legacy scalar pricing inputs."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime, time
from numbers import Real

from ..core.models import (
    MAX_STRIKE_PRICE,
    ExerciseStyle,
    MarketData,
    OptionContract,
    OptionType,
)
from .calendars import (
    WEEKEND_CALENDAR,
    ZERO_SETTLEMENT_LAG,
    BusinessCalendar,
    BusinessDayConvention,
    SettlementLag,
)
from .conventions import MarketConventions
from .curves import CarryCurve, DiscountCurve, FlatDiscountCurve, FlatDividendCurve
from .dates import DayCountConvention, ExpiryDate, ValuationDate
from .forwards import ForwardBuilder, ForwardResult


def _strike(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("strike_price must be a real number")
    strike = float(value)
    if not math.isfinite(strike) or not 0.0 < strike <= MAX_STRIKE_PRICE:
        raise ValueError(f"strike_price must be within (0, {MAX_STRIKE_PRICE:g}]")
    return strike


@dataclass(frozen=True, slots=True)
class DatedOptionContract:
    """Option terms expressed with a civil expiry rather than a year fraction."""

    symbol: str
    strike_price: float
    expiry_date: ExpiryDate
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    contract_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str):
            raise TypeError("symbol must be a string")
        symbol = self.symbol.strip().upper()
        if not symbol or len(symbol) > 64:
            raise ValueError("symbol must contain between 1 and 64 characters")
        if any(unicodedata.category(character).startswith("C") for character in symbol):
            raise ValueError("symbol must not contain control characters")
        if not isinstance(self.expiry_date, ExpiryDate):
            raise TypeError("expiry_date must be an ExpiryDate")
        if not isinstance(self.option_type, OptionType):
            raise TypeError("option_type must be an OptionType")
        if not isinstance(self.exercise_style, ExerciseStyle):
            raise TypeError("exercise_style must be an ExerciseStyle")
        if not isinstance(self.contract_id, str):
            raise TypeError("contract_id must be a string")

        strike = _strike(self.strike_price)
        contract_id = self.contract_id.strip()
        if self.contract_id and (
            not contract_id
            or len(contract_id) > 256
            or any(unicodedata.category(character).startswith("C") for character in contract_id)
        ):
            raise ValueError("contract_id must contain between 1 and 256 printable characters")
        if not contract_id:
            identity = json.dumps(
                {
                    "exercise_style": self.exercise_style.value,
                    "expiry_date": self.expiry_date.value.isoformat(),
                    "option_type": self.option_type.value,
                    "strike_price": strike.hex(),
                    "symbol": symbol,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
            contract_id = (
                f"{symbol}:{self.option_type.value}:{self.exercise_style.value}:dated:{digest}"
            )

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "strike_price", strike)
        object.__setattr__(self, "contract_id", contract_id)

    def resolve(self, time_to_expiry: float) -> OptionContract:
        """Create the scalar contract consumed by numerical model classes."""

        return OptionContract(
            symbol=self.symbol,
            strike_price=self.strike_price,
            time_to_expiry=time_to_expiry,
            option_type=self.option_type,
            exercise_style=self.exercise_style,
            contract_id=self.contract_id,
        )


@dataclass(frozen=True, slots=True)
class ResolvedPricingInputs:
    """Scalar model inputs plus the conventions evidence used to derive them."""

    contract: OptionContract
    market_data: MarketData
    forward: ForwardResult
    conventions: MarketConventions
    discount_curve_id: str
    carry_curve_id: str
    conventions_id: str
    market_id: str

    def diagnostics(self) -> dict[str, object]:
        payload = self.forward.to_dict()
        payload.update(
            {
                "carry_curve_id": self.carry_curve_id,
                "conventions": self.conventions.to_dict(),
                "conventions_id": self.conventions_id,
                "discount_curve_id": self.discount_curve_id,
                "equivalent_continuous_dividend_yield": self.market_data.dividend_yield,
                "equivalent_continuous_risk_free_rate": self.market_data.risk_free_rate,
                "market_id": self.market_id,
                "rate_representation": "endpoint_equivalent_continuous",
                "time_to_expiry": self.contract.time_to_expiry,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class MarketEnvironment:
    """Immutable market snapshot with curves and explicit conventions."""

    spot_price: float
    conventions: MarketConventions
    discount_curve: DiscountCurve
    carry_curve: CarryCurve
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.conventions, MarketConventions):
            raise TypeError("conventions must be MarketConventions")
        builder = ForwardBuilder(
            spot_price=self.spot_price,
            discount_curve=self.discount_curve,
            carry_curve=self.carry_curve,
            conventions=self.conventions,
        )
        object.__setattr__(self, "spot_price", builder.spot_price)

        timestamp = self.timestamp
        if timestamp is None:
            timestamp = datetime.combine(
                self.conventions.valuation_date.value,
                time.min,
                tzinfo=UTC,
            )
        elif not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime or None")
        elif timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        else:
            timestamp = timestamp.astimezone(UTC)
        object.__setattr__(self, "timestamp", timestamp)

    @property
    def market_id(self) -> str:
        if self.timestamp is None:  # pragma: no cover - normalized in __post_init__
            raise AssertionError("timestamp was not normalized")
        payload = {
            "carry_curve_id": self.carry_curve.curve_id,
            "conventions_id": self.conventions.conventions_id,
            "discount_curve_id": self.discount_curve.curve_id,
            "spot_price": self.spot_price.hex(),
            "timestamp": self.timestamp.isoformat(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_scalar_rates(
        cls,
        *,
        spot_price: float,
        valuation_date: ValuationDate,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        day_count: DayCountConvention = DayCountConvention.ACTUAL_365_FIXED,
        calendar: BusinessCalendar = WEEKEND_CALENDAR,
        settlement_lag: SettlementLag = ZERO_SETTLEMENT_LAG,
        settlement_convention: BusinessDayConvention = BusinessDayConvention.FOLLOWING,
        expiry_convention: BusinessDayConvention = BusinessDayConvention.UNADJUSTED,
        timestamp: datetime | None = None,
    ) -> MarketEnvironment:
        """Build a dated environment from the established scalar-rate inputs."""

        conventions = MarketConventions(
            valuation_date=valuation_date,
            day_count=day_count,
            calendar=calendar,
            settlement_lag=settlement_lag,
            settlement_convention=settlement_convention,
            expiry_convention=expiry_convention,
        )
        return cls(
            spot_price=spot_price,
            conventions=conventions,
            discount_curve=FlatDiscountCurve(
                reference_date=valuation_date,
                rate=risk_free_rate,
                day_count=day_count,
            ),
            carry_curve=FlatDividendCurve(
                reference_date=valuation_date,
                rate=dividend_yield,
                day_count=day_count,
            ),
            timestamp=timestamp,
        )

    def resolve(self, contract: DatedOptionContract) -> ResolvedPricingInputs:
        """Resolve dates and curves to the scalar endpoint representation.

        The equivalent rates preserve both the constructed forward and the
        valuation-to-expiry discount factor. Thus existing vanilla kernels can
        consume the snapshot without knowing about calendars or curve shapes.
        """

        if not isinstance(contract, DatedOptionContract):
            raise TypeError("contract must be a DatedOptionContract")
        forward = ForwardBuilder(
            spot_price=self.spot_price,
            discount_curve=self.discount_curve,
            carry_curve=self.carry_curve,
            conventions=self.conventions,
        ).build(contract.expiry_date)
        time_to_expiry = self.conventions.day_count.year_fraction(
            self.conventions.valuation_date.value,
            forward.expiry_date,
        )
        if not math.isfinite(time_to_expiry) or time_to_expiry <= 0.0:
            raise ValueError("day-count year fraction to adjusted expiry must be positive")

        risk_free_rate = -math.log(forward.discount_factor) / time_to_expiry
        forward_growth = math.log(forward.forward_price / self.spot_price) / time_to_expiry
        dividend_yield = risk_free_rate - forward_growth
        if self.timestamp is None:  # pragma: no cover - normalized in __post_init__
            raise AssertionError("timestamp was not normalized")
        market_data = MarketData(
            spot_price=self.spot_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            timestamp=self.timestamp,
        )
        scalar_contract = contract.resolve(time_to_expiry)
        return ResolvedPricingInputs(
            contract=scalar_contract,
            market_data=market_data,
            forward=forward,
            conventions=self.conventions,
            discount_curve_id=self.discount_curve.curve_id,
            carry_curve_id=self.carry_curve.curve_id,
            conventions_id=self.conventions.conventions_id,
            market_id=self.market_id,
        )


__all__ = ["DatedOptionContract", "MarketEnvironment", "ResolvedPricingInputs"]
