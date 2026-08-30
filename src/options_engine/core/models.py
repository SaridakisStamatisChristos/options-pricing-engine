"""Domain models for the options pricing engine."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .replay import ReplayCapsule


MAX_SPOT_PRICE = 1e12
MAX_STRIKE_PRICE = 1e12
MAX_TIME_TO_EXPIRY = 100.0


class OptionType(StrEnum):
    """Supported option contract types."""

    CALL = "call"
    PUT = "put"


class ExerciseStyle(StrEnum):
    """Available exercise styles for an option contract."""

    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market conditions required to price an option."""

    spot_price: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("spot_price", self.spot_price),
            ("risk_free_rate", self.risk_free_rate),
            ("dividend_yield", self.dividend_yield),
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number")
        spot_price = float(self.spot_price)
        risk_free_rate = float(self.risk_free_rate)
        dividend_yield = float(self.dividend_yield)
        if not math.isfinite(spot_price) or not 0.0 < spot_price <= MAX_SPOT_PRICE:
            raise ValueError(f"spot_price must be within (0, {MAX_SPOT_PRICE:g}]")
        if not math.isfinite(risk_free_rate) or not -1.0 <= risk_free_rate <= 1.0:
            raise ValueError("risk_free_rate must be within [-1, 1]")
        if not math.isfinite(dividend_yield) or not -1.0 <= dividend_yield <= 1.0:
            raise ValueError("dividend_yield must be within [-1, 1]")
        object.__setattr__(self, "spot_price", spot_price)
        object.__setattr__(self, "risk_free_rate", risk_free_rate)
        object.__setattr__(self, "dividend_yield", dividend_yield)
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.now(UTC))
        elif not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime")
        elif self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class OptionContract:
    """Immutable description of an option contract to be priced."""

    symbol: str
    strike_price: float
    time_to_expiry: float
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
        if isinstance(self.strike_price, bool) or not isinstance(self.strike_price, Real):
            raise TypeError("strike_price must be a real number")
        if isinstance(self.time_to_expiry, bool) or not isinstance(self.time_to_expiry, Real):
            raise TypeError("time_to_expiry must be a real number")
        strike_price = float(self.strike_price)
        time_to_expiry = float(self.time_to_expiry)
        if not math.isfinite(strike_price) or not 0.0 < strike_price <= MAX_STRIKE_PRICE:
            raise ValueError(f"strike_price must be within (0, {MAX_STRIKE_PRICE:g}]")
        if not math.isfinite(time_to_expiry) or not 0.0 < time_to_expiry <= MAX_TIME_TO_EXPIRY:
            raise ValueError(f"time_to_expiry must be within (0, {MAX_TIME_TO_EXPIRY:g}]")
        if not isinstance(self.option_type, OptionType):
            raise TypeError("option_type must be an OptionType")
        if not isinstance(self.exercise_style, ExerciseStyle):
            raise TypeError("exercise_style must be an ExerciseStyle")

        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "strike_price", strike_price)
        object.__setattr__(self, "time_to_expiry", time_to_expiry)
        if not isinstance(self.contract_id, str):
            raise TypeError("contract_id must be a string")
        if self.contract_id:
            contract_id = self.contract_id.strip()
            if not contract_id or len(contract_id) > 256:
                raise ValueError("contract_id must contain between 1 and 256 characters")
            if any(unicodedata.category(character).startswith("C") for character in contract_id):
                raise ValueError("contract_id must not contain control characters")
        else:
            contract_id = ""

        # Hash the exact economic terms. Rounded, human-readable identifiers caused
        # cache aliases between nearby strikes/maturities and between exercise styles.
        identity = json.dumps(
            {
                "exercise_style": self.exercise_style.value,
                "option_type": self.option_type.value,
                "strike_price": strike_price.hex(),
                "symbol": symbol,
                "time_to_expiry": time_to_expiry.hex(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
        contract_id = contract_id or (
            f"{symbol}:{self.option_type.value}:{self.exercise_style.value}:{digest}"
        )
        object.__setattr__(self, "contract_id", contract_id)


@dataclass(slots=True)
class PricingResult:
    """Container for the outcome of a pricing model evaluation."""

    contract_id: str
    theoretical_price: float
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    rho: float | None = None
    implied_volatility: float | None = None
    computation_time_ms: float = 0.0
    model_used: str = "unknown"
    error: str | None = None
    standard_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    capsule_id: str | None = None
    replay_capsule: ReplayCapsule | None = None
    control_variate_report: dict[str, object] | None = None
    estimate_diagnostics: dict[str, object] | None = None
    numerical_diagnostics: dict[str, object] | None = None
    ci_greeks: dict[str, dict[str, float]] | None = None
    greeks_meta: dict[str, dict[str, object]] | None = None
