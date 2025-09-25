"""Domain models for the options pricing engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Optional, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from .replay import ReplayCapsule


class OptionType(str, Enum):
    """Supported option contract types."""

    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    """Available exercise styles for an option contract."""

    EUROPEAN = "european"
    AMERICAN = "american"


@dataclass(frozen=True, slots=True)
class MarketData:
    """Market conditions required to price an option."""

    spot_price: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass validation
        if self.spot_price <= 0:
            raise ValueError("spot_price must be strictly positive")
        if not -1.0 <= self.risk_free_rate <= 1.0:
            raise ValueError("risk_free_rate must be within [-1, 1]")
        if not 0 <= self.dividend_yield <= 1.0:
            raise ValueError("dividend_yield must be within [0, 1]")
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class OptionContract:
    """Immutable description of an option contract to be priced."""

    symbol: str
    strike_price: float
    time_to_expiry: float
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    contract_id: Optional[str] = None

    def __post_init__(self) -> None:  # pragma: no cover - dataclass validation
        if not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if self.strike_price <= 0:
            raise ValueError("strike_price must be strictly positive")
        if self.time_to_expiry <= 0:
            raise ValueError("time_to_expiry must be strictly positive")

        contract_id = self.contract_id or (
            f"{self.symbol}_{self.strike_price:.4f}_"
            f"{self.time_to_expiry:.6f}_{self.option_type.value}"
        )
        object.__setattr__(self, "contract_id", contract_id)


@dataclass(slots=True)
class PricingResult:
    """Container for the outcome of a pricing model evaluation."""

    contract_id: str
    theoretical_price: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    implied_volatility: Optional[float] = None
    computation_time_ms: float = 0.0
    model_used: str = "unknown"
    error: Optional[str] = None
    standard_error: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    capsule_id: Optional[str] = None
    replay_capsule: Optional["ReplayCapsule"] = None
