from dataclasses import dataclass
from enum import Enum
from datetime import datetime
class OptionType(str, Enum):
    CALL="call"; PUT="put"
class ExerciseStyle(str, Enum):
    EUROPEAN="european"; AMERICAN="american"
@dataclass(frozen=True)
class MarketData:
    spot_price: float; risk_free_rate: float; dividend_yield: float = 0.0; timestamp: float|None=None
    def __post_init__(self):
        if self.spot_price<=0: raise ValueError("S>0")
        if not (0<=self.risk_free_rate<=1): raise ValueError("r in [0,1]")
        if not (0<=self.dividend_yield<=1): raise ValueError("q in [0,1]")
        if self.timestamp is None:
            object.__setattr__(self, "timestamp", datetime.utcnow().timestamp())
@dataclass(frozen=True)
class OptionContract:
    symbol: str; strike_price: float; time_to_expiry: float; option_type: OptionType; exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN; contract_id: str|None=None
    def __post_init__(self):
        if self.strike_price<=0: raise ValueError("K>0")
        if self.time_to_expiry<=0: raise ValueError("T>0")
        cid=self.contract_id or f"{self.symbol}_{self.strike_price:.4f}_{self.time_to_expiry:.6f}_{self.option_type.value}"
        object.__setattr__(self, "contract_id", cid)
@dataclass
class PricingResult:
    contract_id: str; theoretical_price: float
    delta: float|None=None; gamma: float|None=None; theta: float|None=None; vega: float|None=None; rho: float|None=None
    implied_volatility: float|None=None; computation_time_ms: float=0.0; model_used: str="unknown"; error: str|None=None
