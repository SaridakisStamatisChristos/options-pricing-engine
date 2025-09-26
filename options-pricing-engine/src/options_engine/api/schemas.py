"""Pydantic models representing the external API surface."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContractPayload(BaseModel):
    symbol: str
    strike_price: float = Field(gt=0)
    time_to_expiry: float = Field(alias="time_to_expiry", ge=1e-6)
    option_type: str
    exercise_style: str = "EUROPEAN"

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("option_type", mode="before")
    @classmethod
    def _normalise_option_type(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("option_type must be a string")
        normalised = value.strip().upper()
        if normalised not in {"CALL", "PUT"}:
            raise ValueError("unsupported option_type")
        return normalised

    @field_validator("exercise_style", mode="before")
    @classmethod
    def _normalise_exercise_style(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("exercise_style must be a string")
        normalised = value.strip().upper()
        if normalised not in {"EUROPEAN", "AMERICAN"}:
            raise ValueError("unsupported exercise_style")
        return normalised


class MarketPayload(BaseModel):
    spot_price: float = Field(gt=0)
    risk_free_rate: float
    dividend_yield: float = 0.0

    @field_validator("risk_free_rate", "dividend_yield", mode="before")
    @classmethod
    def _clamp_rates(cls, value: float, info) -> float:
        if value is None:
            raise TypeError("rate must be provided")
        rate = float(value)
        if info.field_name == "risk_free_rate" and not (-0.5 <= rate <= 1.0):
            raise ValueError("risk_free_rate out of bounds")
        if info.field_name == "dividend_yield" and not (-0.5 <= rate <= 1.0):
            raise ValueError("dividend_yield out of bounds")
        return rate


class ModelPrecision(BaseModel):
    target_ci_bps: Optional[float] = Field(default=None, gt=0)
    max_paths: Optional[int] = Field(default=None, ge=1_000, le=1_000_000)


class ModelParams(BaseModel):
    paths: Optional[int] = Field(default=None, ge=1_000, le=1_000_000)
    steps: Optional[int] = Field(default=None, ge=10, le=10_000)
    antithetic: Optional[bool] = None
    seed_prefix: Optional[str] = None
    use_qmc: Optional[bool] = None
    use_cv: Optional[bool] = None


class ModelSelector(BaseModel):
    family: str
    params: ModelParams | None = None

    @field_validator("family", mode="before")
    @classmethod
    def _normalise_family(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("model family must be a string")
        normalised = value.strip().lower()
        if normalised not in {"black_scholes", "binomial", "monte_carlo"}:
            raise ValueError("unsupported model family")
        return normalised


class GreeksRequest(BaseModel):
    delta: bool = False
    gamma: bool = False
    vega: bool = False
    theta: bool = False
    rho: bool = False


class QuoteRequest(BaseModel):
    contract: ContractPayload
    market: MarketPayload
    volatility: float = Field(gt=1e-5, le=5.0)
    model: ModelSelector = Field(default_factory=lambda: ModelSelector(family="black_scholes"))
    greeks: Optional[GreeksRequest] = None
    precision: Optional[ModelPrecision] = None
    idempotency_key: Optional[str] = None


class ConfidenceInterval(BaseModel):
    half_width_abs: float
    half_width_bps: float
    paths_used: int
    vr_pipeline: str


class QuoteResponse(BaseModel):
    theoretical_price: float
    greeks: Optional[Dict[str, float]] = None
    ci: Optional[ConfidenceInterval] = None
    capsule_id: str
    model_used: Dict[str, Any]
    surface_id: Optional[str] = None
    seed_lineage: Optional[str] = None


class BatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    greeks_default: Optional[GreeksRequest] = None


class BatchResult(BaseModel):
    index: int
    ok: bool
    value: Optional[QuoteResponse] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    results: List[BatchResult]
    capsule_ids: List[str]


class GreeksOnlyResponse(BaseModel):
    greeks: Dict[str, float]
    capsule_id: str
    model_used: Dict[str, Any]


class VersionResponse(BaseModel):
    build_id: str
    sbom_hash: Optional[str] = None
    library_versions: Dict[str, str]
    flags: Dict[str, bool]
    baseline_tag: Optional[str] = None


class ReplayRequest(BaseModel):
    strict_build: bool = False

