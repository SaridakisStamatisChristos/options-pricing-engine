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
    antithetic: Optional[bool] = Field(
        default=None,
        description="Toggle antithetic variance reduction for Monte Carlo pricers.",
    )
    seed_prefix: Optional[str] = Field(
        default=None,
        description="Prefix incorporated into generated random seeds for repeatability.",
    )
    use_qmc: Optional[bool] = Field(
        default=None,
        description="Enable quasi-Monte Carlo (Sobol) sampling; disables antithetic paths when true.",
    )
    use_cv: Optional[bool] = Field(
        default=None,
        description="Enable control variates to tighten Monte Carlo confidence intervals.",
    )


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


class SurfaceHandle(BaseModel):
    surface_id: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("surface_id", mode="before")
    @classmethod
    def _coerce_surface_id(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value)

    @field_validator("payload", mode="before")
    @classmethod
    def _ensure_payload(cls, value: Any) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("surface payload must be a mapping")
        return dict(value)

    def resolved_id(self) -> Optional[str]:
        if self.surface_id is not None:
            return self.surface_id
        if self.payload and "surface_id" in self.payload:
            return str(self.payload["surface_id"])
        return None


class QuoteRequest(BaseModel):
    contract: ContractPayload
    market: MarketPayload
    volatility: float = Field(gt=1e-5, le=5.0)
    model: ModelSelector = Field(default_factory=lambda: ModelSelector(family="black_scholes"))
    greeks: Optional[GreeksRequest] = None
    precision: Optional[ModelPrecision] = None
    idempotency_key: Optional[str] = None
    surface: Optional[SurfaceHandle] = None

    @field_validator("surface", mode="before")
    @classmethod
    def _normalise_surface(cls, value: Any) -> Optional[SurfaceHandle]:
        if value is None or isinstance(value, SurfaceHandle):
            return value
        if isinstance(value, str):
            return SurfaceHandle(surface_id=value)
        if isinstance(value, dict):
            surface_id = value.get("surface_id") or value.get("id")
            return SurfaceHandle(surface_id=surface_id, payload=value)
        raise TypeError("surface must be a string id or mapping")


class ConfidenceInterval(BaseModel):
    half_width_abs: float = Field(description="95% confidence half-width in absolute price units.")
    half_width_bps: float = Field(description="95% confidence half-width expressed in basis points.")
    paths_used: int = Field(description="Number of Monte Carlo paths consumed to reach the estimate.")
    vr_pipeline: str = Field(description="Variance-reduction techniques applied during pricing.")


class QuoteResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    theoretical_price: float = Field(description="Model-implied fair value of the requested option.")
    greeks: Optional[Dict[str, float]] = Field(
        default=None,
        description="First-order risk sensitivities included on demand.",
    )
    ci: Optional[ConfidenceInterval] = Field(
        default=None,
        description="Confidence interval metadata when Monte Carlo pricing provides uncertainty bounds.",
    )
    capsule_id: str = Field(description="Deterministic identifier representing the pricing capsule configuration.")
    model_used: Dict[str, Any] = Field(description="Resolved model metadata, parameters, and diagnostics.")
    surface_id: Optional[str] = Field(
        default=None,
        description="Identifier of the implied volatility surface leveraged during pricing, if any.",
    )
    seed_lineage: Optional[str] = Field(
        default=None,
        description="Hash of seed components to reproduce Monte Carlo random streams.",
    )


class BatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    greeks_default: Optional[GreeksRequest] = None


class BatchResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    index: int
    ok: bool
    value: Optional[QuoteResponse] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    results: List[BatchResult]
    capsule_ids: List[str]


class GreeksOnlyResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

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

