"""Pydantic models representing the external API surface."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class APIModel(BaseModel):
    """Base schema that rejects misspelled or unsupported fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ContractPayload(APIModel):
    symbol: str = Field(min_length=1, max_length=64)
    strike_price: float = Field(gt=0, le=1e12)
    time_to_expiry: float = Field(alias="time_to_expiry", gt=1e-6, le=100.0)
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


class MarketPayload(APIModel):
    spot_price: float = Field(gt=0, le=1e12)
    risk_free_rate: float
    dividend_yield: float = 0.0

    @field_validator("risk_free_rate", "dividend_yield", mode="before")
    @classmethod
    def _clamp_rates(cls, value: float, info: ValidationInfo) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("rate must be a real number")
        rate = float(value)
        if not math.isfinite(rate):
            raise ValueError("rate must be finite")
        if info.field_name == "risk_free_rate" and not (-0.5 <= rate <= 1.0):
            raise ValueError("risk_free_rate out of bounds")
        if info.field_name == "dividend_yield" and not (-0.5 <= rate <= 1.0):
            raise ValueError("dividend_yield out of bounds")
        return rate


class ModelPrecision(APIModel):
    target_ci_bps: float | None = Field(default=None, gt=0, le=10_000)
    max_paths: int | None = Field(default=None, ge=1_000, le=1_000_000)


class ModelParams(APIModel):
    paths: int | None = Field(default=None, ge=1_000, le=1_000_000)
    steps: int | None = Field(default=None, ge=10, le=4_096)
    antithetic: bool | None = Field(
        default=None,
        description="Toggle antithetic variance reduction for Monte Carlo pricers.",
    )
    seed_prefix: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._:-]+$",
        description="Prefix incorporated into generated random seeds for repeatability.",
    )
    use_qmc: bool | None = Field(
        default=None,
        description=(
            "Enable randomized Sobol sampling with independent scrambles for valid uncertainty "
            "estimation; disables antithetic paths when true."
        ),
    )
    use_cv: bool | None = Field(
        default=None,
        description="Enable control variates to tighten Monte Carlo confidence intervals.",
    )


class ModelSelector(APIModel):
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


class GreeksRequest(APIModel):
    delta: bool = False
    gamma: bool = False
    vega: bool = False
    theta: bool = False
    rho: bool = False


class SurfaceHandle(APIModel):
    surface_id: str | None = Field(default=None, max_length=128)
    payload: dict[str, Any] | None = None

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("surface_id", mode="before")
    @classmethod
    def _coerce_surface_id(cls, value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError("surface_id must be a string")
        return value

    @field_validator("payload", mode="before")
    @classmethod
    def _ensure_payload(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError("surface payload must be a mapping")
        for key in ("surface_id", "id"):
            identifier = value.get(key)
            if identifier is not None and (
                not isinstance(identifier, str) or not 1 <= len(identifier) <= 128
            ):
                raise ValueError(f"surface payload {key} must be a bounded string")
        return dict(value)

    def resolved_id(self) -> str | None:
        if self.surface_id is not None:
            return self.surface_id
        if self.payload and "surface_id" in self.payload:
            return str(self.payload["surface_id"])
        return None


class QuoteRequest(APIModel):
    contract: ContractPayload
    market: MarketPayload
    volatility: float = Field(gt=1e-5, le=5.0)
    model: ModelSelector = Field(default_factory=lambda: ModelSelector(family="black_scholes"))
    greeks: GreeksRequest | None = None
    precision: ModelPrecision | None = None
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    )
    surface: SurfaceHandle | None = None

    @field_validator("surface", mode="before")
    @classmethod
    def _normalise_surface(cls, value: Any) -> SurfaceHandle | None:
        if value is None or isinstance(value, SurfaceHandle):
            return value
        if isinstance(value, str):
            return SurfaceHandle(surface_id=value)
        if isinstance(value, dict):
            surface_id = value.get("surface_id") or value.get("id")
            return SurfaceHandle(surface_id=surface_id, payload=value)
        raise TypeError("surface must be a string id or mapping")


class ConfidenceInterval(APIModel):
    half_width_abs: float = Field(description="95% confidence half-width in absolute price units.")
    half_width_bps: float = Field(
        description="95% confidence half-width expressed in basis points."
    )
    paths_used: int = Field(
        description="Number of Monte Carlo paths consumed to reach the estimate."
    )
    vr_pipeline: str = Field(description="Variance-reduction techniques applied during pricing.")


class QuoteResponse(APIModel):
    model_config = ConfigDict(protected_namespaces=())

    theoretical_price: float = Field(
        description="Model-implied fair value of the requested option."
    )
    greeks: dict[str, float] | None = Field(
        default=None,
        description="First-order risk sensitivities included on demand.",
    )
    ci: ConfidenceInterval | None = Field(
        default=None,
        description="Confidence interval metadata when Monte Carlo pricing provides uncertainty bounds.",
    )
    capsule_id: str = Field(
        description="Deterministic identifier representing the pricing capsule configuration."
    )
    model_used: dict[str, Any] = Field(
        description="Resolved model metadata, parameters, and diagnostics."
    )
    surface_id: str | None = Field(
        default=None,
        description="Identifier of the implied volatility surface leveraged during pricing, if any.",
    )
    seed_lineage: str | None = Field(
        default=None,
        description="Hash of seed components to reproduce Monte Carlo random streams.",
    )


class BatchRequest(APIModel):
    items: list[dict[str, Any]]
    greeks_default: GreeksRequest | None = None


class BatchResult(APIModel):
    model_config = ConfigDict(protected_namespaces=())

    index: int
    ok: bool
    value: QuoteResponse | None = None
    error: str | None = None


class BatchResponse(APIModel):
    results: list[BatchResult]
    capsule_ids: list[str]


class GreeksOnlyResponse(APIModel):
    model_config = ConfigDict(protected_namespaces=())

    greeks: dict[str, float]
    capsule_id: str
    model_used: dict[str, Any]


class VersionResponse(APIModel):
    build_id: str
    sbom_hash: str | None = None
    library_versions: dict[str, str]
    flags: dict[str, bool]
    baseline_tag: str | None = None


class ReplayRequest(APIModel):
    strict_build: bool = False
