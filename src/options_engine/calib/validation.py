"""Model-neutral, serializable calibration validation primitives.

The objects in this module deliberately contain only Python scalars and tuples:
optimizer library result objects and mutable numpy arrays never cross the public
diagnostics boundary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class HoldoutPolicy(StrEnum):
    NONE = "none"
    ALTERNATING = "alternating"
    WINGS = "wings"
    CENTRE = "centre"
    FRACTIONAL = "fractional"


class FitQuality(StrEnum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNSTABLE = "unstable"
    INVALID = "invalid"


class CalibrationFailureReason(StrEnum):
    INSUFFICIENT_OBSERVATIONS = "insufficient_observations"
    OPTIMIZER_FAILED = "optimizer_failed"
    INVALID_MODEL_EVALUATION = "invalid_model_evaluation"
    ADMISSIBILITY_FAILED = "admissibility_failed"
    ARBITRAGE_VIOLATION = "arbitrage_violation"


class CalibrationError(RuntimeError):
    """Calibration failure carrying a stable machine-readable reason code."""

    def __init__(self, reason: CalibrationFailureReason, message: str) -> None:
        self.reason = reason
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ResidualObservation:
    tenor: float
    strike: float
    log_moneyness: float
    market_volatility: float
    fitted_volatility: float
    residual: float
    weighted_residual: float
    is_holdout: bool


@dataclass(frozen=True, slots=True)
class ResidualSummary:
    observations: int
    calibration_observations: int
    holdout_observations: int
    rmse: float
    weighted_rmse: float
    holdout_rmse: float | None
    mae: float
    maximum_absolute_residual: float
    mean_residual: float
    residual_standard_deviation: float
    percentiles: dict[str, float]
    holdout_mae: float | None = None
    holdout_maximum_absolute_residual: float | None = None
    holdout_mean_residual: float | None = None


@dataclass(frozen=True, slots=True)
class OptimizerAttempt:
    seed: int
    initial_parameters: tuple[float, ...]
    success: bool
    status: int
    message: str
    objective: float | None
    parameters: tuple[float, ...] | None
    evaluations: int


@dataclass(frozen=True, slots=True)
class InitializationSensitivity:
    attempts: tuple[OptimizerAttempt, ...]
    objective_spread: float | None
    parameter_spread: tuple[float, ...]
    failed_start_fraction: float
    best_seed_dominance: float | None
    materially_different_similar_solutions: bool
    classification: str
    thresholds: dict[str, float] | None = None


@dataclass(frozen=True, slots=True)
class ConditioningDiagnostics:
    singular_values: tuple[float, ...]
    effective_rank: int
    condition_number: float | None
    weakly_identified: bool
    parameter_correlation: tuple[tuple[float, ...], ...] | None
    note: str = "Local numerical conditioning; not a statistical confidence interval."


@dataclass(frozen=True, slots=True)
class WeightDiagnostics:
    minimum_normalized_weight: float
    maximum_normalized_weight: float
    effective_sample_size: float
    top_three_weight_fraction: float


def deterministic_holdout_mask(
    values: ArrayLike,
    policy: HoldoutPolicy | str = HoldoutPolicy.NONE,
    *,
    fraction: float = 0.2,
    minimum_training: int = 1,
    centre: float | None = None,
) -> np.ndarray:
    """Select holdouts by sorted value, independent of input order or RNG state."""

    try:
        selected = HoldoutPolicy(policy)
    except ValueError as exc:
        raise ValueError(f"unsupported holdout policy: {policy!r}") from exc
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError("holdout values must be a finite one-dimensional array")
    if not 0.0 <= fraction <= 0.5 or minimum_training < 1:
        raise ValueError("fraction must be in [0, 0.5] and minimum_training must be positive")
    size = data.size
    mask = np.zeros(size, dtype=bool)
    if selected is HoldoutPolicy.NONE or size == 0:
        return mask
    available = size - minimum_training
    if available <= 0:
        raise ValueError("holdout would leave fewer than the minimum training observations")
    order = np.argsort(data, kind="stable")
    count = min(available, max(1, math.floor(size * fraction)))
    if selected is HoldoutPolicy.ALTERNATING:
        candidates = order[1::2]
        count = min(available, candidates.size)
    elif selected is HoldoutPolicy.WINGS:
        # Alternate low/high extremes; unlike the old implementation this is
        # balanced and cannot accidentally walk through one wing first.
        candidates = np.array(
            [order[i // 2] if i % 2 == 0 else order[-(i // 2) - 1] for i in range(size)]
        )
    elif selected is HoldoutPolicy.CENTRE:
        centre_value = float(np.median(data)) if centre is None else float(centre)
        if not math.isfinite(centre_value):
            raise ValueError("holdout centre must be finite")
        candidates = np.argsort(np.abs(data - centre_value), kind="stable")
    else:
        # Evenly spaced deterministic indices provide representative coverage.
        candidates = order[np.unique(np.linspace(0, size - 1, count, dtype=int))]
    for index in candidates:
        if int(np.sum(mask)) >= count:
            break
        mask[int(index)] = True
    if size - int(np.sum(mask)) < minimum_training:
        raise ValueError("holdout would underidentify the calibration")
    return mask


def residual_diagnostics(
    *,
    tenors: ArrayLike,
    strikes: ArrayLike,
    forwards: ArrayLike,
    market: ArrayLike,
    fitted: ArrayLike,
    weights: ArrayLike | None = None,
    holdout: ArrayLike | None = None,
) -> tuple[tuple[ResidualObservation, ...], ResidualSummary, WeightDiagnostics]:
    arrays = [np.asarray(item, dtype=float) for item in (tenors, strikes, forwards, market, fitted)]
    if any(item.ndim != 1 for item in arrays) or len({item.size for item in arrays}) != 1:
        raise ValueError("residual inputs must be matching one-dimensional arrays")
    if not all(np.isfinite(item).all() for item in arrays) or np.any(arrays[2] <= 0.0):
        raise ValueError("residual inputs must be finite and forwards positive")
    size = arrays[0].size
    weight = np.ones(size) if weights is None else np.asarray(weights, dtype=float)
    held = np.zeros(size, dtype=bool) if holdout is None else np.asarray(holdout, dtype=bool)
    if (
        weight.shape != (size,)
        or held.shape != (size,)
        or np.any(weight <= 0)
        or not np.isfinite(weight).all()
    ):
        raise ValueError("weights must be finite and positive and holdout must match observations")
    normalized = weight / np.sum(weight)
    residual = arrays[4] - arrays[3]
    training = ~held
    if not np.any(training):
        raise ValueError("at least one calibration observation is required")
    train_weights = weight[training]
    train_residual = residual[training]
    percentiles = {f"p{q:02d}": float(np.percentile(residual, q)) for q in (5, 25, 50, 75, 95)}
    rows = tuple(
        ResidualObservation(
            float(t),
            float(k),
            float(math.log(k / f)),
            float(m),
            float(v),
            float(r),
            float(r * math.sqrt(w)),
            bool(h),
        )
        for t, k, f, m, v, r, w, h in zip(*arrays, residual, weight, held, strict=True)
    )
    summary = ResidualSummary(
        size,
        int(np.sum(training)),
        int(np.sum(held)),
        float(np.sqrt(np.mean(train_residual**2))),
        float(np.sqrt(np.average(train_residual**2, weights=train_weights))),
        float(np.sqrt(np.mean(residual[held] ** 2))) if np.any(held) else None,
        float(np.mean(np.abs(residual))),
        float(np.max(np.abs(residual))),
        float(np.mean(residual)),
        float(np.std(residual)),
        percentiles,
        float(np.mean(np.abs(residual[held]))) if np.any(held) else None,
        float(np.max(np.abs(residual[held]))) if np.any(held) else None,
        float(np.mean(residual[held])) if np.any(held) else None,
    )
    concentration = WeightDiagnostics(
        float(np.min(normalized)),
        float(np.max(normalized)),
        float(1.0 / np.sum(normalized**2)),
        float(np.sum(np.sort(normalized)[-min(3, size) :])),
    )
    return rows, summary, concentration


def analyze_initialization_sensitivity(
    attempts: tuple[OptimizerAttempt, ...] | list[OptimizerAttempt],
    *,
    objective_relative_tolerance: float = 0.01,
    parameter_relative_tolerance: float = 0.10,
    failed_fraction_tolerance: float = 0.25,
) -> InitializationSensitivity:
    """Compare successful starts using scale-aware, documented thresholds.

    Parameter ambiguity is measured relative to the magnitude of the best
    solution (with a unit floor), rather than applying one dimensional threshold
    to unrelated economic parameters.
    """

    records = tuple(attempts)
    if not records:
        raise ValueError("at least one optimizer attempt is required")
    successful = tuple(
        item
        for item in records
        if item.success and item.objective is not None and item.parameters is not None
    )
    failed_fraction = 1.0 - len(successful) / len(records)
    thresholds = {
        "objective_relative_tolerance": objective_relative_tolerance,
        "parameter_relative_tolerance": parameter_relative_tolerance,
        "failed_fraction_tolerance": failed_fraction_tolerance,
    }
    if not successful:
        return InitializationSensitivity(
            records, None, (), 1.0, None, False, "sensitive", thresholds
        )
    objectives = np.asarray([item.objective for item in successful], dtype=float)
    parameters = np.asarray([item.parameters for item in successful], dtype=float)
    best_index = int(np.argmin(objectives))
    best = float(objectives[best_index])
    objective_spread = float(np.ptp(objectives)) if objectives.size > 1 else 0.0
    parameter_spread = tuple(float(value) for value in np.ptp(parameters, axis=0))
    objective_scale = max(abs(best), 1e-12)
    similar = objectives <= best + max(1e-12, objective_relative_tolerance * objective_scale)
    economic_scale = np.maximum(np.abs(parameters[best_index]), 1.0)
    relative_spread = (
        np.ptp(parameters[similar], axis=0) / economic_scale
        if int(np.sum(similar)) > 1
        else np.zeros(parameters.shape[1])
    )
    ambiguous = bool(np.any(relative_spread > parameter_relative_tolerance))
    sorted_objectives = np.sort(objectives)
    dominance = (
        float((sorted_objectives[1] - best) / objective_scale)
        if sorted_objectives.size > 1
        else None
    )
    classification = (
        "multimodal/ambiguous"
        if ambiguous
        else "sensitive"
        if failed_fraction > failed_fraction_tolerance
        else "stable"
    )
    return InitializationSensitivity(
        records,
        objective_spread,
        parameter_spread,
        failed_fraction,
        dominance,
        ambiguous,
        classification,
        thresholds,
    )


def conditioning_from_jacobian(
    jacobian: ArrayLike, *, relative_tolerance: float = 1e-8
) -> ConditioningDiagnostics:
    jac = np.asarray(jacobian, dtype=float)
    if jac.ndim != 2 or not np.isfinite(jac).all():
        raise ValueError("jacobian must be a finite matrix")
    singular = np.linalg.svd(jac, compute_uv=False)
    rank = int(np.sum(singular > (singular[0] * relative_tolerance if singular.size else 0.0)))
    condition = (
        None if not singular.size or singular[-1] <= 0 else float(singular[0] / singular[-1])
    )
    correlation = None
    if rank == jac.shape[1] and jac.shape[0] > jac.shape[1]:
        covariance = np.linalg.pinv(jac.T @ jac)
        scale = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(scale, scale)
        correlation = tuple(
            tuple(float(x) for x in row)
            for row in np.divide(
                covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0
            )
        )
    return ConditioningDiagnostics(
        tuple(float(x) for x in singular),
        rank,
        condition,
        rank < jac.shape[1] or condition is None or condition > 1e8,
        correlation,
    )


def serializable(value: Any) -> Any:
    """Recursively convert validation dataclasses/enums to deterministic JSON data."""
    if hasattr(value, "__dataclass_fields__"):
        return serializable(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): serializable(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [serializable(item) for item in value.tolist()]
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "CalibrationError",
    "CalibrationFailureReason",
    "ConditioningDiagnostics",
    "FitQuality",
    "HoldoutPolicy",
    "InitializationSensitivity",
    "OptimizerAttempt",
    "ResidualObservation",
    "ResidualSummary",
    "WeightDiagnostics",
    "analyze_initialization_sensitivity",
    "conditioning_from_jacobian",
    "deterministic_holdout_mask",
    "residual_diagnostics",
    "serializable",
]
