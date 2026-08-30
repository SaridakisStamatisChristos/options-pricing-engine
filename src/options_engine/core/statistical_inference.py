"""Statistical inference helpers for stochastic pricing estimators.

The pricing engines report uncertainty over *independent sampling units*.
Antithetic paths therefore enter this module after they have been collapsed
into pair averages, while randomized QMC enters as one observation per
independent scramble.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.stats import t as student_t

MAX_INFERENCE_UNITS = 1_000_000


def _optional_bound(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")
    normalised = float(value)
    if not math.isfinite(normalised):
        raise ValueError(f"{name} must be finite")
    return normalised


def _project(value: float, lower_bound: float | None, upper_bound: float | None) -> float:
    projected = value
    if lower_bound is not None:
        projected = max(projected, lower_bound)
    if upper_bound is not None:
        projected = min(projected, upper_bound)
    return projected


@dataclass(frozen=True, slots=True)
class MeanEstimate:
    """Point estimate and uncertainty for a bounded sample mean.

    ``raw_estimate`` and ``raw_confidence_interval`` describe the unconstrained
    sample mean. ``bounded_estimate`` and ``confidence_interval`` apply the
    same monotone projection to the no-arbitrage parameter space, so the
    published point estimate and interval always describe the same statistic.
    """

    raw_estimate: float
    bounded_estimate: float
    standard_error: float | None
    raw_confidence_interval: tuple[float, float] | None
    confidence_interval: tuple[float, float] | None
    confidence_level: float
    critical_value: float | None
    degrees_of_freedom: int | None
    independent_units: int
    lower_bound: float | None
    upper_bound: float | None
    projection_applied: bool
    interval_projection_applied: bool
    method: str | None

    @property
    def raw_half_width(self) -> float | None:
        """Return the half-width before no-arbitrage projection."""

        if self.raw_confidence_interval is None:
            return None
        return 0.5 * (self.raw_confidence_interval[1] - self.raw_confidence_interval[0])

    def diagnostics(
        self,
        *,
        estimator: str,
        raw_estimate_name: str = "raw_estimate",
    ) -> dict[str, object]:
        """Return JSON-safe audit metadata for a pricing result."""

        return {
            "estimator": estimator,
            raw_estimate_name: self.raw_estimate,
            "bounded_estimate": self.bounded_estimate,
            "projection_applied": self.projection_applied,
            "raw_confidence_interval": self.raw_confidence_interval,
            "bounded_confidence_interval": self.confidence_interval,
            "interval_projection_applied": self.interval_projection_applied,
            "confidence_level": self.confidence_level,
            "interval_method": self.method,
            "critical_value": self.critical_value,
            "degrees_of_freedom": self.degrees_of_freedom,
            "independent_units": self.independent_units,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }


def estimate_mean(
    observations: np.ndarray,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    confidence_level: float = 0.95,
) -> MeanEstimate:
    """Estimate a bounded mean using Student-t sampling uncertainty.

    Student-t is used whenever the variance is estimated from the sample. It
    converges to the normal critical value for large samples while retaining
    the correct degrees-of-freedom penalty for small samples. With only one
    independent unit no sampling error can be estimated, so uncertainty fields
    are explicitly unavailable instead of being reported as zero.
    """

    sample = np.asarray(observations, dtype=float)
    if sample.ndim != 1:
        raise ValueError("observations must be one-dimensional")
    if not 1 <= sample.size <= MAX_INFERENCE_UNITS:
        raise ValueError(f"observations must contain between 1 and {MAX_INFERENCE_UNITS} values")
    if not np.isfinite(sample).all():
        raise ValueError("observations must be finite")

    lower = _optional_bound("lower_bound", lower_bound)
    upper = _optional_bound("upper_bound", upper_bound)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("lower_bound must not exceed upper_bound")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise TypeError("confidence_level must be a real number")
    level = float(confidence_level)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("confidence_level must be within (0, 1)")

    independent_units = int(sample.size)
    raw_estimate = float(np.mean(sample))
    bounded_estimate = _project(raw_estimate, lower, upper)
    projection_applied = bounded_estimate != raw_estimate

    standard_error: float | None = None
    raw_interval: tuple[float, float] | None = None
    bounded_interval: tuple[float, float] | None = None
    critical_value: float | None = None
    degrees_of_freedom: int | None = None
    interval_projection_applied = False
    method: str | None = None

    if independent_units > 1:
        degrees_of_freedom = independent_units - 1
        sample_std = float(np.std(sample, ddof=1))
        standard_error = sample_std / math.sqrt(independent_units)
        tail_probability = 0.5 * (1.0 + level)
        critical_value = float(student_t.ppf(tail_probability, df=degrees_of_freedom))
        if not math.isfinite(critical_value):
            raise ValueError("Student-t critical value is non-finite")
        half_width = critical_value * standard_error
        raw_interval = (raw_estimate - half_width, raw_estimate + half_width)
        bounded_interval = (
            _project(raw_interval[0], lower, upper),
            _project(raw_interval[1], lower, upper),
        )
        interval_projection_applied = bounded_interval != raw_interval
        method = "student_t_projected" if interval_projection_applied else "student_t"

    return MeanEstimate(
        raw_estimate=raw_estimate,
        bounded_estimate=bounded_estimate,
        standard_error=standard_error,
        raw_confidence_interval=raw_interval,
        confidence_interval=bounded_interval,
        confidence_level=level,
        critical_value=critical_value,
        degrees_of_freedom=degrees_of_freedom,
        independent_units=independent_units,
        lower_bound=lower,
        upper_bound=upper,
        projection_applied=projection_applied,
        interval_projection_applied=interval_projection_applied,
        method=method,
    )


__all__ = ["MAX_INFERENCE_UNITS", "MeanEstimate", "estimate_mean"]
