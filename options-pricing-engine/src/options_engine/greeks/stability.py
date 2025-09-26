"""Stability guards and helper utilities for Monte Carlo Greeks."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

CI_Z_VALUE = 1.96
TAU_FLOOR = 1e-6
SIGMA_FLOOR = 1e-4
LOG_MONEYNESS_BOUND = 4.0
SPOT_FLOOR = 1e-12


def clamp_tau(value: float) -> float:
    """Apply the maturity epsilon policy."""

    return max(float(value), TAU_FLOOR)


def clamp_sigma(value: float) -> float:
    """Apply the volatility epsilon policy."""

    return max(float(value), SIGMA_FLOOR)


def clamp_log_moneyness(values: np.ndarray) -> np.ndarray:
    """Clamp log-moneyness style terms to avoid extreme tails."""

    return np.clip(values, -LOG_MONEYNESS_BOUND, LOG_MONEYNESS_BOUND)


def safe_spot(value: float) -> float:
    """Return a strictly positive spot used in denominators."""

    return max(float(value), SPOT_FLOOR)


def contributions_finite(contributions: np.ndarray) -> bool:
    """Return True when all per-path contributions are finite."""

    return bool(np.isfinite(contributions).all())


def standard_error(contributions: np.ndarray) -> float:
    """Compute the standard error of a set of contributions."""

    sample = np.asarray(contributions, dtype=float)
    if sample.size <= 1:
        return 0.0
    sample_std = float(np.std(sample, ddof=1))
    if not math.isfinite(sample_std):
        return math.inf
    return sample_std / math.sqrt(sample.size)


def half_width(se: float, z_value: float = CI_Z_VALUE) -> float:
    """Return the half-width of the two-sided confidence interval."""

    return float(abs(z_value) * se)


def is_estimate_unstable(value: float, half_width_abs: float, price: float, *, threshold: float = 0.2) -> bool:
    """Return True if an estimator should fall back to finite differences."""

    if not math.isfinite(value) or not math.isfinite(half_width_abs):
        return True
    if price < 0.5:
        return False
    magnitude = abs(value)
    if magnitude < 1e-12:
        return half_width_abs > 1e-6
    return half_width_abs > threshold * magnitude


def guard_against_pathologies(arrays: Iterable[np.ndarray]) -> bool:
    """Check that all arrays contain finite values and no NaNs/Infs."""

    return all(np.isfinite(np.asarray(array, dtype=float)).all() for array in arrays)
