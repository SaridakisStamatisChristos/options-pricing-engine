"""Shared validation and stochastic infrastructure for pricing models."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.random import Generator, SeedSequence

from .models import MarketData, OptionContract

SQRT_TWO = math.sqrt(2.0)
INV_SQRT_TWO_PI = 1.0 / math.sqrt(2.0 * math.pi)
TAU_MIN = 1e-8
SIGMA_MIN = 1e-6
MAX_BINOMIAL_STEPS = 4_096
MAX_MONTE_CARLO_PATHS = 1_000_000
MAX_LSMC_PATHS = 500_000
MAX_LSMC_STEPS = 512
MAX_LSMC_WORK_ITEMS = 5_000_000
MAX_CV_FOLDS = 64
MAX_RANDOM_SEED = 2**128 - 1
LOG_MONEYNESS_CLAMP = (
    (1e-6, 8.0),
    (1e-4, 10.0),
    (1e-3, 12.0),
    (1e-2, 14.0),
    (1e-1, 16.0),
    (1.0, 18.0),
    (float("inf"), 20.0),
)


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    """Return an integer scalar after strict type and resource-bound validation."""

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalised = int(value)
    if not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return normalised


def _require_boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _validate_seed_sequence(name: str, value: object) -> SeedSequence | None:
    if value is not None and not isinstance(value, SeedSequence):
        raise TypeError(f"{name} must be a numpy.random.SeedSequence or None")
    return value


def _validate_lsmc_workload(*, steps: int, paths: int, antithetic: bool) -> int:
    # Preserve at least two independent sampling units. One antithetic pair is
    # two correlated paths but still supplies zero degrees of freedom for an
    # estimated-variance confidence interval.
    effective_paths = max(4, paths + (paths % 2)) if antithetic else max(2, paths)
    work_items = steps * effective_paths
    if work_items > MAX_LSMC_WORK_ITEMS:
        raise ValueError(
            f"LSMC workload exceeds the supported steps*paths limit of {MAX_LSMC_WORK_ITEMS}"
        )
    return effective_paths


@dataclass(slots=True)
class PriceResult:
    """Simplified pricing result returned by helper pricing functions."""

    price: float
    ci_half_width: float
    meta: dict[str, object]
    standard_error: float | None = None
    confidence_interval: tuple[float, float] | None = None
    estimate_diagnostics: dict[str, object] | None = None


def _normalise_option_type(option_type: str) -> str:
    if not isinstance(option_type, str):
        raise TypeError("option_type must be a string")
    value = option_type.strip().lower()
    if value not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return value


def _build_capsule_id(config: dict[str, object]) -> str:
    digest = hashlib.blake2b(repr(sorted(config.items())).encode("utf8"), digest_size=8).hexdigest()
    return f"american_lsmc::{digest}"


def _antithetic_units(values: np.ndarray, *, antithetic: bool) -> np.ndarray:
    """Collapse antithetic path pairs into statistically independent units."""

    samples = np.asarray(values, dtype=float)
    if antithetic and samples.size >= 2 and samples.size % 2 == 0:
        half = samples.size // 2
        return np.asarray(0.5 * (samples[:half] + samples[half:]), dtype=float)
    return samples.copy()


def _cross_fitted_control_variate(
    samples: np.ndarray,
    control: np.ndarray,
    *,
    control_mean: float,
    folds: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a known-mean control variate without in-sample coefficient bias.

    Each fold is adjusted with a coefficient learned only from the other
    folds. ``samples`` and ``control`` must already represent independent
    simulation units (for example, antithetic-pair averages).
    """

    observations = np.asarray(samples, dtype=float)
    controls = np.asarray(control, dtype=float)
    if observations.shape != controls.shape:
        raise ValueError("control variate samples must have matching shapes")

    count = observations.size
    report: dict[str, object] = {
        "used": False,
        "folds": 0,
        "beta": 0.0,
        "correlation": 0.0,
        "variance_ratio": 1.0,
    }
    if count < 8:
        return observations.copy(), report

    sample_variance = float(np.var(observations, ddof=1))
    control_variance = float(np.var(controls, ddof=1))
    if sample_variance <= 1e-18 or control_variance <= 1e-18:
        return observations.copy(), report

    covariance = float(np.cov(observations, controls, ddof=1)[0, 1])
    correlation = covariance / math.sqrt(sample_variance * control_variance)
    report["correlation"] = float(correlation)
    if not math.isfinite(correlation) or abs(correlation) < 0.05:
        return observations.copy(), report

    fold_count = min(max(2, int(folds)), max(2, count // 4))
    labels = np.arange(count) % fold_count
    adjusted = observations.copy()
    betas: list[float] = []
    for held_out in range(fold_count):
        test = labels == held_out
        train = ~test
        train_control_variance = float(np.var(controls[train], ddof=1))
        if train_control_variance <= 1e-18:
            return observations.copy(), report
        beta = float(
            np.cov(observations[train], controls[train], ddof=1)[0, 1] / train_control_variance
        )
        if not math.isfinite(beta) or abs(beta) > 50.0:
            return observations.copy(), report
        adjusted[test] -= beta * (controls[test] - control_mean)
        betas.append(beta)

    adjusted_variance = float(np.var(adjusted, ddof=1))
    variance_ratio = adjusted_variance / sample_variance
    report.update(
        {
            "folds": fold_count,
            "beta": float(np.mean(betas)),
            "variance_ratio": variance_ratio,
        }
    )
    if not math.isfinite(variance_ratio) or variance_ratio >= 1.0:
        return observations.copy(), report

    report["used"] = True
    return adjusted, report


# --- kept from the feature branch; required by replay capsules ---
def _contract_payload(contract: OptionContract) -> dict[str, object]:
    return {
        "contract_id": contract.contract_id,
        "symbol": contract.symbol,
        "strike_price": contract.strike_price,
        "time_to_expiry": contract.time_to_expiry,
        "option_type": contract.option_type.value,
        "exercise_style": contract.exercise_style.value,
    }


def _market_payload(market_data: MarketData) -> dict[str, object]:
    return {
        "spot_price": market_data.spot_price,
        "risk_free_rate": market_data.risk_free_rate,
        "dividend_yield": market_data.dividend_yield,
    }


# -----------------------------------------------------------------

_THREAD_LOCAL_RNG = threading.local()


def _thread_local_generator(seed_sequence: SeedSequence | None = None) -> Generator:
    """Return a numpy Generator with deterministic seeding when requested."""
    if seed_sequence is not None:
        return np.random.default_rng(seed_sequence)
    generator: Generator | None = getattr(_THREAD_LOCAL_RNG, "generator", None)
    if generator is None:
        generator = np.random.default_rng()
        _THREAD_LOCAL_RNG.generator = generator
    return generator


def _apply_pathwise_control_variates(
    discounted_payoffs: np.ndarray,
    terminal_prices: np.ndarray,
    *,
    contract: OptionContract,
    market_data: MarketData,
    volatility: float,
    antithetic: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply an honest, cross-fitted discounted-underlying control variate.

    The vanilla payoff must not be decomposed into analytically priced indicator
    controls: doing so reconstructs Black-Scholes and makes the Monte Carlo
    confidence interval circular.  Here the sole control is discounted ``S_T``,
    whose expectation follows directly from the risk-neutral martingale.  Betas
    are estimated on the opposite fold to avoid in-sample variance overstatement.
    """

    del volatility  # The martingale control does not require an option formula.
    y = _antithetic_units(discounted_payoffs, antithetic=antithetic)
    terminal = _antithetic_units(terminal_prices, antithetic=antithetic)
    sample_size = y.size
    raw_var = float(np.var(y, ddof=1)) if sample_size > 1 else 0.0
    empty_report: dict[str, object] = {
        "cv_used": False,
        "rho": None,
        "beta": None,
        "raw_var": raw_var,
        "residual_var": raw_var,
        "independent_units": int(sample_size),
    }
    if sample_size < 8 or raw_var < 1e-18:
        return y, empty_report

    discount = math.exp(-market_data.risk_free_rate * contract.time_to_expiry)
    expected_control = market_data.spot_price * math.exp(
        -market_data.dividend_yield * contract.time_to_expiry
    )
    control = discount * terminal
    control_var = float(np.var(control, ddof=1))
    if control_var < 1e-18:
        return y, empty_report

    covariance = float(np.cov(y, control, ddof=1)[0, 1])
    rho = covariance / math.sqrt(max(raw_var * control_var, 1e-24))
    empty_report["rho"] = rho
    if not math.isfinite(rho) or abs(rho) < 0.1:
        return y, empty_report

    folds = np.arange(sample_size) % 2
    adjusted = y.copy()
    betas: list[float] = []
    for held_out in (0, 1):
        train = folds != held_out
        test = ~train
        train_control = control[train]
        train_payoff = y[train]
        variance = float(np.var(train_control, ddof=1))
        if variance < 1e-18:
            return y, empty_report
        beta = float(np.cov(train_payoff, train_control, ddof=1)[0, 1] / variance)
        if not math.isfinite(beta) or abs(beta) > 50.0:
            return y, empty_report
        adjusted[test] = y[test] - beta * (control[test] - expected_control)
        betas.append(beta)

    residual_var = float(np.var(adjusted, ddof=1))
    if not math.isfinite(residual_var) or residual_var >= raw_var:
        return y, empty_report

    return adjusted, {
        "cv_used": True,
        "rho": rho,
        "beta": tuple(betas),
        "raw_var": raw_var,
        "residual_var": residual_var,
        "independent_units": int(sample_size),
    }


__all__ = [
    "INV_SQRT_TWO_PI",
    "LOG_MONEYNESS_CLAMP",
    "MAX_BINOMIAL_STEPS",
    "MAX_CV_FOLDS",
    "MAX_LSMC_PATHS",
    "MAX_LSMC_STEPS",
    "MAX_LSMC_WORK_ITEMS",
    "MAX_MONTE_CARLO_PATHS",
    "MAX_RANDOM_SEED",
    "SIGMA_MIN",
    "SQRT_TWO",
    "TAU_MIN",
    "PriceResult",
]
