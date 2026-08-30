"""Numerical utilities for Monte Carlo style pricing algorithms."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real

import numpy as np

__all__ = [
    "apply_global_clamps",
    "deep_itm_policy",
    "deep_otm_upper_bound",
    "enforce_precision_policy",
    "laguerre_basis3",
    "numerics_policy_hash",
    "stable_regression",
]


def apply_global_clamps(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    r: float,
    q: float,
) -> tuple[tuple[float, float, float, float, float, float], list[str]]:
    """Validate scalar pricing inputs without silently changing the contract.

    The historical function name is retained for API compatibility.  Silent
    clamping is unsafe in a pricing system because the returned value no longer
    corresponds to the request that was audited or replayed.
    """

    raw_values = (spot, strike, tau, sigma, r, q)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_values):
        raise TypeError("pricing inputs must be real numbers")
    values = (
        float(spot),
        float(strike),
        float(tau),
        float(sigma),
        float(r),
        float(q),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("pricing inputs must be finite")
    if not 0.0 < spot <= 1e12:
        raise ValueError("spot must be within (0, 1e12]")
    if not 0.0 < strike <= 1e12:
        raise ValueError("strike must be within (0, 1e12]")
    if not 0.0 < tau <= 100.0:
        raise ValueError("tau must be within (0, 100]")
    if not 0.0 < sigma <= 5.0:
        raise ValueError("sigma must be within (0, 5]")
    if not -1.0 <= r <= 1.0:
        raise ValueError("r must be within [-1, 1]")
    if not -1.0 <= q <= 1.0:
        raise ValueError("q must be within [-1, 1]")
    return values, []


def laguerre_basis3(x: np.ndarray) -> np.ndarray:
    """Return the first four Laguerre basis functions evaluated at ``x``."""

    x = np.asarray(x, dtype=float)
    if x.size > 5_000_000:
        raise ValueError("Laguerre input exceeds the 5000000-value limit")
    if not np.isfinite(x).all():
        raise ValueError("Laguerre input must be finite")
    if x.ndim != 1:
        x = np.ravel(x)

    x_clipped = np.clip(x, 1e-8, 1e8)
    x_scaled = np.minimum(x_clipped, 10.0)

    ones = np.ones_like(x_scaled)
    l1 = 1.0 - x_scaled
    l2 = 1.0 - 2.0 * x_scaled + 0.5 * (x_scaled**2)
    l3 = 1.0 - 3.0 * x_scaled + 1.5 * (x_scaled**2) - (x_scaled**3) / 6.0

    return np.column_stack((ones, l1, l2, l3))


def stable_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge_eps: float = 1e-12,
) -> tuple[np.ndarray, bool]:
    """Solve ``X beta = y`` using a numerically stable approach."""

    design = np.asarray(X, dtype=float)
    targets = np.asarray(y, dtype=float)
    if design.ndim != 2 or targets.ndim != 1:
        raise ValueError("X must be two-dimensional and y must be one-dimensional")
    if design.shape[0] != targets.size:
        raise ValueError("X and y must contain the same number of observations")
    if design.size > 5_000_000 or design.shape[1] > 256:
        raise ValueError("regression design exceeds the supported size")
    if not np.isfinite(design).all() or not np.isfinite(targets).all():
        raise ValueError("regression inputs must be finite")
    if isinstance(ridge_eps, bool) or not isinstance(ridge_eps, Real):
        raise TypeError("ridge_eps must be a real number")
    ridge_eps = float(ridge_eps)
    if not math.isfinite(ridge_eps) or not 0.0 < ridge_eps <= 1.0:
        raise ValueError("ridge_eps must be finite and within (0, 1]")
    if design.size == 0:
        return np.zeros(0, dtype=float), False

    used_ridge = False
    try:
        beta, *_ = np.linalg.lstsq(design, targets, rcond=None)
        cond = np.linalg.cond(design)
        if not np.isfinite(cond) or cond > 1e12:
            raise np.linalg.LinAlgError
        return beta.astype(float, copy=False), used_ridge
    except np.linalg.LinAlgError:
        used_ridge = True
        regulariser = math.sqrt(ridge_eps) * np.eye(design.shape[1])
        augmented_design = np.vstack((design, regulariser))
        augmented_targets = np.concatenate((targets, np.zeros(design.shape[1], dtype=float)))
        try:
            beta, *_ = np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(augmented_design) @ augmented_targets
        if not np.isfinite(beta).all():
            raise ValueError("regression solver produced non-finite coefficients") from None
        return beta.astype(float, copy=False), used_ridge


def enforce_precision_policy(
    price: float, ci_half_width: float
) -> tuple[float, str, float, list[str]]:
    """Classify Monte Carlo precision without altering the confidence interval."""

    if isinstance(price, bool) or not isinstance(price, Real):
        raise TypeError("price must be a real number")
    if isinstance(ci_half_width, bool) or not isinstance(ci_half_width, Real):
        raise TypeError("ci_half_width must be a real number")
    normalised_price = float(price)
    reported = float(ci_half_width)
    if not math.isfinite(normalised_price) or not math.isfinite(reported):
        raise ValueError("price and ci_half_width must be finite")
    if reported < 0.0:
        raise ValueError("ci_half_width must be non-negative")

    precision_limit = max(1e-4, 0.02 * max(abs(normalised_price), 1.0))

    if reported <= 0.25 * precision_limit:
        bucket = "tight"
    elif reported <= precision_limit:
        bucket = "medium"
    else:
        bucket = "loose"

    return reported, bucket, precision_limit, []


def deep_itm_policy(
    spot: float,
    strike: float,
    option_type: str,
) -> tuple[float | None, str | None]:
    """Return the American option's exact immediate-exercise lower bound."""

    spot, strike, option_type, _, _, _ = _validate_bound_inputs(
        spot,
        strike,
        option_type,
    )
    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if intrinsic <= 0.0:
        return 0.0, None
    return intrinsic, "no_arbitrage_floor"


def deep_otm_upper_bound(
    spot: float,
    strike: float,
    option_type: str,
    *,
    tau: float = 0.0,
    r: float = 0.0,
    q: float = 0.0,
) -> tuple[float | None, str | None]:
    """Return a model-independent American option upper bound.

    With non-negative carry rates, an American call cannot be worth more than
    the underlying and an American put cannot be worth more than the strike.
    The exponential factors retain valid bounds under negative rates or
    negative dividend yields. Unlike the former two-percent tail cap, these
    bounds hold for every volatility and maturity.
    """

    spot, strike, option_type, tau, r, q = _validate_bound_inputs(
        spot,
        strike,
        option_type,
        tau=tau,
        r=r,
        q=q,
    )
    return (
        (
            spot * math.exp(max(-q * tau, 0.0)),
            "no_arbitrage_cap",
        )
        if option_type == "call"
        else (
            strike * math.exp(max(-r * tau, 0.0)),
            "no_arbitrage_cap",
        )
    )


def _validate_bound_inputs(
    spot: float,
    strike: float,
    option_type: str,
    *,
    tau: float = 0.0,
    r: float = 0.0,
    q: float = 0.0,
) -> tuple[float, float, str, float, float, float]:
    """Validate model-independent bound inputs without silently coercing them."""

    raw_values = (spot, strike, tau, r, q)
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_values):
        raise TypeError("bound inputs must be real numbers")
    if not isinstance(option_type, str):
        raise TypeError("option_type must be a string")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be either 'call' or 'put'")

    normalised = tuple(float(value) for value in raw_values)
    if not all(math.isfinite(value) for value in normalised):
        raise ValueError("bound inputs must be finite")
    spot_value, strike_value, tau_value, r_value, q_value = normalised
    if not 0.0 < spot_value <= 1e12:
        raise ValueError("spot must be within (0, 1e12]")
    if not 0.0 < strike_value <= 1e12:
        raise ValueError("strike must be within (0, 1e12]")
    if not 0.0 <= tau_value <= 100.0:
        raise ValueError("tau must be within [0, 100]")
    if not -1.0 <= r_value <= 1.0:
        raise ValueError("r must be within [-1, 1]")
    if not -1.0 <= q_value <= 1.0:
        raise ValueError("q must be within [-1, 1]")
    return spot_value, strike_value, option_type, tau_value, r_value, q_value


def numerics_policy_hash() -> str:
    """Return a short hash identifying the current numerics policy."""

    payload = {
        "precision": "reported-v2",
        "inputs": "reject-v2",
        "bounds": "american-no-arbitrage-v2",
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()
