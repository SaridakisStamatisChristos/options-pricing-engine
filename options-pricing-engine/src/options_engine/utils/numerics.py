"""Numerical utilities for Monte Carlo style pricing algorithms."""

from __future__ import annotations

import hashlib
import json
import math
from typing import List, Tuple

import numpy as np

__all__ = [
    "apply_global_clamps",
    "laguerre_basis3",
    "stable_regression",
    "enforce_precision_policy",
    "deep_itm_policy",
    "deep_otm_upper_bound",
    "numerics_policy_hash",
]


def apply_global_clamps(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    r: float,
    q: float,
) -> Tuple[Tuple[float, float, float, float, float, float], List[str]]:
    """Clamp inputs to reasonable numerical ranges."""

    flags: List[str] = []

    if spot <= 0.0:
        spot = 1e-8
        flags.append("clamp_spot")
    elif spot > 1e8:
        spot = 1e8
        flags.append("clamp_spot")

    if strike <= 0.0:
        strike = 1e-8
        flags.append("clamp_strike")
    elif strike > 1e8:
        strike = 1e8
        flags.append("clamp_strike")

    if tau < 1e-6:
        tau = 1e-6
        flags.append("clamp_tau")
    elif tau > 100.0:
        tau = 100.0
        flags.append("clamp_tau")

    if sigma < 1e-6:
        sigma = 1e-6
        flags.append("clamp_sigma")
    elif sigma > 5.0:
        sigma = 5.0
        flags.append("clamp_sigma")

    if r < -5.0:
        r = -5.0
        flags.append("clamp_rate")
    elif r > 5.0:
        r = 5.0
        flags.append("clamp_rate")

    if q < -5.0:
        q = -5.0
        flags.append("clamp_dividend")
    elif q > 5.0:
        q = 5.0
        flags.append("clamp_dividend")

    return (spot, strike, tau, sigma, r, q), flags


def laguerre_basis3(x: np.ndarray) -> np.ndarray:
    """Return the first four Laguerre basis functions evaluated at ``x``."""

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
) -> Tuple[np.ndarray, bool]:
    """Solve ``X beta = y`` using a numerically stable approach."""

    if X.size == 0:
        return np.zeros(0, dtype=float), False

    used_ridge = False
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        cond = np.linalg.cond(X)
        if not np.isfinite(cond) or cond > 1e12:
            raise np.linalg.LinAlgError
        return beta.astype(float, copy=False), used_ridge
    except np.linalg.LinAlgError:
        used_ridge = True
        XtX = X.T @ X
        regulariser = ridge_eps * np.eye(X.shape[1])
        try:
            beta = np.linalg.solve(XtX + regulariser, X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XtX + regulariser) @ (X.T @ y)
        return beta.astype(float, copy=False), used_ridge


def enforce_precision_policy(price: float, ci_half_width: float) -> Tuple[float, str, float, List[str]]:
    """Apply a simple precision policy to Monte Carlo estimates."""

    precision_limit = max(1e-4, 0.02 * max(abs(price), 1.0))
    flags: List[str] = []

    clipped = float(ci_half_width)
    if clipped > precision_limit:
        clipped = precision_limit
        flags.append("ci_clipped")

    if clipped <= 0.25 * precision_limit:
        bucket = "tight"
    elif clipped <= 0.5 * precision_limit:
        bucket = "medium"
    else:
        bucket = "loose"

    return clipped, bucket, precision_limit, flags


def deep_itm_policy(
    spot: float,
    strike: float,
    option_type: str,
) -> Tuple[float | None, str | None]:
    """Return a conservative lower bound for deep in-the-money options."""

    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    lower_bound: float | None = None
    flag: str | None = None

    if intrinsic <= 0.0:
        return None, None

    ratio = spot / strike if strike > 0 else float("inf")
    if option_type == "put":
        ratio = strike / spot if spot > 0 else float("inf")

    if ratio >= 3.0:
        lower_bound = 0.95 * intrinsic
        flag = "deep_itm_floor"

    return lower_bound, flag


def deep_otm_upper_bound(
    spot: float,
    strike: float,
    option_type: str,
) -> Tuple[float | None, str | None]:
    """Return a loose upper bound for deep out-of-the-money options."""

    ratio = strike / spot if spot > 0 else float("inf")
    if option_type == "put":
        ratio = spot / strike if strike > 0 else float("inf")

    if ratio >= 3.0:
        premium = 0.02 * max(spot, strike)
        return premium, "deep_otm_cap"

    return None, None


def numerics_policy_hash() -> str:
    """Return a short hash identifying the current numerics policy."""

    payload = {
        "precision": "v1",
        "clamps": "v1",
        "tails": "v1",
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()
