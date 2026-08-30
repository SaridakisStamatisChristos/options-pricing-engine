"""Pricing model implementations used by the engine."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import ClassVar

import numpy as np
from numpy.random import Generator, SeedSequence
from scipy.stats import norm

from ..greeks.estimators import (
    GreekSummary,
    aggregate_statistics,
    finite_difference_delta,
    finite_difference_gamma,
    finite_difference_rho,
    finite_difference_theta,
    finite_difference_vega,
    pathwise_delta,
    pathwise_gamma,
    pathwise_vega,
    rho_likelihood_ratio,
    theta_likelihood_ratio,
)
from ..greeks.stability import contributions_finite, is_estimate_unstable
from ..utils.numerics import (
    apply_global_clamps,
    deep_itm_policy,
    deep_otm_upper_bound,
    enforce_precision_policy,
    laguerre_basis3,
    numerics_policy_hash,
    stable_regression,
)
from ..utils.validation import validate_pricing_parameters
from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from .replay import ReplayCapsule, build_replay_capsule

LOGGER = logging.getLogger(__name__)

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
    effective_paths = paths + (paths % 2) if antithetic else paths
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


def black_scholes_price(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
) -> PriceResult:
    """Convenience wrapper returning a :class:`PriceResult` for Black-Scholes."""

    opt = _normalise_option_type(option_type)
    contract = OptionContract(
        symbol="BS",
        strike_price=strike,
        time_to_expiry=tau,
        option_type=OptionType.CALL if opt == "call" else OptionType.PUT,
    )
    market = MarketData(spot_price=spot, risk_free_rate=r, dividend_yield=q)
    model = BlackScholesModel()
    result = model.calculate_price(contract, market, sigma)
    meta: dict[str, object] = {
        "method": "black_scholes",
        "option_type": opt,
        "policy_flags": [],
    }
    return PriceResult(
        price=result.theoretical_price, ci_half_width=0.0, meta=meta, standard_error=0.0
    )


def binomial_price(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    *,
    steps: int = 200,
) -> PriceResult:
    """Convenience wrapper around :class:`BinomialModel`."""

    opt = _normalise_option_type(option_type)
    contract = OptionContract(
        symbol="BINOM",
        strike_price=strike,
        time_to_expiry=tau,
        option_type=OptionType.CALL if opt == "call" else OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
    )
    market = MarketData(spot_price=spot, risk_free_rate=r, dividend_yield=q)
    model = BinomialModel(steps=steps)
    result = model.calculate_price(contract, market, sigma)
    meta: dict[str, object] = {
        "method": f"binomial_{steps}",
        "option_type": opt,
        "policy_flags": [],
    }
    return PriceResult(
        price=result.theoretical_price, ci_half_width=0.0, meta=meta, standard_error=0.0
    )


def _runtime_checks(
    *,
    base_price: float,
    option_type: str,
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    r: float,
    q: float,
    steps: int,
    paths: int,
    seed: int,
    basis: str,
    antithetic: bool,
    use_cv: bool,
) -> dict[str, object]:
    if os.getenv("NUMERICS_STRICT") != "1":
        return {"checks_enabled": False}

    diagnostics: dict[str, object] = {"checks_enabled": True}
    reduced_paths = max(2_000, paths // 4)
    bump_seed = seed + 7919

    try:
        bump = max(1e-4, 0.01 * strike)
        low = american_lsmc_price(
            spot,
            max(strike - bump, 1e-8),
            tau,
            sigma,
            r,
            q,
            option_type,
            steps=steps,
            paths=reduced_paths,
            seed=bump_seed,
            basis=basis,
            antithetic=antithetic,
            use_cv=False,
            _skip_checks=True,
        )
        high = american_lsmc_price(
            spot,
            strike + bump,
            tau,
            sigma,
            r,
            q,
            option_type,
            steps=steps,
            paths=reduced_paths,
            seed=bump_seed + 1,
            basis=basis,
            antithetic=antithetic,
            use_cv=False,
            _skip_checks=True,
        )
        convex = low.price - 2.0 * base_price + high.price
        diagnostics["strike_convexity"] = convex >= -1e-3 * max(1.0, abs(base_price))
    except Exception:
        diagnostics["strike_convexity"] = False

    try:
        bump_sigma = 0.05 * sigma if sigma > 0 else 0.01
        higher_sigma = american_lsmc_price(
            spot,
            strike,
            tau,
            sigma + bump_sigma,
            r,
            q,
            option_type,
            steps=steps,
            paths=reduced_paths,
            seed=bump_seed + 2,
            basis=basis,
            antithetic=antithetic,
            use_cv=False,
            _skip_checks=True,
        )
        diagnostics["sigma_monotonic"] = higher_sigma.price >= base_price - 1e-3 * max(
            1.0, abs(base_price)
        )
    except Exception:
        diagnostics["sigma_monotonic"] = False

    try:
        bump_tau = 0.05 * tau if tau > 0 else 0.01
        higher_tau = american_lsmc_price(
            spot,
            strike,
            tau + bump_tau,
            sigma,
            r,
            q,
            option_type,
            steps=steps,
            paths=reduced_paths,
            seed=bump_seed + 3,
            basis=basis,
            antithetic=antithetic,
            use_cv=False,
            _skip_checks=True,
        )
        diagnostics["tau_monotonic"] = higher_tau.price >= base_price - 1e-3 * max(
            1.0, abs(base_price)
        )
    except Exception:
        diagnostics["tau_monotonic"] = False

    return diagnostics


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


def american_lsmc_price(
    spot: float,
    strike: float,
    tau: float,
    sigma: float,
    r: float,
    q: float,
    option_type: str,
    *,
    steps: int = 64,
    paths: int = 20_000,
    seed: int = 0,
    basis: str = "laguerre3",
    antithetic: bool = True,
    use_cv: bool = True,
    _skip_checks: bool = False,
) -> PriceResult:
    """Price an American option using the Longstaff-Schwartz method."""

    steps = _bounded_integer("steps", steps, minimum=1, maximum=MAX_LSMC_STEPS)
    paths = _bounded_integer("paths", paths, minimum=1, maximum=MAX_LSMC_PATHS)
    seed = _bounded_integer("seed", seed, minimum=0, maximum=MAX_RANDOM_SEED)
    antithetic = _require_boolean("antithetic", antithetic)
    use_cv = _require_boolean("use_cv", use_cv)
    _skip_checks = _require_boolean("_skip_checks", _skip_checks)
    effective_path_count = _validate_lsmc_workload(
        steps=steps,
        paths=paths,
        antithetic=antithetic,
    )
    if basis != "laguerre3":
        raise ValueError("only the 'laguerre3' basis is supported")

    opt = _normalise_option_type(option_type)
    (spot, strike, tau, sigma, r, q), clamp_flags = apply_global_clamps(
        spot, strike, tau, sigma, r, q
    )

    policy_flags = list(clamp_flags)
    if antithetic:
        policy_flags.append("antithetic")

    intrinsic_now = max(spot - strike, 0.0) if opt == "call" else max(strike - spot, 0.0)
    if opt == "call" and q <= 0.0 and r >= 0.0:
        # With no positive continuous dividend, early exercise destroys time value.
        # The American call is therefore exactly its European counterpart.
        european = black_scholes_price(spot, strike, tau, sigma, r, q, opt)
        config = {
            "spot": spot,
            "strike": strike,
            "tau": tau,
            "sigma": sigma,
            "r": r,
            "q": q,
            "steps": steps,
            "paths": paths,
            "seed": seed,
            "basis": basis,
            "antithetic": antithetic,
            "use_cv": use_cv,
        }
        meta: dict[str, object] = {
            "method": "american_call_no_early_exercise",
            "option_type": opt,
            "policy_flags": ["no_early_exercise_theorem"],
            "precision_bucket": "exact",
            "precision_limit": 0.0,
            "runtime": {"checks_enabled": False},
            "capsule": {
                "capsule_id": _build_capsule_id(config),
                "policy_hash": numerics_policy_hash(),
                "config": config,
            },
        }
        return PriceResult(
            price=european.price,
            ci_half_width=0.0,
            meta=meta,
            standard_error=0.0,
        )

    if tau <= 1e-6 or sigma <= 1e-8:
        meta = {
            "method": "american_lsmc",
            "option_type": opt,
            "policy_flags": policy_flags,
            "precision_bucket": "tight",
            "precision_limit": 0.0,
            "runtime": {"checks_enabled": False},
            "capsule": {
                "capsule_id": _build_capsule_id({"seed": seed}),
                "policy_hash": numerics_policy_hash(),
                "config": {
                    "spot": spot,
                    "strike": strike,
                    "tau": tau,
                    "sigma": sigma,
                    "r": r,
                    "q": q,
                    "steps": steps,
                    "paths": paths,
                    "seed": seed,
                    "basis": basis,
                    "antithetic": antithetic,
                    "use_cv": use_cv,
                },
            },
        }
        return PriceResult(price=intrinsic_now, ci_half_width=0.0, meta=meta, standard_error=0.0)

    step_count = steps
    dt = tau / step_count
    discount_step = math.exp(-r * dt)

    path_count = effective_path_count
    if antithetic:
        half = path_count // 2
    rng = np.random.default_rng(seed)

    normals = rng.standard_normal((step_count, path_count if not antithetic else half))
    if antithetic:
        normals = np.concatenate([normals, -normals], axis=1)
    path_count = normals.shape[1]

    prices = np.empty((step_count + 1, path_count), dtype=float)
    prices[0, :] = spot
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt)
    for step in range(1, step_count + 1):
        shock = diffusion * normals[step - 1]
        with np.errstate(over="ignore", invalid="ignore"):
            prices[step] = prices[step - 1] * np.exp(drift + shock)
        if not np.isfinite(prices[step]).all():
            raise ValueError("LSMC simulation exceeds the supported floating-point range")

    if opt == "call":
        payoffs = np.maximum(prices - strike, 0.0)
    else:
        payoffs = np.maximum(strike - prices, 0.0)

    cashflows = payoffs[-1].copy()
    regression_guard = False
    im_filter = False

    for step in range(step_count - 1, 0, -1):
        cashflows *= discount_step
        intrinsic = payoffs[step]
        in_the_money = intrinsic > 0.0
        itm_count = int(np.count_nonzero(in_the_money))
        if itm_count < 10:
            im_filter = True
            continue

        scaled = prices[step, in_the_money] / strike
        basis_matrix = laguerre_basis3(scaled)
        targets = cashflows[in_the_money]

        # Cross-fit the stopping rule: each path is evaluated by coefficients
        # estimated on the opposite fold. This removes the classic look-ahead
        # bias from fitting and exercising on the same simulated cashflows.
        continuation = np.empty_like(targets)
        itm_indices = np.flatnonzero(in_the_money)
        if antithetic:
            pair_count = path_count // 2
            unit_indices = np.where(
                itm_indices < pair_count,
                itm_indices,
                itm_indices - pair_count,
            )
        else:
            unit_indices = itm_indices
        fold_count = min(5, max(2, itm_count // 10))
        fold_labels = unit_indices % fold_count
        for held_out in range(fold_count):
            train = fold_labels != held_out
            test = ~train
            if not np.any(test):
                continue
            fold_prediction: np.ndarray | None = None
            for columns in range(basis_matrix.shape[1], 1, -1):
                beta, used_ridge = stable_regression(basis_matrix[train, :columns], targets[train])
                predictions = basis_matrix[test, :columns] @ beta
                if not np.all(np.isfinite(predictions)):
                    regression_guard = True
                    continue
                if used_ridge and columns > 2:
                    regression_guard = True
                    continue
                fold_prediction = predictions
                regression_guard = regression_guard or used_ridge
                break
            if fold_prediction is None:
                fold_prediction = np.full(
                    int(np.count_nonzero(test)),
                    float(np.mean(targets[train])),
                )
                regression_guard = True
            continuation[test] = np.maximum(fold_prediction, 0.0)

        cont_values = np.zeros_like(cashflows)
        cont_values[in_the_money] = continuation
        exercise = np.zeros_like(cashflows, dtype=bool)
        exercise[in_the_money] = intrinsic[in_the_money] >= cont_values[in_the_money]
        cashflows = np.where(exercise, intrinsic, cashflows)

    cashflows *= discount_step

    intrinsic_zero = np.full(path_count, intrinsic_now, dtype=float)
    cashflows = np.where(intrinsic_zero >= cashflows, intrinsic_zero, cashflows)

    effective_cashflows = _antithetic_units(cashflows, antithetic=antithetic)
    cv_report: dict[str, object] = {"used": False}

    bs_reference = black_scholes_price(spot, strike, tau, sigma, r, q, opt)
    if use_cv:
        euro_payoff = payoffs[-1] * math.exp(-r * tau)
        euro_units = _antithetic_units(euro_payoff, antithetic=antithetic)
        effective_cashflows, cv_report = _cross_fitted_control_variate(
            effective_cashflows,
            euro_units,
            control_mean=bs_reference.price,
            folds=5,
        )
        if cv_report["used"]:
            policy_flags.extend(("cv_cross_fitted", "cv_used"))
        else:
            policy_flags.append("cv_skipped")
    else:
        policy_flags.append("cv_skipped")

    price = float(np.mean(effective_cashflows))
    independent_count = effective_cashflows.size
    if independent_count > 1:
        standard_error = float(np.std(effective_cashflows, ddof=1) / math.sqrt(independent_count))
    else:
        standard_error = 0.0
    ci_half_width = 1.96 * standard_error

    ci_half_width, precision_bucket, precision_limit, precision_flags = enforce_precision_policy(
        price, ci_half_width
    )
    policy_flags.extend(precision_flags)

    lower_tail, lower_flag = deep_itm_policy(spot, strike, opt)
    lower_tail = max(lower_tail or 0.0, bs_reference.price)
    upper_tail, upper_flag = deep_otm_upper_bound(spot, strike, opt, tau=tau, r=r, q=q)

    original_price = price
    if lower_tail is not None:
        price = max(price, lower_tail)
    if upper_tail is not None:
        price = min(price, upper_tail)
    if price != original_price:
        if price > original_price and lower_flag:
            policy_flags.append(lower_flag)
        if price < original_price and upper_flag:
            policy_flags.append(upper_flag)
        policy_flags.append("no_arbitrage_projection")

    if im_filter:
        policy_flags.append("lsmc_im_filter")
    if regression_guard:
        policy_flags.append("reg_singular_guard")

    config = {
        "spot": spot,
        "strike": strike,
        "tau": tau,
        "sigma": sigma,
        "r": r,
        "q": q,
        "steps": steps,
        "paths": paths,
        "seed": seed,
        "basis": basis,
        "antithetic": antithetic,
        "use_cv": use_cv,
    }

    runtime = (
        _runtime_checks(
            base_price=price,
            option_type=opt,
            spot=spot,
            strike=strike,
            tau=tau,
            sigma=sigma,
            r=r,
            q=q,
            steps=steps,
            paths=paths,
            seed=seed,
            basis=basis,
            antithetic=antithetic,
            use_cv=use_cv,
        )
        if not _skip_checks
        else {"checks_enabled": False}
    )

    result_meta: dict[str, object] = {
        "method": "american_lsmc",
        "option_type": opt,
        "policy_flags": sorted(set(policy_flags)),
        "precision_bucket": precision_bucket,
        "precision_limit": precision_limit,
        "runtime": runtime,
        "control_variate": cv_report,
        "estimate": {
            "raw_price": original_price,
            "projected_price": price,
            "lower_bound": lower_tail,
            "upper_bound": upper_tail,
            "projection_applied": price != original_price,
        },
        "capsule": {
            "capsule_id": _build_capsule_id(config),
            "policy_hash": numerics_policy_hash(),
            "config": config,
        },
    }

    return PriceResult(
        price=price,
        ci_half_width=ci_half_width,
        meta=result_meta,
        standard_error=standard_error,
    )


def _norm_pdf(value: float) -> float:
    return INV_SQRT_TWO_PI * math.exp(-0.5 * value * value)


def _norm_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / SQRT_TWO)


def _log_moneyness_threshold(time_to_expiry: float) -> float:
    for maturity, clamp in LOG_MONEYNESS_CLAMP:
        if time_to_expiry <= maturity:
            return clamp
    return LOG_MONEYNESS_CLAMP[-1][1]


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


def _black_scholes_payoff(
    contract: OptionContract,
    spot: float,
    strike: float,
) -> float:
    """Return the intrinsic value of an option contract."""
    intrinsic = max(0.0, spot - strike)
    if contract.option_type is OptionType.PUT:
        intrinsic = max(0.0, strike - spot)
    return intrinsic


def _black_scholes_greeks(
    contract: OptionContract,
    market_data: MarketData,
    volatility: float,
) -> tuple[float, float, float, float, float, float]:
    """Calculate the Black-Scholes price and greeks."""
    spot = market_data.spot_price
    strike = contract.strike_price
    time_to_expiry = contract.time_to_expiry
    rate = market_data.risk_free_rate
    dividend = market_data.dividend_yield

    if time_to_expiry <= 0.0 or volatility <= 0.0:
        intrinsic = _black_scholes_payoff(contract, spot, strike)
        return intrinsic, 0.0, 0.0, 0.0, 0.0, 0.0

    time_to_expiry = max(time_to_expiry, TAU_MIN)
    volatility = max(volatility, SIGMA_MIN)

    sqrt_t = math.sqrt(time_to_expiry)
    discount_dividend = math.exp(-dividend * time_to_expiry)
    discount_rate = math.exp(-rate * time_to_expiry)
    parity_anchor = spot * discount_dividend - strike * discount_rate

    log_moneyness = math.log(spot / strike)
    numerator = log_moneyness + (rate - dividend + 0.5 * volatility**2) * time_to_expiry
    denominator = volatility * sqrt_t
    d1 = numerator / denominator
    d2 = d1 - volatility * sqrt_t

    pdf = _norm_pdf(d1)
    cdf_d1 = _norm_cdf(d1)
    cdf_d2 = _norm_cdf(d2)

    clamp_threshold = _log_moneyness_threshold(time_to_expiry)
    if log_moneyness > clamp_threshold:
        # Tail-safe formulation to reduce catastrophic cancellation
        tail_call = strike * discount_rate * _norm_cdf(-d2) - spot * discount_dividend * _norm_cdf(
            -d1
        )
        call_price = parity_anchor + tail_call
    else:
        call_price = spot * discount_dividend * cdf_d1 - strike * discount_rate * cdf_d2

    call_delta = discount_dividend * cdf_d1
    call_gamma = discount_dividend * pdf / (spot * volatility * sqrt_t)
    call_theta = (
        -spot * discount_dividend * pdf * volatility / (2.0 * sqrt_t)
        + dividend * spot * discount_dividend * cdf_d1
        - rate * strike * discount_rate * cdf_d2
    ) / 365.0
    call_vega = spot * discount_dividend * pdf * sqrt_t / 100.0
    call_rho = strike * discount_rate * time_to_expiry * cdf_d2 / 100.0

    if contract.option_type is OptionType.CALL:
        price = call_price
        delta = call_delta
        gamma = call_gamma
        theta = call_theta
        vega = call_vega
        rho = call_rho
    else:
        price = call_price - parity_anchor
        delta = call_delta - discount_dividend
        gamma = call_gamma
        theta = (
            call_theta
            + (-dividend * spot * discount_dividend + rate * strike * discount_rate) / 365.0
        )
        vega = call_vega
        rho = call_rho - time_to_expiry * strike * discount_rate / 100.0

    return price, delta, gamma, theta, vega, rho


@dataclass(slots=True)
class BlackScholesModel:
    """Deterministic Black-Scholes pricing model."""

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> PricingResult:
        start = time.perf_counter()
        try:
            if contract.exercise_style is not ExerciseStyle.EUROPEAN:
                raise ValueError("Black-Scholes supports European exercise only")
            validate_pricing_parameters(contract, market_data, volatility)
            price, delta, gamma, theta, vega, rho = _black_scholes_greeks(
                contract, market_data, volatility
            )
            price = max(0.0, price)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=price,
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                implied_volatility=volatility,
                computation_time_ms=elapsed_ms,
                model_used="black_scholes",
            )
        except Exception:  # pragma: no cover - preserve context for API error mapping
            LOGGER.exception("Black-Scholes pricing failed")
            raise


@dataclass(slots=True)
class BinomialModel:
    """Recombining binomial tree pricing model."""

    MAX_STEPS: ClassVar[int] = MAX_BINOMIAL_STEPS

    steps: int = 200
    _compute_rho: bool = True

    def __post_init__(self) -> None:
        self.steps = _bounded_integer(
            "steps",
            self.steps,
            minimum=2,
            maximum=self.MAX_STEPS,
        )
        self._compute_rho = _require_boolean("_compute_rho", self._compute_rho)

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
    ) -> PricingResult:
        start = time.perf_counter()
        try:
            validate_pricing_parameters(contract, market_data, volatility)

            # The CRR risk-neutral probability is valid only when d < exp((r-q)dt) < u.
            # Increase the resolution until that condition holds; clamping p creates a
            # different, arbitrageable process and can even produce negative vega.
            steps = self.steps
            while True:
                delta_t = contract.time_to_expiry / steps
                log_up = volatility * math.sqrt(delta_t)
                up = math.exp(log_up)
                down = 1.0 / up
                growth = math.exp(
                    (market_data.risk_free_rate - market_data.dividend_yield) * delta_t
                )
                probability = (growth - down) / (up - down)
                if 0.0 < probability < 1.0:
                    break
                if steps >= self.MAX_STEPS:
                    raise ValueError(
                        "CRR tree cannot satisfy no-arbitrage probability; "
                        "increase volatility or use Black-Scholes"
                    )
                steps = min(self.MAX_STEPS, steps * 2)
            discount = math.exp(-market_data.risk_free_rate * delta_t)

            sqrt_dt = math.sqrt(delta_t)
            dup = up * sqrt_dt
            ddown = -down * sqrt_dt
            denom = up - down
            if denom == 0.0:
                raise ZeroDivisionError("up and down factors resulted in zero denominator")
            numerator = growth - down
            dnumerator = -ddown
            ddenom = dup - ddown
            dprobability = (dnumerator * denom - numerator * ddenom) / (denom**2)

            node_indices = np.arange(steps + 1)
            log_price_span = volatility * math.sqrt(contract.time_to_expiry * steps)
            max_terminal_log_price = math.log(market_data.spot_price) + log_price_span
            vega_scale = math.sqrt(contract.time_to_expiry * steps)
            safe_log_limit = math.log(np.finfo(float).max) - math.log1p(vega_scale)
            if max_terminal_log_price > safe_log_limit:
                raise ValueError(
                    "CRR tree exceeds the supported floating-point range; "
                    "reduce volatility, maturity, or steps"
                )
            log_prices = math.log(market_data.spot_price) + (2 * node_indices - steps) * log_up
            prices = np.exp(log_prices)

            price_vega = np.empty_like(prices)
            price_vega[:] = prices * ((2 * node_indices - steps) * sqrt_dt)

            if contract.option_type is OptionType.CALL:
                values = np.maximum(prices - contract.strike_price, 0.0)
                value_vega = np.where(values > 0.0, price_vega, 0.0)
            else:
                values = np.maximum(contract.strike_price - prices, 0.0)
                value_vega = np.where(values > 0.0, -price_vega, 0.0)

            first_step_values: np.ndarray | None = None
            first_step_prices: np.ndarray | None = None
            second_step_values: np.ndarray | None = None
            second_step_prices: np.ndarray | None = None

            for index in range(steps - 1, -1, -1):
                prev_values = values
                prev_value_vega = value_vega

                continuation_values = discount * (
                    probability * prev_values[1:] + (1.0 - probability) * prev_values[:-1]
                )
                continuation_vega = discount * (
                    probability * prev_value_vega[1:] + (1.0 - probability) * prev_value_vega[:-1]
                )
                continuation_vega += discount * dprobability * (prev_values[1:] - prev_values[:-1])

                prev_prices = prices
                prev_price_vega = price_vega
                prices = prev_prices[:-1] * up
                price_vega = prev_price_vega[:-1] * up + prev_prices[:-1] * dup

                values = continuation_values
                value_vega = continuation_vega

                if contract.exercise_style is ExerciseStyle.AMERICAN:
                    if contract.option_type is OptionType.CALL:
                        exercise_value = np.maximum(prices - contract.strike_price, 0.0)
                        exercise_vega = np.where(exercise_value > 0.0, price_vega, 0.0)
                    else:
                        exercise_value = np.maximum(contract.strike_price - prices, 0.0)
                        exercise_vega = np.where(exercise_value > 0.0, -price_vega, 0.0)

                    exercise_mask = exercise_value > values
                    if np.any(exercise_mask):
                        values = np.where(exercise_mask, exercise_value, values)
                        value_vega = np.where(exercise_mask, exercise_vega, value_vega)

                if index == 2:
                    second_step_values = values.copy()
                    second_step_prices = prices.copy()
                if index == 1:
                    first_step_values = values.copy()
                    first_step_prices = prices.copy()

            elapsed_ms = (time.perf_counter() - start) * 1000.0
            delta: float | None = None
            gamma: float | None = None
            theta: float | None = None

            if first_step_values is not None and first_step_prices is not None:
                denom = first_step_prices[1] - first_step_prices[0]
                if abs(denom) > 0:
                    delta = float((first_step_values[1] - first_step_values[0]) / denom)

            if (
                delta is not None
                and second_step_values is not None
                and second_step_prices is not None
                and first_step_prices is not None
            ):
                down_denom = second_step_prices[1] - second_step_prices[0]
                up_denom = second_step_prices[2] - second_step_prices[1]
                root_denom = 0.5 * (second_step_prices[2] - second_step_prices[0])
                if abs(down_denom) > 0 and abs(up_denom) > 0 and abs(root_denom) > 0:
                    delta_down = (second_step_values[1] - second_step_values[0]) / down_denom
                    delta_up = (second_step_values[2] - second_step_values[1]) / up_denom
                    gamma = float((delta_up - delta_down) / root_denom)

            if second_step_values is not None and second_step_values.size >= 2:
                theta = float((second_step_values[1] - values[0]) / (2.0 * delta_t) / 365.0)

            vega = float(value_vega[0] / 100.0)
            rho: float | None = None
            if self._compute_rho:
                bump = 1e-4
                rate_low = max(-1.0, market_data.risk_free_rate - bump)
                rate_high = min(1.0, market_data.risk_free_rate + bump)
                if rate_high > rate_low:
                    low_market = MarketData(
                        spot_price=market_data.spot_price,
                        risk_free_rate=rate_low,
                        dividend_yield=market_data.dividend_yield,
                    )
                    high_market = MarketData(
                        spot_price=market_data.spot_price,
                        risk_free_rate=rate_high,
                        dividend_yield=market_data.dividend_yield,
                    )
                    rho_model = BinomialModel(steps=steps, _compute_rho=False)
                    low_price = rho_model.calculate_price(
                        contract, low_market, volatility
                    ).theoretical_price
                    high_price = rho_model.calculate_price(
                        contract, high_market, volatility
                    ).theoretical_price
                    rho = float((high_price - low_price) / (rate_high - rate_low) / 100.0)

            numerical_outputs = [values[0], value_vega[0]]
            numerical_outputs.extend(
                value for value in (delta, gamma, theta, rho) if value is not None
            )
            if not all(math.isfinite(float(value)) for value in numerical_outputs):
                raise ValueError("CRR tree produced a non-finite price or Greek")

            return PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=float(values[0]),
                delta=delta,
                gamma=gamma,
                theta=theta,
                vega=vega,
                rho=rho,
                computation_time_ms=elapsed_ms,
                model_used=f"binomial_{steps}",
                implied_volatility=volatility,
            )
        except Exception:  # pragma: no cover - preserve context for API error mapping
            LOGGER.exception("Binomial pricing failed")
            raise


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


@dataclass(slots=True)
class MonteCarloModel:
    """Monte Carlo pricer supporting antithetic variates."""

    DEFAULT_PATHS: ClassVar[int] = 20_000
    DEFAULT_ANTITHETIC: ClassVar[bool] = True
    MAX_PATHS: ClassVar[int] = MAX_MONTE_CARLO_PATHS

    paths: int = DEFAULT_PATHS
    antithetic: bool = DEFAULT_ANTITHETIC
    seed_sequence: SeedSequence | None = None  # Optional deterministic seed shared across threads
    use_control_variates: bool = True

    def __post_init__(self) -> None:
        self.paths = _bounded_integer(
            "paths",
            self.paths,
            minimum=1,
            maximum=self.MAX_PATHS,
        )
        self.antithetic = _require_boolean("antithetic", self.antithetic)
        self.seed_sequence = _validate_seed_sequence("seed_sequence", self.seed_sequence)
        self.use_control_variates = _require_boolean(
            "use_control_variates",
            self.use_control_variates,
        )

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        seed_sequence: SeedSequence | None = None,
    ) -> PricingResult:
        start = time.perf_counter()
        try:
            if contract.exercise_style is not ExerciseStyle.EUROPEAN:
                raise ValueError("Terminal-payoff Monte Carlo supports European exercise only")
            validate_pricing_parameters(contract, market_data, volatility)

            simulation_paths = self.paths

            if contract.time_to_expiry <= 1e-12 or volatility <= 1e-12:
                price, delta, gamma, _theta, vega, _rho = _black_scholes_greeks(
                    contract, market_data, max(volatility, 1e-12)
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                return PricingResult(
                    contract_id=contract.contract_id,
                    theoretical_price=max(0.0, price),
                    delta=delta,
                    gamma=gamma,
                    theta=_theta,
                    vega=vega,
                    rho=_rho,
                    computation_time_ms=elapsed_ms,
                    model_used=f"monte_carlo_{simulation_paths}",
                    implied_volatility=volatility,
                )

            sequence = _validate_seed_sequence("seed_sequence", seed_sequence) or self.seed_sequence
            rng = _thread_local_generator(sequence)

            if self.antithetic:
                # At least two independent pairs are required to estimate a
                # sampling error; a single antithetic pair has zero degrees of
                # freedom even though it contains two correlated paths.
                simulation_paths = max(4, simulation_paths + (simulation_paths % 2))
                half_paths = simulation_paths // 2
                base_draws = rng.standard_normal(half_paths)
                draws = np.empty(simulation_paths, dtype=float)
                draws[:half_paths] = base_draws
                draws[half_paths:] = -base_draws
            else:
                draws = rng.standard_normal(simulation_paths)

            time_sqrt = math.sqrt(max(0.0, contract.time_to_expiry))
            drift = (
                market_data.risk_free_rate - market_data.dividend_yield - 0.5 * volatility**2
            ) * contract.time_to_expiry
            diffusion = volatility * time_sqrt * draws
            with np.errstate(over="ignore", invalid="ignore"):
                terminal_prices = market_data.spot_price * np.exp(drift + diffusion)
            if not np.isfinite(terminal_prices).all():
                raise ValueError("Monte Carlo simulation exceeds the floating-point range")

            if contract.option_type is OptionType.CALL:
                payoff = np.maximum(terminal_prices - contract.strike_price, 0.0)
            else:
                payoff = np.maximum(contract.strike_price - terminal_prices, 0.0)

            discount_factor = math.exp(-market_data.risk_free_rate * contract.time_to_expiry)
            discounted_payoffs = discount_factor * payoff
            if self.use_control_variates:
                adjusted_payoffs, cv_report = _apply_pathwise_control_variates(
                    discounted_payoffs,
                    terminal_prices,
                    contract=contract,
                    market_data=market_data,
                    volatility=volatility,
                    antithetic=self.antithetic,
                )
            else:
                adjusted_payoffs = _antithetic_units(discounted_payoffs, antithetic=self.antithetic)
                cv_report = {
                    "cv_used": False,
                    "rho": None,
                    "beta": None,
                    "raw_var": float(np.var(adjusted_payoffs, ddof=1))
                    if adjusted_payoffs.size > 1
                    else 0.0,
                    "residual_var": float(np.var(adjusted_payoffs, ddof=1))
                    if adjusted_payoffs.size > 1
                    else 0.0,
                    "independent_units": int(adjusted_payoffs.size),
                }
            theoretical_price = float(np.mean(adjusted_payoffs))
            if not math.isfinite(theoretical_price):
                raise ValueError("Monte Carlo estimator produced a non-finite price")

            greeks_ci: dict[str, dict[str, float]] = {}
            greeks_meta: dict[str, dict[str, object]] = {}
            greek_values: dict[str, float] = {}

            price_guard = max(0.0, theoretical_price)
            vr_pipeline = "cv" if bool(cv_report["cv_used"]) else "plain"
            bs_reference: dict[str, float] | None = None

            def _ensure_bs_reference() -> dict[str, float]:
                nonlocal bs_reference
                if bs_reference is not None:
                    return bs_reference
                if contract.exercise_style is not ExerciseStyle.EUROPEAN:
                    bs_reference = {}
                    return bs_reference
                try:
                    _, d_ref, g_ref, t_ref, v_ref, r_ref = _black_scholes_greeks(
                        contract, market_data, volatility
                    )
                except Exception:  # pragma: no cover - safeguard
                    bs_reference = {}
                else:
                    bs_reference = {
                        "delta": float(d_ref),
                        "gamma": float(g_ref),
                        "theta": float(t_ref),
                        "vega": float(v_ref),
                        "rho": float(r_ref),
                    }
                return bs_reference

            def _register_greek(
                name: str,
                method: str,
                summary: GreekSummary,
                fallback_factory: Callable[[], GreekSummary] | None = None,
                value_transform: Callable[[GreekSummary], float] | None = None,
            ) -> None:
                transform = value_transform or (lambda item: float(item.value))
                fallback_summary: GreekSummary | None = None
                use_summary = summary
                if fallback_factory is not None:
                    needs_fallback = not contributions_finite(
                        summary.contributions
                    ) or is_estimate_unstable(
                        transform(summary), summary.half_width_abs, price_guard
                    )
                    if needs_fallback:
                        fallback_summary = fallback_factory()
                        use_summary = fallback_summary
                final_value = transform(use_summary)
                greek_values[name] = final_value
                greeks_ci[name] = {
                    "standard_error": float(use_summary.standard_error),
                    "half_width_abs": float(use_summary.half_width_abs),
                }
                meta: dict[str, object] = {
                    "method": method if fallback_summary is None else "fd",
                    "paths_used": int(use_summary.contributions.size),
                    "simulation_paths": simulation_paths,
                    "vr_pipeline": vr_pipeline,
                    "fallback": None if fallback_summary is None else "fd",
                }
                if fallback_summary is not None:
                    meta["primary_method"] = method
                    reference = _ensure_bs_reference()
                    if name in reference:
                        reference_value = reference[name]
                        denominator = max(abs(reference_value), 1e-4)
                        rel_error = abs(final_value - reference_value) / denominator
                        meta["fd_rel_error"] = float(rel_error)
                        if rel_error > 0.1:
                            meta["unstable_fd"] = True
                greeks_meta[name] = meta

            def _independent_summary(contributions: np.ndarray) -> GreekSummary:
                return aggregate_statistics(
                    _antithetic_units(contributions, antithetic=self.antithetic)
                )

            def _collapse_summary(summary: GreekSummary) -> GreekSummary:
                return _independent_summary(summary.contributions)

            delta_summary = _independent_summary(
                pathwise_delta(
                    contract,
                    market_data,
                    discount_factor=discount_factor,
                    terminal_prices=terminal_prices,
                )
            )
            _register_greek(
                "delta",
                "pathwise",
                delta_summary,
                lambda: _collapse_summary(
                    finite_difference_delta(
                        contract,
                        market_data,
                        volatility=volatility,
                        time_to_expiry=contract.time_to_expiry,
                        draws=draws,
                        discounted_payoffs=discounted_payoffs,
                    )
                ),
            )

            gamma_summary = _independent_summary(
                pathwise_gamma(
                    contract,
                    market_data,
                    discount_factor=discount_factor,
                    terminal_prices=terminal_prices,
                    volatility=volatility,
                    time_to_expiry=contract.time_to_expiry,
                )
            )
            _register_greek(
                "gamma",
                "pathwise_lr",
                gamma_summary,
                lambda: _collapse_summary(
                    finite_difference_gamma(
                        contract,
                        market_data,
                        volatility=volatility,
                        time_to_expiry=contract.time_to_expiry,
                        draws=draws,
                        discounted_payoffs=discounted_payoffs,
                    )
                ),
            )

            vega_summary = _independent_summary(
                pathwise_vega(
                    contract,
                    market_data,
                    discount_factor=discount_factor,
                    terminal_prices=terminal_prices,
                    volatility=volatility,
                    time_to_expiry=contract.time_to_expiry,
                    draws=draws,
                )
            )
            _register_greek(
                "vega",
                "pathwise",
                vega_summary,
                lambda: _collapse_summary(
                    finite_difference_vega(
                        contract,
                        market_data,
                        volatility=volatility,
                        time_to_expiry=contract.time_to_expiry,
                        draws=draws,
                        discounted_payoffs=discounted_payoffs,
                    )
                ),
            )

            theta_summary = _independent_summary(
                theta_likelihood_ratio(
                    contract,
                    market_data,
                    payoff=payoff,
                    discount_factor=discount_factor,
                    terminal_prices=terminal_prices,
                    volatility=volatility,
                    time_to_expiry=contract.time_to_expiry,
                )
            )
            _register_greek(
                "theta",
                "lr",
                theta_summary,
                lambda: _collapse_summary(
                    finite_difference_theta(
                        contract,
                        market_data,
                        volatility=volatility,
                        time_to_expiry=contract.time_to_expiry,
                        draws=draws,
                        discounted_payoffs=discounted_payoffs,
                    )
                ),
            )

            rho_summary = _independent_summary(
                rho_likelihood_ratio(
                    contract,
                    market_data,
                    payoff=payoff,
                    discount_factor=discount_factor,
                    terminal_prices=terminal_prices,
                    volatility=volatility,
                    time_to_expiry=contract.time_to_expiry,
                )
            )
            _register_greek(
                "rho",
                "lr",
                rho_summary,
                lambda: _collapse_summary(
                    finite_difference_rho(
                        contract,
                        market_data,
                        volatility=volatility,
                        time_to_expiry=contract.time_to_expiry,
                        draws=draws,
                        discounted_payoffs=discounted_payoffs,
                    )
                ),
            )

            delta_estimate = greek_values.get("delta")
            gamma_estimate = greek_values.get("gamma")
            vega_estimate = greek_values.get("vega")
            theta_estimate = greek_values.get("theta")
            rho_estimate = greek_values.get("rho")

            standard_error: float | None = None
            confidence_interval: tuple[float, float] | None = None
            independent_count = int(adjusted_payoffs.size)
            if independent_count > 1:
                sample_std = float(np.std(adjusted_payoffs, ddof=1))
                standard_error = sample_std / math.sqrt(independent_count)
                z_score = norm.ppf(0.975)  # 95% CI
                half_width = z_score * standard_error
                confidence_interval = (
                    theoretical_price - half_width,
                    theoretical_price + half_width,
                )

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            capsule: ReplayCapsule | None = None
            if sequence is not None:
                try:
                    capsule = build_replay_capsule(
                        seed_sequence=sequence,
                        model_name="monte_carlo",
                        model_config={
                            "paths": simulation_paths,
                            "antithetic": self.antithetic,
                            "use_control_variates": self.use_control_variates,
                        },
                        request={
                            "contract": _contract_payload(contract),
                            "market_data": _market_payload(market_data),
                            "volatility": volatility,
                        },
                    )
                except (TypeError, ValueError):
                    capsule = None

            result = PricingResult(
                contract_id=contract.contract_id,
                theoretical_price=max(0.0, theoretical_price),
                delta=delta_estimate,
                gamma=gamma_estimate,
                theta=theta_estimate,
                vega=vega_estimate,
                rho=rho_estimate,
                computation_time_ms=elapsed_ms,
                model_used=f"monte_carlo_{simulation_paths}",
                implied_volatility=volatility,
                standard_error=standard_error,
                confidence_interval=confidence_interval,
                capsule_id=capsule.capsule_id if capsule else None,
                replay_capsule=capsule,
                control_variate_report=cv_report,
                ci_greeks=greeks_ci,
                greeks_meta=greeks_meta,
            )
            return result
        except Exception:  # pragma: no cover - preserve context for API error mapping
            LOGGER.exception("Monte Carlo pricing failed")
            raise


def _default_basis_factories() -> dict[str, Sequence[Callable[[np.ndarray], np.ndarray]]]:
    """Return a dictionary describing the default LSMC basis candidates."""

    def _constant(_: np.ndarray) -> np.ndarray:
        return np.ones_like(_, dtype=float)

    def _identity(x: np.ndarray) -> np.ndarray:
        return x

    def _square(x: np.ndarray) -> np.ndarray:
        return x**2

    def _cube(x: np.ndarray) -> np.ndarray:
        return x**3

    def _log(x: np.ndarray) -> np.ndarray:
        return np.asarray(np.log(np.maximum(x, 1e-12)), dtype=float)

    def _sqrt(x: np.ndarray) -> np.ndarray:
        return np.asarray(np.sqrt(np.maximum(x, 0.0)), dtype=float)

    return {
        "polynomial_2": (_constant, _identity, _square),
        "polynomial_3": (_constant, _identity, _square, _cube),
        "log_linear": (_constant, _identity, _log),
        "sqrt_polynomial": (_constant, _sqrt, _identity, _square),
    }


def _build_design_matrix(
    basis: Sequence[Callable[[np.ndarray], np.ndarray]], values: np.ndarray
) -> np.ndarray:
    """Evaluate the provided basis functions and build the design matrix."""
    columns: list[np.ndarray] = []
    for function in basis:
        evaluated = function(values)
        if evaluated.ndim != 1:
            evaluated = np.asarray(evaluated, dtype=float).reshape(-1)
        columns.append(np.asarray(evaluated, dtype=float))
    design = np.column_stack(columns)
    if design.shape[0] != values.size or not np.isfinite(design).all():
        raise ValueError("basis functions must return finite values matching the input size")
    return design


def _information_criteria(rss: float, n: int, k: int) -> tuple[float, float]:
    """Compute the AIC and BIC for a linear regression fit."""
    if n <= k or n == 0:
        return float("inf"), float("inf")
    variance = max(rss / n, 1e-16)
    log_likelihood = -0.5 * n * (math.log(2.0 * math.pi) + math.log(variance) + 1.0)
    aic = 2.0 * k - 2.0 * log_likelihood
    bic = math.log(n) * k - 2.0 * log_likelihood
    return float(aic), float(bic)


def _kfold_indices(sample_size: int, folds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Generate shuffled k-fold indices."""
    folds = max(2, min(sample_size, folds))
    indices = np.arange(sample_size)
    rng.shuffle(indices)
    return [fold for fold in np.array_split(indices, folds) if fold.size > 0]


def _fit_linear_regression(design: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a linear regression returning coefficients and residual sum of squares."""
    coefficients, residuals, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if residuals.size:
        rss = float(residuals[0])
    else:
        predictions = design @ coefficients
        rss = float(np.sum((targets - predictions) ** 2))
    if rank < design.shape[1]:
        rss = float("inf")
    return coefficients, rss


@dataclass(slots=True)
class BasisMetrics:
    """Diagnostics describing a fitted basis at a single exercise date."""

    name: str
    coefficients: np.ndarray
    rss: float
    aic: float
    bic: float
    cv_rmse: float


@dataclass(slots=True)
class ExercisePolicyStep:
    """Summary of the extracted early exercise policy at a given time step."""

    time_index: int
    time: float
    basis: BasisMetrics | None
    in_the_money: int
    exercised: int
    exercise_fraction: float
    exercise_spot_mean: float | None


@dataclass(slots=True)
class LSMCAnalysis:
    """Container for diagnostics returned by the Longstaff-Schwartz model."""

    pricing_result: PricingResult
    policy: np.ndarray
    policy_steps: list[ExercisePolicyStep]
    basis_diagnostics: list[list[BasisMetrics]]
    reference_price: float
    reference_model_used: str
    price_diff_bps: float


@dataclass(slots=True)
class LongstaffSchwartzModel:
    """American option pricing using the Longstaff-Schwartz method with diagnostics."""

    MAX_PATHS: ClassVar[int] = MAX_LSMC_PATHS
    MAX_STEPS: ClassVar[int] = MAX_LSMC_STEPS
    MAX_WORK_ITEMS: ClassVar[int] = MAX_LSMC_WORK_ITEMS

    paths: int = 80_000
    steps: int = 60
    cv_folds: int = 5
    antithetic: bool = True
    seed_sequence: SeedSequence | None = None
    basis_factories: dict[str, Sequence[Callable[[np.ndarray], np.ndarray]]] | None = None
    reference_steps: int = 2_000

    def __post_init__(self) -> None:
        self.paths = _bounded_integer(
            "paths",
            self.paths,
            minimum=1,
            maximum=self.MAX_PATHS,
        )
        self.steps = _bounded_integer(
            "steps",
            self.steps,
            minimum=1,
            maximum=self.MAX_STEPS,
        )
        self.cv_folds = _bounded_integer(
            "cv_folds",
            self.cv_folds,
            minimum=2,
            maximum=MAX_CV_FOLDS,
        )
        self.antithetic = _require_boolean("antithetic", self.antithetic)
        self.seed_sequence = _validate_seed_sequence("seed_sequence", self.seed_sequence)
        self.reference_steps = _bounded_integer(
            "reference_steps",
            self.reference_steps,
            minimum=2,
            maximum=MAX_BINOMIAL_STEPS,
        )
        _validate_lsmc_workload(
            steps=self.steps,
            paths=self.paths,
            antithetic=self.antithetic,
        )
        if self.basis_factories is not None:
            if not isinstance(self.basis_factories, dict) or not self.basis_factories:
                raise ValueError("basis_factories must be a non-empty dictionary or None")
            if len(self.basis_factories) > 16:
                raise ValueError("at most 16 LSMC basis families are supported")
            normalized_factories: dict[str, Sequence[Callable[[np.ndarray], np.ndarray]]] = {}
            for name, functions in self.basis_factories.items():
                if not isinstance(name, str) or not name or len(name) > 64:
                    raise ValueError("basis names must contain between 1 and 64 characters")
                if (
                    isinstance(functions, (str, bytes))
                    or not isinstance(functions, Sequence)
                    or not 1 <= len(functions) <= 16
                    or not all(callable(function) for function in functions)
                ):
                    raise ValueError("each basis family must contain 1 to 16 callables")
                normalized_factories[name] = tuple(functions)
            self.basis_factories = normalized_factories

    def _prepare_paths(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        rng: Generator,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Simulate price paths under the risk-neutral measure."""
        time_to_expiry = contract.time_to_expiry
        step_count = self.steps
        dt = time_to_expiry / step_count
        sqrt_dt = math.sqrt(dt)

        path_count = self.paths
        if self.antithetic:
            path_count += path_count % 2
            half = path_count // 2
            base_draws = rng.standard_normal((step_count, half))
            draws = np.concatenate([base_draws, -base_draws], axis=1)
        else:
            draws = rng.standard_normal((step_count, path_count))
            path_count = draws.shape[1]

        prices = np.empty((step_count + 1, path_count), dtype=float)
        prices[0, :] = market_data.spot_price

        drift = (market_data.risk_free_rate - market_data.dividend_yield - 0.5 * volatility**2) * dt
        diffusion = volatility * sqrt_dt

        for index in range(1, step_count + 1):
            shock = diffusion * draws[index - 1, :]
            with np.errstate(over="ignore", invalid="ignore"):
                prices[index, :] = prices[index - 1, :] * np.exp(drift + shock)
            if not np.isfinite(prices[index, :]).all():
                raise ValueError("LSMC simulation exceeds the supported floating-point range")

        times = np.linspace(0.0, time_to_expiry, step_count + 1)
        discount = math.exp(-market_data.risk_free_rate * dt)
        return prices, times, discount

    def _intrinsic_value(self, contract: OptionContract, prices: np.ndarray) -> np.ndarray:
        """Return intrinsic values for the provided price vector."""
        if contract.option_type is OptionType.CALL:
            return np.maximum(prices - contract.strike_price, 0.0)
        return np.maximum(contract.strike_price - prices, 0.0)

    def _evaluate_basis(
        self,
        basis_name: str,
        basis_functions: Sequence[Callable[[np.ndarray], np.ndarray]],
        features: np.ndarray,
        targets: np.ndarray,
        folds: int,
        rng: np.random.Generator,
    ) -> BasisMetrics:
        """Fit a regression basis computing AIC/BIC and cross-validation RMSE."""
        design = _build_design_matrix(basis_functions, features)
        coefficients, rss = _fit_linear_regression(design, targets)

        sample_size = features.size
        parameters = design.shape[1]
        aic, bic = _information_criteria(rss, sample_size, parameters)

        cv_indices = _kfold_indices(sample_size, folds, rng)
        sq_errors: list[float] = []
        for fold in cv_indices:
            train_mask = np.ones(sample_size, dtype=bool)
            train_mask[fold] = False
            if not train_mask.any():
                continue
            train_design = design[train_mask]
            train_targets = targets[train_mask]
            test_design = design[~train_mask]
            test_targets = targets[~train_mask]
            if train_design.size == 0 or test_design.size == 0:
                continue
            fold_coefficients, _ = _fit_linear_regression(train_design, train_targets)
            predictions = test_design @ fold_coefficients
            sq_errors.append(float(np.mean((test_targets - predictions) ** 2)))

        if sq_errors:
            cv_rmse = float(math.sqrt(max(0.0, float(np.mean(sq_errors)))))
        else:
            cv_rmse = float("inf")

        return BasisMetrics(
            name=basis_name,
            coefficients=coefficients,
            rss=rss,
            aic=aic,
            bic=bic,
            cv_rmse=cv_rmse,
        )

    def price_with_diagnostics(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        seed_sequence: SeedSequence | None = None,
    ) -> LSMCAnalysis:
        """Run the Longstaff-Schwartz algorithm returning diagnostics."""
        if contract.exercise_style is not ExerciseStyle.AMERICAN:
            raise ValueError("Longstaff-Schwartz model requires an American option contract")

        validate_pricing_parameters(contract, market_data, volatility)

        start = time.perf_counter()

        if (
            contract.option_type is OptionType.CALL
            and market_data.dividend_yield <= 0.0
            and market_data.risk_free_rate >= 0.0
        ):
            european_contract = OptionContract(
                symbol=contract.symbol,
                strike_price=contract.strike_price,
                time_to_expiry=contract.time_to_expiry,
                option_type=contract.option_type,
                exercise_style=ExerciseStyle.EUROPEAN,
            )
            exact = BlackScholesModel().calculate_price(european_contract, market_data, volatility)
            exact.contract_id = contract.contract_id
            exact.model_used = "american_call_no_early_exercise"
            path_count = self.paths + (self.paths % 2) if self.antithetic else self.paths
            policy = np.zeros((self.steps + 1, path_count), dtype=bool)
            return LSMCAnalysis(
                pricing_result=exact,
                policy=policy,
                policy_steps=[],
                basis_diagnostics=[],
                reference_price=exact.theoretical_price,
                reference_model_used="black_scholes_no_early_exercise_theorem",
                price_diff_bps=0.0,
            )

        basis_factories = self.basis_factories or _default_basis_factories()
        if not basis_factories:
            raise ValueError("At least one basis must be provided for LSMC")

        sequence = _validate_seed_sequence("seed_sequence", seed_sequence) or self.seed_sequence
        rng = _thread_local_generator(sequence)
        diagnostic_rng = np.random.default_rng(42)

        prices, times, discount = self._prepare_paths(contract, market_data, volatility, rng)
        step_count, path_count = prices.shape[0] - 1, prices.shape[1]

        intrinsic_maturity = self._intrinsic_value(contract, prices[-1, :])
        cashflows = intrinsic_maturity.copy()

        policy = np.zeros((step_count + 1, path_count), dtype=bool)
        policy[-1, :] = intrinsic_maturity > 0.0

        basis_diagnostics: list[list[BasisMetrics]] = []
        policy_steps: list[ExercisePolicyStep] = []

        strike = contract.strike_price

        for step in range(step_count - 1, -1, -1):
            spot = prices[step, :]
            intrinsic = self._intrinsic_value(contract, spot)
            in_the_money = intrinsic > 0.0

            continuation = discount * cashflows
            evaluated_bases: list[BasisMetrics] = []
            selected: BasisMetrics | None = None

            if np.any(in_the_money):
                features = spot[in_the_money] / strike
                targets = continuation[in_the_money]
                for name, basis_functions in basis_factories.items():
                    try:
                        metrics = self._evaluate_basis(
                            name, basis_functions, features, targets, self.cv_folds, diagnostic_rng
                        )
                    except (np.linalg.LinAlgError, FloatingPointError, ValueError):
                        continue
                    evaluated_bases.append(metrics)

                basis_diagnostics.append(evaluated_bases)

                if evaluated_bases:
                    valid_bases = [
                        metrics
                        for metrics in evaluated_bases
                        if math.isfinite(metrics.cv_rmse)
                        and math.isfinite(metrics.aic)
                        and math.isfinite(metrics.bic)
                    ]
                    if valid_bases:
                        valid_bases.sort(key=lambda m: (m.cv_rmse, m.bic))
                        selected = valid_bases[0]
            else:
                basis_diagnostics.append([])

            exercised_paths = np.zeros(path_count, dtype=bool)
            exercise_mean: float | None = None

            if selected is not None:
                basis_functions = basis_factories[selected.name]
                features = spot[in_the_money] / strike
                targets = continuation[in_the_money]
                design = _build_design_matrix(basis_functions, features)
                predictions = np.empty_like(targets)
                itm_indices = np.flatnonzero(in_the_money)
                if self.antithetic:
                    pair_count = path_count // 2
                    unit_indices = np.where(
                        itm_indices < pair_count,
                        itm_indices,
                        itm_indices - pair_count,
                    )
                else:
                    unit_indices = itm_indices
                fold_count = min(
                    self.cv_folds,
                    max(2, targets.size // max(2 * design.shape[1], 1)),
                )
                labels = unit_indices % fold_count
                for held_out in range(fold_count):
                    train = labels != held_out
                    test = ~train
                    if not np.any(test):
                        continue
                    if int(np.count_nonzero(train)) < design.shape[1]:
                        predictions[test] = float("inf")
                        continue
                    coefficients, _ = _fit_linear_regression(design[train], targets[train])
                    predictions[test] = design[test] @ coefficients
                predictions = np.maximum(predictions, 0.0)
                exercise_region = intrinsic[in_the_money] >= predictions

                in_money_indices = np.flatnonzero(in_the_money)
                exercised_indices = in_money_indices[exercise_region]
                exercised_paths[exercised_indices] = True
                if exercised_indices.size:
                    exercise_mean = float(np.mean(spot[exercised_indices]))

            policy[step, exercised_paths] = True

            exercise_count = int(np.count_nonzero(exercised_paths))
            if exercise_count:
                cashflows[exercised_paths] = intrinsic[exercised_paths]
            cashflows[~exercised_paths] = continuation[~exercised_paths]

            policy_steps.append(
                ExercisePolicyStep(
                    time_index=step,
                    time=times[step],
                    basis=selected,
                    in_the_money=int(np.count_nonzero(in_the_money)),
                    exercised=exercise_count,
                    exercise_fraction=(exercise_count / path_count) if path_count else 0.0,
                    exercise_spot_mean=exercise_mean,
                )
            )

        independent_cashflows = _antithetic_units(cashflows, antithetic=self.antithetic)
        european_contract = OptionContract(
            symbol=contract.symbol,
            strike_price=contract.strike_price,
            time_to_expiry=contract.time_to_expiry,
            option_type=contract.option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
        )
        european_reference = BlackScholesModel().calculate_price(
            european_contract, market_data, volatility
        )
        discounted_european = intrinsic_maturity * math.exp(
            -market_data.risk_free_rate * contract.time_to_expiry
        )
        independent_european = _antithetic_units(discounted_european, antithetic=self.antithetic)
        independent_cashflows, cv_report = _cross_fitted_control_variate(
            independent_cashflows,
            independent_european,
            control_mean=european_reference.theoretical_price,
            folds=self.cv_folds,
        )

        raw_price = float(np.mean(independent_cashflows))
        intrinsic_now = float(
            self._intrinsic_value(contract, np.array([market_data.spot_price], dtype=float))[0]
        )
        lower_bound = max(intrinsic_now, european_reference.theoretical_price)
        upper_bound = (
            market_data.spot_price
            * math.exp(
                max(
                    -market_data.dividend_yield * contract.time_to_expiry,
                    0.0,
                )
            )
            if contract.option_type is OptionType.CALL
            else contract.strike_price
            * math.exp(
                max(
                    -market_data.risk_free_rate * contract.time_to_expiry,
                    0.0,
                )
            )
        )
        price = min(max(raw_price, lower_bound), upper_bound)
        independent_count = independent_cashflows.size
        if independent_count > 1:
            std_err = float(np.std(independent_cashflows, ddof=1) / math.sqrt(independent_count))
        else:
            std_err = 0.0
        raw_interval = (raw_price - 1.96 * std_err, raw_price + 1.96 * std_err)
        confidence_interval = (
            min(max(raw_interval[0], lower_bound), upper_bound),
            min(max(raw_interval[1], lower_bound), upper_bound),
        )
        cv_report = dict(cv_report)
        cv_report.update(
            {
                "raw_price": raw_price,
                "projected_price": price,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "projection_applied": price != raw_price,
            }
        )

        binomial_model = BinomialModel(steps=self.reference_steps)
        reference = binomial_model.calculate_price(contract, market_data, volatility)
        reference_price = reference.theoretical_price
        price_diff_bps = (price - reference_price) / max(reference_price, 1e-12) * 10_000.0

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        pricing_result = PricingResult(
            contract_id=contract.contract_id,
            theoretical_price=max(0.0, price),
            delta=reference.delta,
            gamma=reference.gamma,
            theta=reference.theta,
            vega=reference.vega,
            rho=reference.rho,
            implied_volatility=volatility,
            computation_time_ms=elapsed_ms,
            model_used=f"lsmc_{path_count}x{step_count}",
            standard_error=std_err,
            confidence_interval=confidence_interval,
            control_variate_report=cv_report,
        )

        policy_steps.reverse()

        return LSMCAnalysis(
            pricing_result=pricing_result,
            policy=policy,
            policy_steps=policy_steps,
            basis_diagnostics=basis_diagnostics[::-1],
            reference_price=reference_price,
            reference_model_used=reference.model_used,
            price_diff_bps=float(price_diff_bps),
        )

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        seed_sequence: SeedSequence | None = None,
    ) -> PricingResult:
        """Return a :class:`PricingResult` for the LSMC model."""
        analysis = self.price_with_diagnostics(
            contract, market_data, volatility, seed_sequence=seed_sequence
        )
        return analysis.pricing_result


def replay_pricing_capsule(capsule: ReplayCapsule) -> PricingResult:
    """Re-run a pricing request described by ``capsule``."""
    if not isinstance(capsule, ReplayCapsule):
        raise TypeError("capsule must be a ReplayCapsule")
    if not capsule.verify_integrity():
        raise ValueError("Replay capsule integrity check failed")
    payload = capsule.payload
    unknown_top_level = set(payload) - {"model", "request", "seed", "surface_id"}
    if unknown_top_level:
        raise ValueError(
            f"Replay capsule contains unsupported entries: {sorted(unknown_top_level)}"
        )
    model_info = payload.get("model", {})
    request_info = payload.get("request", {})
    if not isinstance(model_info, dict) or not isinstance(request_info, dict):
        raise ValueError("Replay capsule model and request entries must be mappings")
    model_name = model_info.get("name")
    config = model_info.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("Replay capsule model config must be a mapping")
    if set(model_info) - {"name", "config"}:
        raise ValueError("Replay capsule model entry contains unsupported fields")
    if set(request_info) - {"contract", "market_data", "volatility"}:
        raise ValueError("Replay capsule request entry contains unsupported fields")

    contract_info = request_info.get("contract") or {}
    market_info = request_info.get("market_data") or {}
    if not isinstance(contract_info, dict) or not isinstance(market_info, dict):
        raise ValueError("Replay capsule contract and market data must be mappings")
    volatility_raw = request_info.get("volatility")
    if isinstance(volatility_raw, bool) or not isinstance(volatility_raw, Real):
        raise ValueError("Invalid volatility in replay capsule")
    volatility = float(volatility_raw)

    if set(contract_info) - {
        "symbol",
        "strike_price",
        "time_to_expiry",
        "option_type",
        "exercise_style",
        "contract_id",
    }:
        raise ValueError("Replay capsule contract contains unsupported fields")
    if set(market_info) - {"spot_price", "risk_free_rate", "dividend_yield"}:
        raise ValueError("Replay capsule market data contains unsupported fields")

    symbol = contract_info.get("symbol")
    strike = contract_info.get("strike_price")
    expiry = contract_info.get("time_to_expiry")
    option_type = contract_info.get("option_type")
    exercise_style = contract_info.get("exercise_style")
    contract_id = contract_info.get("contract_id", "")
    if (
        not isinstance(symbol, str)
        or isinstance(strike, bool)
        or not isinstance(strike, Real)
        or isinstance(expiry, bool)
        or not isinstance(expiry, Real)
        or not isinstance(option_type, str)
        or not isinstance(exercise_style, str)
        or not isinstance(contract_id, str)
    ):
        raise ValueError("Invalid contract parameters in replay capsule")

    try:
        contract = OptionContract(
            symbol=symbol,
            strike_price=float(strike),
            time_to_expiry=float(expiry),
            option_type=OptionType(option_type),
            exercise_style=ExerciseStyle(exercise_style),
            contract_id=contract_id,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid contract parameters in replay capsule") from exc

    spot = market_info.get("spot_price")
    rate = market_info.get("risk_free_rate")
    dividend = market_info.get("dividend_yield")

    def _replay_real(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("Invalid market data in replay capsule")
        return float(value)

    spot_value = _replay_real(spot)
    rate_value = _replay_real(rate)
    dividend_value = _replay_real(dividend)
    try:
        market = MarketData(
            spot_price=spot_value,
            risk_free_rate=rate_value,
            dividend_yield=dividend_value,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid market data in replay capsule") from exc

    seed_sequence = capsule.resolve_seed_sequence()

    if model_name == "monte_carlo":
        if set(config) - {"paths", "antithetic", "use_control_variates"}:
            raise ValueError("Replay capsule model config contains unsupported fields")
        if seed_sequence is None:
            raise ValueError("Monte Carlo replay capsule does not contain a valid seed")
        paths = _bounded_integer(
            "capsule paths",
            config.get("paths", MonteCarloModel.DEFAULT_PATHS),
            minimum=1,
            maximum=MAX_MONTE_CARLO_PATHS,
        )
        antithetic = _require_boolean(
            "capsule antithetic",
            config.get("antithetic", MonteCarloModel.DEFAULT_ANTITHETIC),
        )
        use_control_variates = _require_boolean(
            "capsule use_control_variates",
            config.get("use_control_variates", True),
        )
        model = MonteCarloModel(
            paths=paths,
            antithetic=antithetic,
            use_control_variates=use_control_variates,
        )
        return model.calculate_price(contract, market, volatility, seed_sequence=seed_sequence)

    raise ValueError(f"Replay is not supported for model '{model_name}'")
