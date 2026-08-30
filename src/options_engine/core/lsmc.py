"""Cross-fitted Longstaff-Schwartz American option valuation."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.random import Generator, SeedSequence

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
from .black_scholes import BlackScholesModel, black_scholes_price
from .crr import BinomialModel
from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from .pricing_common import (
    MAX_BINOMIAL_STEPS,
    MAX_CV_FOLDS,
    MAX_LSMC_PATHS,
    MAX_LSMC_STEPS,
    MAX_LSMC_WORK_ITEMS,
    MAX_RANDOM_SEED,
    PriceResult,
    _antithetic_units,
    _bounded_integer,
    _build_capsule_id,
    _cross_fitted_control_variate,
    _normalise_option_type,
    _require_boolean,
    _thread_local_generator,
    _validate_lsmc_workload,
    _validate_seed_sequence,
)
from .statistical_inference import estimate_mean


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

    lower_tail, lower_flag = deep_itm_policy(spot, strike, opt)
    lower_tail = max(lower_tail or 0.0, bs_reference.price)
    upper_tail, upper_flag = deep_otm_upper_bound(spot, strike, opt, tau=tau, r=r, q=q)

    inference = estimate_mean(
        effective_cashflows,
        lower_bound=lower_tail,
        upper_bound=upper_tail,
    )
    original_price = inference.raw_estimate
    price = inference.bounded_estimate
    standard_error = inference.standard_error
    ci_half_width = inference.raw_half_width
    if ci_half_width is None:  # Defensive: workload admission guarantees >= 2 units.
        raise RuntimeError("LSMC uncertainty requires at least two independent sampling units")
    ci_half_width, precision_bucket, precision_limit, precision_flags = enforce_precision_policy(
        price, ci_half_width
    )
    policy_flags.extend(precision_flags)

    if inference.projection_applied:
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
            **inference.diagnostics(
                estimator="cross_fitted_lsmc_policy",
                raw_estimate_name="raw_policy_estimate",
            ),
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
        confidence_interval=inference.confidence_interval,
        estimate_diagnostics=inference.diagnostics(
            estimator="cross_fitted_lsmc_policy",
            raw_estimate_name="raw_policy_estimate",
        ),
    )


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

        path_count = (
            max(4, self.paths + (self.paths % 2)) if self.antithetic else max(2, self.paths)
        )
        if self.antithetic:
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
        inference = estimate_mean(
            independent_cashflows,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        price = inference.bounded_estimate
        std_err = inference.standard_error
        confidence_interval = inference.confidence_interval
        cv_report = dict(cv_report)
        cv_report.update(
            {
                **inference.diagnostics(
                    estimator="cross_fitted_lsmc_policy",
                    raw_estimate_name="raw_policy_estimate",
                ),
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
            estimate_diagnostics=inference.diagnostics(
                estimator="cross_fitted_lsmc_policy",
                raw_estimate_name="raw_policy_estimate",
            ),
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


__all__ = [
    "BasisMetrics",
    "ExercisePolicyStep",
    "LSMCAnalysis",
    "LongstaffSchwartzModel",
    "american_lsmc_price",
]
