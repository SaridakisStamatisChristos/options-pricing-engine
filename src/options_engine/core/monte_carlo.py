"""Terminal-payoff Monte Carlo valuation."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.random import SeedSequence

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
from ..utils.validation import reject_discrete_dividends, validate_pricing_parameters
from .black_scholes import _black_scholes_greeks
from .models import ExerciseStyle, MarketData, OptionContract, OptionType, PricingResult
from .pricing_common import (
    MAX_MONTE_CARLO_PATHS,
    _antithetic_units,
    _apply_pathwise_control_variates,
    _bounded_integer,
    _contract_payload,
    _market_payload,
    _require_boolean,
    _thread_local_generator,
    _validate_seed_sequence,
)
from .replay import ReplayCapsule, build_replay_capsule
from .statistical_inference import estimate_mean

LOGGER = logging.getLogger(__name__)


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
            reject_discrete_dividends(market_data, "terminal-payoff Monte Carlo")

            # At least two independent units are required to estimate sampling
            # variance. The constructor retains the historical ``paths=1``
            # input, but execution resolves it to two ordinary draws (or two
            # antithetic pairs below) and reports that resolved count.
            simulation_paths = max(2, self.paths)

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
            price_inference = estimate_mean(adjusted_payoffs, lower_bound=0.0)

            greeks_ci: dict[str, dict[str, float]] = {}
            greeks_meta: dict[str, dict[str, object]] = {}
            greek_values: dict[str, float] = {}

            price_guard = price_inference.bounded_estimate
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

            standard_error = price_inference.standard_error
            confidence_interval = price_inference.confidence_interval

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
                theoretical_price=price_inference.bounded_estimate,
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
                estimate_diagnostics=price_inference.diagnostics(
                    estimator="terminal_payoff_sample_mean"
                ),
                ci_greeks=greeks_ci,
                greeks_meta=greeks_meta,
            )
            return result
        except Exception:  # pragma: no cover - preserve context for API error mapping
            LOGGER.exception("Monte Carlo pricing failed")
            raise


__all__ = ["MonteCarloModel"]
