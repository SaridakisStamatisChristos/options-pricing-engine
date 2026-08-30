from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.random import SeedSequence

from options_engine.core.models import MarketData, OptionContract, OptionType
from options_engine.core.replay import ReplayCapsule, build_replay_capsule
from options_engine.greeks.estimators import (
    aggregate_statistics,
    ensure_fd_inputs,
    finite_difference_delta,
    finite_difference_gamma,
    finite_difference_rho,
    finite_difference_theta,
    finite_difference_vega,
    pathwise_gamma,
    pathwise_vega,
    rho_likelihood_ratio,
    simulate_terminal_prices,
    theta_likelihood_ratio,
)
from options_engine.greeks.stability import (
    contributions_finite,
    guard_against_pathologies,
    is_estimate_unstable,
    standard_error,
)
from options_engine.utils.numerics import (
    apply_global_clamps,
    deep_itm_policy,
    deep_otm_upper_bound,
    enforce_precision_policy,
    laguerre_basis3,
    numerics_policy_hash,
    stable_regression,
)
from options_engine.utils.validation import validate_pricing_parameters


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((math.nan, 1.0, 1.0, 0.2, 0.0, 0.0), "finite"),
        ((0.0, 1.0, 1.0, 0.2, 0.0, 0.0), "spot"),
        ((1.0, 0.0, 1.0, 0.2, 0.0, 0.0), "strike"),
        ((1.0, 1.0, 0.0, 0.2, 0.0, 0.0), "tau"),
        ((1.0, 1.0, 1.0, 0.0, 0.0, 0.0), "sigma"),
        ((1.0, 1.0, 1.0, 0.2, 6.0, 0.0), "r"),
        ((1.0, 1.0, 1.0, 0.2, 0.0, -6.0), "q"),
    ],
)
def test_scalar_input_policy_rejects_instead_of_clamping(
    values: tuple[float, float, float, float, float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        apply_global_clamps(*values)


@pytest.mark.parametrize(
    ("price", "ci_half_width", "exception"),
    [
        (math.nan, 0.1, ValueError),
        (math.inf, 0.1, ValueError),
        (1.0, math.nan, ValueError),
        (1.0, math.inf, ValueError),
        (1.0, -0.1, ValueError),
        (True, 0.1, TypeError),
        (1.0, False, TypeError),
        ("1.0", 0.1, TypeError),
    ],
)
def test_precision_policy_rejects_malformed_intervals(
    price: object,
    ci_half_width: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        enforce_precision_policy(price, ci_half_width)  # type: ignore[arg-type]


def test_bound_policies_reject_malformed_contracts() -> None:
    with pytest.raises(ValueError, match="option_type"):
        deep_itm_policy(100.0, 90.0, "CALL")
    with pytest.raises(ValueError, match="finite"):
        deep_itm_policy(math.nan, 90.0, "call")
    with pytest.raises(ValueError, match="tau"):
        deep_otm_upper_bound(100.0, 90.0, "put", tau=-1.0)
    with pytest.raises(TypeError, match="real numbers"):
        deep_otm_upper_bound(100.0, 90.0, "put", q=True)


def test_regression_precision_and_bound_policies() -> None:
    matrix = laguerre_basis3(np.array([[0.5, 1.0], [1.5, 2.0]]))
    assert matrix.shape == (4, 4)

    empty_beta, used_ridge = stable_regression(np.empty((0, 2)), np.empty(0))
    assert empty_beta.size == 0
    assert used_ridge is False

    singular = np.ones((5, 3))
    beta, used_ridge = stable_regression(singular, np.arange(5.0))
    assert np.isfinite(beta).all()
    assert used_ridge is True

    assert enforce_precision_policy(10.0, 0.01)[1] == "tight"
    assert enforce_precision_policy(10.0, 0.1)[1] == "medium"
    assert enforce_precision_policy(10.0, 1.0)[1] == "loose"
    assert deep_itm_policy(120.0, 100.0, "call") == (20.0, "no_arbitrage_floor")
    assert deep_itm_policy(100.0, 120.0, "call") == (0.0, None)
    assert deep_otm_upper_bound(100.0, 90.0, "call", tau=2.0, q=-0.1)[0] == pytest.approx(
        100.0 * math.exp(0.2)
    )
    assert deep_otm_upper_bound(100.0, 90.0, "put", tau=2.0, r=-0.1)[0] == pytest.approx(
        90.0 * math.exp(0.2)
    )
    assert len(numerics_policy_hash()) == 16


def test_replay_capsule_round_trip_and_invalid_payloads() -> None:
    sequence = SeedSequence([1, 2, 3], spawn_key=(4, 5))
    capsule = build_replay_capsule(
        seed_sequence=sequence,
        model_name="monte_carlo",
        model_config={"paths": 128},
        request={"values": (2, 1)},
        surface_id="surface-1",
    )
    restored = capsule.resolve_seed_sequence()
    assert restored is not None
    assert restored.spawn_key == sequence.spawn_key
    assert capsule.to_json() == capsule.to_json()

    assert ReplayCapsule("0" * 64, {}).resolve_seed_sequence() is None
    assert ReplayCapsule("0" * 64, {"seed": {"entropy": "bad"}}).resolve_seed_sequence() is None
    array_entropy = build_replay_capsule(
        seed_sequence=SeedSequence(np.array([7, 8], dtype=np.uint32)),
        model_name="monte_carlo",
        model_config={"paths": 128},
        request={"value": 1},
    )
    assert array_entropy.resolve_seed_sequence() is not None
    with pytest.raises(ValueError, match="NaN"):
        build_replay_capsule(
            seed_sequence=None,
            model_name="x",
            model_config={},
            request={"bad": math.nan},
        )


def test_greek_estimators_and_finite_difference_harness() -> None:
    contract = OptionContract("GREEKS", 100.0, 1.0, OptionType.PUT)
    market = MarketData(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    draws = np.linspace(-2.5, 2.5, 2_001)
    terminal = simulate_terminal_prices(100.0, market, 0.25, 1.0, draws)
    payoff = np.maximum(100.0 - terminal, 0.0)
    discount = math.exp(-0.02)
    discounted = discount * payoff

    summaries = [
        finite_difference_delta(
            contract,
            market,
            volatility=0.25,
            time_to_expiry=1.0,
            draws=draws,
            discounted_payoffs=discounted,
        ),
        finite_difference_gamma(
            contract,
            market,
            volatility=0.25,
            time_to_expiry=1.0,
            draws=draws,
            discounted_payoffs=discounted,
        ),
        finite_difference_vega(
            contract,
            market,
            volatility=0.25,
            time_to_expiry=1.0,
            draws=draws,
            discounted_payoffs=discounted,
        ),
        finite_difference_theta(
            contract,
            market,
            volatility=0.25,
            time_to_expiry=1.0,
            draws=draws,
            discounted_payoffs=discounted,
        ),
        finite_difference_rho(
            contract,
            market,
            volatility=0.25,
            time_to_expiry=1.0,
            draws=draws,
            discounted_payoffs=discounted,
        ),
    ]
    assert all(math.isfinite(summary.value) for summary in summaries)
    assert all(summary.contributions.shape == draws.shape for summary in summaries)

    assert np.isfinite(
        pathwise_gamma(
            contract,
            market,
            discount_factor=discount,
            terminal_prices=terminal,
            volatility=0.25,
            time_to_expiry=1.0,
        )
    ).all()
    assert np.isfinite(
        pathwise_vega(
            contract,
            market,
            discount_factor=discount,
            terminal_prices=terminal,
            volatility=0.25,
            time_to_expiry=1.0,
            draws=draws,
        )
    ).all()
    assert np.isfinite(
        theta_likelihood_ratio(
            contract,
            market,
            payoff=payoff,
            discount_factor=discount,
            terminal_prices=terminal,
            volatility=0.25,
            time_to_expiry=1.0,
        )
    ).all()
    assert np.isfinite(
        rho_likelihood_ratio(
            contract,
            market,
            payoff=payoff,
            discount_factor=discount,
            terminal_prices=terminal,
            volatility=0.25,
            time_to_expiry=1.0,
        )
    ).all()

    assert aggregate_statistics(np.array([])).standard_error == math.inf
    one_unit = aggregate_statistics(np.array([1.0]))
    assert one_unit.standard_error == math.inf
    two_units = aggregate_statistics(np.array([1.0, 3.0]))
    assert two_units.standard_error == pytest.approx(1.0)
    assert two_units.half_width_abs > 12.0
    assert np.array_equal(
        simulate_terminal_prices(100.0, market, 0.2, 0.0, draws), np.full_like(draws, 100.0)
    )
    ensure_fd_inputs(
        contract,
        market,
        volatility=0.25,
        time_to_expiry=1.0,
        draws=draws,
        discounted_payoffs=discounted,
    )
    with pytest.raises(ValueError, match="must be finite"):
        ensure_fd_inputs(
            contract,
            market,
            volatility=0.25,
            time_to_expiry=1.0,
            draws=np.array([math.nan]),
            discounted_payoffs=np.array([0.0]),
        )


def test_stability_and_domain_validation_edges() -> None:
    assert standard_error(np.array([1.0])) == 0.0
    assert standard_error(np.array([1.0, math.inf])) == math.inf
    assert contributions_finite(np.array([1.0, 2.0]))
    assert not guard_against_pathologies([np.array([math.nan])])
    assert is_estimate_unstable(math.nan, 0.0, 1.0)
    assert not is_estimate_unstable(0.0, 1.0, 0.1)
    assert is_estimate_unstable(0.0, 1.0, 1.0)

    contract = OptionContract("VALID", 100.0, 1.0, OptionType.CALL)
    market = MarketData(spot_price=100.0, risk_free_rate=0.0)
    validate_pricing_parameters(contract, market, 0.2)
    with pytest.raises(ValueError, match="volatility"):
        validate_pricing_parameters(contract, market, 0.0)
    with pytest.raises(ValueError, match="volatility"):
        validate_pricing_parameters(contract, market, math.nan)
