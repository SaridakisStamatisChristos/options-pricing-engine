"""Regression tests for previously reproduced correctness failures."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.random import SeedSequence

from options_engine.calib.validators import NoArbitrageValidator
from options_engine.core.models import (
    ExerciseStyle,
    MarketData,
    OptionContract,
    OptionType,
)
from options_engine.core.pricing_engine import OptionsEngine
from options_engine.core.pricing_models import (
    BinomialModel,
    BlackScholesModel,
    MonteCarloModel,
    american_lsmc_price,
)


def test_black_scholes_theta_matches_calendar_time_difference() -> None:
    model = BlackScholesModel()
    market = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    contract = OptionContract("THETA", 100.0, 1.0, OptionType.CALL)
    theta = model.calculate_price(contract, market, 0.2).theta
    assert theta is not None

    bump = 1e-4
    shorter = OptionContract("THETA", 100.0, 1.0 - bump, OptionType.CALL)
    longer = OptionContract("THETA", 100.0, 1.0 + bump, OptionType.CALL)
    finite_difference = (
        model.calculate_price(shorter, market, 0.2).theoretical_price
        - model.calculate_price(longer, market, 0.2).theoretical_price
    ) / (2.0 * bump * 365.0)
    assert theta == pytest.approx(finite_difference, rel=1e-7, abs=1e-10)


def test_cache_identity_includes_style_and_random_stream() -> None:
    market = MarketData(spot_price=50.0, risk_free_rate=0.1)
    european = OptionContract("CACHE", 100.0, 1.0, OptionType.PUT, ExerciseStyle.EUROPEAN)
    american = OptionContract("CACHE", 100.0, 1.0, OptionType.PUT, ExerciseStyle.AMERICAN)
    assert european.contract_id != american.contract_id

    with OptionsEngine(num_threads=1) as engine:
        european_result = engine.price_option(
            european, market, "binomial_200", override_volatility=0.2
        )
        american_result = engine.price_option(
            american, market, "binomial_200", override_volatility=0.2
        )
        assert american_result["theoretical_price"] > european_result["theoretical_price"]
        assert american_result["cached"] is False

        call = OptionContract("SEED", 50.0, 1.0, OptionType.CALL)
        first = engine.price_option(
            call, market, "monte_carlo_20k", override_volatility=0.2, seed=1
        )
        second = engine.price_option(
            call, market, "monte_carlo_20k", override_volatility=0.2, seed=2
        )
        replay = engine.price_option(
            call, market, "monte_carlo_20k", override_volatility=0.2, seed=1
        )
        assert first["theoretical_price"] != second["theoretical_price"]
        assert replay["cached"] is True


def test_non_dividend_american_call_is_not_tail_capped() -> None:
    result = american_lsmc_price(
        100.0,
        300.0,
        5.0,
        1.0,
        0.0,
        0.0,
        "call",
        paths=2_000,
        seed=3,
    )
    assert result.price == pytest.approx(57.3268193636, rel=1e-9)
    assert result.ci_half_width == 0.0
    assert "ci_clipped" not in result.meta["policy_flags"]


def test_negative_rate_american_call_does_not_use_no_exercise_theorem() -> None:
    result = american_lsmc_price(
        spot=100.0,
        strike=80.0,
        tau=2.0,
        sigma=0.2,
        r=-0.08,
        q=0.0,
        option_type="call",
        steps=48,
        paths=20_000,
        seed=17,
    )

    assert result.meta["method"] == "american_lsmc"
    assert result.price >= 20.0


def test_strict_lsmc_runtime_diagnostics_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NUMERICS_STRICT", "1")
    result = american_lsmc_price(
        spot=95.0,
        strike=100.0,
        tau=0.75,
        sigma=0.3,
        r=0.02,
        q=0.0,
        option_type="put",
        steps=16,
        paths=2_000,
        seed=41,
    )

    runtime = result.meta["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["checks_enabled"] is True
    assert {"strike_convexity", "sigma_monotonic", "tau_monotonic"} <= runtime.keys()


def test_crr_refines_until_probability_is_arbitrage_free() -> None:
    contract = OptionContract("CRR", 100.0, 1.0, OptionType.CALL)
    market = MarketData(spot_price=100.0, risk_free_rate=0.1)
    tree = BinomialModel(steps=10).calculate_price(contract, market, 0.01)
    analytic = BlackScholesModel().calculate_price(contract, market, 0.01)
    assert tree.theoretical_price == pytest.approx(analytic.theoretical_price, rel=2e-4, abs=2e-4)
    assert tree.vega is not None and tree.vega >= -1e-10
    assert tree.model_used == "binomial_160"


def test_uneven_strike_convexity_uses_slopes() -> None:
    validator = NoArbitrageValidator()
    convex = validator._check_butterfly(
        1.0,
        np.array([99.0, 100.0, 120.0]),
        np.array([441.0, 400.0, 0.0]),
    )
    assert not [violation for violation in convex if violation.kind == "butterfly"]

    concave = validator._check_butterfly(
        1.0,
        np.array([90.0, 100.0, 120.0]),
        np.array([20.0, 15.0, 0.0]),
    )
    assert any(violation.kind == "butterfly" for violation in concave)


def test_model_exercise_compatibility_is_explicit() -> None:
    contract = OptionContract("STYLE", 100.0, 1.0, OptionType.PUT, ExerciseStyle.AMERICAN)
    market = MarketData(spot_price=100.0, risk_free_rate=0.02)
    with pytest.raises(ValueError, match="European exercise only"):
        BlackScholesModel().calculate_price(contract, market, 0.2)
    with pytest.raises(ValueError, match="European exercise only"):
        MonteCarloModel(paths=1_000).calculate_price(contract, market, 0.2)


def test_monte_carlo_control_is_not_an_option_payoff_decomposition() -> None:
    contract = OptionContract("HONESTMC", 100.0, 1.0, OptionType.CALL)
    market = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    result = MonteCarloModel(paths=20_000).calculate_price(
        contract, market, 0.2, seed_sequence=SeedSequence(7)
    )
    analytic = BlackScholesModel().calculate_price(contract, market, 0.2)
    assert result.standard_error is not None
    assert (
        abs(result.theoretical_price - analytic.theoretical_price) <= 1.96 * result.standard_error
    )
    report = result.control_variate_report
    assert report is not None and report["cv_used"] is True
    raw_variance = float(report["raw_var"] or 0.0)
    residual_variance = float(report["residual_var"] or 0.0)
    assert 1e-3 < residual_variance / raw_variance < 1.0


def test_contract_ids_do_not_alias_rounded_terms() -> None:
    first = OptionContract("ID", 100.00001, 1.0000001, OptionType.CALL)
    second = OptionContract("ID", 100.00002, 1.0000002, OptionType.CALL)
    assert first.contract_id != second.contract_id
    assert math.isfinite(first.strike_price)
