import math
import warnings
from numpy.random import default_rng, SeedSequence

from options_engine.core.models import ExerciseStyle, MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import (
    BlackScholesModel,
    LongstaffSchwartzModel,
    MonteCarloModel,
)


def test_bs_runs():
    m = BlackScholesModel()
    c = OptionContract(
        symbol="TEST", strike_price=100.0, time_to_expiry=1.0, option_type=OptionType.CALL
    )
    md = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    r = m.calculate_price(c, md, 0.2)
    assert r.theoretical_price > 0


def test_monte_carlo_handles_small_antithetic_path_counts():
    model = MonteCarloModel(paths=1, antithetic=True)
    contract = OptionContract(
        symbol="MC", strike_price=100.0, time_to_expiry=1.0, option_type=OptionType.CALL
    )
    market = MarketData(spot_price=100.0, risk_free_rate=0.01)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = model.calculate_price(contract, market, 0.2)

    assert not captured
    assert math.isfinite(result.theoretical_price)
    assert result.standard_error is not None
    assert result.standard_error >= 0.0
    assert result.confidence_interval is not None
    lower, upper = result.confidence_interval
    assert lower <= upper


def test_european_call_put_parity_holds() -> None:
    model = BlackScholesModel()
    rng = default_rng(42)

    for _ in range(10):
        spot = float(rng.uniform(50.0, 150.0))
        strike = float(rng.uniform(50.0, 150.0))
        risk_free_rate = float(rng.uniform(0.0, 0.1))
        dividend_yield = float(rng.uniform(0.0, 0.05))
        time_to_expiry = float(rng.uniform(0.1, 2.0))
        volatility = float(rng.uniform(0.05, 0.6))

        call_contract = OptionContract(
            symbol="PCALL",
            strike_price=strike,
            time_to_expiry=time_to_expiry,
            option_type=OptionType.CALL,
        )
        put_contract = OptionContract(
            symbol="PPUT",
            strike_price=strike,
            time_to_expiry=time_to_expiry,
            option_type=OptionType.PUT,
        )
        market = MarketData(
            spot_price=spot,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )

        call_price = model.calculate_price(call_contract, market, volatility).theoretical_price
        put_price = model.calculate_price(put_contract, market, volatility).theoretical_price

        expected = spot * math.exp(-dividend_yield * time_to_expiry) - strike * math.exp(
            -risk_free_rate * time_to_expiry
        )
        assert math.isclose(call_price - put_price, expected, rel_tol=0.0, abs_tol=1e-6)


def test_call_price_monotonic_decreases_with_strike() -> None:
    model = BlackScholesModel()
    market = MarketData(spot_price=110.0, risk_free_rate=0.02, dividend_yield=0.01)
    time_to_expiry = 1.0
    volatility = 0.25

    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    prices = []
    for strike in strikes:
        contract = OptionContract(
            symbol=f"CALL{int(strike)}",
            strike_price=strike,
            time_to_expiry=time_to_expiry,
            option_type=OptionType.CALL,
        )
        prices.append(model.calculate_price(contract, market, volatility).theoretical_price)

    assert prices == sorted(prices, reverse=True)


def test_monte_carlo_confidence_interval_contracts_with_more_paths() -> None:
    base_seed = SeedSequence(2024)
    base_model = MonteCarloModel(paths=4096, seed_sequence=base_seed.spawn(1)[0])
    refined_model = MonteCarloModel(paths=8192, seed_sequence=base_seed.spawn(1)[0])

    contract = OptionContract(
        symbol="MC",
        strike_price=100.0,
        time_to_expiry=1.0,
        option_type=OptionType.CALL,
    )
    market = MarketData(spot_price=105.0, risk_free_rate=0.01)

    base_result = base_model.calculate_price(contract, market, 0.2)
    refined_result = refined_model.calculate_price(contract, market, 0.2)

    assert base_result.confidence_interval is not None
    assert refined_result.confidence_interval is not None

    def _half_width(interval: tuple[float, float]) -> float:
        lower, upper = interval
        return abs(upper - lower) / 2.0

    assert _half_width(refined_result.confidence_interval) < _half_width(
        base_result.confidence_interval
    )


def test_lsmc_matches_high_res_binomial_within_five_bps() -> None:
    seed = SeedSequence(2025)
    model = LongstaffSchwartzModel(
        paths=80_000,
        steps=60,
        cv_folds=5,
        seed_sequence=seed.spawn(1)[0],
        reference_steps=2_000,
    )

    contract = OptionContract(
        symbol="AM_PUT",
        strike_price=100.0,
        time_to_expiry=1.0,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
    )
    market = MarketData(spot_price=90.0, risk_free_rate=0.03, dividend_yield=0.0)

    analysis = model.price_with_diagnostics(contract, market, 0.25)

    assert math.isfinite(analysis.reference_price)
    assert analysis.reference_model_used.startswith("binomial")
    assert abs(analysis.price_diff_bps) <= 5.0


def test_lsmc_provides_basis_selection_and_policy_details() -> None:
    seed = SeedSequence(2026)
    model = LongstaffSchwartzModel(
        paths=50_000,
        steps=50,
        cv_folds=4,
        seed_sequence=seed.spawn(1)[0],
        reference_steps=2_000,
    )

    contract = OptionContract(
        symbol="AM_PUT_DIAG",
        strike_price=100.0,
        time_to_expiry=1.0,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
    )
    market = MarketData(spot_price=80.0, risk_free_rate=0.02, dividend_yield=0.0)

    analysis = model.price_with_diagnostics(contract, market, 0.3)

    assert analysis.policy.shape[0] == model.steps + 1
    if model.antithetic:
        expected_paths = max(2, int(model.paths) + (int(model.paths) % 2))
    else:
        expected_paths = max(1, int(model.paths))
    assert analysis.policy.shape[1] == expected_paths

    candidate_names = {
        metrics.name for step_metrics in analysis.basis_diagnostics for metrics in step_metrics
    }
    assert "polynomial_2" in candidate_names
    assert "log_linear" in candidate_names

    selected_bases = [
        step.basis for step in analysis.policy_steps if step.basis is not None and step.in_the_money > 0
    ]
    assert selected_bases, "Expected at least one regression step with a selected basis"

    for basis in selected_bases:
        assert basis is not None
        assert math.isfinite(basis.aic)
        assert math.isfinite(basis.bic)
        assert math.isfinite(basis.cv_rmse)

    early_exercise_steps = [
        step for step in analysis.policy_steps[:-1] if step.exercised > 0 and step.in_the_money > 0
    ]
    assert early_exercise_steps, "Expected to observe early exercise before maturity"
