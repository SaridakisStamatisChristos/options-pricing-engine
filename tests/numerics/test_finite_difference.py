from __future__ import annotations

import math

import numpy as np
import pytest

from options_engine.core.finite_difference import (
    FiniteDifferenceDiagnostics,
    FiniteDifferenceModel,
)
from options_engine.core.models import ExerciseStyle, MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import BinomialModel, BlackScholesModel


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_crank_nicolson_matches_black_scholes(option_type: OptionType) -> None:
    contract = OptionContract("PDE-EU", 100.0, 1.0, option_type)
    market = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    model = FiniteDifferenceModel(space_steps=300, time_steps=300)

    result = model.calculate_price(contract, market, 0.2)
    reference = BlackScholesModel().calculate_price(contract, market, 0.2)

    assert result.theoretical_price == pytest.approx(
        reference.theoretical_price, rel=0.0, abs=0.006
    )
    assert result.delta is not None and math.isfinite(result.delta)
    assert result.gamma is not None and result.gamma >= 0.0
    assert result.theta is not None and math.isfinite(result.theta)
    assert result.model_used == "finite_difference_cn_300x300"


def test_psor_american_put_matches_independent_crr_family() -> None:
    contract = OptionContract(
        "PDE-AM",
        100.0,
        1.0,
        OptionType.PUT,
        ExerciseStyle.AMERICAN,
    )
    market = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    model = FiniteDifferenceModel(space_steps=300, time_steps=300)

    analysis = model.price_with_diagnostics(contract, market, 0.2)
    reference = BinomialModel(steps=1_000).calculate_price(contract, market, 0.2)

    assert analysis.pricing_result.theoretical_price == pytest.approx(
        reference.theoretical_price, rel=0.0, abs=0.01
    )
    assert analysis.pricing_result.theoretical_price >= 0.0
    assert analysis.diagnostics.exercise_solver == "psor"
    assert analysis.diagnostics.psor_converged is True
    assert analysis.diagnostics.psor_max_iterations_used > 0
    assert analysis.diagnostics.lcp_residual <= model.psor_tolerance
    assert analysis.diagnostics.rannacher_half_steps == 2
    assert analysis.spot_grid.shape == analysis.value_grid.shape == (301,)
    assert (analysis.value_grid >= 0.0).all()


def test_american_solution_dominates_european_solution() -> None:
    market = MarketData(spot_price=85.0, risk_free_rate=0.04, dividend_yield=0.0)
    european = OptionContract("PDE-DOM", 100.0, 0.75, OptionType.PUT)
    american = OptionContract(
        "PDE-DOM",
        100.0,
        0.75,
        OptionType.PUT,
        ExerciseStyle.AMERICAN,
    )
    model = FiniteDifferenceModel(space_steps=220, time_steps=220)

    european_price = model.calculate_price(european, market, 0.25).theoretical_price
    american_price = model.calculate_price(american, market, 0.25).theoretical_price

    assert american_price >= european_price
    assert american_price >= american.strike_price - market.spot_price


def test_sinh_grid_is_non_uniform_and_contains_spot_and_strike() -> None:
    contract = OptionContract("PDE-GRID", 100.0, 1.0, OptionType.CALL)
    market = MarketData(spot_price=93.0, risk_free_rate=0.03, dividend_yield=0.01)

    analysis = FiniteDifferenceModel(space_steps=80, time_steps=80).price_with_diagnostics(
        contract, market, 0.3
    )
    spacing = np.diff(analysis.spot_grid)

    assert analysis.diagnostics.grid_type == "sinh"
    assert analysis.diagnostics.spot_on_grid is True
    assert analysis.diagnostics.strike_on_grid is True
    assert np.any(analysis.spot_grid == market.spot_price)
    assert np.any(analysis.spot_grid == contract.strike_price)
    assert float(np.max(spacing)) > 2.0 * float(np.min(spacing))
    assert analysis.diagnostics.min_spot_step == pytest.approx(float(np.min(spacing)))
    assert analysis.diagnostics.max_spot_step == pytest.approx(float(np.max(spacing)))


def test_uniform_grid_remains_available_for_reproducibility() -> None:
    contract = OptionContract("PDE-UNIFORM", 100.0, 0.5, OptionType.PUT)
    market = MarketData(spot_price=95.0, risk_free_rate=0.02, dividend_yield=0.01)
    analysis = FiniteDifferenceModel(
        space_steps=60,
        time_steps=60,
        grid_type="uniform",
    ).price_with_diagnostics(contract, market, 0.25)

    spacing = np.diff(analysis.spot_grid)
    assert np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-13)
    assert analysis.diagnostics.grid_type == "uniform"


def test_v210_positional_constructors_remain_compatible() -> None:
    model = FiniteDifferenceModel(80, 80, 3.5, True, 1.2, 1e-10, 10_000)
    diagnostics = FiniteDifferenceDiagnostics(
        80,
        80,
        400.0,
        5.0,
        0.0125,
        2,
        "psor",
        True,
        20,
        1_600,
        1e-11,
        0.0,
        100.0,
        False,
    )

    assert model.psor_omega == 1.2
    assert model.grid_type == "sinh"
    assert diagnostics.grid_type == "uniform"
    assert diagnostics.level_prices == ()


def test_non_uniform_operator_is_exact_for_a_quadratic() -> None:
    spot_grid = np.array([0.0, 0.4, 1.1, 2.3, 4.8, 8.0])
    market = MarketData(spot_price=1.0, risk_free_rate=0.03, dividend_yield=0.01)
    volatility = 0.4
    lower, diagonal, upper = FiniteDifferenceModel._operator_coefficients(
        spot_grid, market, volatility
    )
    quadratic = spot_grid**2
    discrete = lower * quadratic[:-2] + diagonal * quadratic[1:-1] + upper * quadratic[2:]
    expected = (volatility**2 + market.risk_free_rate - 2.0 * market.dividend_yield) * spot_grid[
        1:-1
    ] ** 2

    assert discrete == pytest.approx(expected, rel=0.0, abs=2e-14)


def test_european_refinement_reports_credible_richardson_error() -> None:
    contract = OptionContract("PDE-REFINE-EU", 100.0, 1.0, OptionType.CALL)
    market = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    model = FiniteDifferenceModel(
        space_steps=50,
        time_steps=50,
        refinement_levels=3,
    )

    analysis = model.price_with_diagnostics(contract, market, 0.2)
    reference = BlackScholesModel().calculate_price(contract, market, 0.2)
    diagnostics = analysis.diagnostics
    level_errors = [abs(price - reference.theoretical_price) for price in diagnostics.level_prices]

    assert diagnostics.level_space_steps == (50, 100, 200)
    assert diagnostics.level_time_steps == (50, 100, 200)
    assert level_errors[2] < level_errors[1] < level_errors[0]
    assert diagnostics.level_differences[1] < diagnostics.level_differences[0]
    assert diagnostics.observed_order == pytest.approx(2.0, abs=0.15)
    assert diagnostics.error_estimate_method == "formal_second_order"
    assert diagnostics.richardson_error_estimate is not None
    assert level_errors[-1] <= 1.1 * diagnostics.richardson_error_estimate
    assert diagnostics.richardson_extrapolated_price is not None
    assert (
        abs(diagnostics.richardson_extrapolated_price - reference.theoretical_price)
        < level_errors[-1]
    )
    assert diagnostics.level_projection_applied == (False, False, False)


def test_projection_disables_formal_richardson_estimate() -> None:
    contract = OptionContract("PDE-PROJECTION", 100.0, 1.0, OptionType.CALL)
    model = FiniteDifferenceModel(space_steps=40, time_steps=40, refinement_levels=2)

    _, _, error, extrapolated, method = model._convergence_diagnostics(
        contract,
        (0.0, 0.0),
        (True, True),
    )

    assert error is None
    assert extrapolated is None
    assert method == "grid_difference_only"


@pytest.mark.parametrize("exercise_solver", ["psor", "penalty"])
def test_american_refinement_reports_observed_convergence(exercise_solver: str) -> None:
    contract = OptionContract(
        "PDE-REFINE-AM",
        100.0,
        1.0,
        OptionType.PUT,
        ExerciseStyle.AMERICAN,
    )
    market = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    analysis = FiniteDifferenceModel(
        space_steps=60,
        time_steps=60,
        refinement_levels=3,
        exercise_solver=exercise_solver,
    ).price_with_diagnostics(contract, market, 0.2)
    diagnostics = analysis.diagnostics

    assert diagnostics.level_differences[1] < diagnostics.level_differences[0]
    assert diagnostics.observed_order == pytest.approx(2.0, abs=0.25)
    assert diagnostics.error_estimate_method == "observed_order"
    assert diagnostics.richardson_error_estimate is not None
    assert len(diagnostics.level_lcp_residuals) == 3
    assert max(diagnostics.level_lcp_residuals) <= 2e-9


@pytest.mark.parametrize(
    ("case_id", "option_type", "spot", "strike", "maturity", "rate", "dividend", "vol"),
    [
        ("atm-put", OptionType.PUT, 100.0, 100.0, 1.0, 0.05, 0.02, 0.2),
        ("div-call", OptionType.CALL, 110.0, 100.0, 1.0, 0.01, 0.12, 0.35),
        ("negative-rate", OptionType.PUT, 70.0, 100.0, 1.5, -0.02, 0.01, 0.5),
        ("short-high-vol", OptionType.PUT, 95.0, 100.0, 7.0 / 365.0, 0.02, 0.01, 1.2),
    ],
)
def test_psor_and_penalty_are_independent_cross_validation_families(
    case_id: str,
    option_type: OptionType,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: float,
) -> None:
    contract = OptionContract(
        case_id,
        strike,
        maturity,
        option_type,
        ExerciseStyle.AMERICAN,
    )
    market = MarketData(spot, rate, dividend)
    analyses = {
        solver: FiniteDifferenceModel(
            space_steps=140,
            time_steps=180,
            exercise_solver=solver,
        ).price_with_diagnostics(contract, market, vol)
        for solver in ("psor", "penalty")
    }
    psor = analyses["psor"]
    penalty = analyses["penalty"]

    assert psor.pricing_result.theoretical_price == pytest.approx(
        penalty.pricing_result.theoretical_price, rel=0.0, abs=1e-5
    )
    assert psor.diagnostics.psor_converged is True
    assert psor.diagnostics.penalty_converged is None
    assert psor.diagnostics.lcp_residual <= 1e-10
    assert penalty.diagnostics.psor_converged is None
    assert penalty.diagnostics.penalty_converged is True
    assert penalty.diagnostics.penalty_parameter == 1e7
    assert penalty.diagnostics.penalty_max_iterations_used > 0
    assert penalty.diagnostics.lcp_residual <= 2e-9
    assert penalty.diagnostics.penalty_equation_residual <= 1e-8
    assert penalty.pricing_result.model_used.startswith("finite_difference_cn_penalty_")


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"space_steps": 19}, ValueError),
        ({"time_steps": 9}, ValueError),
        ({"space_steps": True}, TypeError),
        ({"rannacher_smoothing": 1}, TypeError),
        ({"psor_omega": 2.0}, ValueError),
        ({"psor_tolerance": math.nan}, ValueError),
        ({"grid_type": "log"}, ValueError),
        ({"grid_concentration": 0.001}, ValueError),
        ({"exercise_solver": "policy"}, ValueError),
        ({"refinement_levels": 5}, ValueError),
        ({"penalty_parameter": 1e11}, ValueError),
        ({"penalty_max_iterations": 1}, ValueError),
        ({"s_max_override": math.nan}, ValueError),
        ({"space_steps": 1_000, "time_steps": 6_000}, ValueError),
        ({"space_steps": 1_000, "time_steps": 1_000, "refinement_levels": 3}, ValueError),
    ],
)
def test_finite_difference_configuration_guardrails(kwargs, exception) -> None:
    with pytest.raises(exception):
        FiniteDifferenceModel(**kwargs)


def test_s_max_override_is_validated_against_contract_domain() -> None:
    contract = OptionContract("PDE-DOMAIN", 100.0, 1.0, OptionType.PUT)
    market = MarketData(spot_price=110.0, risk_free_rate=0.02)
    model = FiniteDifferenceModel(space_steps=40, time_steps=40, s_max_override=105.0)

    with pytest.raises(ValueError, match="exceed both spot and strike"):
        model.calculate_price(contract, market, 0.2)
