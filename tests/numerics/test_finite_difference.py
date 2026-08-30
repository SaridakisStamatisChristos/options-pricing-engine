from __future__ import annotations

import math

import pytest

from options_engine.core.finite_difference import FiniteDifferenceModel
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


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"space_steps": 19}, ValueError),
        ({"time_steps": 9}, ValueError),
        ({"space_steps": True}, TypeError),
        ({"rannacher_smoothing": 1}, TypeError),
        ({"psor_omega": 2.0}, ValueError),
        ({"psor_tolerance": math.nan}, ValueError),
        ({"space_steps": 1_000, "time_steps": 6_000}, ValueError),
    ],
)
def test_finite_difference_configuration_guardrails(kwargs, exception) -> None:
    with pytest.raises(exception):
        FiniteDifferenceModel(**kwargs)
