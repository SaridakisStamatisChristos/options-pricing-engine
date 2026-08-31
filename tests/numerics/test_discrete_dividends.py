from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from options_engine.core.crr import BinomialModel
from options_engine.core.dividends import CashDividend, CashDividendSchedule
from options_engine.core.finite_difference import FiniteDifferenceModel
from options_engine.core.models import (
    ExerciseStyle,
    MarketData,
    OptionContract,
    OptionType,
)

REFERENCE_PATH = (
    Path(__file__).resolve().parents[1] / "reference" / "quantlib_discrete_dividends_v1.json"
)
REFERENCE = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = REFERENCE["cases"]


def _inputs(case: dict[str, Any]) -> tuple[OptionContract, MarketData]:
    contract = OptionContract(
        case["id"],
        case["strike"],
        case["time_to_expiry"],
        OptionType(case["type"]),
        ExerciseStyle(case["style"]),
    )
    schedule = CashDividendSchedule(
        tuple(
            CashDividend(dividend["ex_time"], dividend["amount"]) for dividend in case["dividends"]
        )
    )
    return contract, MarketData(
        case["spot"],
        case["r"],
        case["q"],
        cash_dividends=schedule,
    )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_penalty_pde_against_quantlib_spot_dividend_fixtures(
    case: dict[str, Any],
) -> None:
    contract, market = _inputs(case)

    result = FiniteDifferenceModel(
        space_steps=280,
        time_steps=360,
        exercise_solver="penalty",
    ).calculate_price(contract, market, case["vol"])

    # QuantLib 1.43 uses an independently implemented 2,400 x 1,200 finite-
    # difference engine with its Spot cash-dividend model. This tolerance was
    # fixed from the committed refinement study and includes both grid errors.
    assert result.theoretical_price == pytest.approx(case["reference_price"], abs=0.001)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_crr_cash_jump_tree_against_quantlib_fixtures(case: dict[str, Any]) -> None:
    contract, market = _inputs(case)

    result = BinomialModel(steps=4_096).calculate_price(contract, market, case["vol"])

    # Piecewise-linear event interpolation is first-order at cash jumps. The
    # difficult two-year, 80%-vol American case determines this tolerance.
    assert result.theoretical_price == pytest.approx(case["reference_price"], abs=0.01)
    assert result.numerical_diagnostics is not None
    assert result.numerical_diagnostics["cash_dividend_interpolation"] == "piecewise_linear"


@pytest.mark.parametrize("case_index", [3, 4])
def test_psor_and_penalty_cash_dividend_solvers_cross_validate(case_index: int) -> None:
    case = CASES[case_index]
    contract, market = _inputs(case)
    prices: dict[str, float] = {}

    for solver in ("psor", "penalty"):
        result = FiniteDifferenceModel(
            space_steps=180,
            time_steps=220,
            exercise_solver=solver,
        ).calculate_price(contract, market, case["vol"])
        prices[solver] = result.theoretical_price
        assert result.numerical_diagnostics is not None
        assert result.numerical_diagnostics["cash_dividend_jumps_applied"] == len(case["dividends"])
        assert result.numerical_diagnostics["solver_converged"] is True

    assert prices["psor"] == pytest.approx(prices["penalty"], abs=2e-5)
    assert prices["psor"] == pytest.approx(case["reference_price"], abs=0.002)


@pytest.mark.parametrize("case_index", [0, 3])
def test_pde_cash_event_refinement_has_observed_convergence(case_index: int) -> None:
    case = CASES[case_index]
    contract, market = _inputs(case)

    analysis = FiniteDifferenceModel(
        space_steps=80,
        time_steps=100,
        refinement_levels=3,
        exercise_solver="penalty",
    ).price_with_diagnostics(contract, market, case["vol"])
    diagnostics = analysis.diagnostics

    assert diagnostics.level_differences[1] < diagnostics.level_differences[0]
    assert diagnostics.error_estimate_method == "observed_order"
    assert diagnostics.observed_order is not None
    assert diagnostics.cash_dividend_event_aligned is True
    assert diagnostics.cash_dividend_jumps_applied == len(case["dividends"])
    assert diagnostics.rannacher_half_steps == 2 * (len(case["dividends"]) + 1)


def test_crr_difficult_cash_event_case_converges_to_reference() -> None:
    case = CASES[-1]
    contract, market = _inputs(case)
    reference = case["reference_price"]

    errors = [
        abs(
            BinomialModel(steps=steps)
            .calculate_price(contract, market, case["vol"])
            .theoretical_price
            - reference
        )
        for steps in (400, 1_600, 4_096)
    ]

    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.01


def test_quantlib_cash_dividend_fixture_covers_required_regimes() -> None:
    regimes = {regime for case in CASES for regime in case["regimes"]}

    assert REFERENCE["source"]["cash_dividend_model"] == "Spot"
    assert regimes >= {
        "american_call",
        "american_put",
        "continuous_and_cash_dividends",
        "deep_itm",
        "deep_otm",
        "high_volatility",
        "multiple_dividends",
        "negative_rates",
        "short_maturity",
    }
