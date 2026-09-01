"""True deterministic term-structure pricing and independent validation."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from options_engine.core.black_scholes import BlackScholesModel
from options_engine.core.crr import BinomialModel
from options_engine.core.finite_difference import FiniteDifferenceModel
from options_engine.core.models import ExerciseStyle, MarketData, OptionContract, OptionType
from options_engine.market import (
    ALL_DAYS_CALENDAR,
    ContinuousDividendCurve,
    ContinuousZeroCurve,
    DatedCashDividend,
    DatedCashDividendSchedule,
    DatedOptionContract,
    ExDividendDate,
    ExpiryDate,
    MarketConventions,
    MarketEnvironment,
    ValuationDate,
    ZeroRateNode,
)

REFERENCE_PATH = Path(__file__).resolve().parents[1] / "reference" / "quantlib_curve_aware_v1.json"
REFERENCE = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
CASES: list[dict[str, Any]] = REFERENCE["cases"]
VALUATION = ValuationDate(date.fromisoformat(REFERENCE["valuation_date"]))


def _inputs(case: dict[str, Any]):
    funding = ContinuousZeroCurve(
        VALUATION,
        tuple(
            ZeroRateNode(date.fromisoformat(node["date"]), node["rate"])
            for node in case["funding_nodes"]
        ),
    )
    carry = ContinuousDividendCurve(
        VALUATION,
        tuple(
            ZeroRateNode(date.fromisoformat(node["date"]), node["rate"])
            for node in case["carry_nodes"]
        ),
    )
    schedule = DatedCashDividendSchedule(
        tuple(
            DatedCashDividend(
                ExDividendDate(date.fromisoformat(dividend["date"])),
                dividend["amount"],
            )
            for dividend in case["dividends"]
        )
    )
    environment = MarketEnvironment(
        case["spot"],
        MarketConventions(VALUATION, calendar=ALL_DAYS_CALENDAR),
        funding,
        carry,
        cash_dividends=schedule,
    )
    resolved = environment.resolve_curve_aware(
        DatedOptionContract(
            case["id"],
            case["strike"],
            ExpiryDate(date.fromisoformat(case["expiry_date"])),
            OptionType(case["type"]),
            ExerciseStyle(case["style"]),
        )
    )
    return environment, resolved


def _same_endpoint_environment(mid_rate: float) -> MarketEnvironment:
    expiry = date(2027, 1, 1)
    return MarketEnvironment(
        100.0,
        MarketConventions(VALUATION, calendar=ALL_DAYS_CALENDAR),
        ContinuousZeroCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), mid_rate),
                ZeroRateNode(expiry, 0.04),
            ),
        ),
        ContinuousDividendCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), 0.01),
                ZeroRateNode(expiry, 0.01),
            ),
        ),
    )


def test_curve_aware_black_scholes_preserves_factors_and_put_call_parity() -> None:
    environment = _same_endpoint_environment(0.07)
    prices: dict[OptionType, float] = {}
    for option_type in (OptionType.CALL, OptionType.PUT):
        resolved = environment.resolve_curve_aware(
            DatedOptionContract(
                "PARITY",
                100.0,
                ExpiryDate(date(2027, 1, 1)),
                option_type,
            )
        )
        result = BlackScholesModel().calculate_price_curve_aware(
            resolved.contract,
            resolved.market_data,
            0.22,
            resolved.term_structure,
        )
        prices[option_type] = result.theoretical_price
        assert result.numerical_diagnostics is not None
        assert result.numerical_diagnostics["curve_aware_mode"] is True
        assert result.theta is None
        assert result.rho is None

    context = environment.resolve_curve_aware(
        DatedOptionContract(
            "PARITY",
            100.0,
            ExpiryDate(date(2027, 1, 1)),
            OptionType.CALL,
        )
    ).term_structure
    parity = 100.0 * context.carry_factor(0.0, 1.0) - 100.0 * context.discount_factor(0.0, 1.0)
    assert prices[OptionType.CALL] - prices[OptionType.PUT] == pytest.approx(parity, abs=2e-14)


def test_same_terminal_factors_same_european_but_different_american_prices() -> None:
    environments = (_same_endpoint_environment(0.01), _same_endpoint_environment(0.07))
    european_prices: list[float] = []
    american_prices: list[float] = []
    for environment in environments:
        european = environment.resolve_curve_aware(
            DatedOptionContract(
                "SHAPE",
                100.0,
                ExpiryDate(date(2027, 1, 1)),
                OptionType.PUT,
            )
        )
        american = environment.resolve_curve_aware(
            DatedOptionContract(
                "SHAPE",
                100.0,
                ExpiryDate(date(2027, 1, 1)),
                OptionType.PUT,
                ExerciseStyle.AMERICAN,
            )
        )
        european_prices.append(
            BlackScholesModel()
            .calculate_price_curve_aware(
                european.contract,
                european.market_data,
                0.22,
                european.term_structure,
            )
            .theoretical_price
        )
        american_prices.append(
            FiniteDifferenceModel(
                space_steps=180,
                time_steps=220,
                exercise_solver="penalty",
            )
            .calculate_price_curve_aware(
                american.contract,
                american.market_data,
                0.22,
                american.term_structure,
            )
            .theoretical_price
        )

    assert european_prices[0] == pytest.approx(european_prices[1], abs=2e-14)
    # This is the central regression: endpoint flattening cannot produce this.
    assert abs(american_prices[0] - american_prices[1]) > 0.5


@pytest.mark.parametrize("model_name", ["black_scholes", "finite_difference", "binomial"])
def test_flat_curves_reproduce_scalar_model_prices(model_name: str) -> None:
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.04,
        dividend_yield=0.01,
        calendar=ALL_DAYS_CALENDAR,
    )
    style = ExerciseStyle.EUROPEAN if model_name == "black_scholes" else ExerciseStyle.AMERICAN
    resolved = environment.resolve_curve_aware(
        DatedOptionContract(
            "FLAT",
            100.0,
            ExpiryDate(date(2027, 1, 1)),
            OptionType.PUT,
            style,
        )
    )
    if model_name == "black_scholes":
        model = BlackScholesModel()
    elif model_name == "finite_difference":
        model = FiniteDifferenceModel(space_steps=100, time_steps=100)
    else:
        model = BinomialModel(steps=300)
    scalar = model.calculate_price(resolved.contract, resolved.market_data, 0.2)
    curved = model.calculate_price_curve_aware(
        resolved.contract,
        resolved.market_data,
        0.2,
        resolved.term_structure,
    )
    assert curved.theoretical_price == pytest.approx(scalar.theoretical_price, rel=0.0, abs=5e-13)


def test_curve_aware_european_pde_converges_to_exact_analytic_value() -> None:
    environment = MarketEnvironment(
        100.0,
        MarketConventions(VALUATION, calendar=ALL_DAYS_CALENDAR),
        ContinuousZeroCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), 0.01),
                ZeroRateNode(date(2027, 1, 1), 0.04),
            ),
        ),
        ContinuousDividendCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), 0.03),
                ZeroRateNode(date(2027, 1, 1), 0.01),
            ),
        ),
    )
    resolved = environment.resolve_curve_aware(
        DatedOptionContract(
            "EU-CONVERGENCE",
            105.0,
            ExpiryDate(date(2027, 1, 1)),
            OptionType.CALL,
        )
    )
    reference = BlackScholesModel().calculate_price_curve_aware(
        resolved.contract,
        resolved.market_data,
        0.27,
        resolved.term_structure,
    )
    errors = []
    for steps in (50, 100, 200):
        result = FiniteDifferenceModel(
            space_steps=steps,
            time_steps=steps,
        ).calculate_price_curve_aware(
            resolved.contract,
            resolved.market_data,
            0.27,
            resolved.term_structure,
        )
        errors.append(abs(result.theoretical_price - reference.theoretical_price))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.0012

    tree_errors = []
    for steps in (200, 800, 3_200):
        result = BinomialModel(steps=steps).calculate_price_curve_aware(
            resolved.contract,
            resolved.market_data,
            0.27,
            resolved.term_structure,
        )
        tree_errors.append(abs(result.theoretical_price - reference.theoretical_price))
    assert tree_errors[2] < tree_errors[1] < tree_errors[0]
    assert tree_errors[2] < 0.0003


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_penalty_pde_against_quantlib_shaped_curve_fixtures(case: dict[str, Any]) -> None:
    _environment, resolved = _inputs(case)
    result = FiniteDifferenceModel(
        space_steps=280,
        time_steps=360,
        exercise_solver="penalty",
    ).calculate_price_curve_aware(
        resolved.contract,
        resolved.market_data,
        case["vol"],
        resolved.term_structure,
    )
    tolerance = 0.0025 if "high_volatility" in case["regimes"] else 0.0008
    assert result.theoretical_price == pytest.approx(case["reference_price"], abs=tolerance)


@pytest.mark.parametrize("case_index", [0, 4])
def test_curve_aware_psor_and_penalty_solvers_agree(case_index: int) -> None:
    case = CASES[case_index]
    _environment, resolved = _inputs(case)
    prices = {}
    for solver in ("psor", "penalty"):
        result = FiniteDifferenceModel(
            space_steps=160,
            time_steps=200,
            exercise_solver=solver,
        ).calculate_price_curve_aware(
            resolved.contract,
            resolved.market_data,
            case["vol"],
            resolved.term_structure,
        )
        prices[solver] = result.theoretical_price
        assert result.numerical_diagnostics is not None
        assert result.numerical_diagnostics["solver_converged"] is True
    assert prices["psor"] == pytest.approx(prices["penalty"], abs=1e-5)


@pytest.mark.parametrize("case_index", [1, 2, 4, 5])
def test_curve_aware_tree_converges_to_quantlib_reference(case_index: int) -> None:
    case = CASES[case_index]
    _environment, resolved = _inputs(case)
    errors = []
    for steps in (200, 800, 3_200):
        result = BinomialModel(steps=steps).calculate_price_curve_aware(
            resolved.contract,
            resolved.market_data,
            case["vol"],
            resolved.term_structure,
        )
        errors.append(abs(result.theoretical_price - case["reference_price"]))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < (0.003 if case["dividends"] else 0.001)


def test_curve_and_cash_nodes_are_exact_pde_anchors_with_reproducible_rates() -> None:
    case = CASES[4]
    environment, resolved = _inputs(case)
    result = FiniteDifferenceModel(
        space_steps=120,
        time_steps=140,
        exercise_solver="penalty",
    ).calculate_price_curve_aware(
        resolved.contract,
        resolved.market_data,
        case["vol"],
        resolved.term_structure,
    )
    diagnostics = result.numerical_diagnostics
    assert diagnostics is not None
    assert diagnostics["curve_node_alignment_status"] is True
    assert diagnostics["cash_dividend_alignment_status"] is True
    assert diagnostics["cash_dividend_event_aligned"] is True
    mesh = diagnostics["actual_time_mesh"]
    assert isinstance(mesh, tuple)
    for anchor in (
        *resolved.term_structure.curve_node_times,
        *resolved.term_structure.cash_dividend_times,
    ):
        assert any(math.isclose(anchor, value, rel_tol=0.0, abs_tol=2e-13) for value in mesh)
    assert len(diagnostics["effective_step_funding_rates"]) == len(mesh) - 1
    assert diagnostics["min_local_funding_rate"] < diagnostics["max_local_funding_rate"]
    assert diagnostics["min_local_carry_rate"] < diagnostics["max_local_carry_rate"]
    assert diagnostics["discount_curve_id"] == environment.discount_curve.curve_id
    assert diagnostics["carry_curve_id"] == environment.carry_curve.curve_id


def test_curve_aware_american_dominates_european_and_outputs_are_finite() -> None:
    environment = _same_endpoint_environment(0.01)
    european = environment.resolve_curve_aware(
        DatedOptionContract(
            "DOMINANCE",
            100.0,
            ExpiryDate(date(2027, 1, 1)),
            OptionType.PUT,
        )
    )
    american = environment.resolve_curve_aware(
        DatedOptionContract(
            "DOMINANCE",
            100.0,
            ExpiryDate(date(2027, 1, 1)),
            OptionType.PUT,
            ExerciseStyle.AMERICAN,
        )
    )
    european_price = (
        BlackScholesModel()
        .calculate_price_curve_aware(
            european.contract,
            european.market_data,
            0.22,
            european.term_structure,
        )
        .theoretical_price
    )
    american_result = FiniteDifferenceModel(
        space_steps=160,
        time_steps=180,
        exercise_solver="penalty",
    ).calculate_price_curve_aware(
        american.contract,
        american.market_data,
        0.22,
        american.term_structure,
    )
    assert math.isfinite(american_result.theoretical_price)
    assert american_result.theoretical_price >= european_price
    assert american_result.theoretical_price >= max(
        american.contract.strike_price - american.market_data.spot_price, 0.0
    )


def test_curve_aware_european_call_is_monotone_in_spot_and_within_bounds() -> None:
    prices = []
    for spot in (80.0, 100.0, 120.0):
        base = _same_endpoint_environment(0.07)
        environment = MarketEnvironment(
            spot,
            base.conventions,
            base.discount_curve,
            base.carry_curve,
        )
        resolved = environment.resolve_curve_aware(
            DatedOptionContract(
                "MONOTONE",
                100.0,
                ExpiryDate(date(2027, 1, 1)),
                OptionType.CALL,
            )
        )
        result = BlackScholesModel().calculate_price_curve_aware(
            resolved.contract,
            resolved.market_data,
            0.3,
            resolved.term_structure,
        )
        upper_bound = spot * resolved.term_structure.carry_factor(0.0, 1.0)
        assert 0.0 <= result.theoretical_price <= upper_bound
        assert math.isfinite(result.theoretical_price)
        prices.append(result.theoretical_price)
    assert prices[0] < prices[1] < prices[2]


def test_curve_aware_crr_adapts_or_rejects_without_probability_clipping() -> None:
    expiry = ExpiryDate(date(2027, 1, 1))
    adaptive_environment = MarketEnvironment(
        100.0,
        MarketConventions(VALUATION, calendar=ALL_DAYS_CALENDAR),
        ContinuousZeroCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), 0.0),
                ZeroRateNode(expiry.value, 0.5),
            ),
        ),
        ContinuousDividendCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), 0.0),
                ZeroRateNode(expiry.value, 0.0),
            ),
        ),
    )
    adaptive = adaptive_environment.resolve_curve_aware(
        DatedOptionContract("ADAPT", 100.0, expiry, OptionType.CALL)
    )
    result = BinomialModel(steps=8).calculate_price_curve_aware(
        adaptive.contract,
        adaptive.market_data,
        0.1,
        adaptive.term_structure,
    )
    assert result.numerical_diagnostics is not None
    assert result.numerical_diagnostics["effective_steps"] > 8
    assert result.numerical_diagnostics["probability_clipping"] is False
    assert result.numerical_diagnostics["min_risk_neutral_probability"] > 0.0
    assert result.numerical_diagnostics["max_risk_neutral_probability"] < 1.0

    rejecting_environment = MarketEnvironment(
        100.0,
        MarketConventions(VALUATION, calendar=ALL_DAYS_CALENDAR),
        ContinuousZeroCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), -1.0),
                ZeroRateNode(expiry.value, 1.0),
            ),
        ),
        ContinuousDividendCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), 0.0),
                ZeroRateNode(expiry.value, 0.0),
            ),
        ),
    )
    rejecting = rejecting_environment.resolve_curve_aware(
        DatedOptionContract("REJECT", 100.0, expiry, OptionType.CALL)
    )
    with pytest.raises(ValueError, match="maximum supported resolution reached"):
        BinomialModel(steps=4_096).calculate_price_curve_aware(
            rejecting.contract,
            rejecting.market_data,
            0.01,
            rejecting.term_structure,
        )


def test_curve_aware_inputs_reject_invalid_intervals_and_maturity_mismatch() -> None:
    resolved = _same_endpoint_environment(0.01).resolve_curve_aware(
        DatedOptionContract(
            "INVALID",
            100.0,
            ExpiryDate(date(2027, 1, 1)),
            OptionType.PUT,
        )
    )
    with pytest.raises(ValueError, match="end_time on or after"):
        resolved.term_structure.discount_factor(0.8, 0.2)
    mismatched_contract = OptionContract("INVALID", 100.0, 0.5, OptionType.PUT)
    mismatched_market = MarketData(100.0, 0.04, 0.01)
    with pytest.raises(ValueError, match="maturity must match"):
        FiniteDifferenceModel(space_steps=40, time_steps=40).calculate_price_curve_aware(
            mismatched_contract,
            mismatched_market,
            0.2,
            resolved.term_structure,
        )


def test_quantlib_curve_fixture_covers_required_regimes() -> None:
    regimes = {regime for case in CASES for regime in case["regimes"]}
    assert REFERENCE["source"] == {
        "library": "QuantLib",
        "version": "1.43",
        "engine": "FdBlackScholesVanillaEngine",
        "scheme": "CrankNicolson",
        "time_steps": 2400,
        "space_steps": 1200,
        "damping_steps": 4,
        "cash_dividend_model": "Spot",
        "curve_interpolation": "linear_continuous_zero",
        "generator": "reports/generate_quantlib_references.py",
    }
    assert regimes >= {
        "american_call",
        "american_put",
        "cash_dividends",
        "deep_itm",
        "deep_otm",
        "downward_funding",
        "high_volatility",
        "long_maturity",
        "negative_rates",
        "nonflat_carry",
        "short_maturity",
        "upward_funding",
    }
