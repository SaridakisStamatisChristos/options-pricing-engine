"""Model-time factor-adapter and dated curve-dispatch invariants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pytest

from options_engine.core.finite_difference import FiniteDifferenceModel
from options_engine.core.models import ExerciseStyle, OptionType
from options_engine.core.pricing_engine import OptionsEngine
from options_engine.market import (
    ALL_DAYS_CALENDAR,
    ContinuousDividendCurve,
    ContinuousZeroCurve,
    DatedCashDividend,
    DatedCashDividendSchedule,
    DatedOptionContract,
    DayCountConvention,
    ExDividendDate,
    ExpiryDate,
    FlatDiscountCurve,
    FlatDividendCurve,
    MarketConventions,
    MarketEnvironment,
    SettlementLag,
    ValuationDate,
    ZeroRateNode,
)

VALUATION = ValuationDate(date(2026, 1, 1))
EXPIRY = ExpiryDate(date(2027, 1, 1))


def _contract(
    *,
    option_type: OptionType = OptionType.PUT,
    style: ExerciseStyle = ExerciseStyle.AMERICAN,
) -> DatedOptionContract:
    return DatedOptionContract("CURVE", 100.0, EXPIRY, option_type, style)


def test_interval_factors_compose_and_step_rates_reproduce_ratios_exactly() -> None:
    environment = MarketEnvironment(
        100.0,
        MarketConventions(VALUATION, calendar=ALL_DAYS_CALENDAR),
        ContinuousZeroCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 4, 1), -0.01),
                ZeroRateNode(date(2026, 9, 1), 0.035),
                ZeroRateNode(EXPIRY.value, 0.05),
            ),
        ),
        ContinuousDividendCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 4, 1), 0.04),
                ZeroRateNode(date(2026, 9, 1), 0.015),
                ZeroRateNode(EXPIRY.value, 0.01),
            ),
        ),
    )
    context = environment.resolve_curve_aware(_contract()).term_structure
    t0, t1, t2 = 0.0, 0.4, 1.0

    for factor in (context.discount_factor, context.carry_factor):
        assert factor(t0, t2) == pytest.approx(factor(t0, t1) * factor(t1, t2), rel=2e-15)
    funding_rate, carry_rate = context.step_rates(t1, t2)
    assert math.exp(-funding_rate * (t2 - t1)) == pytest.approx(
        context.discount_factor(t1, t2), rel=2e-15
    )
    assert math.exp(-carry_rate * (t2 - t1)) == pytest.approx(
        context.carry_factor(t1, t2), rel=2e-15
    )
    assert (
        context.context_id == environment.resolve_curve_aware(_contract()).term_structure.context_id
    )


def test_settlement_basis_preserves_forward_but_not_post_settlement_curve_ratios() -> None:
    settlement_lag = SettlementLag(2)
    conventions = MarketConventions(
        VALUATION,
        calendar=ALL_DAYS_CALENDAR,
        settlement_lag=settlement_lag,
    )
    event_date = date(2026, 4, 1)
    funding = FlatDiscountCurve(VALUATION, 0.05)
    carry = FlatDividendCurve(VALUATION, 0.02)
    environment = MarketEnvironment(
        100.0,
        conventions,
        funding,
        carry,
        cash_dividends=DatedCashDividendSchedule(
            (DatedCashDividend(ExDividendDate(event_date), 1.5),)
        ),
    )
    resolved = environment.resolve_curve_aware(_contract(option_type=OptionType.CALL))
    context = resolved.term_structure
    maturity = resolved.contract.time_to_expiry
    event_time = conventions.day_count.year_fraction(VALUATION.value, event_date)

    assert 0.0 < context.settlement_time < event_time
    assert 100.0 * context.growth_factor(0.0, maturity) == pytest.approx(
        resolved.scalar.forward.continuous_carry_forward_price,
        rel=2e-14,
    )
    assert context.discount_factor(event_time, maturity) == pytest.approx(
        funding.forward_discount_factor(event_date, EXPIRY.value), rel=2e-14
    )
    assert context.carry_factor(event_time, maturity) == pytest.approx(
        carry.forward_carry_factor(event_date, EXPIRY.value), rel=2e-14
    )
    diagnostics = resolved.diagnostics()
    assert diagnostics["cash_dividend_curve_aware_forward_deduction_mismatch"] == pytest.approx(
        0.0, abs=2e-14
    )


@dataclass(frozen=True)
class _CustomFlatCurve:
    reference_date: ValuationDate
    rate: float
    day_count: DayCountConvention = DayCountConvention.ACTUAL_365_FIXED

    @property
    def curve_id(self) -> str:
        return "custom-flat"

    def discount_factor(self, maturity: date) -> float:
        return math.exp(
            -self.rate * self.day_count.year_fraction(self.reference_date.value, maturity)
        )

    def carry_factor(self, maturity: date) -> float:
        return self.discount_factor(maturity)

    def zero_rate(self, maturity: date) -> float:
        self.discount_factor(maturity)
        return self.rate

    def forward_discount_factor(self, start: date, end: date) -> float:
        return self.discount_factor(end) / self.discount_factor(start)

    def forward_carry_factor(self, start: date, end: date) -> float:
        return self.forward_discount_factor(start, end)


def test_custom_protocol_curves_keep_endpoint_compatibility_but_fail_true_curve_mode() -> None:
    curve = _CustomFlatCurve(VALUATION, 0.03)
    environment = MarketEnvironment(
        100.0,
        MarketConventions(VALUATION, calendar=ALL_DAYS_CALENDAR),
        curve,
        curve,
    )

    assert environment.resolve(_contract()).market_data.risk_free_rate == pytest.approx(0.03)
    with pytest.raises(TypeError, match="custom endpoint-only curve"):
        environment.resolve_curve_aware(_contract())


def test_dated_orchestration_defaults_to_curves_and_rejects_unsupported_models() -> None:
    environment = MarketEnvironment(
        100.0,
        MarketConventions(VALUATION, calendar=ALL_DAYS_CALENDAR),
        ContinuousZeroCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), 0.01),
                ZeroRateNode(EXPIRY.value, 0.04),
            ),
        ),
        ContinuousDividendCurve(
            VALUATION,
            (
                ZeroRateNode(date(2026, 7, 2), 0.01),
                ZeroRateNode(EXPIRY.value, 0.01),
            ),
        ),
    )
    with OptionsEngine(num_threads=1) as engine:
        engine.models["finite_difference_400"] = FiniteDifferenceModel(
            space_steps=100,
            time_steps=120,
            exercise_solver="penalty",
        )
        curve_result = engine.price_dated_option(
            _contract(),
            environment,
            model_name="finite_difference_400",
            override_volatility=0.22,
        )
        compatibility_result = engine.price_dated_option(
            _contract(),
            environment,
            model_name="finite_difference_400",
            override_volatility=0.22,
            curve_aware=False,
        )
        with pytest.raises(ValueError, match="does not support true curve-aware pricing"):
            engine.price_dated_option(
                _contract(),
                environment,
                model_name="longstaff_schwartz_20k",
                override_volatility=0.22,
            )
        with pytest.raises(ValueError, match="does not support true curve-aware pricing"):
            engine.price_dated_option(
                _contract(
                    option_type=OptionType.CALL,
                    style=ExerciseStyle.EUROPEAN,
                ),
                environment,
                model_name="monte_carlo_20k",
                override_volatility=0.22,
            )

    assert curve_result["market_conventions"]["curve_aware_mode"] is True
    assert curve_result["market_conventions"]["rate_representation"] == (
        "deterministic_term_structure"
    )
    assert compatibility_result["market_conventions"]["rate_representation"] == (
        "endpoint_equivalent_continuous"
    )
    assert (
        abs(
            float(curve_result["theoretical_price"])
            - float(compatibility_result["theoretical_price"])
        )
        > 0.1
    )


def test_dated_orchestration_validates_curve_aware_flag() -> None:
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.03,
        calendar=ALL_DAYS_CALENDAR,
    )
    with (
        OptionsEngine(num_threads=1) as engine,
        pytest.raises(TypeError, match="curve_aware must be a boolean"),
    ):
        engine.price_dated_option(
            _contract(),
            environment,
            curve_aware="yes",  # type: ignore[arg-type]
        )


def test_curve_aware_dated_orchestration_preserves_seed_validation() -> None:
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.03,
        calendar=ALL_DAYS_CALENDAR,
    )
    with (
        OptionsEngine(num_threads=1) as engine,
        pytest.raises(ValueError, match="seed must be within"),
    ):
        engine.price_dated_option(
            _contract(),
            environment,
            model_name="finite_difference_400",
            override_volatility=0.2,
            seed=-1,
        )
