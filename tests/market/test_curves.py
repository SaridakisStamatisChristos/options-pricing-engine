"""Interpolation, extrapolation, and validation tests for market curves."""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import pytest

from options_engine.market import (
    CarryCurve,
    ContinuousDividendCurve,
    ContinuousZeroCurve,
    DayCountConvention,
    DiscountCurve,
    DiscountFactorCurve,
    DiscountFactorNode,
    DividendFactorCurve,
    ExtrapolationMethod,
    FlatDiscountCurve,
    FlatDividendCurve,
    ValuationDate,
    ZeroRateNode,
)

REFERENCE = ValuationDate(date(2025, 1, 1))


def test_flat_curves_support_negative_rates_and_forward_factors() -> None:
    funding = FlatDiscountCurve(REFERENCE, -0.01)
    dividend = FlatDividendCurve(REFERENCE, 0.02)
    maturity = date(2026, 1, 1)

    assert isinstance(funding, DiscountCurve)
    assert isinstance(dividend, CarryCurve)
    assert funding.discount_factor(maturity) == pytest.approx(math.exp(0.01))
    assert dividend.carry_factor(maturity) == pytest.approx(math.exp(-0.02))
    assert funding.forward_discount_factor(REFERENCE.value, maturity) == pytest.approx(
        math.exp(0.01)
    )
    assert funding.zero_rate(REFERENCE.value) == -0.01


def test_continuous_zero_curve_interpolates_linearly_in_zero_rates() -> None:
    first = date(2026, 1, 1)
    second = date(2027, 1, 1)
    middle = date(2026, 7, 2)
    curve = ContinuousZeroCurve(
        REFERENCE,
        (ZeroRateNode(first, 0.02), ZeroRateNode(second, 0.04)),
    )
    day_count = DayCountConvention.ACTUAL_365_FIXED
    first_time = day_count.year_fraction(REFERENCE.value, first)
    second_time = day_count.year_fraction(REFERENCE.value, second)
    middle_time = day_count.year_fraction(REFERENCE.value, middle)
    weight = (middle_time - first_time) / (second_time - first_time)
    expected_rate = 0.02 + weight * 0.02

    assert curve.zero_rate(date(2025, 7, 1)) == pytest.approx(0.02)
    assert curve.zero_rate(middle) == pytest.approx(expected_rate)
    assert curve.discount_factor(middle) == pytest.approx(math.exp(-expected_rate * middle_time))
    assert curve.zero_rate(second) == pytest.approx(0.04)


def test_zero_curve_extrapolation_policies_are_distinct() -> None:
    nodes = (
        ZeroRateNode(date(2026, 1, 1), 0.02),
        ZeroRateNode(date(2027, 1, 1), 0.04),
    )
    maturity = date(2028, 1, 1)
    strict = ContinuousZeroCurve(REFERENCE, nodes)
    flat_zero = ContinuousZeroCurve(REFERENCE, nodes, extrapolation=ExtrapolationMethod.FLAT_ZERO)
    flat_forward = ContinuousZeroCurve(
        REFERENCE, nodes, extrapolation=ExtrapolationMethod.FLAT_FORWARD
    )

    with pytest.raises(ValueError, match="extrapolation='raise'"):
        strict.discount_factor(maturity)
    assert flat_zero.zero_rate(maturity) == pytest.approx(0.04)
    assert flat_forward.zero_rate(maturity) > flat_zero.zero_rate(maturity)


def test_discount_factor_curve_is_log_linear_from_the_reference_date() -> None:
    first = date(2026, 1, 1)
    second = date(2027, 1, 1)
    first_df = math.exp(-0.02)
    second_df = math.exp(-0.06)
    curve = DiscountFactorCurve(
        REFERENCE,
        (
            DiscountFactorNode(first, first_df),
            DiscountFactorNode(second, second_df),
        ),
    )
    half_first = date(2025, 7, 2)
    between = date(2026, 7, 2)
    t1 = curve.day_count.year_fraction(REFERENCE.value, first)
    t2 = curve.day_count.year_fraction(REFERENCE.value, second)
    th = curve.day_count.year_fraction(REFERENCE.value, half_first)
    tb = curve.day_count.year_fraction(REFERENCE.value, between)
    expected_first_log = math.log(first_df) * th / t1
    weight = (tb - t1) / (t2 - t1)
    expected_between_log = math.log(first_df) + weight * (math.log(second_df) - math.log(first_df))

    assert curve.discount_factor(REFERENCE.value) == 1.0
    assert curve.discount_factor(half_first) == pytest.approx(math.exp(expected_first_log))
    assert curve.discount_factor(between) == pytest.approx(math.exp(expected_between_log))
    assert curve.discount_factor(second) == pytest.approx(second_df)


def test_discount_factor_extrapolation_can_be_flat_zero_or_flat_forward() -> None:
    nodes = (
        DiscountFactorNode(date(2026, 1, 1), math.exp(-0.02)),
        DiscountFactorNode(date(2027, 1, 1), math.exp(-0.06)),
    )
    maturity = date(2028, 1, 1)
    flat_zero = DiscountFactorCurve(REFERENCE, nodes, extrapolation=ExtrapolationMethod.FLAT_ZERO)
    flat_forward = DiscountFactorCurve(
        REFERENCE, nodes, extrapolation=ExtrapolationMethod.FLAT_FORWARD
    )

    assert flat_forward.discount_factor(maturity) < flat_zero.discount_factor(maturity)
    assert flat_forward.zero_rate(maturity) > flat_zero.zero_rate(maturity)


def test_dividend_curves_expose_carry_semantics_for_both_quote_styles() -> None:
    expiry = date(2026, 1, 1)
    continuous = ContinuousDividendCurve(
        REFERENCE,
        (ZeroRateNode(expiry, 0.015),),
    )
    factor = DividendFactorCurve(
        REFERENCE,
        (DiscountFactorNode(expiry, math.exp(-0.015)),),
    )

    assert continuous.carry_factor(expiry) == pytest.approx(math.exp(-0.015))
    assert factor.carry_factor(expiry) == pytest.approx(math.exp(-0.015))
    assert continuous.forward_carry_factor(REFERENCE.value, expiry) == pytest.approx(
        factor.forward_carry_factor(REFERENCE.value, expiry)
    )


def test_curve_ids_are_stable_and_semantically_typed() -> None:
    nodes = (ZeroRateNode(date(2026, 1, 1), 0.02),)
    first = ContinuousZeroCurve(REFERENCE, nodes)
    second = ContinuousZeroCurve(REFERENCE, tuple(nodes))
    dividend = ContinuousDividendCurve(REFERENCE, nodes)

    assert first.curve_id == second.curve_id
    assert first.curve_id != dividend.curve_id
    assert (
        FlatDiscountCurve(REFERENCE, 0.02).curve_id != FlatDividendCurve(REFERENCE, 0.02).curve_id
    )


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    [
        (
            lambda: ContinuousZeroCurve(REFERENCE, ()),
            ValueError,
            "between 1",
        ),
        (
            lambda: ContinuousZeroCurve(
                REFERENCE,
                (
                    ZeroRateNode(date(2027, 1, 1), 0.02),
                    ZeroRateNode(date(2026, 1, 1), 0.03),
                ),
            ),
            ValueError,
            "strictly increasing",
        ),
        (
            lambda: DiscountFactorCurve(
                REFERENCE,
                (DiscountFactorNode(date(2026, 1, 1), math.exp(-2.0)),),
            ),
            ValueError,
            "implied zero rate",
        ),
        (lambda: ZeroRateNode(date(2026, 1, 1), True), TypeError, "real number"),
        (lambda: DiscountFactorNode(date(2026, 1, 1), 0.0), ValueError, "positive"),
        (lambda: FlatDiscountCurve(REFERENCE, 1.1), ValueError, "within"),
    ],
)
def test_curve_validation(factory: Any, error_type: type[Exception], message: str) -> None:
    with pytest.raises(error_type, match=message):
        factory()


def test_curve_queries_reject_dates_before_reference_and_reverse_forwards() -> None:
    curve = FlatDiscountCurve(REFERENCE, 0.02)
    with pytest.raises(ValueError, match="precede"):
        curve.discount_factor(date(2024, 12, 31))
    with pytest.raises(ValueError, match="end on or after start"):
        curve.forward_discount_factor(date(2026, 1, 1), date(2025, 1, 1))


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    [
        (
            lambda: FlatDiscountCurve(date(2025, 1, 1), 0.02),
            TypeError,
            "ValuationDate",
        ),
        (
            lambda: FlatDiscountCurve(REFERENCE, 0.02, "actual_365_fixed"),
            TypeError,
            "DayCountConvention",
        ),
        (
            lambda: ContinuousZeroCurve(REFERENCE, "nodes"),
            TypeError,
            "iterable",
        ),
        (
            lambda: ContinuousZeroCurve(REFERENCE, (object(),)),
            TypeError,
            "ZeroRateNode",
        ),
        (
            lambda: ContinuousZeroCurve(
                REFERENCE,
                (ZeroRateNode(date(2026, 1, 1), 0.02),),
                extrapolation="raise",
            ),
            TypeError,
            "ExtrapolationMethod",
        ),
        (
            lambda: DiscountFactorCurve(REFERENCE, "nodes"),
            TypeError,
            "iterable",
        ),
        (
            lambda: DiscountFactorCurve(REFERENCE, (object(),)),
            TypeError,
            "DiscountFactorNode",
        ),
        (
            lambda: DiscountFactorNode(date(2026, 1, 1), True),
            TypeError,
            "real number",
        ),
    ],
)
def test_curve_type_validation(factory: Any, error_type: type[Exception], message: str) -> None:
    with pytest.raises(error_type, match=message):
        factory()


def test_single_node_flat_forward_extrapolation_matches_terminal_zero() -> None:
    zero = ContinuousZeroCurve(
        REFERENCE,
        (ZeroRateNode(date(2026, 1, 1), 0.025),),
        extrapolation=ExtrapolationMethod.FLAT_FORWARD,
    )
    discount = DiscountFactorCurve(
        REFERENCE,
        (DiscountFactorNode(date(2026, 1, 1), math.exp(-0.025)),),
        extrapolation=ExtrapolationMethod.FLAT_FORWARD,
    )

    assert zero.zero_rate(date(2028, 1, 1)) == pytest.approx(0.025)
    assert discount.zero_rate(date(2028, 1, 1)) == pytest.approx(0.025)


def test_curve_evaluation_fails_closed_on_overflow_or_underflow() -> None:
    early_reference = ValuationDate(date.min)
    with pytest.raises(ValueError, match="floating-point range"):
        FlatDiscountCurve(early_reference, 1.0).discount_factor(date.max)
    with pytest.raises(ValueError, match="floating-point range"):
        FlatDiscountCurve(early_reference, -1.0).discount_factor(date.max)


def test_curve_nodes_and_queries_reject_datetime_maturities() -> None:
    with pytest.raises(TypeError, match="not a datetime"):
        ZeroRateNode(datetime(2026, 1, 1), 0.02)
    with pytest.raises(TypeError, match="not a datetime"):
        FlatDiscountCurve(REFERENCE, 0.02).discount_factor(datetime(2026, 1, 1))
