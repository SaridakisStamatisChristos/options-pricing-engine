"""Adapt supported dated market curves to model-time pricing factors."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from itertools import pairwise

from ..term_structure import (
    DeterministicFactorCurve,
    DeterministicTermStructure,
    LogFactorSegment,
)
from .conventions import MarketConventions
from .curves import (
    CarryCurve,
    ContinuousZeroCurve,
    DiscountCurve,
    DiscountFactorCurve,
    FlatDiscountCurve,
)


def _supported_curve_nodes(curve: DiscountCurve | CarryCurve) -> tuple[date, ...]:
    if isinstance(curve, FlatDiscountCurve):
        return ()
    if isinstance(curve, (ContinuousZeroCurve, DiscountFactorCurve)):
        return tuple(node.maturity for node in curve.nodes)
    raise TypeError(
        "true curve-aware pricing supports FlatDiscountCurve, ContinuousZeroCurve, "
        "DiscountFactorCurve, and their dividend/carry counterparts; use "
        "curve_aware=False for a custom endpoint-only curve"
    )


def _source_log_factor(curve: DiscountCurve | CarryCurve, curve_time: float) -> float:
    """Evaluate a built-in curve at its native continuous year fraction."""

    if isinstance(curve, FlatDiscountCurve):
        value = -curve.rate * curve_time
    elif isinstance(curve, ContinuousZeroCurve):
        value = -curve._integrated_rate(curve_time)
    elif isinstance(curve, DiscountFactorCurve):
        value = curve._log_discount(curve_time)
    else:  # pragma: no cover - rejected by _supported_curve_nodes first
        raise TypeError("unsupported curve type for true curve-aware pricing")
    if not math.isfinite(value):
        raise ValueError("curve log factor is non-finite on the pricing horizon")
    return value


def _quadratic_segment(
    start_time: float,
    end_time: float,
    start_value: float,
    middle_value: float,
    end_value: float,
) -> LogFactorSegment:
    quadratic = 2.0 * (start_value + end_value - 2.0 * middle_value)
    linear = end_value - start_value - quadratic
    scale = max(1.0, abs(start_value), abs(middle_value), abs(end_value))
    if abs(quadratic) <= 64.0 * math.ulp(scale):
        quadratic = 0.0
        linear = end_value - start_value
    return LogFactorSegment(
        start_time=start_time,
        end_time=end_time,
        constant=start_value,
        linear=linear,
        quadratic=quadratic,
    )


def _adapt_factor_curve(
    curve: DiscountCurve | CarryCurve,
    conventions: MarketConventions,
    expiry_date: date,
    *,
    extra_anchor_dates: tuple[date, ...] = (),
    log_factor_adjustment: Callable[[float], float] | None = None,
) -> DeterministicFactorCurve:
    reference_date = conventions.valuation_date.value
    if curve.reference_date.value != reference_date:
        raise ValueError("curve reference date must equal the valuation date")
    if expiry_date <= reference_date:
        raise ValueError("pricing expiry must follow the valuation date")

    curve_node_dates = _supported_curve_nodes(curve)
    anchors = sorted(
        {
            reference_date,
            expiry_date,
            *(node for node in curve_node_dates if reference_date < node < expiry_date),
            *(node for node in extra_anchor_dates if reference_date < node < expiry_date),
        }
    )
    model_times = tuple(
        conventions.day_count.year_fraction(reference_date, anchor) for anchor in anchors
    )
    source_times = tuple(
        curve.day_count.year_fraction(reference_date, anchor) for anchor in anchors
    )
    maturity = model_times[-1]
    if not math.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("curve-aware model maturity must be finite and positive")
    if any(right <= left for left, right in pairwise(model_times)):
        raise ValueError(
            "curve anchor dates do not map to strictly increasing model times under "
            f"{conventions.day_count.value}"
        )
    if any(right <= left for left, right in pairwise(source_times)):
        raise ValueError(
            "curve anchor dates do not map to strictly increasing source-curve times under "
            f"{curve.day_count.value}"
        )

    adjustment = log_factor_adjustment or (lambda _time: 0.0)

    def log_value(model_time: float, source_time: float) -> float:
        value = _source_log_factor(curve, source_time) + adjustment(model_time)
        if not math.isfinite(value):
            raise ValueError("adjusted curve log factor is non-finite")
        return value

    segments: list[LogFactorSegment] = []
    for index in range(len(anchors) - 1):
        model_start = model_times[index]
        model_end = model_times[index + 1]
        source_start = source_times[index]
        source_end = source_times[index + 1]
        model_middle = 0.5 * (model_start + model_end)
        source_middle = 0.5 * (source_start + source_end)
        segments.append(
            _quadratic_segment(
                model_start,
                model_end,
                log_value(model_start, source_start),
                log_value(model_middle, source_middle),
                log_value(model_end, source_end),
            )
        )

    internal_anchor_times = tuple(model_times[1:-1])
    return DeterministicFactorCurve(
        curve_id=curve.curve_id,
        maturity=maturity,
        segments=tuple(segments),
        node_times=internal_anchor_times,
    )


def build_deterministic_term_structure(
    *,
    discount_curve: DiscountCurve,
    carry_curve: CarryCurve,
    conventions: MarketConventions,
    expiry_date: date,
    cash_dividend_times: tuple[float, ...] = (),
) -> DeterministicTermStructure:
    """Build exact model-time factor ratios from supported immutable curves.

    Quoted spot settles on ``conventions.settlement_date``.  For a non-zero
    lag, the small settlement basis ``log(D(0,s)/Q(0,s))`` is accrued only over
    ``[0,s]`` and then held constant.  Consequently the supplied dated forward
    is preserved, while every interval after settlement—including every valid
    cash-dividend event-to-expiry interval—uses the original carry curve ratio
    exactly.
    """

    if not isinstance(conventions, MarketConventions):
        raise TypeError("conventions must be MarketConventions")
    reference_date = conventions.valuation_date.value
    settlement_date = conventions.settlement_date
    settlement_time = conventions.day_count.year_fraction(reference_date, settlement_date)
    maturity = conventions.day_count.year_fraction(reference_date, expiry_date)
    if not 0.0 <= settlement_time < maturity:
        raise ValueError("spot settlement must lie in [valuation, expiry)")

    funding = _adapt_factor_curve(discount_curve, conventions, expiry_date)
    if settlement_time == 0.0:
        basis_adjustment: Callable[[float], float] | None = None
        carry_extra_anchors: tuple[date, ...] = ()
    else:
        settlement_discount = discount_curve.forward_discount_factor(
            reference_date, settlement_date
        )
        settlement_carry = carry_curve.forward_carry_factor(reference_date, settlement_date)
        if (
            not math.isfinite(settlement_discount)
            or settlement_discount <= 0.0
            or not math.isfinite(settlement_carry)
            or settlement_carry <= 0.0
        ):
            raise ValueError("settlement curve factors must be finite and positive")
        terminal_basis = math.log(settlement_discount / settlement_carry)

        def basis_adjustment(model_time: float) -> float:
            return terminal_basis * min(max(model_time / settlement_time, 0.0), 1.0)

        carry_extra_anchors = (settlement_date,)

    carry = _adapt_factor_curve(
        carry_curve,
        conventions,
        expiry_date,
        extra_anchor_dates=carry_extra_anchors,
        log_factor_adjustment=basis_adjustment,
    )
    context = DeterministicTermStructure(
        funding=funding,
        carry=carry,
        cash_dividend_times=cash_dividend_times,
        settlement_time=settlement_time,
    )

    expected_discount = discount_curve.forward_discount_factor(reference_date, expiry_date)
    expected_growth = carry_curve.forward_carry_factor(
        settlement_date, expiry_date
    ) / discount_curve.forward_discount_factor(settlement_date, expiry_date)
    if not math.isclose(
        context.discount_factor(0.0, maturity),
        expected_discount,
        rel_tol=2e-13,
        abs_tol=0.0,
    ):
        raise RuntimeError("funding adapter failed to preserve the supplied expiry factor")
    if not math.isclose(
        context.growth_factor(0.0, maturity),
        expected_growth,
        rel_tol=2e-13,
        abs_tol=0.0,
    ):
        raise RuntimeError("carry adapter failed to preserve the supplied dated forward")
    return context


__all__ = ["build_deterministic_term_structure"]
