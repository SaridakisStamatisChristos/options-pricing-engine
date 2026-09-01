"""Date-free deterministic term structures for numerical pricing kernels.

Market curves are quoted on civil dates.  Numerical models, in contrast, work
on a continuous model-time axis.  This module is the deliberately small seam
between those two domains: it stores the logarithm of each positive factor as
immutable piecewise-quadratic segments and evaluates interval factors by exact
log-factor differences.

For ``0 <= t0 < t1 <= T`` the conventions are

``D(t0, t1) = D(0, t1) / D(0, t0)`` and
``Q(t0, t1) = Q(0, t1) / Q(0, t0)``.

The step-equivalent continuously compounded rates are therefore
``r = -log(D(t0, t1)) / (t1 - t0)`` and
``q = -log(Q(t0, t1)) / (t1 - t0)``.  Pricing schemes consume these interval
quantities directly; no numerical differentiation or expiry-flat reduction is
required.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import pairwise
from numbers import Real

MAX_TERM_STRUCTURE_SEGMENTS = 20_000
_TIME_TOLERANCE = 64.0 * math.ulp(1.0)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _model_time(name: str, value: object, *, maturity: float) -> float:
    model_time = _finite_float(name, value)
    tolerance = _TIME_TOLERANCE * max(1.0, maturity)
    if model_time < -tolerance or model_time > maturity + tolerance:
        raise ValueError(f"{name} must be within [0, {maturity:g}]")
    if abs(model_time) <= tolerance:
        return 0.0
    if abs(model_time - maturity) <= tolerance:
        return maturity
    return model_time


@dataclass(frozen=True, slots=True)
class LogFactorSegment:
    """One local quadratic representation of ``log(factor(0, t))``.

    With ``x = (t - start_time) / (end_time - start_time)``, the segment is
    ``constant + linear*x + quadratic*x*x``.  A quadratic is sufficient for
    both log-linear factor curves and linearly interpolated continuous zero
    rates, including an affine re-parameterisation from civil-date curve time
    to model time.
    """

    start_time: float
    end_time: float
    constant: float
    linear: float
    quadratic: float = 0.0

    def __post_init__(self) -> None:
        start = _finite_float("segment start_time", self.start_time)
        end = _finite_float("segment end_time", self.end_time)
        if start < 0.0 or end <= start:
            raise ValueError("factor segments require 0 <= start_time < end_time")
        object.__setattr__(self, "start_time", start)
        object.__setattr__(self, "end_time", end)
        for name in ("constant", "linear", "quadratic"):
            object.__setattr__(self, name, _finite_float(f"segment {name}", getattr(self, name)))

    def log_factor(self, model_time: float) -> float:
        """Evaluate the segment at a validated in-segment model time."""

        tolerance = _TIME_TOLERANCE * max(1.0, self.end_time)
        if model_time < self.start_time - tolerance or model_time > self.end_time + tolerance:
            raise ValueError("model time is outside the selected factor segment")
        if model_time <= self.start_time:
            coordinate = 0.0
        elif model_time >= self.end_time:
            coordinate = 1.0
        else:
            coordinate = (model_time - self.start_time) / (self.end_time - self.start_time)
        value = self.constant + coordinate * (self.linear + coordinate * self.quadratic)
        if not math.isfinite(value):
            raise ValueError("term-structure log factor exceeds floating-point range")
        return value

    def instantaneous_rate(self, model_time: float) -> float:
        """Return ``-d log(factor(0,t))/dt`` inside this exact segment."""

        tolerance = _TIME_TOLERANCE * max(1.0, self.end_time)
        if model_time < self.start_time - tolerance or model_time > self.end_time + tolerance:
            raise ValueError("model time is outside the selected factor segment")
        coordinate = min(
            1.0,
            max(
                0.0,
                (model_time - self.start_time) / (self.end_time - self.start_time),
            ),
        )
        derivative = (self.linear + 2.0 * self.quadratic * coordinate) / (
            self.end_time - self.start_time
        )
        rate = -derivative
        if not math.isfinite(rate):
            raise ValueError("term-structure instantaneous rate is non-finite")
        return rate


@dataclass(frozen=True, slots=True)
class DeterministicFactorCurve:
    """Positive deterministic factor curve on a bounded model-time horizon."""

    curve_id: str
    maturity: float
    segments: tuple[LogFactorSegment, ...]
    node_times: tuple[float, ...] = ()
    _segment_ends: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.curve_id, str) or not self.curve_id:
            raise ValueError("curve_id must be a non-empty string")
        maturity = _finite_float("curve maturity", self.maturity)
        if maturity <= 0.0:
            raise ValueError("curve maturity must be strictly positive")
        object.__setattr__(self, "maturity", maturity)
        if not isinstance(self.segments, tuple):
            raise TypeError("segments must be a tuple of LogFactorSegment values")
        if not 1 <= len(self.segments) <= MAX_TERM_STRUCTURE_SEGMENTS:
            raise ValueError(
                f"factor curve must contain between 1 and {MAX_TERM_STRUCTURE_SEGMENTS} segments"
            )
        if any(not isinstance(segment, LogFactorSegment) for segment in self.segments):
            raise TypeError("every factor-curve segment must be a LogFactorSegment")

        tolerance = _TIME_TOLERANCE * max(1.0, maturity)
        if abs(self.segments[0].start_time) > tolerance:
            raise ValueError("factor-curve segments must begin at model time zero")
        previous_end = 0.0
        previous_log = 0.0
        for index, segment in enumerate(self.segments):
            if abs(segment.start_time - previous_end) > tolerance:
                raise ValueError("factor-curve segments must be contiguous")
            start_log = segment.log_factor(segment.start_time)
            if index == 0 and abs(start_log) > 1e-12:
                raise ValueError("factor curve must equal one at model time zero")
            if index and not math.isclose(
                start_log,
                previous_log,
                rel_tol=0.0,
                abs_tol=2e-12 * max(1.0, abs(start_log), abs(previous_log)),
            ):
                raise ValueError("factor-curve segments must be continuous")
            previous_end = segment.end_time
            previous_log = segment.log_factor(segment.end_time)
        if abs(previous_end - maturity) > tolerance:
            raise ValueError("factor-curve segments must end at the pricing maturity")

        if not isinstance(self.node_times, tuple):
            raise TypeError("node_times must be a tuple")
        normalised_nodes = tuple(
            _model_time("curve node time", node, maturity=maturity) for node in self.node_times
        )
        if any(node <= 0.0 or node >= maturity for node in normalised_nodes):
            raise ValueError("curve node times must lie strictly inside the pricing horizon")
        if any(right <= left for left, right in pairwise(normalised_nodes)):
            raise ValueError("curve node times must be strictly increasing and unique")
        boundaries = tuple(segment.end_time for segment in self.segments[:-1])
        for node in normalised_nodes:
            if not any(
                math.isclose(node, value, rel_tol=0.0, abs_tol=tolerance) for value in boundaries
            ):
                raise ValueError("every curve node time must be a factor-segment boundary")
        object.__setattr__(self, "node_times", normalised_nodes)
        object.__setattr__(
            self,
            "_segment_ends",
            tuple(segment.end_time for segment in self.segments),
        )

    def _segment(self, model_time: float) -> LogFactorSegment:
        index = bisect.bisect_left(self._segment_ends, model_time)
        return self.segments[min(index, len(self.segments) - 1)]

    def log_factor(self, model_time: float) -> float:
        """Return ``log(factor(0,t))`` without exponentiation loss."""

        time = _model_time("model_time", model_time, maturity=self.maturity)
        if time == 0.0:
            return 0.0
        return self._segment(time).log_factor(time)

    def factor(self, model_time: float) -> float:
        """Return the positive factor from valuation to ``model_time``."""

        return self._exp_log_factor(self.log_factor(model_time))

    @staticmethod
    def _exp_log_factor(log_factor: float) -> float:
        try:
            factor = math.exp(log_factor)
        except OverflowError as exc:
            raise ValueError("term-structure factor exceeds floating-point range") from exc
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("term-structure factor exceeds floating-point range")
        return factor

    def interval_factor(self, start_time: float, end_time: float) -> float:
        """Return the exact supplied factor ratio over ``[start_time,end_time]``."""

        start = _model_time("start_time", start_time, maturity=self.maturity)
        end = _model_time("end_time", end_time, maturity=self.maturity)
        if end < start:
            raise ValueError("factor interval requires end_time on or after start_time")
        if end == start:
            return 1.0
        return self._exp_log_factor(self.log_factor(end) - self.log_factor(start))

    def step_rate(self, start_time: float, end_time: float) -> float:
        """Return the exact interval-equivalent continuous rate."""

        start = _model_time("start_time", start_time, maturity=self.maturity)
        end = _model_time("end_time", end_time, maturity=self.maturity)
        if end <= start:
            raise ValueError("step-rate interval requires end_time after start_time")
        rate = -(self.log_factor(end) - self.log_factor(start)) / (end - start)
        if not math.isfinite(rate):
            raise ValueError("term-structure step rate is non-finite")
        return rate

    def instantaneous_rate(self, model_time: float) -> float:
        """Return the exact one-sided segment rate at ``model_time``."""

        time = _model_time("model_time", model_time, maturity=self.maturity)
        return self._segment(time).instantaneous_rate(time)


@dataclass(frozen=True, slots=True)
class DeterministicTermStructure:
    """Funding/carry factors and exact event anchors for one pricing request."""

    funding: DeterministicFactorCurve
    carry: DeterministicFactorCurve
    cash_dividend_times: tuple[float, ...] = ()
    settlement_time: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.funding, DeterministicFactorCurve):
            raise TypeError("funding must be a DeterministicFactorCurve")
        if not isinstance(self.carry, DeterministicFactorCurve):
            raise TypeError("carry must be a DeterministicFactorCurve")
        if self.funding.maturity != self.carry.maturity:
            raise ValueError("funding and carry horizons must match exactly")
        maturity = self.maturity
        if not isinstance(self.cash_dividend_times, tuple):
            raise TypeError("cash_dividend_times must be a tuple")
        dividend_times = tuple(
            _model_time("cash-dividend time", value, maturity=maturity)
            for value in self.cash_dividend_times
        )
        if any(value <= 0.0 or value >= maturity for value in dividend_times):
            raise ValueError("cash-dividend times must lie strictly inside the pricing horizon")
        if any(right <= left for left, right in pairwise(dividend_times)):
            raise ValueError("cash-dividend times must be strictly increasing and unique")
        object.__setattr__(self, "cash_dividend_times", dividend_times)
        settlement = _model_time("settlement_time", self.settlement_time, maturity=maturity)
        if settlement >= maturity:
            raise ValueError("settlement_time must precede pricing maturity")
        object.__setattr__(self, "settlement_time", settlement)

    @property
    def maturity(self) -> float:
        return self.funding.maturity

    @property
    def discount_curve_id(self) -> str:
        return self.funding.curve_id

    @property
    def carry_curve_id(self) -> str:
        return self.carry.curve_id

    @property
    def curve_node_times(self) -> tuple[float, ...]:
        return tuple(sorted({*self.funding.node_times, *self.carry.node_times}))

    @property
    def context_id(self) -> str:
        def curve_payload(curve: DeterministicFactorCurve) -> dict[str, object]:
            return {
                "curve_id": curve.curve_id,
                "maturity": curve.maturity.hex(),
                "node_times": [value.hex() for value in curve.node_times],
                "segments": [
                    {
                        "start": segment.start_time.hex(),
                        "end": segment.end_time.hex(),
                        "constant": segment.constant.hex(),
                        "linear": segment.linear.hex(),
                        "quadratic": segment.quadratic.hex(),
                    }
                    for segment in curve.segments
                ],
            }

        payload = {
            "funding": curve_payload(self.funding),
            "carry": curve_payload(self.carry),
            "cash_dividend_times": [value.hex() for value in self.cash_dividend_times],
            "settlement_time": self.settlement_time.hex(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def discount_factor(self, start_time: float, end_time: float) -> float:
        return self.funding.interval_factor(start_time, end_time)

    def carry_factor(self, start_time: float, end_time: float) -> float:
        return self.carry.interval_factor(start_time, end_time)

    def step_rates(self, start_time: float, end_time: float) -> tuple[float, float]:
        """Return exact interval-equivalent ``(funding_rate, carry_rate)``."""

        return (
            self.funding.step_rate(start_time, end_time),
            self.carry.step_rate(start_time, end_time),
        )

    def growth_factor(self, start_time: float, end_time: float) -> float:
        """Return risk-neutral stock growth ``Q(t0,t1) / D(t0,t1)``."""

        growth = self.carry_factor(start_time, end_time) / self.discount_factor(
            start_time, end_time
        )
        if not math.isfinite(growth) or growth <= 0.0:
            raise ValueError("term-structure growth factor must be finite and positive")
        return growth

    def diagnostics(self) -> dict[str, object]:
        return {
            "curve_aware_mode": True,
            "pricing_context_id": self.context_id,
            "discount_curve_id": self.discount_curve_id,
            "carry_curve_id": self.carry_curve_id,
            "funding_curve_intervals": len(self.funding.segments),
            "carry_curve_intervals": len(self.carry.segments),
            "curve_node_times": self.curve_node_times,
            "cash_dividend_times": self.cash_dividend_times,
            "settlement_time": self.settlement_time,
            "expiry_discount_factor": self.discount_factor(0.0, self.maturity),
            "expiry_carry_factor": self.carry_factor(0.0, self.maturity),
            "expiry_growth_factor": self.growth_factor(0.0, self.maturity),
            "factor_convention": "interval_ratio_exact",
            "curve_flattening": False,
        }


__all__ = [
    "DeterministicFactorCurve",
    "DeterministicTermStructure",
    "LogFactorSegment",
]
