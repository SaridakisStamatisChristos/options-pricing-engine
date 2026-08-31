"""Deterministic continuously compounded and discount-factor curves."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from numbers import Real
from typing import Protocol, runtime_checkable

from .dates import DayCountConvention, ValuationDate

MAX_ABS_CONTINUOUS_RATE = 1.0
MAX_CURVE_NODES = 10_000


def _civil_date(name: str, value: object) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError(f"{name} must be a datetime.date, not a datetime")
    return value


def _continuous_rate(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    rate = float(value)
    if not math.isfinite(rate) or not -MAX_ABS_CONTINUOUS_RATE <= rate <= MAX_ABS_CONTINUOUS_RATE:
        raise ValueError(
            f"{name} must be finite and within "
            f"[-{MAX_ABS_CONTINUOUS_RATE:g}, {MAX_ABS_CONTINUOUS_RATE:g}]"
        )
    return rate


def _positive_factor(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    factor = float(value)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return factor


def _discount_from_integrated_rate(integrated_rate: float) -> float:
    try:
        factor = math.exp(-integrated_rate)
    except OverflowError as exc:
        raise ValueError("curve evaluation exceeds floating-point range") from exc
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("curve evaluation exceeds floating-point range")
    return factor


def _fingerprint(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ExtrapolationMethod(StrEnum):
    """Explicit behavior after the final curve node."""

    RAISE = "raise"
    FLAT_ZERO = "flat_zero"
    FLAT_FORWARD = "flat_forward"


@runtime_checkable
class DiscountCurve(Protocol):
    """Structural interface required of a funding/discount curve."""

    @property
    def reference_date(self) -> ValuationDate: ...

    @property
    def day_count(self) -> DayCountConvention: ...

    @property
    def curve_id(self) -> str: ...

    def discount_factor(self, maturity: date) -> float: ...

    def zero_rate(self, maturity: date) -> float: ...

    def forward_discount_factor(self, start: date, end: date) -> float: ...


@runtime_checkable
class CarryCurve(Protocol):
    """Structural interface for continuous dividend or other carry factors."""

    @property
    def reference_date(self) -> ValuationDate: ...

    @property
    def day_count(self) -> DayCountConvention: ...

    @property
    def curve_id(self) -> str: ...

    def carry_factor(self, maturity: date) -> float: ...

    def zero_rate(self, maturity: date) -> float: ...

    def forward_carry_factor(self, start: date, end: date) -> float: ...


@dataclass(frozen=True, slots=True)
class ZeroRateNode:
    """One continuously compounded zero-rate observation."""

    maturity: date
    rate: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "maturity", _civil_date("node maturity", self.maturity))
        object.__setattr__(self, "rate", _continuous_rate("node rate", self.rate))


@dataclass(frozen=True, slots=True)
class DiscountFactorNode:
    """One positive discount-factor observation."""

    maturity: date
    discount_factor: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "maturity", _civil_date("node maturity", self.maturity))
        object.__setattr__(
            self,
            "discount_factor",
            _positive_factor("node discount_factor", self.discount_factor),
        )


class _DiscountCurveMethods:
    reference_date: ValuationDate

    def discount_factor(self, maturity: date) -> float:
        raise NotImplementedError

    def forward_discount_factor(self, start: date, end: date) -> float:
        start_date = _civil_date("start", start)
        end_date = _civil_date("end", end)
        if end_date < start_date:
            raise ValueError("forward discount interval requires end on or after start")
        start_factor = self.discount_factor(start_date)
        end_factor = self.discount_factor(end_date)
        return _positive_factor("forward discount factor", end_factor / start_factor)


@dataclass(frozen=True, slots=True)
class FlatDiscountCurve(_DiscountCurveMethods):
    """A flat continuously compounded funding curve."""

    reference_date: ValuationDate
    rate: float
    day_count: DayCountConvention = DayCountConvention.ACTUAL_365_FIXED

    def __post_init__(self) -> None:
        if not isinstance(self.reference_date, ValuationDate):
            raise TypeError("reference_date must be a ValuationDate")
        if not isinstance(self.day_count, DayCountConvention):
            raise TypeError("day_count must be a DayCountConvention")
        object.__setattr__(self, "rate", _continuous_rate("rate", self.rate))

    @property
    def curve_id(self) -> str:
        return _fingerprint(
            {
                "curve_type": type(self).__name__,
                "day_count": self.day_count.value,
                "rate": self.rate.hex(),
                "reference_date": self.reference_date.value.isoformat(),
            }
        )

    def _time(self, maturity: date) -> float:
        maturity_date = _civil_date("maturity", maturity)
        if maturity_date < self.reference_date.value:
            raise ValueError("curve maturity cannot precede its reference date")
        return self.day_count.year_fraction(self.reference_date.value, maturity_date)

    def discount_factor(self, maturity: date) -> float:
        return _discount_from_integrated_rate(self.rate * self._time(maturity))

    def zero_rate(self, maturity: date) -> float:
        self._time(maturity)
        return self.rate


class FlatDividendCurve(FlatDiscountCurve):
    """A flat continuous dividend/carry curve."""

    __slots__ = ()

    def carry_factor(self, maturity: date) -> float:
        return self.discount_factor(maturity)

    def forward_carry_factor(self, start: date, end: date) -> float:
        return self.forward_discount_factor(start, end)


@dataclass(frozen=True, slots=True)
class ContinuousZeroCurve(_DiscountCurveMethods):
    """Zero curve with linear interpolation in continuous zero rates.

    The first quoted zero rate is flat back to the reference date. Beyond the
    last node, callers must choose rejection, flat-zero, or flat-forward
    extrapolation explicitly.
    """

    reference_date: ValuationDate
    nodes: tuple[ZeroRateNode, ...]
    day_count: DayCountConvention = DayCountConvention.ACTUAL_365_FIXED
    extrapolation: ExtrapolationMethod = ExtrapolationMethod.RAISE
    _times: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference_date, ValuationDate):
            raise TypeError("reference_date must be a ValuationDate")
        if not isinstance(self.day_count, DayCountConvention):
            raise TypeError("day_count must be a DayCountConvention")
        if not isinstance(self.extrapolation, ExtrapolationMethod):
            raise TypeError("extrapolation must be an ExtrapolationMethod")
        if isinstance(self.nodes, (str, bytes)):
            raise TypeError("nodes must be an iterable of ZeroRateNode values")
        try:
            nodes = tuple(self.nodes)
        except TypeError as exc:
            raise TypeError("nodes must be an iterable of ZeroRateNode values") from exc
        if not 1 <= len(nodes) <= MAX_CURVE_NODES:
            raise ValueError(f"zero curve must contain between 1 and {MAX_CURVE_NODES} nodes")
        if any(not isinstance(node, ZeroRateNode) for node in nodes):
            raise TypeError("every zero curve node must be a ZeroRateNode")

        times: list[float] = []
        previous_date = self.reference_date.value
        previous_time = 0.0
        for node in nodes:
            if node.maturity <= previous_date:
                raise ValueError("zero curve node maturities must be strictly increasing")
            time = self.day_count.year_fraction(self.reference_date.value, node.maturity)
            if not math.isfinite(time) or time <= previous_time:
                raise ValueError("zero curve node year fractions must be strictly increasing")
            times.append(time)
            previous_date = node.maturity
            previous_time = time

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "_times", tuple(times))

    @property
    def curve_id(self) -> str:
        return _fingerprint(
            {
                "curve_type": type(self).__name__,
                "day_count": self.day_count.value,
                "extrapolation": self.extrapolation.value,
                "nodes": [
                    {"maturity": node.maturity.isoformat(), "rate": node.rate.hex()}
                    for node in self.nodes
                ],
                "reference_date": self.reference_date.value.isoformat(),
            }
        )

    def _time(self, maturity: date) -> float:
        maturity_date = _civil_date("maturity", maturity)
        if maturity_date < self.reference_date.value:
            raise ValueError("curve maturity cannot precede its reference date")
        return self.day_count.year_fraction(self.reference_date.value, maturity_date)

    def _integrated_rate(self, time: float) -> float:
        rates = tuple(node.rate for node in self.nodes)
        if time <= self._times[0]:
            return rates[0] * time

        index = bisect.bisect_left(self._times, time)
        if index < len(self._times):
            left_time = self._times[index - 1]
            right_time = self._times[index]
            weight = (time - left_time) / (right_time - left_time)
            rate = rates[index - 1] + weight * (rates[index] - rates[index - 1])
            return rate * time

        if self.extrapolation is ExtrapolationMethod.RAISE:
            raise ValueError("curve maturity exceeds the final node and extrapolation='raise'")
        last_time = self._times[-1]
        last_integrated = rates[-1] * last_time
        if self.extrapolation is ExtrapolationMethod.FLAT_ZERO:
            return rates[-1] * time
        if len(self._times) == 1:
            last_forward = rates[-1]
        else:
            previous_integrated = rates[-2] * self._times[-2]
            last_forward = (last_integrated - previous_integrated) / (last_time - self._times[-2])
        return last_integrated + last_forward * (time - last_time)

    def discount_factor(self, maturity: date) -> float:
        return _discount_from_integrated_rate(self._integrated_rate(self._time(maturity)))

    def zero_rate(self, maturity: date) -> float:
        time = self._time(maturity)
        if time == 0.0:
            return self.nodes[0].rate
        return self._integrated_rate(time) / time


class ContinuousDividendCurve(ContinuousZeroCurve):
    """Continuous dividend/carry zero curve with linear zero interpolation."""

    __slots__ = ()

    def carry_factor(self, maturity: date) -> float:
        return self.discount_factor(maturity)

    def forward_carry_factor(self, start: date, end: date) -> float:
        return self.forward_discount_factor(start, end)


@dataclass(frozen=True, slots=True)
class DiscountFactorCurve(_DiscountCurveMethods):
    """Curve with log-linear interpolation of positive discount factors."""

    reference_date: ValuationDate
    nodes: tuple[DiscountFactorNode, ...]
    day_count: DayCountConvention = DayCountConvention.ACTUAL_365_FIXED
    extrapolation: ExtrapolationMethod = ExtrapolationMethod.RAISE
    _times: tuple[float, ...] = field(init=False, repr=False, compare=False)
    _log_factors: tuple[float, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference_date, ValuationDate):
            raise TypeError("reference_date must be a ValuationDate")
        if not isinstance(self.day_count, DayCountConvention):
            raise TypeError("day_count must be a DayCountConvention")
        if not isinstance(self.extrapolation, ExtrapolationMethod):
            raise TypeError("extrapolation must be an ExtrapolationMethod")
        if isinstance(self.nodes, (str, bytes)):
            raise TypeError("nodes must be an iterable of DiscountFactorNode values")
        try:
            nodes = tuple(self.nodes)
        except TypeError as exc:
            raise TypeError("nodes must be an iterable of DiscountFactorNode values") from exc
        if not 1 <= len(nodes) <= MAX_CURVE_NODES:
            raise ValueError(
                f"discount-factor curve must contain between 1 and {MAX_CURVE_NODES} nodes"
            )
        if any(not isinstance(node, DiscountFactorNode) for node in nodes):
            raise TypeError("every discount-factor curve node must be a DiscountFactorNode")

        times: list[float] = []
        logs: list[float] = []
        previous_date = self.reference_date.value
        previous_time = 0.0
        for node in nodes:
            if node.maturity <= previous_date:
                raise ValueError(
                    "discount-factor curve node maturities must be strictly increasing"
                )
            time = self.day_count.year_fraction(self.reference_date.value, node.maturity)
            if not math.isfinite(time) or time <= previous_time:
                raise ValueError(
                    "discount-factor curve node year fractions must be strictly increasing"
                )
            log_factor = math.log(node.discount_factor)
            _continuous_rate("node implied zero rate", -log_factor / time)
            times.append(time)
            logs.append(log_factor)
            previous_date = node.maturity
            previous_time = time

        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "_times", tuple(times))
        object.__setattr__(self, "_log_factors", tuple(logs))

    @property
    def curve_id(self) -> str:
        return _fingerprint(
            {
                "curve_type": type(self).__name__,
                "day_count": self.day_count.value,
                "extrapolation": self.extrapolation.value,
                "nodes": [
                    {
                        "discount_factor": node.discount_factor.hex(),
                        "maturity": node.maturity.isoformat(),
                    }
                    for node in self.nodes
                ],
                "reference_date": self.reference_date.value.isoformat(),
            }
        )

    def _time(self, maturity: date) -> float:
        maturity_date = _civil_date("maturity", maturity)
        if maturity_date < self.reference_date.value:
            raise ValueError("curve maturity cannot precede its reference date")
        return self.day_count.year_fraction(self.reference_date.value, maturity_date)

    def _log_discount(self, time: float) -> float:
        if time <= self._times[0]:
            return self._log_factors[0] * time / self._times[0]

        index = bisect.bisect_left(self._times, time)
        if index < len(self._times):
            left_time = self._times[index - 1]
            right_time = self._times[index]
            weight = (time - left_time) / (right_time - left_time)
            return self._log_factors[index - 1] + weight * (
                self._log_factors[index] - self._log_factors[index - 1]
            )

        if self.extrapolation is ExtrapolationMethod.RAISE:
            raise ValueError("curve maturity exceeds the final node and extrapolation='raise'")
        last_time = self._times[-1]
        last_log = self._log_factors[-1]
        if self.extrapolation is ExtrapolationMethod.FLAT_ZERO:
            return last_log * time / last_time
        if len(self._times) == 1:
            last_forward_slope = last_log / last_time
        else:
            last_forward_slope = (last_log - self._log_factors[-2]) / (last_time - self._times[-2])
        return last_log + last_forward_slope * (time - last_time)

    def discount_factor(self, maturity: date) -> float:
        log_discount = self._log_discount(self._time(maturity))
        return _discount_from_integrated_rate(-log_discount)

    def zero_rate(self, maturity: date) -> float:
        time = self._time(maturity)
        if time == 0.0:
            return -self._log_factors[0] / self._times[0]
        return -self._log_discount(time) / time


class DividendFactorCurve(DiscountFactorCurve):
    """Continuous dividend/carry curve quoted as positive carry factors."""

    __slots__ = ()

    def carry_factor(self, maturity: date) -> float:
        return self.discount_factor(maturity)

    def forward_carry_factor(self, start: date, end: date) -> float:
        return self.forward_discount_factor(start, end)


__all__ = [
    "CarryCurve",
    "ContinuousDividendCurve",
    "ContinuousZeroCurve",
    "DiscountCurve",
    "DiscountFactorCurve",
    "DiscountFactorNode",
    "DividendFactorCurve",
    "ExtrapolationMethod",
    "FlatDiscountCurve",
    "FlatDividendCurve",
    "ZeroRateNode",
]
