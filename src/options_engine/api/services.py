"""Utility helpers shared across API routes."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from numbers import Integral, Real
from typing import Any


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def enrich_pricing_result(result: dict[str, Any], quantity: int) -> dict[str, Any]:
    """Augment a single pricing result with position-level analytics.

    Expects `result` to potentially include:
      - theoretical_price: float
      - standard_error: float | None
      - confidence_interval: (lower: float, upper: float) | None
      - greeks: keys may be direct (delta, gamma, theta, vega, rho) in result
    """
    enriched: dict[str, Any] = dict(result)  # copy so we don't mutate input

    # Quantity: must be integer-like and >= 1
    if isinstance(quantity, bool) or not isinstance(quantity, Integral):
        raise ValueError("quantity must be an integer between 1 and 1000000")
    q_int = int(quantity)
    if not 1 <= q_int <= 1_000_000:
        raise ValueError("quantity must be an integer between 1 and 1000000")

    qty = float(q_int)
    enriched["quantity"] = qty

    # Position value
    theoretical_price = _finite_number("theoretical_price", enriched.get("theoretical_price"))
    if theoretical_price < 0.0:
        raise ValueError("theoretical_price must be non-negative")
    enriched["position_value"] = theoretical_price * qty

    # Standard error (scaled)
    standard_error = enriched.get("standard_error")
    if standard_error is not None:
        standard_error_value = _finite_number("standard_error", standard_error)
        if standard_error_value < 0.0:
            raise ValueError("standard_error must be non-negative")
        enriched["position_standard_error"] = standard_error_value * qty

    # Confidence interval (scaled)
    confidence_interval = enriched.get("confidence_interval")
    if confidence_interval is not None:
        if (
            isinstance(confidence_interval, Sequence)
            and not isinstance(confidence_interval, (str, bytes))
            and len(confidence_interval) == 2
        ):
            lower, upper = confidence_interval
            lower_value = _finite_number("confidence_interval lower bound", lower)
            upper_value = _finite_number("confidence_interval upper bound", upper)
            if lower_value > upper_value:
                raise ValueError("confidence_interval lower bound must not exceed upper bound")
            enriched["position_confidence_interval"] = (
                lower_value * qty,
                upper_value * qty,
            )
        else:
            # If malformed, surface a clear error rather than failing later
            raise ValueError("confidence_interval must be a 2-item (lower, upper) sequence")

    # Scale greeks to position-level metrics
    for greek in ("delta", "gamma", "theta", "vega", "rho"):
        value = enriched.get(greek)
        if value is not None:
            enriched[f"position_{greek}"] = _finite_number(greek, value) * qty

    return enriched


def annotate_results_with_quantity(
    results: Iterable[dict[str, Any]], quantities: Iterable[int]
) -> tuple[dict[str, Any], ...]:
    """Zip results with quantities, enrich each, and return an immutable tuple.

    Defensively copies both iterables so generators are consumed exactly once
    and verifies lengths match to avoid silent truncation.
    """
    result_items = tuple(results)
    quantity_items = tuple(quantities)

    if len(result_items) != len(quantity_items):
        raise ValueError("The number of pricing results does not match the number of quantities")

    enriched_results = [
        enrich_pricing_result(result, quantity)
        for result, quantity in zip(result_items, quantity_items, strict=True)
    ]
    return tuple(enriched_results)
