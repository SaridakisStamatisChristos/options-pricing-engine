"""Utility helpers shared across API routes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple


def enrich_pricing_result(result: Dict[str, Any], quantity: int) -> Dict[str, Any]:
    """Augment a single pricing result with position-level analytics.

    Expects `result` to potentially include:
      - theoretical_price: float
      - standard_error: float | None
      - confidence_interval: (lower: float, upper: float) | None
      - greeks: keys may be direct (delta, gamma, theta, vega, rho) in result
    """
    enriched: Dict[str, Any] = dict(result)  # copy so we don't mutate input

    # Quantity: must be integer-like and >= 1
    try:
        q_int = int(quantity)
    except (TypeError, ValueError):
        raise ValueError("quantity must be an integer ≥ 1")
    if q_int < 1:
        raise ValueError("quantity must be ≥ 1")

    qty = float(q_int)
    enriched["quantity"] = qty

    # Position value
    theoretical_price = float(enriched.get("theoretical_price") or 0.0)
    enriched["position_value"] = theoretical_price * qty

    # Standard error (scaled)
    standard_error = enriched.get("standard_error")
    if standard_error is not None:
        enriched["position_standard_error"] = float(standard_error) * qty

    # Confidence interval (scaled)
    confidence_interval = enriched.get("confidence_interval")
    if confidence_interval is not None:
        if (isinstance(confidence_interval, Sequence)
                and len(confidence_interval) == 2):
            lower, upper = confidence_interval  # type: ignore[misc]
            enriched["position_confidence_interval"] = (
                float(lower) * qty,
                float(upper) * qty,
            )
        else:
            # If malformed, surface a clear error rather than failing later
            raise ValueError("confidence_interval must be a 2-item (lower, upper) sequence")

    # Scale greeks to position-level metrics
    for greek in ("delta", "gamma", "theta", "vega", "rho"):
        value = enriched.get(greek)
        if value is not None:
            enriched[f"position_{greek}"] = float(value) * qty

    return enriched


def annotate_results_with_quantity(
    results: Iterable[Dict[str, Any]], quantities: Iterable[int]
) -> Tuple[Dict[str, Any], ...]:
    """Zip results with quantities, enrich each, and return an immutable tuple.

    Defensively copies both iterables so generators are consumed exactly once
    and verifies lengths match to avoid silent truncation.
    """
    result_items = tuple(results)
    quantity_items = tuple(quantities)

    if len(result_items) != len(quantity_items):
        raise ValueError(
            "The number of pricing results does not match the number of quantities"
        )

    enriched_results = [
        enrich_pricing_result(result, quantity)
        for result, quantity in zip(result_items, quantity_items)
    ]
    return tuple(enriched_results)

