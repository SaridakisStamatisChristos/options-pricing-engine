"""Utility helpers shared across API routes."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple


def enrich_pricing_result(result: Dict[str, object], quantity: int) -> Dict[str, object]:
    """Augment a pricing result with position-level analytics."""

    enriched: Dict[str, object] = dict(result)
    qty = float(quantity)
    enriched["quantity"] = qty

    theoretical_price = float(enriched.get("theoretical_price") or 0.0)
    enriched["position_value"] = theoretical_price * qty

    standard_error = enriched.get("standard_error")
    if standard_error is not None:
        enriched["position_standard_error"] = float(standard_error) * qty

    confidence_interval = enriched.get("confidence_interval")
    if confidence_interval is not None:
        lower, upper = confidence_interval  # type: ignore[misc]
        enriched["position_confidence_interval"] = (
            float(lower) * qty,
            float(upper) * qty,
        )

    for greek in ("delta", "gamma", "theta", "vega", "rho"):
        value = enriched.get(greek)
        if value is not None:
            enriched[f"position_{greek}"] = float(value) * qty

    return enriched


def annotate_results_with_quantity(
    results: Iterable[Dict[str, object]], quantities: Iterable[int]
) -> Tuple[Dict[str, object], ...]:
    """Return immutable enriched pricing results zipped with quantities."""

    enriched_results = []
    for result, quantity in zip(results, quantities):
        enriched_results.append(enrich_pricing_result(result, quantity))
    return tuple(enriched_results)
