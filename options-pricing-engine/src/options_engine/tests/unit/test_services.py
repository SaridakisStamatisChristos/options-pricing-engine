from __future__ import annotations

import pytest

from options_engine.api.services import enrich_pricing_result


def test_enrich_pricing_result_adds_position_metrics() -> None:
    base_result = {
        "contract_id": "TST",
        "theoretical_price": 12.5,
        "delta": 0.6,
        "gamma": 0.02,
        "theta": -0.1,
        "vega": 0.15,
        "rho": 0.05,
        "standard_error": 0.3,
        "confidence_interval": (11.5, 13.5),
    }

    enriched = enrich_pricing_result(base_result, 4)

    assert enriched["quantity"] == pytest.approx(4.0)
    assert enriched["position_value"] == pytest.approx(50.0)
    assert enriched["position_delta"] == pytest.approx(2.4)
    assert enriched["position_gamma"] == pytest.approx(0.08)
    assert enriched["position_theta"] == pytest.approx(-0.4)
    assert enriched["position_vega"] == pytest.approx(0.6)
    assert enriched["position_rho"] == pytest.approx(0.2)
    assert enriched["position_standard_error"] == pytest.approx(1.2)
    lower, upper = enriched["position_confidence_interval"]
    assert lower == pytest.approx(46.0)
    assert upper == pytest.approx(54.0)
