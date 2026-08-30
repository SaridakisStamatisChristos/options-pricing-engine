"""Integration tests covering the minimal API surface."""

from __future__ import annotations

import pytest

from options_engine.core.models import MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import BlackScholesModel
from tests.simple_client import SimpleTestClient


def _quote_payload(model: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract": {
            "symbol": "AAPL",
            "strike_price": 150.0,
            "time_to_expiry": 0.5,
            "option_type": "call",
            "exercise_style": "european",
        },
        "market": {
            "spot_price": 147.0,
            "risk_free_rate": 0.01,
            "dividend_yield": 0.0,
        },
        "volatility": 0.25,
        "model": model or {"family": "black_scholes"},
    }
    return payload


def test_quote_black_scholes_returns_price_and_greeks(client: SimpleTestClient) -> None:
    response = client.post("/quote", json=_quote_payload())
    assert response.status_code == 200
    data = response.json()

    contract = OptionContract(
        symbol="AAPL",
        strike_price=150.0,
        time_to_expiry=0.5,
        option_type=OptionType.CALL,
    )
    market = MarketData(spot_price=147.0, risk_free_rate=0.01, dividend_yield=0.0)
    expected = BlackScholesModel().calculate_price(contract, market, 0.25)

    assert pytest.approx(expected.theoretical_price, rel=1e-10) == data["theoretical_price"]
    assert "capsule_id" in data
    assert "ci" not in data
    greeks = data.get("greeks")
    assert greeks is not None
    for key in ("delta", "gamma", "theta", "vega", "rho"):
        assert key in greeks


def test_quote_monte_carlo_returns_confidence_interval(client: SimpleTestClient) -> None:
    payload = _quote_payload(model={"family": "monte_carlo", "params": {"paths": 8192}})
    response = client.post("/quote", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "ci" in data
    ci = data["ci"]
    assert ci["paths_used"] == 8192
    assert ci["vr_pipeline"] in {"antithetic+cv", "antithetic", "baseline", "cv"}
    assert ci["half_width_abs"] > 0.0
    assert ci["half_width_bps"] > 0.0
    assert ci["method"] in {"student_t", "student_t_projected"}
    assert ci["degrees_of_freedom"] == ci["independent_units"] - 1
    assert ci["lower_bound"] <= data["theoretical_price"] <= ci["upper_bound"]
    diagnostics = data["estimate_diagnostics"]
    assert diagnostics["bounded_estimate"] == data["theoretical_price"]
    assert diagnostics["raw_confidence_interval"] is not None


def test_quote_monte_carlo_accepts_variance_reduction_flags(client: SimpleTestClient) -> None:
    payload = _quote_payload(
        model={
            "family": "monte_carlo",
            "params": {"paths": 4096, "use_qmc": True, "use_cv": True, "antithetic": False},
        }
    )
    response = client.post("/quote", json=payload)
    assert response.status_code == 200
    data = response.json()

    ci = data.get("ci")
    assert ci is not None
    assert ci["paths_used"] == 4096
    assert ci["vr_pipeline"] == "rqmc+cv"


def test_batch_endpoint_handles_partial_failures(client: SimpleTestClient) -> None:
    payload = {
        "items": [
            _quote_payload(),
            {
                **_quote_payload(),
                "market": {"spot_price": -1.0, "risk_free_rate": 0.01, "dividend_yield": 0.0},
            },
        ]
    }
    response = client.post("/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert len(data["results"]) == 2
    first, second = data["results"]
    assert first["ok"] is True
    assert second["ok"] is False
    assert second["error"] == "invalid_request"
    assert len(data["capsule_ids"]) == 1


def test_batch_enforces_item_limit(client: SimpleTestClient) -> None:
    payload = {"items": [_quote_payload() for _ in range(101)]}
    response = client.post("/batch", json=payload)
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "1"


def test_greeks_endpoint_matches_analytic(client: SimpleTestClient) -> None:
    payload = _quote_payload()
    response = client.post("/greeks", json=payload)
    assert response.status_code == 200
    data = response.json()

    contract = OptionContract(
        symbol="AAPL",
        strike_price=150.0,
        time_to_expiry=0.5,
        option_type=OptionType.CALL,
    )
    market = MarketData(spot_price=147.0, risk_free_rate=0.01, dividend_yield=0.0)
    expected = BlackScholesModel().calculate_price(contract, market, 0.25)

    for key, value in data["greeks"].items():
        expected_value = getattr(expected, key)
        assert expected_value is not None
        assert pytest.approx(expected_value, rel=1e-8) == value


def test_version_endpoint_contains_library_versions(client: SimpleTestClient) -> None:
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "build_id" in data
    assert "numpy" in data["library_versions"]


def test_idempotency_returns_identical_body(client: SimpleTestClient) -> None:
    payload = _quote_payload()
    payload["idempotency_key"] = "test-key"

    first = client.post("/quote", json=payload)
    second = client.post("/quote", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.text == second.text


def test_idempotency_key_reuse_with_different_request_conflicts(
    client: SimpleTestClient,
) -> None:
    first_payload = _quote_payload()
    first_payload["idempotency_key"] = "conflict-test-key"
    second_payload = _quote_payload()
    second_payload["idempotency_key"] = "conflict-test-key"
    contract = dict(second_payload["contract"])
    contract["strike_price"] = 151.0
    second_payload["contract"] = contract

    assert client.post("/quote", json=first_payload).status_code == 200
    conflict = client.post("/quote", json=second_payload)

    assert conflict.status_code == 409
    assert conflict.json()["detail"] == "idempotency_conflict"


def test_legacy_api_rejects_unknown_model_parameters(client: SimpleTestClient) -> None:
    payload = _quote_payload(
        model={"family": "monte_carlo", "params": {"paths": 4_096, "use_cvv": True}}
    )

    response = client.post("/quote", json=payload)

    assert response.status_code == 422
