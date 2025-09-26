"""Integration tests covering the minimal API surface."""

from __future__ import annotations

from typing import Dict

import pytest
from fastapi.testclient import TestClient

from options_engine.api.server import create_app
from options_engine.core.pricing_models import BlackScholesModel
from options_engine.core.models import MarketData, OptionContract, OptionType
from options_engine.tests.utils import make_token


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    token = make_token(scopes=["pricing:read"])
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


def _quote_payload(model: Dict[str, object] | None = None) -> Dict[str, object]:
    payload: Dict[str, object] = {
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


def test_quote_black_scholes_returns_price_and_greeks(client: TestClient) -> None:
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


def test_quote_monte_carlo_returns_confidence_interval(client: TestClient) -> None:
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


def test_quote_monte_carlo_accepts_variance_reduction_flags(client: TestClient) -> None:
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
    assert ci["vr_pipeline"] == "qmc+cv"


def test_batch_endpoint_handles_partial_failures(client: TestClient) -> None:
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


def test_batch_enforces_item_limit(client: TestClient) -> None:
    payload = {"items": [_quote_payload() for _ in range(101)]}
    response = client.post("/batch", json=payload)
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "1"


def test_greeks_endpoint_matches_analytic(client: TestClient) -> None:
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


def test_version_endpoint_contains_library_versions(client: TestClient) -> None:
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert "build_id" in data
    assert "numpy" in data["library_versions"]


def test_idempotency_returns_identical_body(client: TestClient) -> None:
    payload = _quote_payload()
    payload["idempotency_key"] = "test-key"

    first = client.post("/quote", json=payload)
    second = client.post("/quote", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.text == second.text

