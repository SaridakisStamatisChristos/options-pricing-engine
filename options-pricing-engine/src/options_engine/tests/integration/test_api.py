"""Integration tests for the FastAPI application."""

from __future__ import annotations

import json
from typing import Iterator, cast

import pytest
from fastapi.testclient import TestClient

from options_engine.api.config import get_settings
from options_engine.api.fastapi_app import create_app
from options_engine.api.security import _get_authenticator
from options_engine.tests.utils import make_token

PRICING_PAYLOAD = {
    "contracts": [
        {
            "symbol": "TQA",
            "strike_price": 100.0,
            "time_to_expiry": 1.0,
            "option_type": "call",
            "exercise_style": "european",
            "quantity": 1,
        }
    ],
    "market_data": {
        "spot_price": 105.0,
        "risk_free_rate": 0.01,
        "dividend_yield": 0.0,
        "volatility": 0.2,
    },
    "model": "black_scholes",
    "calculate_greeks": False,
}


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture()
def auth_header() -> Iterator[dict[str, str]]:
    token = make_token(scopes=["pricing:read", "market-data:read"])
    yield {"Authorization": f"Bearer {token}"}


def test_authz_denied_without_token(client) -> None:
    response = client.post("/api/v1/pricing/single", json=PRICING_PAYLOAD)
    assert response.status_code == 401


def test_authz_denied_without_scope(client) -> None:
    token = make_token(scopes=["market-data:read"])
    response = client.post(
        "/api/v1/pricing/single",
        json=PRICING_PAYLOAD,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403
    assert response.json()["detail"].startswith("Permission")


def test_authorized_pricing_request_succeeds(client, auth_header) -> None:
    response = client.post("/api/v1/pricing/single", json=PRICING_PAYLOAD, headers=auth_header)
    assert response.status_code == 200
    payload = response.json()
    assert "results" in payload
    assert payload["results"]


def test_headers_present_hsts_xcto(client) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    headers = {key.lower(): value for key, value in response.headers.items()}
    assert "strict-transport-security" in headers
    assert headers["strict-transport-security"].startswith("max-age=")
    assert headers["x-content-type-options"].lower() == "nosniff"
    assert "x-request-id" in headers


def test_metrics_exposed(client, auth_header) -> None:
    _ = client.post("/api/v1/pricing/single", json=PRICING_PAYLOAD, headers=auth_header)
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    body = metrics.text
    assert "ope_request_latency_seconds" in body
    assert "ope_auth_failures_total" in body
    assert "ope_rate_limit_rejections_total" in body


def test_rate_limit_trips(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "2/minute")
    get_settings.cache_clear()
    _get_authenticator.cache_clear()
    app = create_app()
    token = make_token(scopes=["pricing:read"])
    with TestClient(app) as limited_client:
        for _ in range(2):
            ok = limited_client.post(
                "/api/v1/pricing/single",
                json=PRICING_PAYLOAD,
                headers={"Authorization": f"Bearer {token}"},
            )
            assert ok.status_code == 200
        burst = limited_client.post(
            "/api/v1/pricing/single",
            json=PRICING_PAYLOAD,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert burst.status_code == 429
        body = cast(dict[str, str], burst.json())
        assert body["detail"] == "Rate limit exceeded"
        assert "X-Request-ID" in burst.headers

    monkeypatch.delenv("RATE_LIMIT_DEFAULT")
    get_settings.cache_clear()
    _get_authenticator.cache_clear()


def test_payload_too_large_rejected(monkeypatch) -> None:
    monkeypatch.setenv("MAX_BODY_BYTES", "2048")
    get_settings.cache_clear()
    _get_authenticator.cache_clear()
    app = create_app()
    token = make_token(scopes=["pricing:read"])
    oversized_payload = PRICING_PAYLOAD.copy()
    oversized_contract = dict(oversized_payload["contracts"][0])
    oversized_contract["symbol"] = "X" * 200
    oversized_payload = {
        **oversized_payload,
        "contracts": [oversized_contract],
    }

    payload_bytes = json.dumps(oversized_payload).encode()

    with TestClient(app) as limited_client:
        response = limited_client.post(
            "/api/v1/pricing/single",
            data=payload_bytes,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Content-Length": str(len(payload_bytes)),
            },
        )

    assert response.status_code == 413
    body = cast(dict[str, str], response.json())
    assert body["detail"] == "Payload too large"

    monkeypatch.delenv("MAX_BODY_BYTES")
    get_settings.cache_clear()
    _get_authenticator.cache_clear()
