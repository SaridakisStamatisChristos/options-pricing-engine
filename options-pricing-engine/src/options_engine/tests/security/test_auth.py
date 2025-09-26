import base64
import hashlib
import hmac
import json
import time

import pytest

from options_engine.api.config import get_settings
from options_engine.api.security import _get_authenticator
from options_engine.api.server import build_app
from options_engine.tests.simple_client import SimpleTestClient

DEV_SECRET = "dev-secret-value-at-least-32-bytes!!!"


def _hs256(jwt_payload: dict, secret: str, kid: str = "dev1") -> str:
    header = {"alg": "HS256", "typ": "JWT", "kid": kid}

    def b64(data: dict) -> bytes:
        return base64.urlsafe_b64encode(json.dumps(data, separators=(",", ":")).encode()).rstrip(b"=")

    signing_input = b".".join([b64(header), b64(jwt_payload)])
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return b".".join([signing_input, base64.urlsafe_b64encode(signature).rstrip(b"=")]).decode()


@pytest.fixture(autouse=True)
def dev_env(monkeypatch):
    get_settings.cache_clear()
    _get_authenticator.cache_clear()
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("DEV_JWT_SECRET", DEV_SECRET)
    monkeypatch.setenv("OIDC_AUDIENCE", "options-engine")
    monkeypatch.setenv("OIDC_ISSUER", "https://issuer.dev")
    monkeypatch.setenv("OIDC_CLOCK_SKEW_S", "60")
    yield
    get_settings.cache_clear()
    _get_authenticator.cache_clear()


def _token(scopes: str = "pricing:read") -> str:
    now = int(time.time())
    payload = {
        "iss": "https://issuer.dev",
        "aud": "options-engine",
        "sub": "user-123",
        "iat": now,
        "nbf": now - 10,
        "exp": now + 300,
        "scope": scopes,
    }
    return _hs256(payload, DEV_SECRET)


def _quote_body() -> dict:
    return {
        "contract": {
            "symbol": "AAPL",
            "strike_price": 100,
            "time_to_expiry": 0.5,
            "option_type": "CALL",
            "exercise_style": "EUROPEAN",
        },
        "market": {"spot_price": 100, "risk_free_rate": 0.01, "dividend_yield": 0.0},
        "volatility": 0.2,
        "model": {"family": "black_scholes"},
    }


def test_missing_token_rejected():
    app = build_app()
    with SimpleTestClient(app) as client:
        response = client.post("/quote", json={})
    assert response.status_code in (401, 403)


def test_scope_enforced():
    app = build_app()
    token = _token(scopes="market-data:write")
    with SimpleTestClient(app) as client:
        response = client.post(
            "/quote",
            headers={"Authorization": f"Bearer {token}"},
            json={},
        )
    assert response.status_code in (401, 403)


def test_valid_token_allows():
    app = build_app()
    token = _token(scopes="pricing:read extra:ok")
    body = _quote_body()
    with SimpleTestClient(app) as client:
        response = client.post(
            "/quote",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
        )
    assert response.status_code == 200
    assert "theoretical_price" in response.json()


def test_expired_token_rejected():
    app = build_app()
    now = int(time.time())
    payload = {
        "iss": "https://issuer.dev",
        "aud": "options-engine",
        "sub": "user-123",
        "iat": now - 1800,
        "nbf": now - 1200,
        "exp": now - 300,
        "scope": "pricing:read",
    }
    token = _hs256(payload, DEV_SECRET)
    with SimpleTestClient(app) as client:
        response = client.post(
            "/quote",
            headers={"Authorization": f"Bearer {token}"},
            json=_quote_body(),
        )
    assert response.status_code in (401, 403)
