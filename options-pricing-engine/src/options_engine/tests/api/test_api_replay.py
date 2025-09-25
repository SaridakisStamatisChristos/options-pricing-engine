"""Replay endpoint behaviour tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from options_engine.api import routes
from options_engine.api.server import create_app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def _monte_carlo_payload() -> dict[str, object]:
    return {
        "contract": {
            "symbol": "MSFT",
            "strike_price": 300.0,
            "time_to_expiry": 0.75,
            "option_type": "call",
            "exercise_style": "european",
        },
        "market": {
            "spot_price": 305.0,
            "risk_free_rate": 0.02,
            "dividend_yield": 0.0,
        },
        "volatility": 0.3,
        "model": {"family": "monte_carlo", "params": {"paths": 4096}},
    }


def test_replay_returns_identical_results(client: TestClient) -> None:
    quote = client.post("/quote", json=_monte_carlo_payload())
    assert quote.status_code == 200
    quote_body = quote.json()

    capsule_id = quote_body["capsule_id"]
    replay = client.post(f"/replay/{capsule_id}", json={"strict_build": True})
    assert replay.status_code == 200
    replay_body = replay.json()

    assert replay_body["capsule_id"] == capsule_id
    assert replay_body["replayed"] is True
    assert pytest.approx(quote_body["theoretical_price"], rel=0, abs=1e-12) == replay_body["theoretical_price"]


def test_replay_strict_build_conflict(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    quote = client.post("/quote", json=_monte_carlo_payload())
    capsule_id = quote.json()["capsule_id"]

    monkeypatch.setattr(routes, "BUILD_ID", "different-build")

    replay = client.post(f"/replay/{capsule_id}", json={"strict_build": True})
    assert replay.status_code == 409

