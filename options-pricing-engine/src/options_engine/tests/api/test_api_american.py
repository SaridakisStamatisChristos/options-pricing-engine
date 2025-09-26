from options_engine.tests.simple_client import SimpleTestClient


def test_quote_american_routes_to_lsmc(client: SimpleTestClient) -> None:
    payload = {
        "contract": {
            "symbol": "SPY",
            "strike_price": 100.0,
            "time_to_expiry": 0.5,
            "option_type": "PUT",
            "exercise_style": "AMERICAN",
        },
        "market": {"spot_price": 100.0, "risk_free_rate": 0.01, "dividend_yield": 0.0},
        "volatility": 0.2,
        "model": {"family": "monte_carlo", "params": {"paths": 20000, "steps": 64}},
        "greeks": {"delta": True},
    }
    response = client.post("/quote", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["theoretical_price"] > 0
    assert "ci" in data and data["ci"]["half_width_abs"] >= 0
