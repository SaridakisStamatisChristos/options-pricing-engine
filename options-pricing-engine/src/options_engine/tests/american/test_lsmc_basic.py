import math

from options_engine.core.pricing_models import (
    american_lsmc_price,
    binomial_price,
    black_scholes_price,
)


def test_lsmc_matches_binomial_within_tolerance() -> None:
    spot = 100.0
    strike = 100.0
    tau = 0.5
    sigma = 0.2
    r = 0.01
    q = 0.0

    lsmc = american_lsmc_price(
        spot,
        strike,
        tau,
        sigma,
        r,
        q,
        "put",
        steps=64,
        paths=50_000,
        seed=7,
    )
    binom = binomial_price(
        spot,
        strike,
        tau,
        sigma,
        r,
        q,
        "put",
        steps=400,
    )
    euro = black_scholes_price(spot, strike, tau, sigma, r, q, "put")

    assert lsmc.price >= euro.price
    diff = abs(lsmc.price - binom.price)
    assert diff / max(binom.price, 1e-3) <= 0.02

    assert math.isfinite(lsmc.ci_half_width)
    assert lsmc.ci_half_width <= lsmc.meta["precision_limit"]

    flags = set(lsmc.meta.get("policy_flags", []))
    assert "antithetic" in flags
    assert {"cv_used", "cv_skipped"} & flags
