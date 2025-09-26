import math

from options_engine.core.pricing_models import american_lsmc_price


DEFAULT_ARGS = {
    "spot": 100.0,
    "strike": 100.0,
    "tau": 0.75,
    "sigma": 0.2,
    "r": 0.01,
    "q": 0.0,
    "option_type": "put",
    "steps": 64,
    "paths": 30_000,
    "seed": 17,
}


def test_sigma_monotonicity() -> None:
    lower = american_lsmc_price(**{**DEFAULT_ARGS, "sigma": 0.15})
    higher = american_lsmc_price(**{**DEFAULT_ARGS, "sigma": 0.25})
    assert higher.price >= lower.price - 1e-3


def test_tau_monotonicity() -> None:
    base = american_lsmc_price(**DEFAULT_ARGS)
    bumped = american_lsmc_price(**{**DEFAULT_ARGS, "tau": DEFAULT_ARGS["tau"] * 1.2, "seed": 18})
    assert bumped.price >= base.price - 1e-3


def test_strike_convexity() -> None:
    delta = 2.0
    lower = american_lsmc_price(**{**DEFAULT_ARGS, "strike": DEFAULT_ARGS["strike"] - delta, "seed": 19})
    centre = american_lsmc_price(**{**DEFAULT_ARGS, "seed": 19})
    upper = american_lsmc_price(**{**DEFAULT_ARGS, "strike": DEFAULT_ARGS["strike"] + delta, "seed": 19})
    lhs = centre.price
    rhs = 0.5 * (lower.price + upper.price)
    assert lhs <= rhs + 2e-3


def test_step_convergence() -> None:
    coarse = american_lsmc_price(**{**DEFAULT_ARGS, "steps": 48, "seed": 21})
    fine = american_lsmc_price(**{**DEFAULT_ARGS, "steps": 96, "seed": 21})
    assert fine.price >= coarse.price - 5e-4


def test_deep_otm_flag_present() -> None:
    deep_call = american_lsmc_price(
        spot=80.0,
        strike=200.0,
        tau=0.5,
        sigma=0.25,
        r=0.01,
        q=0.0,
        option_type="call",
        steps=64,
        paths=20_000,
        seed=5,
    )
    assert "lsmc_im_filter" in set(deep_call.meta.get("policy_flags", []))
