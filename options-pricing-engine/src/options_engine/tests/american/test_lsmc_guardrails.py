from options_engine.core.pricing_models import american_lsmc_price, black_scholes_price


def test_american_put_above_euro_put(
    bs=black_scholes_price, lsmc=american_lsmc_price
) -> None:
    params = dict(spot=100.0, strike=100.0, tau=0.5, sigma=0.25, r=0.01, q=0.0)
    euro_put = bs(**params, option_type="put").price
    amer_put = lsmc(**params, option_type="put", steps=64, paths=30000, seed=11).price
    assert amer_put >= euro_put - 1e-4
