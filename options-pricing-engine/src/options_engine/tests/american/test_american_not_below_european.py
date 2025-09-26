from options_engine.core.models import MarketData, OptionContract, OptionType, ExerciseStyle
from options_engine.core.pricing_models import BlackScholesModel, american_lsmc_price


def test_american_put_not_below_european():
    m = MarketData(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.0)
    K, tau, vol = 100.0, 0.5, 0.25
    put_eur = OptionContract("E", K, tau, OptionType.PUT, ExerciseStyle.EUROPEAN)
    bs = BlackScholesModel().calculate_price(put_eur, m, vol).theoretical_price
    price = american_lsmc_price(100.0, K, tau, vol, 0.02, 0.0, "put", paths=20000, seed=7).price
    assert price >= bs - 5e-4
