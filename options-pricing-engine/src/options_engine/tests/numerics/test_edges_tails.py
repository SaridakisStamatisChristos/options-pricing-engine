import math

from options_engine.core.models import MarketData, OptionContract, OptionType, ExerciseStyle
from options_engine.core.pricing_models import BlackScholesModel, MonteCarloModel


def _intrinsic(opt_type: OptionType, spot: float, strike: float, tau: float, r: float, q: float) -> float:
    disc_r = math.exp(-r * tau)
    disc_q = math.exp(-q * tau)
    if opt_type is OptionType.CALL:
        return max(disc_q * spot - disc_r * strike, 0.0)
    return max(disc_r * strike - disc_q * spot, 0.0)


def test_deep_itm_small_tau_near_intrinsic():
    """Deep ITM & small-tau European call ~ intrinsic value."""
    r, q, tau, sigma, K = 0.01, 0.0, 0.01, 0.2, 100.0
    S = K * math.exp(5.0)
    c = OptionContract("DEEP_ITM", K, tau, OptionType.CALL, ExerciseStyle.EUROPEAN)
    m = MarketData(spot_price=S, risk_free_rate=r, dividend_yield=q)
    price = BlackScholesModel().calculate_price(c, m, sigma).theoretical_price
    intrinsic = _intrinsic(c.option_type, S, K, tau, r, q)
    assert abs(price - intrinsic) <= 1e-3 * max(1.0, intrinsic)


def test_mc_ci_shrinks_with_more_paths():
    """MC standard error should decrease with more paths."""
    r, q, tau, sigma, S, K = 0.01, 0.0, 0.5, 0.25, 100.0, 105.0
    c = OptionContract("CI_SHRINK", K, tau, OptionType.CALL, ExerciseStyle.EUROPEAN)
    m = MarketData(spot_price=S, risk_free_rate=r, dividend_yield=q)
    mc_small = MonteCarloModel(paths=4000, antithetic=True)
    mc_large = MonteCarloModel(paths=32000, antithetic=True)
    r_small = mc_small.calculate_price(c, m, sigma)
    r_large = mc_large.calculate_price(c, m, sigma)
    assert r_small.standard_error is not None and r_large.standard_error is not None
    assert r_large.standard_error < r_small.standard_error
