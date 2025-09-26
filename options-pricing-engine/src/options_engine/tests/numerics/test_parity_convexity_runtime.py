import math

from options_engine.core.models import MarketData, OptionContract, OptionType, ExerciseStyle
from options_engine.core.pricing_models import BlackScholesModel


def test_put_call_parity_identity():
    call_c = OptionContract("PARITY", 95.0, 0.4, OptionType.CALL, ExerciseStyle.EUROPEAN)
    put_c  = OptionContract("PARITY", 95.0, 0.4, OptionType.PUT,  ExerciseStyle.EUROPEAN)
    m = MarketData(spot_price=100.0, risk_free_rate=0.01, dividend_yield=0.0)
    vol = 0.2

    bs = BlackScholesModel()
    call = bs.calculate_price(call_c, m, vol).theoretical_price
    put  = bs.calculate_price(put_c,  m, vol).theoretical_price

    forward_anchor = m.spot_price * math.exp(-m.dividend_yield * call_c.time_to_expiry) \
        - call_c.strike_price * math.exp(-m.risk_free_rate * call_c.time_to_expiry)
    assert abs(call - put - forward_anchor) < 1e-8


def test_convexity_in_strike_european_calls():
    m = MarketData(spot_price=100.0, risk_free_rate=0.01, dividend_yield=0.0)
    vol, tau, K = 0.25, 0.5, 100.0
    dK = max(1e-3 * K, 1e-4)
    Ks = [K - dK, K, K + dK]
    bs = BlackScholesModel()
    prices = []
    for strike in Ks:
        c = OptionContract(f"C_{strike:.4f}", strike, tau, OptionType.CALL, ExerciseStyle.EUROPEAN)
        prices.append(bs.calculate_price(c, m, vol).theoretical_price)
    convexity_metric = prices[0] + prices[2] - 2.0 * prices[1]
    assert convexity_metric >= -1e-10
