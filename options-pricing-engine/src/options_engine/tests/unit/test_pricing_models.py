from options_engine.core.models import OptionContract, MarketData, OptionType
from options_engine.core.pricing_models import BlackScholesModel
def test_bs_runs():
    m=BlackScholesModel()
    c=OptionContract(symbol="TEST", strike_price=100.0, time_to_expiry=1.0, option_type=OptionType.CALL)
    md=MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    r=m.calculate_price(c, md, 0.2)
    assert r.theoretical_price > 0
