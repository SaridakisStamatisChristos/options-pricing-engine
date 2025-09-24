import math
import warnings

from options_engine.core.models import MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import BlackScholesModel, MonteCarloModel


def test_bs_runs():
    m = BlackScholesModel()
    c = OptionContract(
        symbol="TEST", strike_price=100.0, time_to_expiry=1.0, option_type=OptionType.CALL
    )
    md = MarketData(spot_price=100.0, risk_free_rate=0.05, dividend_yield=0.02)
    r = m.calculate_price(c, md, 0.2)
    assert r.theoretical_price > 0


def test_monte_carlo_handles_small_antithetic_path_counts():
    model = MonteCarloModel(paths=1, antithetic=True)
    contract = OptionContract(
        symbol="MC", strike_price=100.0, time_to_expiry=1.0, option_type=OptionType.CALL
    )
    market = MarketData(spot_price=100.0, risk_free_rate=0.01)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = model.calculate_price(contract, market, 0.2)

    assert not captured
    assert math.isfinite(result.theoretical_price)
