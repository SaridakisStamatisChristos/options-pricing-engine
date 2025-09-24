from ..core.models import OptionContract, MarketData
def validate_pricing_parameters(c: OptionContract, md: MarketData, vol: float)->None:
    if vol<=0 or vol>5.0: raise ValueError("vol out of range")
    if c.time_to_expiry<=1e-8: raise ValueError("T too small")
    if md.spot_price<=0: raise ValueError("S>0 required")
    if c.strike_price<=0: raise ValueError("K>0 required")
