import math, time, logging
import numpy as np
from scipy.stats import norm
from .models import OptionContract, MarketData, PricingResult, OptionType
from ..utils.validation import validate_pricing_parameters
log=logging.getLogger(__name__)
def _bs(S,K,T,r,sigma,q,is_call):
    if T<=1e-12 or sigma<=1e-12:
        intrinsic = max(0.0, S-K) if is_call else max(0.0, K-S)
        return intrinsic,0,0,0,0,0
    st=math.sqrt(T); d1=(math.log(S/K)+(r-q+0.5*sigma*sigma)*T)/(sigma*st); d2=d1-sigma*st
    if is_call:
        price=S*math.exp(-q*T)*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2); delta=math.exp(-q*T)*norm.cdf(d1); rho=K*T*math.exp(-r*T)*norm.cdf(d2)/100.0
    else:
        price=K*math.exp(-r*T)*norm.cdf(-d2)-S*math.exp(-q*T)*norm.cdf(-d1); delta=-math.exp(-q*T)*norm.cdf(-d1); rho=-K*T*math.exp(-r*T)*norm.cdf(-d2)/100.0
    pdf=norm.pdf(d1); gamma=math.exp(-q*T)*pdf/(S*sigma*st); vega=S*math.exp(-q*T)*pdf*st/100.0
    theta=(-S*pdf*sigma*math.exp(-q*T)/(2*st) + (q*S*math.exp(-q*T)*(norm.cdf(d1) if is_call else -norm.cdf(-d1))) - (r*K*math.exp(-r*T)*(norm.cdf(d2) if is_call else -norm.cdf(-d2))))/365.0
    return price,delta,gamma,theta,vega,rho
class BlackScholesModel:
    def calculate_price(self,c:OptionContract,md:MarketData,vol:float)->PricingResult:
        t0=time.perf_counter()
        try:
            validate_pricing_parameters(c,md,vol)
            price,delta,gamma,theta,vega,rho=_bs(md.spot_price,c.strike_price,c.time_to_expiry,md.risk_free_rate,vol,md.dividend_yield,c.option_type==OptionType.CALL)
            return PricingResult(c.contract_id,max(0.0,price),delta,gamma,theta,vega,rho,vol,(time.perf_counter()-t0)*1000.0,"black_scholes")
        except Exception as e:
            log.exception("BS failed"); return PricingResult(c.contract_id,0.0,error=str(e),computation_time_ms=(time.perf_counter()-t0)*1000.0,model_used="black_scholes")
class BinomialModel:
    def __init__(self,steps:int=200): self.steps=steps
    def calculate_price(self,c:OptionContract,md:MarketData,vol:float)->PricingResult:
        t0=time.perf_counter()
        try:
            validate_pricing_parameters(c,md,vol); N=self.steps; dt=c.time_to_expiry/N
            u=math.exp(vol*math.sqrt(dt)); d=1/u; p=(math.exp((md.risk_free_rate-md.dividend_yield)*dt)-d)/(u-d); p=min(1,max(0,p)); disc=math.exp(-md.risk_free_rate*dt)
            prices=np.array([md.spot_price*(u**j)*(d**(N-j)) for j in range(N+1)])
            vals=np.maximum(prices-c.strike_price,0.0) if c.option_type==OptionType.CALL else np.maximum(c.strike_price-prices,0.0)
            for i in range(N-1,-1,-1):
                vals=disc*(p*vals[1:]+(1-p)*vals[:-1])
                prices=prices[:-1]/u
                if c.exercise_style.name=="AMERICAN":
                    ex=np.maximum(prices-c.strike_price,0.0) if c.option_type==OptionType.CALL else np.maximum(c.strike_price-prices,0.0)
                    vals=np.maximum(vals,ex)
            return PricingResult(c.contract_id,float(vals[0]),model_used=f"binomial_{self.steps}",computation_time_ms=(time.perf_counter()-t0)*1000.0)
        except Exception as e:
            log.exception("BIN failed"); return PricingResult(c.contract_id,0.0,error=str(e),computation_time_ms=(time.perf_counter()-t0)*1000.0,model_used=f"binomial_{self.steps}")
class MonteCarloModel:
    def __init__(self,n:int=20000,antithetic:bool=True): self.n=n; self.antithetic=antithetic
    def calculate_price(self,c:OptionContract,md:MarketData,vol:float)->PricingResult:
        t0=time.perf_counter()
        try:
            validate_pricing_parameters(c,md,vol); n=self.n//2 if self.antithetic else self.n
            z=np.random.standard_normal(n); z=np.concatenate([z,-z]) if self.antithetic else z
            ST=md.spot_price*np.exp((md.risk_free_rate-md.dividend_yield-0.5*vol**2)*c.time_to_expiry + vol*math.sqrt(c.time_to_expiry)*z)
            pay=np.maximum(ST-c.strike_price,0.0) if c.option_type==OptionType.CALL else np.maximum(c.strike_price-ST,0.0)
            price=math.exp(-md.risk_free_rate*c.time_to_expiry)*float(np.mean(pay))
            return PricingResult(c.contract_id,max(0.0,price),model_used=f"monte_carlo_{self.n}",computation_time_ms=(time.perf_counter()-t0)*1000.0)
        except Exception as e:
            log.exception("MC failed"); return PricingResult(c.contract_id,0.0,error=str(e),computation_time_ms=(time.perf_counter()-t0)*1000.0,model_used=f"monte_carlo_{self.n}")
