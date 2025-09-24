import time, logging, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from .models import OptionContract, MarketData
from .pricing_models import BlackScholesModel, BinomialModel, MonteCarloModel
from .volatility_surface import VolatilitySurface, VolatilitySurface as VS
log=logging.getLogger(__name__)
class OptionsEngine:
    def __init__(self,num_threads:int=8,cache_size:int=10000):
        self.vol_surface=VS(); self.models={"black_scholes": BlackScholesModel(), "binomial_200": BinomialModel(200), "monte_carlo_20k": MonteCarloModel(20000)}
        self.num_threads=num_threads; self.cache_size=cache_size; self.cache: Dict[str, Tuple[Dict,float]]={}; self.lock=threading.RLock(); self.total=0
    def _key(self,c:OptionContract,md:MarketData,model:str)->str:
        return f"{c.contract_id}|{md.spot_price:.6f}|{md.risk_free_rate:.6f}|{md.dividend_yield:.6f}|{model}"
    def _get(self,k:str)->Optional[Dict]:
        with self.lock:
            v=self.cache.get(k); 
            if v and time.time()-v[1]<5.0: return v[0]
        return None
    def _put(self,k:str,res:Dict)->None:
        with self.lock:
            if len(self.cache)>=self.cache_size:
                oldest=min(self.cache.items(),key=lambda kv:kv[1][1])[0]; self.cache.pop(oldest,None)
            self.cache[k]=(res,time.time())
    def price_option(self,c:OptionContract,md:MarketData,model_name:str="black_scholes")->Dict:
        if model_name not in self.models: raise ValueError("unknown model")
        k=self._key(c,md,model_name); hit=self._get(k)
        if hit: return {**hit,"cached":True}
        vol=self.vol_surface.get_volatility(c.strike_price,c.time_to_expiry,md.spot_price)
        r=self.models[model_name].calculate_price(c,md,vol)
        out={"contract_id":r.contract_id,"theoretical_price":r.theoretical_price,"delta":r.delta,"gamma":r.gamma,"theta":r.theta,"vega":r.vega,"rho":r.rho,"implied_volatility":r.implied_volatility,"model_used":model_name,"volatility_used":vol,"computation_time_ms":r.computation_time_ms,"error":r.error}
        self._put(k,out); self.total+=1; return {**out,"cached":False}
    def price_portfolio(self,contracts:List[OptionContract],md:MarketData,model_name:str="black_scholes")->List[Dict]:
        if not contracts: return []
        res=[]; 
        with ThreadPoolExecutor(max_workers=self.num_threads) as ex:
            futures=[ex.submit(self.price_option,c,md,model_name) for c in contracts]
            for f in as_completed(futures):
                try: res.append(f.result())
                except Exception as e: log.exception("pricing failed"); res.append({"theoretical_price":0.0,"error":str(e),"model_used":model_name})
        return res
    def calculate_portfolio_greeks(self,results:List[Dict])->Dict[str,float]:
        if not results: return {k:0.0 for k in ["delta","gamma","theta","vega","rho","total_value","total_vega_exposure","position_count"]}
        td=sum((x.get("delta") or 0.0) for x in results); tg=sum((x.get("gamma") or 0.0) for x in results)
        tt=sum((x.get("theta") or 0.0) for x in results); tv=sum((x.get("vega") or 0.0) for x in results)
        tr=sum((x.get("rho") or 0.0) for x in results); tvl=sum((x.get("theoretical_price") or 0.0) for x in results)
        return {"delta":td,"gamma":tg,"theta":tt,"vega":tv,"rho":tr,"total_value":tvl,"total_vega_exposure":tv*100.0,"position_count":float(len(results))}
