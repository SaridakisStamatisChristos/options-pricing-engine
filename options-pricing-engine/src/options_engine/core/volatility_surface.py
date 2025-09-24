import time, logging, numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import griddata, RegularGridInterpolator
log=logging.getLogger(__name__)
@dataclass
class VolatilityPoint:
    strike: float; maturity: float; volatility: float; timestamp: float; source: str
class VolatilitySurface:
    def __init__(self, interpolation_method='linear'):
        self.points: List[VolatilityPoint]=[]; self.interpolation_method=interpolation_method; self.interpolator=None; self.last_update=0.0; self.surface_cache: Dict[tuple,float]={}; self.cache_ttl=60.0
    def update_volatility(self,strike:float,maturity:float,vol:float,source='market'):
        ts=time.time(); 
        for i,p in enumerate(self.points):
            if abs(p.strike-strike)<1e-9 and abs(p.maturity-maturity)<1e-9:
                self.points[i]=VolatilityPoint(strike,maturity,vol,ts,source); break
        else: self.points.append(VolatilityPoint(strike,maturity,vol,ts,source))
        self.surface_cache.clear(); self._build(); self.last_update=ts
    def _build(self):
        if len(self.points)<4: self.interpolator=None; return
        strikes=sorted({p.strike for p in self.points}); mats=sorted({p.maturity for p in self.points})
        grid=np.full((len(strikes),len(mats)), np.nan); mp={(p.strike,p.maturity):p.volatility for p in self.points}
        for i,s in enumerate(strikes):
            for j,m in enumerate(mats): grid[i,j]=mp.get((s,m), np.nan)
        if np.any(np.isnan(grid)):
            pts=np.array([[p.strike,p.maturity] for p in self.points]); vals=np.array([p.volatility for p in self.points])
            def scat(X):
                out=griddata(pts,vals,X,method=self.interpolation_method,fill_value=np.nan)
                if np.any(np.isnan(out)): out[np.isnan(out)]=griddata(pts,vals,X[np.isnan(out)],method='nearest')
                return out
            self.interpolator=scat
        else:
            self.interpolator=RegularGridInterpolator((strikes,mats),grid,method=self.interpolation_method,bounds_error=False,fill_value=None)
    def get_volatility(self,strike:float,maturity:float,spot:float)->float:
        key=(round(strike,4),round(maturity,4)); now=time.time()
        if key in self.surface_cache and now-self.last_update<self.cache_ttl: return self.surface_cache[key]
        if self.interpolator is None or len(self.points)<4: vol=0.20
        else:
            try:
                vol=float(self.interpolator([[strike,maturity]])) if not callable(self.interpolator) else float(self.interpolator(np.array([[strike,maturity]])))
                if not (0.01<vol<3.0) or np.isnan(vol): vol=self._fallback(strike,maturity)
            except Exception as e:
                log.warning("vol interp failed: %s", e); vol=self._fallback(strike,maturity)
        self.surface_cache[key]=vol; return vol
    def _fallback(self,strike,maturity):
        if not self.points: return 0.20
        d=[(((p.strike-strike)/max(strike,1e-6))**2 + ((p.maturity-maturity)/max(maturity,1e-6))**2, p.volatility) for p in self.points]
        d.sort(key=lambda x:x[0]); near=d[:min(5,len(d))]; w=[1/(di+1e-6) for di,_ in near]; v=[vi for _,vi in near]; 
        return float(np.average(v,weights=w))
