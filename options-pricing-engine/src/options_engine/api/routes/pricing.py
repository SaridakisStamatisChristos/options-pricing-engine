from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import time, logging
from ..schemas.request import PricingRequest
from ..schemas.response import PricingBatchResponse
from ...core.models import OptionContract, MarketData
from ...core.pricing_engine import OptionsEngine
from ..security import require_permission
router=APIRouter(prefix="/pricing",tags=["pricing"])
log=logging.getLogger(__name__); engine=OptionsEngine(num_threads=8)
@router.post("/single", response_model=PricingBatchResponse, dependencies=[Depends(require_permission("pricing:read"))])
async def single(request: PricingRequest, background_tasks: BackgroundTasks):
    try:
        t0=time.perf_counter(); c0=request.contracts[0]
        c=OptionContract(symbol=c0.symbol,strike_price=c0.strike_price,time_to_expiry=c0.time_to_expiry,option_type=c0.option_type,exercise_style=c0.exercise_style)
        m=request.market_data; md=MarketData(spot_price=m.spot_price,risk_free_rate=m.risk_free_rate,dividend_yield=m.dividend_yield)
        r=engine.price_option(c,md,model_name=request.model.value); dt=(time.perf_counter()-t0)*1000.0
        background_tasks.add_task(lambda: log.info("priced %s", r.get("contract_id")))
        return PricingBatchResponse(results=[r], total_computation_time_ms=dt, options_per_second=1000/dt if dt>0 else float("inf"))
    except Exception as e:
        log.exception("single failed"); raise HTTPException(status_code=500, detail=str(e))
@router.post("/batch", response_model=PricingBatchResponse, dependencies=[Depends(require_permission("pricing:read"))])
async def batch(request: PricingRequest, background_tasks: BackgroundTasks):
    try:
        t0=time.perf_counter()
        m=request.market_data; md=MarketData(spot_price=m.spot_price,risk_free_rate=m.risk_free_rate,dividend_yield=m.dividend_yield)
        cs=[OptionContract(symbol=x.symbol,strike_price=x.strike_price,time_to_expiry=x.time_to_expiry,option_type=x.option_type,exercise_style=x.exercise_style) for x in request.contracts]
        rs=engine.price_portfolio(cs,md,model_name=request.model.value); dt=(time.perf_counter()-t0)*1000.0; ops=len(rs)/(dt/1000) if dt>0 else float("inf")
        pg=engine.calculate_portfolio_greeks(rs) if request.calculate_greeks else None
        background_tasks.add_task(lambda: log.info("batch priced %d", len(rs)))
        return PricingBatchResponse(results=rs, total_computation_time_ms=dt, options_per_second=ops, portfolio_greeks=pg)
    except Exception as e:
        log.exception("batch failed"); raise HTTPException(status_code=500, detail=str(e))
