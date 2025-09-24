from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import psutil, logging
from .routes import pricing, risk, market_data
app=FastAPI(title="UCTG/1 Options Pricing Engine", version="1.0.1", docs_url="/docs", redoc_url="/redoc")
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(pricing.router, prefix="/api/v1")
app.include_router(risk.router, prefix="/api/v1")
app.include_router(market_data.router, prefix="/api/v1")
@app.get("/health", tags=["monitoring"])
async def health():
    try:
        return {"status":"healthy","timestamp":datetime.utcnow().isoformat(),"version":"1.0.1","system":{"memory_usage_percent": psutil.virtual_memory().percent,"disk_usage_percent": psutil.disk_usage('/').percent}}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service unhealthy")
@app.exception_handler(Exception)
async def global_error(req, exc):
    return JSONResponse(status_code=500, content={"error":"Internal server error"})
