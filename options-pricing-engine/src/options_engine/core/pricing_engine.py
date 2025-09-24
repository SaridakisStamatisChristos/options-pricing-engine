"""Threaded options pricing engine with caching."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .models import MarketData, OptionContract, PricingResult
from .pricing_models import BlackScholesModel, BinomialModel, MonteCarloModel
from .volatility_surface import VolatilitySurface

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _CacheEntry:
    """Internal representation of a cached pricing result."""

    payload: Dict[str, object]
    timestamp: float


class _ResultCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 10_000, ttl_seconds: float = 5.0) -> None:
        self._max_size = max(1, max_size)
        self._ttl = max(0.0, ttl_seconds)
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, _CacheEntry]" = OrderedDict()

    def get(self, key: str, now: Optional[float] = None) -> Optional[Dict[str, object]]:
        if not key:
            return None

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            current_time = now or time.time()
            if self._ttl and current_time - entry.timestamp > self._ttl:
                self._entries.pop(key, None)
                return None

            self._entries.move_to_end(key)
            return dict(entry.payload)

    def put(self, key: str, payload: Dict[str, object], now: Optional[float] = None) -> None:
        if not key:
            return

        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)

            current_time = now or time.time()
            self._entries[key] = _CacheEntry(dict(payload), current_time)

            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)


class OptionsEngine:
    """Coordinates pricing model execution across a pool of workers."""

    def __init__(
        self,
        *,
        num_threads: int = 8,
        cache_size: int = 10_000,
        cache_ttl_seconds: float = 5.0,
        volatility_surface: Optional[VolatilitySurface] = None,
    ) -> None:
        self.num_threads = max(1, num_threads)
        self.vol_surface = volatility_surface or VolatilitySurface()
        self.models: Dict[str, object] = {
            "black_scholes": BlackScholesModel(),
            "binomial_200": BinomialModel(steps=200),
            "monte_carlo_20k": MonteCarloModel(paths=20_000),
        }
        self._cache = _ResultCache(max_size=cache_size, ttl_seconds=cache_ttl_seconds)

    def _make_cache_key(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model_name: str,
        volatility: float,
    ) -> str:
        return (
            f"{contract.contract_id}|{market_data.spot_price:.6f}|"
            f"{market_data.risk_free_rate:.6f}|{market_data.dividend_yield:.6f}|"
            f"{model_name}|{volatility:.6f}"
        )

    def _prepare_result(
        self, result: PricingResult, model_name: str, volatility: float
    ) -> Dict[str, object]:
        return {
            "contract_id": result.contract_id,
            "theoretical_price": result.theoretical_price,
            "delta": result.delta,
            "gamma": result.gamma,
            "theta": result.theta,
            "vega": result.vega,
            "rho": result.rho,
            "implied_volatility": result.implied_volatility or volatility,
            "model_used": model_name,
            "volatility_used": volatility,
            "computation_time_ms": result.computation_time_ms,
            "error": result.error,
        }

    def price_option(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model_name: str = "black_scholes",
        override_volatility: Optional[float] = None,
    ) -> Dict[str, object]:
        if model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'")

        model = self.models[model_name]
        volatility = override_volatility
        if volatility is None:
            volatility = self.vol_surface.get_volatility(
                strike=contract.strike_price,
                maturity=contract.time_to_expiry,
                spot=market_data.spot_price,
            )

        cache_key = self._make_cache_key(contract, market_data, model_name, volatility)
        cached = self._cache.get(cache_key)
        if cached is not None:
            cached["cached"] = True
            return cached

        result = model.calculate_price(contract, market_data, volatility)
        payload = self._prepare_result(result, model_name, volatility)
        payload["cached"] = False
        self._cache.put(cache_key, payload)
        return payload

    def price_portfolio(
        self,
        contracts: Iterable[OptionContract],
        market_data: MarketData,
        model_name: str = "black_scholes",
        override_volatility: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        contract_list = list(contracts)
        if not contract_list:
            return []

        futures: Dict[Future[Dict[str, object]], int] = {}
        results: List[Optional[Dict[str, object]]] = [None] * len(contract_list)

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for index, contract in enumerate(contract_list):
                future = executor.submit(
                    self.price_option,
                    contract,
                    market_data,
                    model_name,
                    override_volatility,
                )
                futures[future] = index

            for future in as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:  # pragma: no cover - defensive programming
                    LOGGER.exception(
                        "Pricing failed for contract %s", contract_list[index].contract_id
                    )
                    results[index] = {
                        "contract_id": contract_list[index].contract_id,
                        "theoretical_price": 0.0,
                        "model_used": model_name,
                        "error": str(exc),
                    }

        return [result for result in results if result is not None]

    @staticmethod
    def calculate_portfolio_greeks(results: Iterable[Dict[str, object]]) -> Dict[str, float]:
        totals = {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "rho": 0.0,
            "total_value": 0.0,
            "total_vega_exposure": 0.0,
            "position_count": 0.0,
        }

        count = 0
        for result in results:
            totals["delta"] += float(result.get("delta") or 0.0)
            totals["gamma"] += float(result.get("gamma") or 0.0)
            totals["theta"] += float(result.get("theta") or 0.0)
            totals["vega"] += float(result.get("vega") or 0.0)
            totals["rho"] += float(result.get("rho") or 0.0)
            totals["total_value"] += float(result.get("theoretical_price") or 0.0)
            count += 1

        totals["total_vega_exposure"] = totals["vega"] * 100.0
        totals["position_count"] = float(count)
        return totals
