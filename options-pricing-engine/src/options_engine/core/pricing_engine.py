# mypy: ignore-errors
"""Threaded options pricing engine with caching."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from numpy.random import SeedSequence

from ..observability.metrics import (
    MODEL_ERRORS,
    MODEL_LATENCY,
    THREADPOOL_IN_FLIGHT,
    THREADPOOL_QUEUE_DEPTH,
    THREADPOOL_QUEUE_WAIT,
    THREADPOOL_REJECTIONS,
    THREADPOOL_SATURATION,
    THREADPOOL_WORKERS,
)
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
        queue_size: int = 32,
        queue_timeout_seconds: float = 0.5,
        task_timeout_seconds: float = 30.0,
        name: str = "default",
        volatility_surface: Optional[VolatilitySurface] = None,
        monte_carlo_seed: Optional[int] = None,
    ) -> None:
        self.num_threads = max(1, num_threads)
        self.queue_size = max(0, queue_size)
        self.queue_timeout_seconds = max(0.0, queue_timeout_seconds)
        self.task_timeout_seconds = max(0.0, task_timeout_seconds)
        self.name = name
        self.vol_surface = volatility_surface or VolatilitySurface()
        self._base_seed_sequence = SeedSequence(monte_carlo_seed) if monte_carlo_seed is not None else None
        self.models: Dict[str, object] = {
            "black_scholes": BlackScholesModel(),
            "binomial_200": BinomialModel(steps=200),
            "monte_carlo_20k": MonteCarloModel(paths=20_000, seed_sequence=self._base_seed_sequence),
        }
        self._cache = _ResultCache(max_size=cache_size, ttl_seconds=cache_ttl_seconds)
        self._executor_lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=self.num_threads,
            thread_name_prefix="options-engine",
        )
        self._queue_capacity = threading.BoundedSemaphore(self.num_threads + self.queue_size)
        self._pending_lock = threading.Lock()
        self._pending_tasks = 0
        THREADPOOL_WORKERS.labels(engine=self.name).set(self.num_threads)

    def __enter__(self) -> "OptionsEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.shutdown(wait=False)
        except Exception:
            pass

    def shutdown(self, wait: bool = True) -> None:
        with self._executor_lock:
            if self._executor is not None:
                self._executor.shutdown(wait=wait, cancel_futures=True)
                self._executor = None

    def _get_executor(self) -> ThreadPoolExecutor:
        with self._executor_lock:
            executor = self._executor
        if executor is None:
            raise RuntimeError("OptionsEngine has been shut down")
        return executor

    def _update_queue_metrics(self) -> None:
        running = min(self._pending_tasks, self.num_threads)
        waiting = max(0, self._pending_tasks - self.num_threads)
        THREADPOOL_IN_FLIGHT.labels(engine=self.name).set(running)
        THREADPOOL_QUEUE_DEPTH.labels(engine=self.name).set(waiting)

    def _submit_task(self, func, *args, **kwargs) -> Future:
        start = time.perf_counter()
        if self.queue_timeout_seconds == 0:
            acquired = self._queue_capacity.acquire(blocking=False)
        else:
            acquired = self._queue_capacity.acquire(timeout=self.queue_timeout_seconds)
        wait_time = time.perf_counter() - start
        THREADPOOL_QUEUE_WAIT.labels(engine=self.name).observe(wait_time)
        if not acquired:
            THREADPOOL_REJECTIONS.labels(engine=self.name).inc()
            THREADPOOL_SATURATION.labels(engine=self.name).inc()
            raise RuntimeError("Pricing engine is saturated")

        with self._pending_lock:
            self._pending_tasks += 1
            self._update_queue_metrics()

        def _finalise(_: Future) -> None:
            self._queue_capacity.release()
            with self._pending_lock:
                self._pending_tasks = max(0, self._pending_tasks - 1)
                self._update_queue_metrics()

        executor = self._get_executor()
        future = executor.submit(func, *args, **kwargs)
        future.add_done_callback(_finalise)
        return future

    def _resolve_seed_sequence(self, override: Optional[SeedSequence]) -> Optional[SeedSequence]:
        if override is not None:
            return override
        if self._base_seed_sequence is None:
            return None
        # Spawn a new child sequence for each invocation to avoid correlated draws
        return self._base_seed_sequence.spawn(1)[0]

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
            "standard_error": result.standard_error,
            "confidence_interval": result.confidence_interval,
            "capsule_id": result.capsule_id,
        }

    def _run_pricing(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model_name: str,
        override_volatility: Optional[float],
        seed_sequence: Optional[SeedSequence],
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

        start = time.perf_counter()
        sequence = self._resolve_seed_sequence(seed_sequence)

        if isinstance(model, MonteCarloModel):
            result = model.calculate_price(
                contract,
                market_data,
                volatility,
                seed_sequence=sequence,
            )
        else:
            result = model.calculate_price(contract, market_data, volatility)

        duration = time.perf_counter() - start
        MODEL_LATENCY.labels(model=model_name).observe(duration)
        if result.error:
            MODEL_ERRORS.labels(model=model_name).inc()

        payload = self._prepare_result(result, model_name, volatility)
        payload["cached"] = False
        self._cache.put(cache_key, payload)
        return payload

    def price_option(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model_name: str = "black_scholes",
        override_volatility: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        seed_sequence = SeedSequence(seed) if seed is not None else None
        future = self._submit_task(
            self._run_pricing,
            contract,
            market_data,
            model_name,
            override_volatility,
            seed_sequence,
        )
        timeout = None if self.task_timeout_seconds == 0 else self.task_timeout_seconds
        try:
            return future.result(timeout=timeout)
        except TimeoutError as exc:
            future.cancel()
            raise RuntimeError("Pricing task timed out") from exc

    def price_portfolio(
        self,
        contracts: Iterable[OptionContract],
        market_data: MarketData,
        model_name: str = "black_scholes",
        override_volatility: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        contract_list = list(contracts)
        if not contract_list:
            return []

        futures: Dict[Future[Dict[str, object]], int] = {}
        results: List[Optional[Dict[str, object]]] = [None] * len(contract_list)

        seed_sequences: Optional[List[SeedSequence]] = None
        if seed is not None:
            base = SeedSequence(seed)
            seed_sequences = base.spawn(len(contract_list))

        for index, contract in enumerate(contract_list):
            seq = seed_sequences[index] if seed_sequences is not None else None
            future = self._submit_task(
                self._run_pricing,
                contract,
                market_data,
                model_name,
                override_volatility,
                seq,
            )
            futures[future] = index

        for future in as_completed(futures):
            index = futures[future]
            try:
                timeout = None if self.task_timeout_seconds == 0 else self.task_timeout_seconds
                results[index] = future.result(timeout=timeout)
            except Exception as exc:  # pragma: no cover - defensive programming
                if isinstance(exc, TimeoutError):
                    future.cancel()
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

        for result in results:
            quantity = float(result.get("quantity") or 1.0)

            delta_value = result.get("position_delta")
            if delta_value is None:
                delta_value = float(result.get("delta") or 0.0) * quantity
            totals["delta"] += float(delta_value)

            gamma_value = result.get("position_gamma")
            if gamma_value is None:
                gamma_value = float(result.get("gamma") or 0.0) * quantity
            totals["gamma"] += float(gamma_value)

            theta_value = result.get("position_theta")
            if theta_value is None:
                theta_value = float(result.get("theta") or 0.0) * quantity
            totals["theta"] += float(theta_value)

            vega_value = result.get("position_vega")
            if vega_value is None:
                vega_value = float(result.get("vega") or 0.0) * quantity
            totals["vega"] += float(vega_value)

            rho_value = result.get("position_rho")
            if rho_value is None:
                rho_value = float(result.get("rho") or 0.0) * quantity
            totals["rho"] += float(rho_value)

            total_value = result.get("position_value")
            if total_value is None:
                total_value = float(result.get("theoretical_price") or 0.0) * quantity
            totals["total_value"] += float(total_value)

            totals["position_count"] += quantity

        totals["total_vega_exposure"] = totals["vega"] * 100.0
        return totals
