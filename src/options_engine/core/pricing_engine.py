"""Threaded options pricing engine with caching."""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import logging
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, TimeoutError, wait
from dataclasses import dataclass
from numbers import Integral, Real
from types import TracebackType
from typing import Any

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
from .black_scholes import BlackScholesModel
from .crr import BinomialModel
from .finite_difference import FiniteDifferenceModel
from .lsmc import LongstaffSchwartzModel
from .models import MarketData, OptionContract, PricingResult
from .monte_carlo import MonteCarloModel
from .volatility_surface import VolatilitySurface

LOGGER = logging.getLogger(__name__)
PricingModel = (
    BlackScholesModel
    | BinomialModel
    | MonteCarloModel
    | LongstaffSchwartzModel
    | FiniteDifferenceModel
)

MAX_ENGINE_THREADS = 256
MAX_ENGINE_QUEUE_SIZE = 100_000
MAX_RESULT_CACHE_SIZE = 1_000_000
MAX_ENGINE_TIMEOUT_SECONDS = 86_400.0
MAX_ENGINE_NAME_LENGTH = 128
MAX_MONTE_CARLO_SEED = 2**128 - 1
MAX_PORTFOLIO_CONTRACTS = 100_000


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalised = int(value)
    if not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return normalised


def _bounded_float(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalised = float(value)
    if not math.isfinite(normalised) or not minimum <= normalised <= maximum:
        raise ValueError(f"{name} must be within [{minimum:g}, {maximum:g}]")
    return normalised


@dataclass(slots=True)
class _CacheEntry:
    """Internal representation of a cached pricing result."""

    payload: dict[str, object]
    timestamp: float


class _ResultCache:
    """Thread-safe LRU cache with TTL support."""

    def __init__(self, max_size: int = 10_000, ttl_seconds: float = 5.0) -> None:
        self._max_size = _bounded_integer(
            "max_size",
            max_size,
            minimum=1,
            maximum=MAX_RESULT_CACHE_SIZE,
        )
        self._ttl = _bounded_float(
            "ttl_seconds",
            ttl_seconds,
            minimum=0.0,
            maximum=MAX_ENGINE_TIMEOUT_SECONDS,
        )
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()

    def get(self, key: str, now: float | None = None) -> dict[str, object] | None:
        if not key:
            return None

        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None

            current_time = time.monotonic() if now is None else now
            if self._ttl and current_time - entry.timestamp > self._ttl:
                self._entries.pop(key, None)
                return None

            self._entries.move_to_end(key)
            return dict(entry.payload)

    def put(self, key: str, payload: dict[str, object], now: float | None = None) -> None:
        if not key:
            return

        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)

            current_time = time.monotonic() if now is None else now
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
        volatility_surface: VolatilitySurface | None = None,
        monte_carlo_seed: int | None = None,
    ) -> None:
        self.num_threads = _bounded_integer(
            "num_threads",
            num_threads,
            minimum=1,
            maximum=MAX_ENGINE_THREADS,
        )
        self.queue_size = _bounded_integer(
            "queue_size",
            queue_size,
            minimum=0,
            maximum=MAX_ENGINE_QUEUE_SIZE,
        )
        self.queue_timeout_seconds = _bounded_float(
            "queue_timeout_seconds",
            queue_timeout_seconds,
            minimum=0.0,
            maximum=MAX_ENGINE_TIMEOUT_SECONDS,
        )
        self.task_timeout_seconds = _bounded_float(
            "task_timeout_seconds",
            task_timeout_seconds,
            minimum=0.0,
            maximum=MAX_ENGINE_TIMEOUT_SECONDS,
        )
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        self.name = name.strip()
        if (
            not self.name
            or len(self.name) > MAX_ENGINE_NAME_LENGTH
            or any(ord(character) < 32 or ord(character) == 127 for character in self.name)
        ):
            raise ValueError(f"name must contain between 1 and {MAX_ENGINE_NAME_LENGTH} characters")
        if volatility_surface is not None and not isinstance(volatility_surface, VolatilitySurface):
            raise TypeError("volatility_surface must be a VolatilitySurface or None")
        self.vol_surface = (
            volatility_surface if volatility_surface is not None else VolatilitySurface()
        )
        if monte_carlo_seed is not None:
            monte_carlo_seed = _bounded_integer(
                "monte_carlo_seed",
                monte_carlo_seed,
                minimum=0,
                maximum=MAX_MONTE_CARLO_SEED,
            )
        self._base_seed_sequence = (
            SeedSequence(monte_carlo_seed) if monte_carlo_seed is not None else None
        )
        self.models: dict[str, PricingModel] = {
            "black_scholes": BlackScholesModel(),
            "binomial_200": BinomialModel(steps=200),
            "monte_carlo_20k": MonteCarloModel(
                paths=20_000, seed_sequence=self._base_seed_sequence
            ),
            "longstaff_schwartz_20k": LongstaffSchwartzModel(
                paths=20_000,
                steps=64,
                seed_sequence=self._base_seed_sequence,
                reference_steps=1_000,
            ),
            "finite_difference_400": FiniteDifferenceModel(),
        }
        self._cache = _ResultCache(max_size=cache_size, ttl_seconds=cache_ttl_seconds)
        self._executor_lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=self.num_threads,
            thread_name_prefix="options-engine",
        )
        self._queue_capacity = threading.BoundedSemaphore(self.num_threads + self.queue_size)
        self._pending_lock = threading.Lock()
        self._pending_tasks = 0
        self._seed_lock = threading.Lock()
        THREADPOOL_WORKERS.labels(engine=self.name).set(self.num_threads)

    def __enter__(self) -> OptionsEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.shutdown()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        with contextlib.suppress(Exception):
            self.shutdown(wait=False)

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

    def _submit_task(
        self,
        func: Callable[..., dict[str, object]],
        *args: object,
        admission_timeout_seconds: float | None = None,
    ) -> Future[dict[str, object]]:
        start = time.perf_counter()
        admission_timeout = self.queue_timeout_seconds
        if admission_timeout_seconds is not None:
            admission_timeout = min(admission_timeout, max(0.0, admission_timeout_seconds))
        if admission_timeout == 0:
            acquired = self._queue_capacity.acquire(blocking=False)
        else:
            acquired = self._queue_capacity.acquire(timeout=admission_timeout)
        wait_time = time.perf_counter() - start
        THREADPOOL_QUEUE_WAIT.labels(engine=self.name).observe(wait_time)
        if not acquired:
            THREADPOOL_REJECTIONS.labels(engine=self.name).inc()
            THREADPOOL_SATURATION.labels(engine=self.name).inc()
            raise RuntimeError("Pricing engine is saturated")

        with self._pending_lock:
            self._pending_tasks += 1
            self._update_queue_metrics()

        def _finalise(_: Future[dict[str, object]]) -> None:
            self._queue_capacity.release()
            with self._pending_lock:
                self._pending_tasks = max(0, self._pending_tasks - 1)
                self._update_queue_metrics()

        try:
            executor = self._get_executor()
            future = executor.submit(func, *args)
        except Exception:
            self._queue_capacity.release()
            with self._pending_lock:
                self._pending_tasks = max(0, self._pending_tasks - 1)
                self._update_queue_metrics()
            raise
        future.add_done_callback(_finalise)
        return future

    def _resolve_seed_sequence(self, override: SeedSequence | None) -> SeedSequence | None:
        if override is not None:
            return override
        if self._base_seed_sequence is None:
            return None
        # SeedSequence.spawn mutates its child counter; serialize access so concurrent
        # requests cannot receive overlapping or implementation-dependent streams.
        with self._seed_lock:
            return self._base_seed_sequence.spawn(1)[0]

    @staticmethod
    def _seed_identity(seed_sequence: SeedSequence | None) -> dict[str, object] | None:
        if seed_sequence is None:
            return None
        entropy = seed_sequence.entropy
        if isinstance(entropy, Sequence):
            entropy_value: object = [int(item) for item in entropy]
        elif entropy is not None:
            entropy_value = int(entropy)
        else:
            entropy_value = None
        return {
            "entropy": entropy_value,
            "spawn_key": [int(item) for item in seed_sequence.spawn_key],
            "pool_size": int(seed_sequence.pool_size),
        }

    @staticmethod
    def _model_config(model: PricingModel) -> dict[str, object]:
        if isinstance(model, MonteCarloModel):
            return {
                "paths": int(model.paths),
                "antithetic": bool(model.antithetic),
                "use_control_variates": bool(model.use_control_variates),
            }
        if isinstance(model, LongstaffSchwartzModel):
            return {
                "paths": int(model.paths),
                "steps": int(model.steps),
                "antithetic": bool(model.antithetic),
                "cv_folds": int(model.cv_folds),
            }
        if isinstance(model, BinomialModel):
            return {"steps": int(model.steps), "tree": "crr-adaptive"}
        if isinstance(model, FiniteDifferenceModel):
            return {
                "space_steps": int(model.space_steps),
                "time_steps": int(model.time_steps),
                "scheme": (
                    "crank_nicolson_rannacher" if model.rannacher_smoothing else "crank_nicolson"
                ),
                "rannacher_smoothing": bool(model.rannacher_smoothing),
                "grid_type": model.grid_type,
                "grid_concentration": float(model.grid_concentration),
                "tail_standard_deviations": float(model.tail_standard_deviations),
                "s_max_override": model.s_max_override,
                "refinement_levels": int(model.refinement_levels),
                "refinement_ratio": int(model.refinement_ratio),
                "american_solver": model.exercise_solver,
                "psor_omega": float(model.psor_omega),
                "psor_tolerance": float(model.psor_tolerance),
                "psor_max_iterations": int(model.psor_max_iterations),
                "penalty_parameter": float(model.penalty_parameter),
                "penalty_tolerance": float(model.penalty_tolerance),
                "penalty_max_iterations": int(model.penalty_max_iterations),
            }
        return {}

    def _make_cache_key(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model_name: str,
        volatility: float,
        model: PricingModel,
        seed_sequence: SeedSequence | None,
    ) -> str:
        payload: dict[str, Any] = {
            "contract": {
                "contract_id": contract.contract_id,
                "symbol": contract.symbol,
                "strike_price": float(contract.strike_price).hex(),
                "time_to_expiry": float(contract.time_to_expiry).hex(),
                "option_type": contract.option_type.value,
                "exercise_style": contract.exercise_style.value,
            },
            "market": {
                "spot_price": float(market_data.spot_price).hex(),
                "risk_free_rate": float(market_data.risk_free_rate).hex(),
                "dividend_yield": float(market_data.dividend_yield).hex(),
            },
            "model": model_name,
            "model_config": self._model_config(model),
            "volatility": float(volatility).hex(),
            "seed": self._seed_identity(seed_sequence),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _prepare_result(
        self, result: PricingResult, model_name: str, volatility: float
    ) -> dict[str, object]:
        return {
            "contract_id": result.contract_id,
            "theoretical_price": result.theoretical_price,
            "delta": result.delta,
            "gamma": result.gamma,
            "theta": result.theta,
            "vega": result.vega,
            "rho": result.rho,
            "implied_volatility": (
                result.implied_volatility if result.implied_volatility is not None else volatility
            ),
            "model_used": result.model_used,
            "model_requested": model_name,
            "volatility_used": volatility,
            "computation_time_ms": result.computation_time_ms,
            "error": result.error,
            "standard_error": result.standard_error,
            "confidence_interval": result.confidence_interval,
            "estimate_diagnostics": result.estimate_diagnostics,
            "numerical_diagnostics": result.numerical_diagnostics,
            "capsule_id": result.capsule_id,
        }

    def _run_pricing(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model_name: str,
        override_volatility: float | None,
        seed_sequence: SeedSequence | None,
    ) -> dict[str, object]:
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

        sequence = seed_sequence
        stochastic = isinstance(model, (MonteCarloModel, LongstaffSchwartzModel))
        cacheable = not stochastic or sequence is not None
        cache_key = self._make_cache_key(
            contract,
            market_data,
            model_name,
            volatility,
            model,
            sequence,
        )
        if cacheable:
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached["cached"] = True
                return cached

        start = time.perf_counter()

        if isinstance(model, (MonteCarloModel, LongstaffSchwartzModel)):
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
        if cacheable:
            self._cache.put(cache_key, payload)
        return payload

    def price_option(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model_name: str = "black_scholes",
        override_volatility: float | None = None,
        seed: int | None = None,
    ) -> dict[str, object]:
        self._validate_request_inputs(
            contract,
            market_data,
            model_name,
            override_volatility,
        )
        model = self.models[model_name]
        requested_seed = self._request_seed_sequence(seed)
        seed_sequence = (
            self._resolve_seed_sequence(requested_seed)
            if isinstance(model, (MonteCarloModel, LongstaffSchwartzModel))
            else None
        )
        deadline = (
            time.perf_counter() + self.task_timeout_seconds
            if self.task_timeout_seconds > 0
            else None
        )
        future = self._submit_task(
            self._run_pricing,
            contract,
            market_data,
            model_name,
            override_volatility,
            seed_sequence,
            admission_timeout_seconds=(
                max(0.0, deadline - time.perf_counter()) if deadline is not None else None
            ),
        )
        timeout = max(0.0, deadline - time.perf_counter()) if deadline is not None else None
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
        override_volatility: float | None = None,
        seed: int | None = None,
    ) -> list[dict[str, object]]:
        if isinstance(contracts, (str, bytes)) or not isinstance(contracts, Iterable):
            raise TypeError("contracts must be an iterable of OptionContract values")
        contract_list = list(itertools.islice(contracts, MAX_PORTFOLIO_CONTRACTS + 1))
        if len(contract_list) > MAX_PORTFOLIO_CONTRACTS:
            raise ValueError(f"portfolio exceeds the {MAX_PORTFOLIO_CONTRACTS}-contract limit")
        if not contract_list:
            return []
        if any(not isinstance(contract, OptionContract) for contract in contract_list):
            raise TypeError("every portfolio entry must be an OptionContract")
        # Preflight every contract before allocating random streams or
        # submitting any work. A mixed-style invalid portfolio must fail
        # atomically rather than executing a valid prefix first.
        for contract in contract_list:
            self._validate_request_inputs(
                contract,
                market_data,
                model_name,
                override_volatility,
            )

        deadline = (
            time.perf_counter() + self.task_timeout_seconds
            if self.task_timeout_seconds > 0
            else None
        )

        futures: dict[Future[dict[str, object]], int] = {}
        results: list[dict[str, object] | None] = [None] * len(contract_list)

        seed_sequences: list[SeedSequence] | None = None
        model = self.models[model_name]
        explicit_base = self._request_seed_sequence(seed)
        if isinstance(model, (MonteCarloModel, LongstaffSchwartzModel)):
            if explicit_base is not None:
                seed_sequences = explicit_base.spawn(len(contract_list))
            elif self._base_seed_sequence is not None:
                with self._seed_lock:
                    seed_sequences = self._base_seed_sequence.spawn(len(contract_list))

        try:
            for index, contract in enumerate(contract_list):
                remaining = (
                    max(0.0, deadline - time.perf_counter()) if deadline is not None else None
                )
                if remaining == 0.0:
                    raise RuntimeError("Portfolio pricing timed out during admission")
                seq = seed_sequences[index] if seed_sequences is not None else None
                future = self._submit_task(
                    self._run_pricing,
                    contract,
                    market_data,
                    model_name,
                    override_volatility,
                    seq,
                    admission_timeout_seconds=remaining,
                )
                futures[future] = index
        except Exception:
            for submitted in futures:
                submitted.cancel()
            raise

        timeout = max(0.0, deadline - time.perf_counter()) if deadline is not None else None
        done, not_done = wait(futures, timeout=timeout, return_when=FIRST_EXCEPTION)
        failed = next((future for future in done if future.exception() is not None), None)
        if failed is not None:
            for future in not_done:
                future.cancel()
            # Re-raise the model exception with its original traceback.
            failed.result()
        if not_done:
            for future in not_done:
                future.cancel()
            raise RuntimeError(
                f"Portfolio pricing timed out with {len(not_done)} unfinished task(s)"
            )

        for future in done:
            index = futures[future]
            results[index] = future.result()

        if any(result is None for result in results):
            raise RuntimeError("Portfolio pricing did not produce a result for every contract")
        return [result for result in results if result is not None]

    @staticmethod
    def _request_seed_sequence(seed: int | None) -> SeedSequence | None:
        if seed is None:
            return None
        normalised = _bounded_integer(
            "seed",
            seed,
            minimum=0,
            maximum=MAX_MONTE_CARLO_SEED,
        )
        return SeedSequence(normalised)

    def _validate_request_inputs(
        self,
        contract: OptionContract,
        market_data: MarketData,
        model_name: str,
        override_volatility: float | None,
    ) -> None:
        if not isinstance(contract, OptionContract):
            raise TypeError("contract must be an OptionContract")
        if not isinstance(market_data, MarketData):
            raise TypeError("market_data must be MarketData")
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string")
        if not model_name or len(model_name) > 128 or model_name not in self.models:
            raise ValueError(f"Unknown model '{model_name}'")
        if override_volatility is not None:
            volatility = _bounded_float(
                "override_volatility",
                override_volatility,
                minimum=0.0,
                maximum=5.0,
            )
            if volatility <= 1e-6:
                raise ValueError("override_volatility must be greater than 1e-6")
        model = self.models[model_name]
        if isinstance(model, (BlackScholesModel, MonteCarloModel)):
            if contract.exercise_style.value != "european":
                raise ValueError(f"{model_name} supports European exercise only")
        elif (
            isinstance(model, LongstaffSchwartzModel)
            and contract.exercise_style.value != "american"
        ):
            raise ValueError("Longstaff-Schwartz requires American exercise")

    @staticmethod
    def calculate_portfolio_greeks(results: Iterable[dict[str, object]]) -> dict[str, float]:
        def numeric(name: str, value: object, default: float = 0.0) -> float:
            if value is None:
                return default
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite real number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"{name} must be a finite real number")
            return number

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

        if isinstance(results, (str, bytes)) or not isinstance(results, Iterable):
            raise TypeError("results must be an iterable of mappings")
        result_rows = list(itertools.islice(results, MAX_PORTFOLIO_CONTRACTS + 1))
        if len(result_rows) > MAX_PORTFOLIO_CONTRACTS:
            raise ValueError(
                f"portfolio results exceed the {MAX_PORTFOLIO_CONTRACTS}-position limit"
            )

        def add(total_name: str, value: float) -> None:
            updated = totals[total_name] + value
            if not math.isfinite(updated):
                raise ValueError(f"portfolio aggregate '{total_name}' exceeds floating-point range")
            totals[total_name] = updated

        for result in result_rows:
            if not isinstance(result, Mapping):
                raise TypeError("every portfolio result must be a mapping")
            quantity = numeric("quantity", result.get("quantity"), 1.0)
            if not 0.0 < quantity <= 1_000_000.0:
                raise ValueError("quantity must be within (0, 1000000]")

            delta_value = result.get("position_delta")
            if delta_value is None:
                delta_value = numeric("delta", result.get("delta")) * quantity
            add("delta", numeric("position_delta", delta_value))

            gamma_value = result.get("position_gamma")
            if gamma_value is None:
                gamma_value = numeric("gamma", result.get("gamma")) * quantity
            add("gamma", numeric("position_gamma", gamma_value))

            theta_value = result.get("position_theta")
            if theta_value is None:
                theta_value = numeric("theta", result.get("theta")) * quantity
            add("theta", numeric("position_theta", theta_value))

            vega_value = result.get("position_vega")
            if vega_value is None:
                vega_value = numeric("vega", result.get("vega")) * quantity
            add("vega", numeric("position_vega", vega_value))

            rho_value = result.get("position_rho")
            if rho_value is None:
                rho_value = numeric("rho", result.get("rho")) * quantity
            add("rho", numeric("position_rho", rho_value))

            total_value = result.get("position_value")
            if total_value is None:
                total_value = (
                    numeric("theoretical_price", result.get("theoretical_price")) * quantity
                )
            add("total_value", numeric("position_value", total_value))

            add("position_count", quantity)

        total_vega_exposure = totals["vega"] * 100.0
        if not math.isfinite(total_vega_exposure):
            raise ValueError("total vega exposure exceeds floating-point range")
        totals["total_vega_exposure"] = total_vega_exposure
        return totals
