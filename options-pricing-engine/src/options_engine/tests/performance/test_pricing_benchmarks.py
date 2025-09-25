"""Micro-benchmarks that enforce minimal performance gates for pricing models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, List

import numpy as np
import pytest

from options_engine.core.models import ExerciseStyle, MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import BlackScholesModel, BinomialModel, MonteCarloModel
from numpy.random import SeedSequence


@dataclass(frozen=True)
class BenchmarkScenario:
    """Container describing an option pricing scenario for benchmarking."""

    contract: OptionContract
    market: MarketData
    volatility: float


def _golden_grid() -> List[BenchmarkScenario]:
    """Return the golden grid of benchmark scenarios covering diverse regimes."""

    base_spot = 100.0
    risk_free_rate = 0.02
    dividend_yield = 0.01

    maturities = [0.05, 0.5, 1.5]
    volatilities = [0.15, 0.3]
    strike_multipliers = [0.85, 1.0, 1.15]
    option_types = [OptionType.CALL, OptionType.PUT]

    scenarios: List[BenchmarkScenario] = []
    scenario_id = 0

    for time_to_expiry in maturities:
        market = MarketData(
            spot_price=base_spot,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
        )
        for volatility in volatilities:
            for multiplier in strike_multipliers:
                strike = base_spot * multiplier
                for option_type in option_types:
                    scenario_id += 1
                    contract = OptionContract(
                        symbol=f"GOLDEN_{scenario_id}",
                        strike_price=strike,
                        time_to_expiry=time_to_expiry,
                        option_type=option_type,
                        exercise_style=ExerciseStyle.EUROPEAN,
                    )
                    scenarios.append(
                        BenchmarkScenario(
                            contract=contract,
                            market=market,
                            volatility=volatility,
                        )
                    )
    return scenarios


PERF_TARGETS = {
    "black_scholes": 75.0,  # milliseconds (p99)
    "binomial": 75.0,  # milliseconds (p99)
    "monte_carlo": 500.0,  # milliseconds (p99)
}

REGRESSION_TOLERANCE = 0.10  # Allow up to 10% regression versus the golden baseline.
BASELINE_PATH = Path(__file__).with_name("benchmarks_baseline.json")


@pytest.fixture(scope="module")
def golden_grid() -> List[BenchmarkScenario]:
    return _golden_grid()


@pytest.fixture(scope="module")
def benchmark_baseline() -> dict[str, dict[str, float]]:
    with BASELINE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _regression_limit(baseline_value: float, *, cushion: float = 1.0) -> float:
    """Return the regression guard value for a metric."""

    # Provide a small absolute cushion so that extremely fast baselines (for example <10ms)
    # are not overly strict when running on shared CI hardware. The cushion can be disabled
    # for deterministic metrics (such as CI widths) by passing ``cushion=0.0``.
    return max(baseline_value * (1.0 + REGRESSION_TOLERANCE), baseline_value + cushion)


def _run_latency_benchmark(
    model_name: str,
    model,
    scenarios: Iterable[BenchmarkScenario],
    *,
    iterations: int,
    seed_prefix: int | None = None,
) -> dict[str, float | list[float]]:
    durations: list[float] = []
    ci_half_widths: list[float] = []

    for scenario_index, scenario in enumerate(scenarios):
        for iteration in range(iterations):
            kwargs = {}
            if seed_prefix is not None:
                # Stabilise Monte Carlo runs so CI width comparisons are meaningful across builds.
                kwargs["seed_sequence"] = SeedSequence(seed_prefix + scenario_index * iterations + iteration)
            result = model.calculate_price(scenario.contract, scenario.market, scenario.volatility, **kwargs)
            durations.append(result.computation_time_ms)
            if result.confidence_interval is not None:
                lower, upper = result.confidence_interval
                ci_half_widths.append((upper - lower) / 2.0)

    metrics: dict[str, float | list[float]] = {
        "samples": durations,
        "p99_latency_ms": float(np.percentile(durations, 99)),
        "max_latency_ms": max(durations),
    }
    if ci_half_widths:
        metrics["median_ci_half_width"] = float(median(ci_half_widths))
    return metrics


def _print_summary(model_key: str, metrics: dict[str, float | list[float]], baseline: dict[str, float]) -> None:
    target = PERF_TARGETS[model_key]
    p99_latency = metrics["p99_latency_ms"]
    regression_limit = _regression_limit(baseline["p99_latency_ms"])
    summary_parts = [
        f"model={model_key}",
        f"p99_ms={p99_latency:.2f}",
        f"target_ms={target:.0f}",
        f"baseline_ms={baseline['p99_latency_ms']:.2f}",
        f"regression_guard_ms={regression_limit:.2f}",
    ]
    median_ci = baseline.get("median_ci_half_width")
    if "median_ci_half_width" in metrics and median_ci is not None:
        current_ci = metrics["median_ci_half_width"]
        ci_limit = _regression_limit(median_ci, cushion=0.0)
        summary_parts.append(f"median_ci_width={current_ci:.4f}")
        summary_parts.append(f"baseline_ci_width={median_ci:.4f}")
        summary_parts.append(f"ci_regression_guard={ci_limit:.4f}")
    print("[bench] " + " ".join(summary_parts))


def test_black_scholes_latency_gate(golden_grid: List[BenchmarkScenario], benchmark_baseline: dict[str, dict[str, float]]) -> None:
    model_key = "black_scholes"
    model = BlackScholesModel()
    metrics = _run_latency_benchmark(model_key, model, golden_grid, iterations=5)

    target = PERF_TARGETS[model_key]
    baseline = benchmark_baseline[model_key]

    _print_summary(model_key, metrics, baseline)

    assert metrics["p99_latency_ms"] <= target, (
        f"Black-Scholes p99 latency {metrics['p99_latency_ms']:.2f}ms exceeded target {target:.2f}ms"
    )
    assert metrics["p99_latency_ms"] <= _regression_limit(baseline["p99_latency_ms"]), (
        "Black-Scholes p99 latency regressed by more than 10% against the baseline"
    )


def test_binomial_latency_gate(golden_grid: List[BenchmarkScenario], benchmark_baseline: dict[str, dict[str, float]]) -> None:
    model_key = "binomial"
    model = BinomialModel()
    metrics = _run_latency_benchmark(model_key, model, golden_grid, iterations=4)

    target = PERF_TARGETS[model_key]
    baseline = benchmark_baseline[model_key]

    _print_summary(model_key, metrics, baseline)

    assert metrics["p99_latency_ms"] <= target, (
        f"Binomial p99 latency {metrics['p99_latency_ms']:.2f}ms exceeded target {target:.2f}ms"
    )
    assert metrics["p99_latency_ms"] <= _regression_limit(baseline["p99_latency_ms"]), (
        "Binomial p99 latency regressed by more than 10% against the baseline"
    )


def test_monte_carlo_latency_and_ci_gate(
    golden_grid: List[BenchmarkScenario], benchmark_baseline: dict[str, dict[str, float]]
) -> None:
    model_key = "monte_carlo"
    model = MonteCarloModel(paths=8_000, antithetic=True)
    metrics = _run_latency_benchmark(
        model_key,
        model,
        golden_grid,
        iterations=3,
        seed_prefix=2024,
    )

    target = PERF_TARGETS[model_key]
    baseline = benchmark_baseline[model_key]

    _print_summary(model_key, metrics, baseline)

    assert metrics["p99_latency_ms"] <= target, (
        f"Monte Carlo p99 latency {metrics['p99_latency_ms']:.2f}ms exceeded target {target:.2f}ms"
    )
    assert metrics["p99_latency_ms"] <= _regression_limit(baseline["p99_latency_ms"]), (
        "Monte Carlo p99 latency regressed by more than 10% against the baseline"
    )

    assert "median_ci_half_width" in metrics, "Monte Carlo benchmark did not record CI width"

    ci_baseline = baseline["median_ci_half_width"]
    assert metrics["median_ci_half_width"] <= _regression_limit(ci_baseline, cushion=0.0), (
        "Monte Carlo CI half-width regressed by more than 10% against the baseline"
    )
