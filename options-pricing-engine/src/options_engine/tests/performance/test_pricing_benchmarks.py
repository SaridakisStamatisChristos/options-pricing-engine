"""Micro-benchmarks that enforce minimal performance gates for pricing models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, List

import numpy as np
import pytest
import scipy

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


def _extract_paths_used(model_used: str) -> int | None:
    match = re.search(r"(\d+)$", model_used)
    if match is None:
        return None
    return int(match.group(1))


def _run_latency_benchmark(
    model_name: str,
    model,
    scenarios: Iterable[BenchmarkScenario],
    *,
    warmup_iterations: int = 10,
    measurement_iterations: int = 100,
    seed_prefix: int | None = None,
) -> dict[str, float | list[float]]:
    durations: list[float] = []
    ci_half_widths: list[float] = []
    ci_bps_values: list[float] = []
    measurement_seeds: list[int] = []
    cell_metrics: list[dict[str, object]] = []

    total_iterations = warmup_iterations + measurement_iterations

    for scenario_index, scenario in enumerate(scenarios):
        cell_durations: list[float] = []
        cell_ci_half_widths: list[float] = []
        cell_ci_bps: list[float] = []
        paths_used: int | None = None
        for iteration in range(total_iterations):
            kwargs = {}
            seed_value: int | None = None
            if seed_prefix is not None:
                seed_value = seed_prefix + scenario_index * total_iterations + iteration
                kwargs["seed_sequence"] = SeedSequence(seed_value)
            result = model.calculate_price(
                scenario.contract, scenario.market, scenario.volatility, **kwargs
            )

            if iteration < warmup_iterations:
                continue

            durations.append(result.computation_time_ms)
            cell_durations.append(result.computation_time_ms)
            if seed_value is not None:
                measurement_seeds.append(seed_value)

            if result.confidence_interval is not None:
                lower, upper = result.confidence_interval
                half_width = (upper - lower) / 2.0
                ci_half_widths.append(half_width)
                cell_ci_half_widths.append(half_width)
                if result.theoretical_price > 0:
                    ci_bps = (half_width / result.theoretical_price) * 10_000.0
                    ci_bps_values.append(ci_bps)
                    cell_ci_bps.append(ci_bps)

            if paths_used is None and hasattr(result, "model_used") and result.model_used:
                extracted = _extract_paths_used(result.model_used)
                if extracted is not None:
                    paths_used = extracted

        if not cell_durations:
            continue

        cell_summary: dict[str, object] = {
            "scenario": scenario.contract.symbol,
            "p50_latency_ms": float(np.percentile(cell_durations, 50)),
            "p95_latency_ms": float(np.percentile(cell_durations, 95)),
            "p99_latency_ms": float(np.percentile(cell_durations, 99)),
        }
        if cell_ci_half_widths:
            cell_summary["median_ci_half_width"] = float(median(cell_ci_half_widths))
        if cell_ci_bps:
            cell_summary["ci_bps"] = float(median(cell_ci_bps))
        if paths_used is not None:
            cell_summary["paths_used"] = paths_used
        if hasattr(model, "antithetic"):
            cell_summary["vr_pipeline"] = "antithetic" if getattr(model, "antithetic") else "plain"
        cell_metrics.append(cell_summary)

    metrics: dict[str, float | list[float] | dict[str, object]] = {
        "samples": durations,
        "p50_latency_ms": float(np.percentile(durations, 50)),
        "p95_latency_ms": float(np.percentile(durations, 95)),
        "p99_latency_ms": float(np.percentile(durations, 99)),
        "max_latency_ms": max(durations),
        "cell_metrics": cell_metrics,
        "libraries": {"numpy": np.__version__, "scipy": scipy.__version__},
    }

    if ci_half_widths:
        metrics["median_ci_half_width"] = float(median(ci_half_widths))
    if ci_bps_values:
        metrics["median_ci_bps"] = float(median(ci_bps_values))
    if measurement_seeds:
        measurement_seeds.sort()
        metrics["seed_lineage"] = {
            "first": measurement_seeds[0],
            "last": measurement_seeds[-1],
            "count": len(measurement_seeds),
        }
    if hasattr(model, "antithetic"):
        metrics["vr_pipeline"] = "antithetic" if getattr(model, "antithetic") else "plain"
    if hasattr(model, "paths"):
        metrics["paths_used"] = int(getattr(model, "paths"))
    return metrics


def _print_summary(model_key: str, metrics: dict[str, float | list[float]], baseline: dict[str, float]) -> None:
    target = PERF_TARGETS[model_key]
    p99_latency = metrics["p99_latency_ms"]
    regression_limit = _regression_limit(baseline["p99_latency_ms"])
    summary_parts = [
        f"model={model_key}",
        f"p50_ms={metrics['p50_latency_ms']:.2f}",
        f"p95_ms={metrics['p95_latency_ms']:.2f}",
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
    if metrics.get("seed_lineage"):
        seed_info = metrics["seed_lineage"]
        summary_parts.append(
            "seed_lineage="
            f"{seed_info['first']}->{seed_info['last']} ({seed_info['count']} seeds)"
        )
    if metrics.get("paths_used") is not None:
        summary_parts.append(f"paths_used={metrics['paths_used']}")
    if metrics.get("vr_pipeline"):
        summary_parts.append(f"vr_pipeline={metrics['vr_pipeline']}")
    libraries = metrics.get("libraries")
    if isinstance(libraries, dict):
        lib_summary = ",".join(f"{name}={version}" for name, version in sorted(libraries.items()))
        summary_parts.append(f"lib_versions={lib_summary}")
    print("[bench] " + " ".join(summary_parts))

    cell_metrics = metrics.get("cell_metrics")
    if isinstance(cell_metrics, list) and cell_metrics:
        top_cells = sorted(
            (cell for cell in cell_metrics if "p99_latency_ms" in cell),
            key=lambda cell: cell["p99_latency_ms"],
            reverse=True,
        )[:5]
        for cell in top_cells:
            cell_parts = [
                "[bench-cell]",
                f"model={model_key}",
                f"scenario={cell.get('scenario', 'unknown')}",
                f"p99_ms={cell['p99_latency_ms']:.2f}",
            ]
            ci_bps = cell.get("ci_bps")
            if ci_bps is not None:
                cell_parts.append(f"ci_bps={ci_bps:.1f}")
            paths_used = cell.get("paths_used")
            if paths_used is not None:
                cell_parts.append(f"paths_used={paths_used}")
            vr_pipeline = cell.get("vr_pipeline")
            if vr_pipeline:
                cell_parts.append(f"vr_pipeline={vr_pipeline}")
            print(" ".join(cell_parts))


def test_black_scholes_latency_gate(golden_grid: List[BenchmarkScenario], benchmark_baseline: dict[str, dict[str, float]]) -> None:
    model_key = "black_scholes"
    model = BlackScholesModel()
    metrics = _run_latency_benchmark(model_key, model, golden_grid)

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
    metrics = _run_latency_benchmark(model_key, model, golden_grid)

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

    assert "seed_lineage" in metrics, "Monte Carlo benchmark must record seed lineage"
    assert metrics.get("paths_used") == baseline.get("paths_used"), (
        "Monte Carlo benchmark paths differ from baseline; cannot compare CI widths fairly"
    )
