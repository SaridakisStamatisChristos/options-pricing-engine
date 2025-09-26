"""Micro-benchmarks enforcing latency and precision gates for pricing models."""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable, List, Sequence

import numpy as np
import pytest
from numpy.random import SeedSequence
from scipy import stats

from options_engine.core.models import (
    ExerciseStyle,
    MarketData,
    OptionContract,
    OptionType,
    PricingResult,
)
from options_engine.core.pricing_models import BlackScholesModel, BinomialModel, MonteCarloModel
from options_engine.core.variance_reduction import VarianceReductionToolkit


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
    "monte_carlo": 2.80,  # milliseconds (p99)
}

LATENCY_ENV_VARS = {
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}

BASELINE_PATH = Path(__file__).with_name("benchmarks_baseline.json")
REPORT_PATH = Path(__file__).with_name("benchmarks_report.json")
UPDATE_BASELINE = os.environ.get("UPDATE_BASELINE", "0") == "1"

EPS_PRICE = 1e-6
CI_Z_SCORE = 1.96

PRECISION_BUCKETS = {
    "A": {"lower": 0.50, "upper": float("inf"), "threshold": 2.0, "cap": 65_536},
    "B": {"lower": 0.10, "upper": 0.50, "threshold": 10.0, "cap": 65_536},
    "C": {"lower": 0.0, "upper": 0.10, "threshold": 0.0005, "cap": 262_144},
}


_LATENCY_RESULTS: dict[str, dict[str, float]] = {}
_PRECISION_RESULTS: dict[str, object] | None = None


def _record_latency_summary(model_key: str, metrics: dict[str, object]) -> None:
    _LATENCY_RESULTS[model_key] = {
        "p50_latency_ms": float(metrics["p50_latency_ms"]),
        "p95_latency_ms": float(metrics["p95_latency_ms"]),
        "p99_latency_ms": float(metrics["p99_latency_ms"]),
    }


def _record_precision_summary(metrics: dict[str, object]) -> None:
    global _PRECISION_RESULTS
    _PRECISION_RESULTS = metrics


def _write_benchmarks_report() -> None:
    if _PRECISION_RESULTS is None:
        return

    report: dict[str, object] = {}
    for model_key, summary in _LATENCY_RESULTS.items():
        report[model_key] = dict(summary)

    precision = _PRECISION_RESULTS.get("monte_carlo")
    if isinstance(precision, dict):
        monte_carlo_entry = report.setdefault("monte_carlo", {})
        for key, value in precision.items():
            if key in {"cell_metrics"}:
                continue
            monte_carlo_entry[key] = value
        if "cell_metrics" in precision:
            monte_carlo_entry["cell_metrics"] = precision["cell_metrics"]

    target_path = BASELINE_PATH if UPDATE_BASELINE else REPORT_PATH
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


@pytest.fixture(scope="module")
def golden_grid() -> List[BenchmarkScenario]:
    return _golden_grid()


@pytest.fixture(scope="module")
def benchmark_baseline() -> dict[str, dict[str, float]]:
    with BASELINE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@contextlib.contextmanager
def _latency_environment() -> Iterable[None]:
    """Ensure the latency benchmark runs in a deterministic, single-threaded setup."""

    original_values = {key: os.environ.get(key) for key in LATENCY_ENV_VARS}
    for key, value in LATENCY_ENV_VARS.items():
        os.environ[key] = value
    previous_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        yield
    finally:
        logging.disable(previous_level)
        for key, previous in original_values.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _spawn_sequences(
    seed_prefix: int | None, scenario_index: int, total_iterations: int
) -> Sequence[SeedSequence | None]:
    if seed_prefix is None:
        return [None] * total_iterations
    base = SeedSequence(seed_prefix + scenario_index)
    return base.spawn(total_iterations)


def _extract_paths_used(model_used: str | None) -> int | None:
    if not model_used:
        return None
    match = re.search(r"(\d+)$", model_used)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _assign_bucket(price: float | None) -> str | None:
    if price is None:
        return None
    for name, config in PRECISION_BUCKETS.items():
        if config["lower"] <= price < config["upper"]:
            return name
    return "A" if price >= 0.50 else "C"


def _ci_half_width(result: PricingResult) -> float | None:
    if result.standard_error is not None:
        return CI_Z_SCORE * result.standard_error
    if result.confidence_interval is not None:
        lower, upper = result.confidence_interval
        return (upper - lower) / 2.0
    return None


def _summarise_cv_reports(reports: Sequence[dict[str, float | bool | None]]) -> dict[str, float | bool | None]:
    if not reports:
        return {
            "cv_used": False,
            "rho": None,
            "beta": None,
            "raw_var": None,
            "residual_var": None,
        }

    cv_used = any(bool(report.get("cv_used")) for report in reports)

    def _median(values: Sequence[float | None]) -> float | None:
        filtered = [value for value in values if value is not None and math.isfinite(value)]
        if not filtered:
            return None
        return float(median(filtered))

    beta_values = [report.get("beta") for report in reports if report.get("beta") is not None]
    beta_summary: float | tuple[float, ...] | None
    if beta_values:
        first = beta_values[0]
        if isinstance(first, (list, tuple)):
            columns = list(zip(*beta_values))
            beta_summary = tuple(float(median(column)) for column in columns)
        else:
            beta_summary = _median(beta_values)
    else:
        beta_summary = None

    return {
        "cv_used": cv_used,
        "rho": _median([report.get("rho") for report in reports]),
        "beta": beta_summary,
        "raw_var": _median([report.get("raw_var") for report in reports]),
        "residual_var": _median([report.get("residual_var") for report in reports]),
    }


def _latency_benchmark(
    model_key: str,
    model,
    scenarios: Iterable[BenchmarkScenario],
    *,
    warmup_iterations: int = 10,
    measurement_iterations: int = 100,
    seed_prefix: int | None = None,
) -> dict[str, object]:
    scenarios = list(scenarios)
    durations: list[float] = []
    cell_metrics: list[dict[str, object]] = []

    with _latency_environment():
        for scenario_index, scenario in enumerate(scenarios):
            total_iterations = warmup_iterations + measurement_iterations
            sequences = _spawn_sequences(seed_prefix, scenario_index, total_iterations)
            cell_durations: list[float] = []
            cell_prices: list[float] = []
            cell_half_widths: list[float] = []

            for iteration, sequence in enumerate(sequences):
                kwargs = {}
                if sequence is not None:
                    kwargs["seed_sequence"] = sequence
                start = time.perf_counter()
                result = model.calculate_price(
                    scenario.contract,
                    scenario.market,
                    scenario.volatility,
                    **kwargs,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if iteration < warmup_iterations:
                    continue

                durations.append(elapsed_ms)
                cell_durations.append(elapsed_ms)
                cell_prices.append(result.theoretical_price)
                half_width = _ci_half_width(result)
                if half_width is not None:
                    cell_half_widths.append(half_width)

            if not cell_durations:
                continue

            cell_summary: dict[str, object] = {
                "scenario": scenario.contract.symbol,
                "p50_latency_ms": float(np.percentile(cell_durations, 50)),
                "p95_latency_ms": float(np.percentile(cell_durations, 95)),
                "p99_latency_ms": float(np.percentile(cell_durations, 99)),
            }
            if cell_prices:
                cell_price = float(median(cell_prices))
                cell_summary["cell_price"] = cell_price
                bucket = _assign_bucket(cell_price)
                if bucket is not None:
                    cell_summary["bucket"] = bucket
            if cell_half_widths:
                cell_summary["median_ci_half_width"] = float(median(cell_half_widths))
            cell_metrics.append(cell_summary)

    if not durations:
        raise RuntimeError("Latency benchmark produced no samples")

    metrics: dict[str, object] = {
        "latency_mode": True,
        "samples": durations,
        "p50_latency_ms": float(np.percentile(durations, 50)),
        "p95_latency_ms": float(np.percentile(durations, 95)),
        "p99_latency_ms": float(np.percentile(durations, 99)),
        "max_latency_ms": float(np.max(durations)),
        "cell_metrics": cell_metrics,
        "libraries": {"numpy": np.__version__},
    }
    if model_key == "monte_carlo":
        metrics["paths"] = 8_000
        metrics["vr_pipeline"] = "antithetic_control"
    return metrics


@dataclass(slots=True)
class PrecisionCellResult:
    scenario: BenchmarkScenario
    bucket: str | None
    cell_price: float | None
    ci_abs: float | None
    ci_bps: float | None
    paths_used: int
    vr_pipeline: str | None
    price_samples: list[float]
    ci_samples: list[float]
    cv_summary: dict[str, float | bool | None]


def _run_precision_cell(
    scenario: BenchmarkScenario,
    *,
    initial_paths: int,
    warmup_iterations: int,
    measurement_iterations: int,
    seed_prefix: int,
) -> PrecisionCellResult:
    total_iterations = warmup_iterations + measurement_iterations
    bs_model = BlackScholesModel()
    bs_price = bs_model.calculate_price(
        scenario.contract, scenario.market, scenario.volatility
    ).theoretical_price
    bucket_hint = _assign_bucket(bs_price)
    bucket_config = PRECISION_BUCKETS.get(bucket_hint or "A", PRECISION_BUCKETS["A"])
    cap = bucket_config["cap"]
    threshold = bucket_config["threshold"]
    if bucket_hint in ("A", "B"):
        target_half_width = (threshold / 10_000.0) * max(bs_price, EPS_PRICE)
    else:
        target_half_width = threshold

    root_sequence = SeedSequence(seed_prefix)
    sequences = root_sequence.spawn(total_iterations)
    report_prices: list[float] = []
    report_half_widths: list[float] = []
    cv_reports: list[dict[str, float | bool | None]] = []
    used_paths: list[int] = []
    strategies: list[str] = []

    for iteration, sequence in enumerate(sequences):
        toolkit = VarianceReductionToolkit(
            baseline_model=MonteCarloModel(paths=initial_paths, antithetic=True),
            seed_sequence=sequence,
        )
        if "sobol_stratified_control" in toolkit.strategies:
            toolkit.strategies = {
                "sobol_stratified_control": toolkit.strategies["sobol_stratified_control"]
            }
        report = toolkit.price_with_diagnostics(
            scenario.contract,
            scenario.market,
            scenario.volatility,
            target_ci_half_width=target_half_width,
            initial_paths=initial_paths,
            max_paths=cap,
        )
        if iteration < warmup_iterations:
            continue
        result = report.pricing_result
        report_prices.append(result.theoretical_price)
        half_width = _ci_half_width(result)
        if half_width is not None:
            report_half_widths.append(half_width)
        cv_report = result.control_variate_report
        if isinstance(cv_report, dict):
            cv_reports.append(cv_report)
        used_paths.append(int(report.diagnostics.used_paths))
        strategies.append(report.diagnostics.strategy)

    cell_price = float(median(report_prices)) if report_prices else None
    median_half_width = float(median(report_half_widths)) if report_half_widths else None
    bucket = _assign_bucket(cell_price)
    ci_bps = None
    ci_abs = median_half_width
    if (
        bucket in ("A", "B")
        and median_half_width is not None
        and cell_price is not None
        and cell_price >= EPS_PRICE
    ):
        ci_bps = 10_000.0 * median_half_width / cell_price
    cv_summary = _summarise_cv_reports(cv_reports)
    vr_label = strategies[-1] if strategies else "sobol_stratified_control"
    paths_reported = int(max(used_paths)) if used_paths else initial_paths

    return PrecisionCellResult(
        scenario=scenario,
        bucket=bucket,
        cell_price=cell_price,
        ci_abs=ci_abs,
        ci_bps=ci_bps,
        paths_used=paths_reported,
        vr_pipeline=vr_label,
        price_samples=report_prices,
        ci_samples=report_half_widths,
        cv_summary=cv_summary,
    )


def _precision_benchmark(
    scenarios: Iterable[BenchmarkScenario],
    *,
    initial_paths: int = 16_384,
    warmup_iterations: int = 5,
    measurement_iterations: int = 20,
    seed_prefix: int = 2024,
) -> dict[str, object]:
    scenarios = list(scenarios)
    cell_results: list[PrecisionCellResult] = []
    price_samples_by_cell: dict[str, list[float]] = {}
    bucket_values: dict[str, list[float]] = {"A": [], "B": [], "C": []}
    bucket_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}
    all_bps: list[float] = []

    for index, scenario in enumerate(scenarios):
        cell = _run_precision_cell(
            scenario,
            initial_paths=initial_paths,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
            seed_prefix=seed_prefix + index * 10_000,
        )
        cell_results.append(cell)
        price_samples_by_cell[scenario.contract.symbol] = list(cell.price_samples)
        if cell.bucket is not None:
            bucket_counts[cell.bucket] += 1
        if cell.bucket in ("A", "B") and cell.ci_bps is not None:
            bucket_values[cell.bucket].append(cell.ci_bps)
            all_bps.append(cell.ci_bps)
        elif cell.bucket == "C" and cell.ci_abs is not None:
            bucket_values["C"].append(cell.ci_abs)
            if cell.cell_price and cell.cell_price >= EPS_PRICE:
                all_bps.append(10_000.0 * cell.ci_abs / cell.cell_price)

    total_bucket_count = sum(bucket_counts.values())
    assert total_bucket_count > 0, "No precision buckets populated"
    for bucket_name in ("A", "B", "C"):
        assert (
            bucket_values[bucket_name]
        ), f"No precision data collected for bucket {bucket_name}"

    ci_bps_values = [value for value in all_bps if value is not None]
    all_ci_abs = [cell.ci_abs for cell in cell_results if cell.ci_abs is not None]
    paths_used_values = [cell.paths_used for cell in cell_results]
    combined_bucket_bps = bucket_values["A"] + bucket_values["B"]

    assert ci_bps_values, "No CI (bps) values captured"
    assert all_ci_abs, "No CI half-width values captured"
    assert combined_bucket_bps, "No CI (bps) values captured for buckets A/B"

    monte_carlo_metrics: dict[str, object] = {
        "median_ci_bps_bucket_A": float(median(bucket_values["A"])),
        "median_ci_bps_bucket_B": float(median(bucket_values["B"])),
        "median_ci_abs_bucket_C": float(median(bucket_values["C"])),
        "bucket_counts": dict(bucket_counts),
        "median_ci_bps_all_cells": float(median(ci_bps_values)),
        "median_ci_half_width": float(median(all_ci_abs)),
        "median_ci_bps": float(median(combined_bucket_bps)),
        "paths_used": int(median(paths_used_values)) if paths_used_values else None,
        "cell_metrics": [],
    }

    for cell in cell_results:
        cell_entry: dict[str, object] = {
            "scenario": cell.scenario.contract.symbol,
            "bucket": cell.bucket,
            "cell_price": cell.cell_price,
            "ci_abs": cell.ci_abs,
            "ci_bps": cell.ci_bps,
            "paths_used": cell.paths_used,
            "vr_pipeline": cell.vr_pipeline,
        }
        cell_entry.update(cell.cv_summary)
        monte_carlo_metrics["cell_metrics"].append(cell_entry)

    worst_cells: list[dict[str, object]] = []
    for bucket in ("A", "B", "C"):
        candidates = [
            (cell.ci_bps if bucket in ("A", "B") else cell.ci_abs, cell)
            for cell in cell_results
            if cell.bucket == bucket
        ]
        candidates = [item for item in candidates if item[0] is not None]
        candidates.sort(key=lambda item: item[0], reverse=True)
        for _metric, cell in candidates[:5]:
            entry: dict[str, object] = {
                "scenario": cell.scenario.contract.symbol,
                "bucket": cell.bucket,
                "cell_price": cell.cell_price,
                "ci_abs": cell.ci_abs,
                "ci_bps": cell.ci_bps,
                "paths_used": cell.paths_used,
                "vr_pipeline": cell.vr_pipeline,
            }
            entry.update(cell.cv_summary)
            worst_cells.append(entry)
    monte_carlo_metrics["worst_cells"] = worst_cells

    subset_count = min(5, len(scenarios))
    baseline_model = MonteCarloModel(paths=initial_paths, antithetic=False)
    baseline_prices_all: list[float] = []
    vr_prices_all: list[float] = []

    for scenario_index, scenario in enumerate(scenarios[:subset_count]):
        total_iterations = warmup_iterations + measurement_iterations
        sequences = _spawn_sequences(seed_prefix + 99, scenario_index, total_iterations)
        prices: list[float] = []
        for iteration, sequence in enumerate(sequences):
            kwargs = {}
            if sequence is not None:
                kwargs["seed_sequence"] = sequence
            result = baseline_model.calculate_price(
                scenario.contract,
                scenario.market,
                scenario.volatility,
                **kwargs,
            )
            if iteration < warmup_iterations:
                continue
            prices.append(result.theoretical_price)
        baseline_prices_all.extend(prices)
        vr_prices_all.extend(price_samples_by_cell.get(scenario.contract.symbol, [])[: len(prices)])

    if baseline_prices_all and vr_prices_all:
        t_stat = stats.ttest_ind(vr_prices_all, baseline_prices_all, equal_var=False)
        monte_carlo_metrics["bias_test"] = {
            "p_value": float(t_stat.pvalue),
            "vr_mean": float(np.mean(vr_prices_all)),
            "baseline_mean": float(np.mean(baseline_prices_all)),
            "sample_size": len(vr_prices_all),
        }

    return {"monte_carlo": monte_carlo_metrics}


def _print_latency_summary(model_key: str, metrics: dict[str, object], baseline: dict[str, float]) -> None:
    parts = [
        f"model={model_key}",
        f"latency_mode={metrics.get('latency_mode', False)}",
        f"p50_ms={metrics['p50_latency_ms']:.2f}",
        f"p95_ms={metrics['p95_latency_ms']:.2f}",
        f"p99_ms={metrics['p99_latency_ms']:.2f}",
        f"target_ms={PERF_TARGETS[model_key]:.2f}",
        f"baseline_ms={baseline['p99_latency_ms']:.2f}",
    ]
    if model_key == "monte_carlo":
        parts.append("paths=8000")
        parts.append("vr_pipeline=antithetic_control")
    print("[bench] " + " ".join(parts))


def _print_precision_summary(metrics: dict[str, object]) -> None:
    monte_carlo = metrics["monte_carlo"]
    parts = ["model=monte_carlo", "latency_mode=False"]
    for key in (
        "median_ci_bps_bucket_A",
        "median_ci_bps_bucket_B",
        "median_ci_abs_bucket_C",
        "median_ci_bps_all_cells",
    ):
        value = monte_carlo.get(key)
        if value is not None:
            parts.append(f"{key}={value:.6f}")
    parts.append(f"bucket_counts={monte_carlo['bucket_counts']}")
    print("[bench] " + " ".join(parts))


def test_black_scholes_latency_gate(
    golden_grid: List[BenchmarkScenario], benchmark_baseline: dict[str, dict[str, float]]
) -> None:
    metrics = _latency_benchmark("black_scholes", BlackScholesModel(), golden_grid)
    _print_latency_summary("black_scholes", metrics, benchmark_baseline["black_scholes"])
    _record_latency_summary("black_scholes", metrics)
    assert metrics["p99_latency_ms"] <= PERF_TARGETS["black_scholes"]


def test_binomial_latency_gate(
    golden_grid: List[BenchmarkScenario], benchmark_baseline: dict[str, dict[str, float]]
) -> None:
    metrics = _latency_benchmark("binomial", BinomialModel(), golden_grid)
    _print_latency_summary("binomial", metrics, benchmark_baseline["binomial"])
    _record_latency_summary("binomial", metrics)
    assert metrics["p99_latency_ms"] <= PERF_TARGETS["binomial"]


def test_monte_carlo_latency_gate(
    golden_grid: List[BenchmarkScenario], benchmark_baseline: dict[str, dict[str, float]]
) -> None:
    # Latency mode: disable CV to remove small setup/accumulation overhead.
    # Precision is enforced separately in test_monte_carlo_precision_gate.
    model = MonteCarloModel(paths=8_000, antithetic=True, use_control_variates=False)
    metrics = _latency_benchmark(
        "monte_carlo",
        model,
        golden_grid,
        seed_prefix=2024,
    )
    _print_latency_summary("monte_carlo", metrics, benchmark_baseline["monte_carlo"])
    _record_latency_summary("monte_carlo", metrics)
    assert metrics["p99_latency_ms"] <= PERF_TARGETS["monte_carlo"]


def test_monte_carlo_precision_gate(golden_grid: List[BenchmarkScenario]) -> None:
    metrics = _precision_benchmark(golden_grid)
    _print_precision_summary(metrics)
    _record_precision_summary(metrics)
    _write_benchmarks_report()

    monte_carlo = metrics["monte_carlo"]
    for key in (
        "median_ci_bps_bucket_A",
        "median_ci_bps_bucket_B",
        "median_ci_abs_bucket_C",
    ):
        assert key in monte_carlo, f"Missing precision metric: {key}"
        value = monte_carlo[key]
        assert isinstance(value, (int, float)), f"Precision metric {key} is not numeric"
        assert math.isfinite(float(value)), f"Precision metric {key} is not finite"

    bucket_counts = monte_carlo.get("bucket_counts")
    assert isinstance(bucket_counts, dict), "bucket_counts missing"
    total_buckets = sum(bucket_counts.get(name, 0) for name in ("A", "B", "C"))
    assert total_buckets > 0

    assert monte_carlo["median_ci_bps_bucket_A"] <= 2.0
    assert monte_carlo["median_ci_bps_bucket_B"] <= 10.0
    assert monte_carlo["median_ci_abs_bucket_C"] <= 0.0005

    bias = monte_carlo.get("bias_test")
    assert isinstance(bias, dict) and bias.get("p_value", 0.0) > 0.05

    worst_cells = monte_carlo.get("worst_cells")
    assert isinstance(worst_cells, list)
    assert len(worst_cells) <= 15

    for cell in monte_carlo.get("cell_metrics", []):
        if cell.get("cv_used"):
            raw_var = cell.get("raw_var")
            residual_var = cell.get("residual_var")
            assert raw_var is None or residual_var is None or residual_var < raw_var
