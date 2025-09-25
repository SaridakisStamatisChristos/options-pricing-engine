"""Micro-benchmarks that enforce minimal performance gates for pricing models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable, List

import numpy as np
import pytest
import scipy
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

EPS_PRICE = 1e-6
CI_Z_SCORE = 1.96
PRECISION_BUCKETS = (
    ("A", 0.50, float("inf"), 200.0),
    ("B", 0.10, 0.50, 1_000.0),
    ("C", 0.0, 0.10, 0.0020),
)


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


def _assign_bucket(price: float | None) -> str | None:
    if price is None:
        return None
    for bucket, lower, upper, _ in PRECISION_BUCKETS:
        if lower <= price < upper:
            return bucket
    return "A" if price >= 0.50 else "C"


def _collect_baseline_prices(
    model: MonteCarloModel,
    scenario: BenchmarkScenario,
    *,
    scenario_index: int,
    warmup_iterations: int,
    measurement_iterations: int,
    seed_prefix: int | None,
) -> list[float]:
    total_iterations = warmup_iterations + measurement_iterations
    prices: list[float] = []
    for iteration in range(total_iterations):
        kwargs = {}
        if seed_prefix is not None:
            seed_value = seed_prefix + scenario_index * total_iterations + iteration
            kwargs["seed_sequence"] = SeedSequence(seed_value)
        result = model.calculate_price(
            scenario.contract,
            scenario.market,
            scenario.volatility,
            **kwargs,
        )
        if iteration < warmup_iterations:
            continue
        prices.append(result.theoretical_price)
    return prices


@dataclass(slots=True)
class VarianceReducedMonteCarloModel:
    """Wrapper executing a variance-reduction strategy via the toolkit."""

    paths: int
    strategy: str = "sobol_stratified_control"
    antithetic: bool = True
    vr_pipeline_label: str = field(init=False)
    last_report: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.vr_pipeline_label = self.strategy

    def calculate_price(
        self,
        contract: OptionContract,
        market_data: MarketData,
        volatility: float,
        *,
        seed_sequence: SeedSequence | None = None,
    ) -> PricingResult:
        toolkit_seed = seed_sequence or SeedSequence()
        baseline_model = MonteCarloModel(
            paths=self.paths,
            antithetic=self.antithetic,
            seed_sequence=toolkit_seed,
        )
        toolkit = VarianceReductionToolkit(
            baseline_model=baseline_model,
            seed_sequence=toolkit_seed,
        )
        report = toolkit.run_strategy(
            self.strategy,
            self.paths,
            contract,
            market_data,
            volatility,
        )
        self.last_report = report
        return report.pricing_result


def _run_latency_benchmark(
    model_name: str,
    model,
    scenarios: Iterable[BenchmarkScenario],
    *,
    warmup_iterations: int = 10,
    measurement_iterations: int = 100,
    seed_prefix: int | None = None,
) -> dict[str, float | list[float]]:
    scenarios = list(scenarios)

    durations: list[float] = []
    ci_half_widths: list[float] = []
    ci_abs_values: list[float] = []
    ci_bps_values: list[float] = []
    measurement_seeds: list[int] = []
    cell_metrics: list[dict[str, object]] = []
    price_samples_by_cell: dict[str, list[float]] = {}

    total_iterations = warmup_iterations + measurement_iterations
    bucket_bps_values: dict[str, list[float]] = {"A": [], "B": []}
    bucket_abs_values: dict[str, list[float]] = {"C": []}

    for scenario_index, scenario in enumerate(scenarios):
        cell_durations: list[float] = []
        cell_half_widths: list[float] = []
        cell_prices: list[float] = []
        paths_used: int | None = None
        for iteration in range(total_iterations):
            kwargs = {}
            seed_value: int | None = None
            if seed_prefix is not None:
                seed_value = seed_prefix + scenario_index * total_iterations + iteration
                kwargs["seed_sequence"] = SeedSequence(seed_value)
            result = model.calculate_price(
                scenario.contract,
                scenario.market,
                scenario.volatility,
                **kwargs,
            )

            if iteration < warmup_iterations:
                continue

            durations.append(result.computation_time_ms)
            cell_durations.append(result.computation_time_ms)
            if seed_value is not None:
                measurement_seeds.append(seed_value)

            price = result.theoretical_price
            cell_prices.append(price)

            half_width: float | None = None
            if result.standard_error is not None:
                half_width = CI_Z_SCORE * result.standard_error
            elif result.confidence_interval is not None:
                lower, upper = result.confidence_interval
                half_width = (upper - lower) / 2.0

            if half_width is not None:
                ci_half_widths.append(half_width)
                ci_abs_values.append(half_width)
                cell_half_widths.append(half_width)

            if paths_used is None and hasattr(result, "model_used") and result.model_used:
                extracted = _extract_paths_used(result.model_used)
                if extracted is not None:
                    paths_used = extracted

        if not cell_durations:
            continue

        cell_price = float(median(cell_prices)) if cell_prices else None
        cell_median_half_width = (
            float(median(cell_half_widths)) if cell_half_widths else None
        )

        cell_summary: dict[str, object] = {
            "scenario": scenario.contract.symbol,
            "p50_latency_ms": float(np.percentile(cell_durations, 50)),
            "p95_latency_ms": float(np.percentile(cell_durations, 95)),
            "p99_latency_ms": float(np.percentile(cell_durations, 99)),
        }

        price_samples_by_cell[scenario.contract.symbol] = list(cell_prices)

        if cell_price is not None:
            cell_summary["cell_price"] = cell_price
        if cell_median_half_width is not None:
            cell_summary["ci_abs"] = cell_median_half_width
            cell_summary["median_ci_half_width"] = cell_median_half_width

        bucket = _assign_bucket(cell_price)
        if bucket is not None:
            cell_summary["bucket"] = bucket

        ci_bps_value: float | None = None
        if (
            cell_median_half_width is not None
            and cell_price is not None
            and cell_price >= EPS_PRICE
            and bucket in ("A", "B")
        ):
            ci_bps_value = 10_000.0 * cell_median_half_width / cell_price
            ci_bps_values.append(ci_bps_value)
            bucket_bps_values[bucket].append(ci_bps_value)
        elif cell_median_half_width is not None:
            bucket_abs_values.setdefault("C", []).append(cell_median_half_width)

        if ci_bps_value is not None:
            cell_summary["ci_bps"] = float(ci_bps_value)
        if paths_used is not None:
            cell_summary["paths_used"] = paths_used
        pipeline_label = getattr(model, "vr_pipeline_label", None)
        if pipeline_label is None and hasattr(model, "antithetic"):
            pipeline_label = "antithetic" if getattr(model, "antithetic") else "plain"
        if pipeline_label:
            cell_summary["vr_pipeline"] = pipeline_label
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
    if ci_abs_values:
        metrics["median_ci_abs"] = float(median(ci_abs_values))
    if ci_bps_values:
        metrics["median_ci_bps"] = float(median(ci_bps_values))
    if measurement_seeds:
        measurement_seeds.sort()
        metrics["seed_lineage"] = {
            "first": measurement_seeds[0],
            "last": measurement_seeds[-1],
            "count": len(measurement_seeds),
        }
    pipeline_label = getattr(model, "vr_pipeline_label", None)
    if pipeline_label is None and hasattr(model, "antithetic"):
        pipeline_label = "antithetic" if getattr(model, "antithetic") else "plain"
    if pipeline_label:
        metrics["vr_pipeline"] = pipeline_label
    if hasattr(model, "paths"):
        metrics["paths_used"] = int(getattr(model, "paths"))

    precision_buckets: dict[str, dict[str, float]] = {}
    for bucket_name, _lower, _upper, threshold in PRECISION_BUCKETS:
        bucket_info: dict[str, float] = {}
        if bucket_name in ("A", "B"):
            values = bucket_bps_values.get(bucket_name, [])
            if values:
                bucket_info["median_ci_bps"] = float(median(values))
                bucket_info["count"] = len(values)
                bucket_info["threshold"] = threshold
        else:
            values = bucket_abs_values.get("C", [])
            if values:
                bucket_info["median_ci_abs"] = float(median(values))
                bucket_info["count"] = len(values)
                bucket_info["threshold"] = threshold
        if bucket_info:
            precision_buckets[bucket_name] = bucket_info
    if precision_buckets:
        metrics["precision_buckets"] = precision_buckets

    ranked_cells: list[tuple[float, dict[str, object]]] = []
    for cell in cell_metrics:
        bucket = cell.get("bucket")
        if bucket == "C" and cell.get("ci_abs") is not None:
            ranked_cells.append((float(cell["ci_abs"]), cell))
        elif bucket in ("A", "B") and cell.get("ci_bps") is not None:
            ranked_cells.append((float(cell["ci_bps"]), cell))
    ranked_cells.sort(key=lambda item: item[0], reverse=True)
    metrics["worst_cells"] = [item[1] for item in ranked_cells[:5]]

    if model_name == "monte_carlo" and seed_prefix is not None and price_samples_by_cell:
        subset_count = min(5, len(scenarios))
        baseline_model = MonteCarloModel(paths=getattr(model, "paths", 20_000), antithetic=False)
        baseline_prices_all: list[float] = []
        vr_prices_all: list[float] = []
        variance_entries: list[dict[str, float]] = []
        for scenario_index, scenario in enumerate(scenarios[:subset_count]):
            baseline_prices = _collect_baseline_prices(
                baseline_model,
                scenario,
                scenario_index=scenario_index,
                warmup_iterations=warmup_iterations,
                measurement_iterations=measurement_iterations,
                seed_prefix=seed_prefix,
            )
            vr_prices = price_samples_by_cell.get(scenario.contract.symbol, [])
            if not baseline_prices or not vr_prices:
                continue
            baseline_prices_all.extend(baseline_prices)
            vr_prices_all.extend(vr_prices[: len(baseline_prices)])
            baseline_var = float(np.var(baseline_prices, ddof=1)) if len(baseline_prices) > 1 else 0.0
            vr_var = float(np.var(vr_prices, ddof=1)) if len(vr_prices) > 1 else 0.0
            ratio = baseline_var / vr_var if vr_var > 0 else float("inf")
            variance_entries.append(
                {
                    "scenario": scenario.contract.symbol,
                    "baseline_var": baseline_var,
                    "vr_var": vr_var,
                    "cv_coeff": ratio,
                }
            )
        if vr_prices_all and baseline_prices_all:
            t_stat = stats.ttest_ind(vr_prices_all, baseline_prices_all, equal_var=False)
            metrics["bias_test"] = {
                "p_value": float(t_stat.pvalue),
                "vr_mean": float(np.mean(vr_prices_all)),
                "baseline_mean": float(np.mean(baseline_prices_all)),
                "sample_size": len(vr_prices_all),
            }
        if variance_entries:
            metrics["variance_reduction"] = {"scenarios": variance_entries}
            finite_coeffs = [
                entry["cv_coeff"] for entry in variance_entries if np.isfinite(entry["cv_coeff"])
            ]
            if finite_coeffs:
                metrics["variance_reduction"]["median_cv_coeff"] = float(median(finite_coeffs))

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
    precision_buckets = metrics.get("precision_buckets")
    if isinstance(precision_buckets, dict):
        for bucket_name, info in sorted(precision_buckets.items()):
            if "median_ci_bps" in info:
                summary_parts.append(
                    f"bucket_{bucket_name}_median_bps={info['median_ci_bps']:.2f}"
                )
            if "median_ci_abs" in info:
                summary_parts.append(
                    f"bucket_{bucket_name}_median_abs={info['median_ci_abs']:.6f}"
                )
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
    bias_test = metrics.get("bias_test")
    if isinstance(bias_test, dict) and "p_value" in bias_test:
        summary_parts.append(f"bias_p={bias_test['p_value']:.4f}")
    variance_reduction = metrics.get("variance_reduction")
    if isinstance(variance_reduction, dict) and "median_cv_coeff" in variance_reduction:
        summary_parts.append(
            f"cv_coeff={variance_reduction['median_cv_coeff']:.3f}"
        )
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
            bucket = cell.get("bucket")
            if bucket:
                cell_parts.append(f"bucket={bucket}")
            price = cell.get("cell_price")
            if price is not None:
                cell_parts.append(f"price={price:.4f}")
            ci_bps = cell.get("ci_bps")
            if ci_bps is not None:
                cell_parts.append(f"ci_bps={ci_bps:.1f}")
            ci_abs = cell.get("ci_abs")
            if ci_abs is not None:
                cell_parts.append(f"ci_abs={ci_abs:.6f}")
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
    model = VarianceReducedMonteCarloModel(paths=16_384, strategy="sobol_stratified_control")
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

    assert "seed_lineage" in metrics, "Monte Carlo benchmark must record seed lineage"
    assert metrics.get("paths_used") == baseline.get("paths_used"), (
        "Monte Carlo benchmark paths differ from baseline; cannot compare CI widths fairly"
    )

    precision = metrics.get("precision_buckets")
    assert isinstance(precision, dict), "Monte Carlo benchmark must report precision buckets"

    bucket_a = precision.get("A")
    if bucket_a and "median_ci_bps" in bucket_a:
        assert bucket_a["median_ci_bps"] <= bucket_a["threshold"], (
            "Bucket A precision gate failed: median CI bps exceeded threshold"
        )

    bucket_b = precision.get("B")
    if bucket_b and "median_ci_bps" in bucket_b:
        assert bucket_b["median_ci_bps"] <= bucket_b["threshold"], (
            "Bucket B precision gate failed: median CI bps exceeded threshold"
        )

    bucket_c = precision.get("C")
    if bucket_c and "median_ci_abs" in bucket_c:
        assert bucket_c["median_ci_abs"] <= bucket_c["threshold"], (
            "Bucket C precision gate failed: median absolute CI exceeded threshold"
        )

    bias_test = metrics.get("bias_test")
    assert isinstance(bias_test, dict) and "p_value" in bias_test, "Bias test results missing"
    assert bias_test["p_value"] > 0.05, "Variance-reduced Monte Carlo fails unbiasedness test"

    variance_reduction = metrics.get("variance_reduction")
    assert isinstance(variance_reduction, dict), "Variance reduction diagnostics missing"
    scenarios_vr = variance_reduction.get("scenarios", [])
    assert scenarios_vr, "Variance reduction scenarios missing"
    cell_buckets = {
        cell.get("scenario"): cell.get("bucket")
        for cell in metrics.get("cell_metrics", [])
        if isinstance(cell, dict)
    }
    for entry in scenarios_vr:
        scenario_bucket = cell_buckets.get(entry.get("scenario"))
        if scenario_bucket == "C":
            continue
        assert entry["vr_var"] <= entry["baseline_var"], (
            "Variance reduction failed to reduce variance for scenario"
        )
    median_cv = variance_reduction.get("median_cv_coeff")
    if median_cv is not None:
        assert median_cv >= 1.0, "Variance reduction coefficient should be >= 1"

    worst_cells = metrics.get("worst_cells")
    assert isinstance(worst_cells, list) and worst_cells, "Worst cells summary missing"
    assert len(worst_cells) <= 5, "Worst cells summary must contain at most five entries"
