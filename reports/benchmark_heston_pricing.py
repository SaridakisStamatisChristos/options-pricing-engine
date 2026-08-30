"""Reproducible Heston pricing accuracy/latency benchmark.

Run from the repository root with::

    python reports/benchmark_heston_pricing.py

Wall-clock results are descriptive, not CI gates.  Controlled performance
smoke tests live under ``tests/performance`` and remain opt-in.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import scipy

from options_engine.calib.heston import heston_call_prices
from options_engine.calib.heston_cos import HestonCOSConfig, heston_cos_call_prices

PARAMS = {
    "v0": 0.04,
    "theta": 0.05,
    "kappa": 1.7,
    "vol_of_vol": 0.45,
    "rho": -0.6,
}


def _quadrature_prices(strikes: np.ndarray) -> np.ndarray:
    return heston_call_prices(
        100.0,
        strikes,
        1.0,
        v0=PARAMS["v0"],
        theta=PARAMS["theta"],
        kappa=PARAMS["kappa"],
        vol_of_vol=PARAMS["vol_of_vol"],
        rho=PARAMS["rho"],
    )


def _cos_prices(strikes: np.ndarray, config: HestonCOSConfig | None = None) -> np.ndarray:
    return heston_cos_call_prices(
        100.0,
        strikes,
        1.0,
        v0=PARAMS["v0"],
        theta=PARAMS["theta"],
        kappa=PARAMS["kappa"],
        vol_of_vol=PARAMS["vol_of_vol"],
        rho=PARAMS["rho"],
        config=config,
    )


def _median_microseconds(operation: Callable[[], object], repetitions: int = 50) -> float:
    operation()
    observations: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        observations.append((time.perf_counter_ns() - started) / 1_000.0)
    return statistics.median(observations)


def _quantlib_errors(config: HestonCOSConfig | None) -> tuple[float, float]:
    fixture_path = Path(__file__).resolve().parents[1] / "tests/reference/quantlib_heston_v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    cos_errors: list[float] = []
    quadrature_errors: list[float] = []
    for case in fixture["cases"]:
        arguments = {
            "v0": case["v0"],
            "theta": case["theta"],
            "kappa": case["kappa"],
            "vol_of_vol": case["vol_of_vol"],
            "rho": case["rho"],
        }
        cos_price = heston_cos_call_prices(
            case["forward"],
            [case["strike"]],
            case["time_to_expiry"],
            config=config,
            **arguments,
        )[0]
        quadrature_price = heston_call_prices(
            case["forward"],
            [case["strike"]],
            case["time_to_expiry"],
            **arguments,
        )[0]
        cos_errors.append(abs(float(cos_price) - case["reference_price"]))
        quadrature_errors.append(abs(float(quadrature_price) - case["reference_price"]))
    return max(cos_errors), max(quadrature_errors)


def main() -> None:
    fixed = HestonCOSConfig(
        terms=256,
        truncation=12.0,
        adaptive=False,
        max_terms=256,
        max_truncation=12.0,
    )
    methods: dict[str, Callable[[np.ndarray], object]] = {
        "Gauss-Laguerre-64": _quadrature_prices,
        "COS-fixed-256-L12": partial(_cos_prices, config=fixed),
        "COS-adaptive": _cos_prices,
    }

    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}; SciPy: {scipy.__version__}")
    print(f"Platform: {platform.platform()}")
    print("\n| Batch | Method | Median µs |")
    print("|---:|---|---:|")
    for batch in (1, 9, 64, 256):
        strikes = np.geomspace(70.0, 140.0, batch)
        for name, operation in methods.items():
            duration = _median_microseconds(partial(operation, strikes))
            print(f"| {batch} | {name} | {duration:.1f} |")

    adaptive_cos, quadrature = _quantlib_errors(None)
    fixed_cos, _ = _quantlib_errors(fixed)
    print("\n| Method | Max abs error vs QuantLib 1.43 fixture |")
    print("|---|---:|")
    print(f"| Gauss-Laguerre-64 | {quadrature:.3e} |")
    print(f"| COS-fixed-256-L12 | {fixed_cos:.3e} |")
    print(f"| COS-adaptive | {adaptive_cos:.3e} |")


if __name__ == "__main__":
    main()
