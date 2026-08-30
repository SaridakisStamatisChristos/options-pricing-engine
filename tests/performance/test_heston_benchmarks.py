"""Opt-in, loose-budget Heston pricing latency smoke tests."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from statistics import median

import numpy as np
import pytest

from options_engine.calib.heston import heston_call_prices
from options_engine.calib.heston_cos import HestonCOSConfig, heston_cos_call_prices

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        os.getenv("RUN_PERFORMANCE_TESTS") != "1",
        reason="performance tests require a controlled runner",
    ),
]

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


def _median_ms(operation: Callable[[], object]) -> float:
    operation()
    observations: list[float] = []
    for _ in range(20):
        started = time.perf_counter()
        operation()
        observations.append((time.perf_counter() - started) * 1_000.0)
    return median(observations)


def test_heston_pricer_latency_smoke() -> None:
    strikes = np.geomspace(70.0, 140.0, 64)
    fixed = HestonCOSConfig(
        terms=256,
        truncation=12.0,
        adaptive=False,
        max_terms=256,
        max_truncation=12.0,
    )
    operations = {
        "heston_gauss_laguerre": (
            lambda: _quadrature_prices(strikes),
            10.0,
        ),
        "heston_cos_fixed": (
            lambda: _cos_prices(strikes, fixed),
            30.0,
        ),
        "heston_cos_adaptive": (
            lambda: _cos_prices(strikes),
            100.0,
        ),
    }

    for name, (operation, default_budget) in operations.items():
        measured = _median_ms(operation)
        budget = float(os.getenv(f"OPE_{name.upper()}_P50_MS", default_budget))
        assert measured < budget
