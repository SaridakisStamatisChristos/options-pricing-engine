"""Accuracy and stability tests for Monte Carlo greek estimators."""

from __future__ import annotations

import math
from statistics import median
from typing import Dict, List

import numpy as np
import pytest
from numpy.random import SeedSequence

import options_engine.core.pricing_models as pricing_models
from options_engine.core.models import ExerciseStyle, MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import BlackScholesModel, MonteCarloModel
from options_engine.tests.performance.test_pricing_benchmarks import BenchmarkScenario, _golden_grid


@pytest.fixture(scope="module")
def golden_grid() -> List[BenchmarkScenario]:
    return _golden_grid()


def _relative_error(value: float, reference: float) -> float:
    denominator = max(abs(reference), 1e-3)
    return abs(value - reference) / denominator


@pytest.mark.parametrize("paths", [120_000])
def test_mc_greeks_accuracy_and_ci(golden_grid: List[BenchmarkScenario], paths: int) -> None:
    model = MonteCarloModel(paths=paths, antithetic=True, seed_sequence=None, use_control_variates=True)
    bs_model = BlackScholesModel()

    seed_root = SeedSequence(2024)
    scenario_seeds = seed_root.spawn(len(golden_grid))

    error_buckets: Dict[str, List[float]] = {
        "delta": [],
        "gamma": [],
        "theta": [],
        "vega": [],
        "rho": [],
    }

    for scenario, seed in zip(golden_grid, scenario_seeds):
        contract = scenario.contract
        market = scenario.market
        volatility = scenario.volatility

        mc_result = model.calculate_price(contract, market, volatility, seed_sequence=seed)
        bs_result = bs_model.calculate_price(contract, market, volatility)

        ci = mc_result.ci_greeks
        assert ci is not None, "Monte Carlo result must expose greek confidence intervals"
        assert set(ci.keys()) == set(error_buckets.keys()), "Unexpected CI keys"
        for stats in ci.values():
            assert math.isfinite(stats["standard_error"])
            assert math.isfinite(stats["half_width_abs"])
            assert stats["standard_error"] >= 0.0
            assert stats["half_width_abs"] >= 0.0

        meta = mc_result.greeks_meta
        assert meta is not None
        for info in meta.values():
            assert isinstance(info["paths_used"], int)
            assert info["paths_used"] > 0
            assert info["vr_pipeline"] in {"cv", "plain"}
            assert "method" in info

        error_buckets["delta"].append(_relative_error(mc_result.delta or 0.0, bs_result.delta or 0.0))
        error_buckets["gamma"].append(_relative_error(mc_result.gamma or 0.0, bs_result.gamma or 0.0))
        error_buckets["theta"].append(_relative_error(mc_result.theta or 0.0, bs_result.theta or 0.0))
        error_buckets["vega"].append(_relative_error(mc_result.vega or 0.0, bs_result.vega or 0.0))
        error_buckets["rho"].append(_relative_error(mc_result.rho or 0.0, bs_result.rho or 0.0))

    assert median(error_buckets["delta"]) <= 0.015
    assert median(error_buckets["vega"]) <= 0.015
    assert median(error_buckets["rho"]) <= 0.015
    assert median(error_buckets["theta"]) <= 0.015
    assert median(error_buckets["gamma"]) <= 0.03


def test_fd_fallback_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = OptionContract(
        symbol="FALLBACK",
        strike_price=100.0,
        time_to_expiry=0.75,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
    )
    market = MarketData(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.0)
    volatility = 0.25

    model = MonteCarloModel(paths=512, antithetic=False, use_control_variates=False)
    seed = SeedSequence(77)

    def _gamma_nan(*args, **kwargs):  # type: ignore[unused-argument]
        terminal_prices = kwargs.get("terminal_prices")
        if terminal_prices is None and len(args) >= 4:
            terminal_prices = args[3]
        assert terminal_prices is not None
        return np.full_like(terminal_prices, np.nan, dtype=float)

    monkeypatch.setattr(pricing_models, "pathwise_gamma", _gamma_nan)

    result = model.calculate_price(contract, market, volatility, seed_sequence=seed)

    assert result.greeks_meta is not None
    gamma_meta = result.greeks_meta["gamma"]
    assert gamma_meta["method"] == "fd"
    assert gamma_meta["fallback"] == "fd"
    assert gamma_meta.get("primary_method") == "pathwise_lr"
    assert "fd_rel_error" in gamma_meta
    assert result.ci_greeks is not None
    gamma_ci = result.ci_greeks["gamma"]
    assert math.isfinite(gamma_ci["standard_error"])
    assert math.isfinite(gamma_ci["half_width_abs"])
