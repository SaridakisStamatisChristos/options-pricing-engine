"""Opt-in latency smoke tests for controlled benchmark runners.

Wall-clock gates are intentionally excluded from the default unit-test job. Set
``RUN_PERFORMANCE_TESTS=1`` on pinned, otherwise idle hardware to run them.
"""

from __future__ import annotations

import os
import time
from statistics import median

import pytest
from numpy.random import SeedSequence

from options_engine.core.models import MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import BinomialModel, BlackScholesModel, MonteCarloModel

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(
        os.getenv("RUN_PERFORMANCE_TESTS") != "1",
        reason="performance tests require a controlled runner",
    ),
]


def _duration_ms(operation) -> float:
    start = time.perf_counter()
    operation()
    return (time.perf_counter() - start) * 1_000.0


def test_model_latency_smoke() -> None:
    contract = OptionContract("PERF", 100.0, 1.0, OptionType.CALL)
    market = MarketData(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    models = {
        "black_scholes": (BlackScholesModel(), 2.0),
        "binomial": (BinomialModel(steps=200), 100.0),
        "monte_carlo": (MonteCarloModel(paths=20_000), 25.0),
    }

    for name, (model, budget_ms) in models.items():
        sequence = SeedSequence(2028)

        def price(
            current_model: BlackScholesModel | BinomialModel | MonteCarloModel = model,
            current_sequence: SeedSequence = sequence,
        ) -> None:
            if isinstance(current_model, MonteCarloModel):
                current_model.calculate_price(contract, market, 0.2, seed_sequence=current_sequence)
            else:
                current_model.calculate_price(contract, market, 0.2)

        price()
        timings = [_duration_ms(price) for _ in range(20)]
        measured = median(timings)
        assert measured < float(os.getenv(f"OPE_{name.upper()}_P50_MS", budget_ms))
