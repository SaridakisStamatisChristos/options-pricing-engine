"""Shared deterministic pricing scenarios for numerical tests."""

from __future__ import annotations

from dataclasses import dataclass

from options_engine.core.models import (
    ExerciseStyle,
    MarketData,
    OptionContract,
    OptionType,
)


@dataclass(frozen=True)
class BenchmarkScenario:
    contract: OptionContract
    market: MarketData
    volatility: float


def golden_grid() -> list[BenchmarkScenario]:
    scenarios: list[BenchmarkScenario] = []
    market = MarketData(spot_price=100.0, risk_free_rate=0.02, dividend_yield=0.01)
    index = 0
    for maturity in (0.05, 0.5, 1.5):
        for volatility in (0.15, 0.3):
            for strike in (85.0, 100.0, 115.0):
                for option_type in (OptionType.CALL, OptionType.PUT):
                    index += 1
                    scenarios.append(
                        BenchmarkScenario(
                            contract=OptionContract(
                                symbol=f"GOLDEN{index}",
                                strike_price=strike,
                                time_to_expiry=maturity,
                                option_type=option_type,
                                exercise_style=ExerciseStyle.EUROPEAN,
                            ),
                            market=market,
                            volatility=volatility,
                        )
                    )
    return scenarios


__all__ = ["BenchmarkScenario", "golden_grid"]
