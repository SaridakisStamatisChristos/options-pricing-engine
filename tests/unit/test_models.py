"""Domain-model validation and identity tests."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pytest

from options_engine.core.models import ExerciseStyle, MarketData, OptionContract, OptionType


def test_market_data_defaults_to_aware_utc_timestamp() -> None:
    market = MarketData(100.0, 0.02, 0.01)

    assert market.timestamp is not None
    assert market.timestamp.tzinfo is UTC


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"spot_price": 0.0, "risk_free_rate": 0.0}, "spot_price"),
        ({"spot_price": math.nan, "risk_free_rate": 0.0}, "spot_price"),
        ({"spot_price": 1e12 + 1.0, "risk_free_rate": 0.0}, "spot_price"),
        ({"spot_price": 100.0, "risk_free_rate": 1.01}, "risk_free_rate"),
        ({"spot_price": 100.0, "risk_free_rate": math.inf}, "risk_free_rate"),
        (
            {"spot_price": 100.0, "risk_free_rate": 0.0, "dividend_yield": -1.01},
            "dividend_yield",
        ),
        (
            {"spot_price": 100.0, "risk_free_rate": 0.0, "dividend_yield": math.nan},
            "dividend_yield",
        ),
        (
            {
                "spot_price": 100.0,
                "risk_free_rate": 0.0,
                "timestamp": datetime(2026, 1, 1),
            },
            "timezone-aware",
        ),
    ],
)
def test_market_data_rejects_invalid_domains(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MarketData(**kwargs)


def test_market_data_rejects_non_datetime_timestamp() -> None:
    with pytest.raises(TypeError, match="datetime"):
        MarketData(100.0, 0.0, timestamp="2026-01-01")


def test_contract_normalizes_symbol_and_preserves_explicit_identifier() -> None:
    contract = OptionContract(
        "  aapl  ",
        100.0,
        1.0,
        OptionType.CALL,
        ExerciseStyle.AMERICAN,
        contract_id="external-id",
    )

    assert contract.symbol == "AAPL"
    assert contract.contract_id == "external-id"


@pytest.mark.parametrize(
    ("args", "error_type", "message"),
    [
        ((" ", 100.0, 1.0, OptionType.CALL), ValueError, "symbol"),
        ((None, 100.0, 1.0, OptionType.CALL), TypeError, "symbol"),
        (("X", 0.0, 1.0, OptionType.CALL), ValueError, "strike_price"),
        (("X", math.inf, 1.0, OptionType.CALL), ValueError, "strike_price"),
        (("X", 1e12 + 1.0, 1.0, OptionType.CALL), ValueError, "strike_price"),
        (("X", 100.0, 0.0, OptionType.CALL), ValueError, "time_to_expiry"),
        (("X", 100.0, math.nan, OptionType.CALL), ValueError, "time_to_expiry"),
        (("X", 100.0, 100.01, OptionType.CALL), ValueError, "time_to_expiry"),
        (("X", 100.0, 1.0, "call"), TypeError, "OptionType"),
        (("X", 100.0, 1.0, OptionType.CALL, "european"), TypeError, "ExerciseStyle"),
        (
            ("X", 100.0, 1.0, OptionType.CALL, ExerciseStyle.EUROPEAN, " "),
            ValueError,
            "contract_id",
        ),
        (
            ("X", 100.0, 1.0, OptionType.CALL, ExerciseStyle.EUROPEAN, 123),
            TypeError,
            "contract_id",
        ),
    ],
)
def test_contract_rejects_invalid_domains(
    args: tuple[Any, ...], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        OptionContract(*args)
