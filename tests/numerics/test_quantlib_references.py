from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from options_engine.calib.heston import heston_call_prices
from options_engine.core.finite_difference import FiniteDifferenceModel
from options_engine.core.models import ExerciseStyle, MarketData, OptionContract, OptionType
from options_engine.core.pricing_models import BlackScholesModel

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "reference"


def _load(name: str) -> dict[str, Any]:
    with (REFERENCE_DIR / name).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["schema_version"] == 1
    assert payload["source"]["library"] == "QuantLib"
    return payload


VANILLA_REFERENCE = _load("quantlib_vanilla_v1.json")
AMERICAN_CASES = [case for case in VANILLA_REFERENCE["cases"] if case["style"] == "american"]


@pytest.mark.parametrize(
    "case",
    [case for case in VANILLA_REFERENCE["cases"] if case["style"] == "european"],
    ids=lambda case: case["id"],
)
def test_black_scholes_against_committed_quantlib_reference(case: dict[str, Any]) -> None:
    contract = OptionContract(
        case["id"],
        case["strike"],
        case["time_to_expiry"],
        OptionType(case["type"]),
    )
    market = MarketData(case["spot"], case["r"], case["q"])

    result = BlackScholesModel().calculate_price(contract, market, case["vol"])

    assert result.theoretical_price == pytest.approx(case["reference_price"], abs=2e-11)


@pytest.mark.parametrize(
    "case",
    AMERICAN_CASES,
    ids=lambda case: case["id"],
)
@pytest.mark.parametrize("exercise_solver", ["psor", "penalty"])
def test_finite_difference_against_committed_quantlib_reference(
    case: dict[str, Any], exercise_solver: str
) -> None:
    contract = OptionContract(
        case["id"],
        case["strike"],
        case["time_to_expiry"],
        OptionType(case["type"]),
        ExerciseStyle.AMERICAN,
    )
    market = MarketData(case["spot"], case["r"], case["q"])

    result = FiniteDifferenceModel(
        space_steps=240,
        time_steps=320,
        exercise_solver=exercise_solver,
    ).calculate_price(contract, market, case["vol"])

    # QuantLib uses a separately implemented 2,000 x 800 finite-difference
    # engine. The 0.002 currency-unit tolerance includes both engines' grid
    # error and was fixed from the committed convergence study.
    assert result.theoretical_price == pytest.approx(case["reference_price"], abs=0.002)


def test_quantlib_american_fixture_covers_required_difficult_regimes() -> None:
    regimes = {regime for case in AMERICAN_CASES for regime in case.get("regimes", [])}

    assert regimes >= {
        "deep_itm",
        "deep_otm",
        "short_maturity",
        "high_volatility",
        "dividends",
        "negative_rates",
    }


@pytest.mark.parametrize(
    "case",
    _load("quantlib_heston_v1.json")["cases"],
    ids=lambda case: case["id"],
)
def test_heston_against_committed_quantlib_reference(case: dict[str, Any]) -> None:
    price = heston_call_prices(
        case["forward"],
        [case["strike"]],
        case["time_to_expiry"],
        v0=case["v0"],
        theta=case["theta"],
        kappa=case["kappa"],
        vol_of_vol=case["vol_of_vol"],
        rho=case["rho"],
    )[0]

    assert price == pytest.approx(case["reference_price"], abs=5e-7)
