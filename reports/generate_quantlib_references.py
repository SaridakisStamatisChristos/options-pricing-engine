"""Regenerate committed QuantLib cross-engine reference fixtures.

QuantLib is intentionally not a project dependency. Run this script in an
isolated environment, review the fixture diff, and then run the normal test
suite::

    uv run --with 'QuantLib==1.43' reports/generate_quantlib_references.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import QuantLib as ql

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "tests" / "reference"
VALUATION_DATE = ql.Date(1, ql.January, 2026)
DAY_COUNT = ql.Actual365Fixed()
CALENDAR = ql.NullCalendar()


def _expiry(time_to_expiry: float) -> ql.Date:
    return VALUATION_DATE + round(float(time_to_expiry) * 365.0)


def _vanilla_price(case: dict[str, Any]) -> float:
    expiry = _expiry(case["time_to_expiry"])
    spot = ql.QuoteHandle(ql.SimpleQuote(case["spot"]))
    risk_free = ql.YieldTermStructureHandle(
        ql.FlatForward(VALUATION_DATE, case["r"], DAY_COUNT, ql.Continuous)
    )
    dividend = ql.YieldTermStructureHandle(
        ql.FlatForward(VALUATION_DATE, case["q"], DAY_COUNT, ql.Continuous)
    )
    volatility = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(VALUATION_DATE, CALENDAR, case["vol"], DAY_COUNT)
    )
    process = ql.BlackScholesMertonProcess(spot, dividend, risk_free, volatility)
    option_kind = ql.Option.Call if case["type"] == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(option_kind, case["strike"])
    if case["style"] == "american":
        exercise = ql.AmericanExercise(VALUATION_DATE, expiry)
        engine = ql.FdBlackScholesVanillaEngine(process, 2_000, 800, 2)
    else:
        exercise = ql.EuropeanExercise(expiry)
        engine = ql.AnalyticEuropeanEngine(process)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return float(option.NPV())


def _discrete_dividend_price(case: dict[str, Any]) -> float:
    """Price the documented limited-liability spot-jump cash-dividend model."""

    expiry = _expiry(case["time_to_expiry"])
    spot = ql.QuoteHandle(ql.SimpleQuote(case["spot"]))
    risk_free = ql.YieldTermStructureHandle(
        ql.FlatForward(VALUATION_DATE, case["r"], DAY_COUNT, ql.Continuous)
    )
    dividend = ql.YieldTermStructureHandle(
        ql.FlatForward(VALUATION_DATE, case["q"], DAY_COUNT, ql.Continuous)
    )
    volatility = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(VALUATION_DATE, CALENDAR, case["vol"], DAY_COUNT)
    )
    process = ql.BlackScholesMertonProcess(spot, dividend, risk_free, volatility)
    dividends = ql.DividendVector(
        [_expiry(item["ex_time"]) for item in case["dividends"]],
        [item["amount"] for item in case["dividends"]],
    )
    option_kind = ql.Option.Call if case["type"] == "call" else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(option_kind, case["strike"])
    exercise = (
        ql.AmericanExercise(VALUATION_DATE, expiry)
        if case["style"] == "american"
        else ql.EuropeanExercise(expiry)
    )
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(
        ql.FdBlackScholesVanillaEngine(
            process,
            dividends,
            2_400,
            1_200,
            4,
            ql.FdmSchemeDesc.CrankNicolson(),
            False,
            ql.nullDouble(),
            ql.FdBlackScholesVanillaEngine.Spot,
        )
    )
    return float(option.NPV())


def _heston_price(case: dict[str, Any]) -> float:
    expiry = _expiry(case["time_to_expiry"])
    zero_curve = ql.YieldTermStructureHandle(ql.FlatForward(VALUATION_DATE, 0.0, DAY_COUNT))
    process = ql.HestonProcess(
        zero_curve,
        zero_curve,
        ql.QuoteHandle(ql.SimpleQuote(case["forward"])),
        case["v0"],
        case["kappa"],
        case["theta"],
        case["vol_of_vol"],
        case["rho"],
    )
    engine = ql.AnalyticHestonEngine(ql.HestonModel(process), 192)
    option = ql.VanillaOption(
        ql.PlainVanillaPayoff(ql.Option.Call, case["strike"]),
        ql.EuropeanExercise(expiry),
    )
    option.setPricingEngine(engine)
    return float(option.NPV())


def _regenerate(filename: str, pricing_function) -> None:
    path = REFERENCE_DIR / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source"]["version"] = ql.__version__
    for case in payload["cases"]:
        case["reference_price"] = pricing_function(case)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ql.Settings.instance().evaluationDate = VALUATION_DATE
    _regenerate("quantlib_vanilla_v1.json", _vanilla_price)
    _regenerate("quantlib_discrete_dividends_v1.json", _discrete_dividend_price)
    _regenerate("quantlib_heston_v1.json", _heston_price)


if __name__ == "__main__":
    main()
