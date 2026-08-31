"""Dated-contract resolution and scalar compatibility tests."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta, timezone
from typing import Any

import pytest

from options_engine.core.black_scholes import BlackScholesModel
from options_engine.core.models import ExerciseStyle, MarketData, OptionContract, OptionType
from options_engine.core.pricing_engine import OptionsEngine
from options_engine.market import (
    BusinessCalendar,
    BusinessDayConvention,
    ContinuousDividendCurve,
    ContinuousZeroCurve,
    DatedOptionContract,
    ExpiryDate,
    FlatDiscountCurve,
    FlatDividendCurve,
    ForwardBuilder,
    MarketConventions,
    MarketEnvironment,
    SettlementLag,
    ValuationDate,
    ZeroRateNode,
)

VALUATION = ValuationDate(date(2025, 1, 1))
EXPIRY = ExpiryDate(date(2026, 1, 1))


def _dated_contract(expiry: ExpiryDate = EXPIRY) -> DatedOptionContract:
    return DatedOptionContract(
        symbol="test",
        strike_price=100.0,
        expiry_date=expiry,
        option_type=OptionType.CALL,
    )


def test_flat_dated_environment_resolves_to_established_scalar_inputs() -> None:
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.04,
        dividend_yield=0.01,
    )
    resolved = environment.resolve(_dated_contract())

    assert resolved.contract.time_to_expiry == 1.0
    assert resolved.market_data.risk_free_rate == pytest.approx(0.04)
    assert resolved.market_data.dividend_yield == pytest.approx(0.01)
    assert resolved.forward.forward_price == pytest.approx(100.0 * math.exp(0.03))
    assert resolved.forward.discount_factor == pytest.approx(math.exp(-0.04))
    assert resolved.market_data.timestamp == datetime(2025, 1, 1, tzinfo=UTC)
    assert resolved.diagnostics()["rate_representation"] == "endpoint_equivalent_continuous"


def test_dated_and_scalar_black_scholes_paths_are_numerically_equivalent() -> None:
    scalar_contract = OptionContract("TEST", 100.0, 1.0, OptionType.CALL)
    scalar_market = MarketData(
        100.0,
        0.04,
        0.01,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
    )
    dated_market = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.04,
        dividend_yield=0.01,
    )
    scalar_result = BlackScholesModel().calculate_price(scalar_contract, scalar_market, 0.25)
    resolved = dated_market.resolve(_dated_contract())
    dated_result = BlackScholesModel().calculate_price(
        resolved.contract, resolved.market_data, 0.25
    )

    assert dated_result.theoretical_price == pytest.approx(
        scalar_result.theoretical_price, abs=1e-13
    )
    assert dated_result.delta == pytest.approx(scalar_result.delta, abs=1e-13)


def test_options_engine_resolves_at_orchestration_boundary_only() -> None:
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.04,
        dividend_yield=0.01,
    )
    scalar_contract = OptionContract("TEST", 100.0, 1.0, OptionType.CALL)
    scalar_market = MarketData(100.0, 0.04, 0.01)

    with OptionsEngine(num_threads=1) as engine:
        scalar = engine.price_option(
            scalar_contract,
            scalar_market,
            override_volatility=0.25,
        )
        dated = engine.price_dated_option(
            _dated_contract(),
            environment,
            override_volatility=0.25,
        )

    assert dated["theoretical_price"] == pytest.approx(scalar["theoretical_price"], abs=1e-13)
    assert "market_conventions" not in scalar
    diagnostics = dated["market_conventions"]
    assert isinstance(diagnostics, dict)
    assert diagnostics["discount_curve_id"] == environment.discount_curve.curve_id
    assert diagnostics["expiry_date"] == "2026-01-01"


def test_non_flat_curves_preserve_forward_and_discount_factor_endpoints() -> None:
    funding = ContinuousZeroCurve(
        VALUATION,
        (
            ZeroRateNode(date(2025, 7, 1), 0.02),
            ZeroRateNode(EXPIRY.value, 0.05),
        ),
    )
    dividends = ContinuousDividendCurve(
        VALUATION,
        (
            ZeroRateNode(date(2025, 7, 1), 0.005),
            ZeroRateNode(EXPIRY.value, 0.015),
        ),
    )
    environment = MarketEnvironment(
        spot_price=120.0,
        conventions=MarketConventions(VALUATION),
        discount_curve=funding,
        carry_curve=dividends,
    )
    resolved = environment.resolve(DatedOptionContract("CURVE", 125.0, EXPIRY, OptionType.CALL))
    tau = resolved.contract.time_to_expiry
    rate = resolved.market_data.risk_free_rate
    dividend = resolved.market_data.dividend_yield

    assert math.exp(-rate * tau) == pytest.approx(resolved.forward.discount_factor)
    assert 120.0 * math.exp((rate - dividend) * tau) == pytest.approx(
        resolved.forward.forward_price
    )
    assert resolved.forward.forward_price == pytest.approx(120.0 * math.exp(0.05 - 0.015))


def test_settlement_lag_and_expiry_adjustment_change_forward_explicitly() -> None:
    valuation = ValuationDate(date(2025, 1, 3))  # Friday
    calendar = BusinessCalendar(
        name="desk",
        holidays=frozenset({date(2025, 1, 6)}),
    )
    conventions = MarketConventions(
        valuation_date=valuation,
        calendar=calendar,
        settlement_lag=SettlementLag(2),
        expiry_convention=BusinessDayConvention.MODIFIED_FOLLOWING,
    )
    environment = MarketEnvironment(
        spot_price=100.0,
        conventions=conventions,
        discount_curve=FlatDiscountCurve(valuation, 0.05),
        carry_curve=FlatDividendCurve(valuation, 0.02),
    )
    resolved = environment.resolve(_dated_contract(ExpiryDate(date(2025, 2, 1))))

    assert resolved.forward.settlement_date == date(2025, 1, 8)
    assert resolved.forward.expiry_date == date(2025, 2, 3)
    assert resolved.contract.time_to_expiry == pytest.approx(31.0 / 365.0)
    assert resolved.forward.forward_price == pytest.approx(100.0 * math.exp(0.03 * 26.0 / 365.0))


def test_negative_rate_curve_resolves_without_special_cases() -> None:
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=-0.02,
        dividend_yield=0.01,
    )
    resolved = environment.resolve(_dated_contract())

    assert resolved.market_data.risk_free_rate == pytest.approx(-0.02)
    assert resolved.forward.discount_factor > 1.0
    assert resolved.forward.forward_price < 100.0


def test_dated_contract_identity_uses_expiry_and_preserves_explicit_ids() -> None:
    first = _dated_contract(ExpiryDate(date(2026, 1, 1)))
    same = _dated_contract(ExpiryDate(date(2026, 1, 1)))
    different = _dated_contract(ExpiryDate(date(2026, 1, 2)))
    explicit = DatedOptionContract(
        "test",
        100.0,
        EXPIRY,
        OptionType.PUT,
        ExerciseStyle.AMERICAN,
        contract_id="external-contract",
    )

    assert first.contract_id == same.contract_id
    assert first.contract_id != different.contract_id
    assert explicit.contract_id == "external-contract"
    assert first.symbol == "TEST"


def test_environment_and_convention_ids_are_deterministic() -> None:
    first = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.03,
    )
    second = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.03,
    )

    assert first.market_id == second.market_id
    assert first.conventions.conventions_id == second.conventions.conventions_id


def test_environment_rejects_curve_reference_mismatch_and_bad_timestamps() -> None:
    other_reference = ValuationDate(date(2025, 1, 2))
    conventions = MarketConventions(VALUATION)
    with pytest.raises(ValueError, match="discount_curve reference"):
        MarketEnvironment(
            100.0,
            conventions,
            FlatDiscountCurve(other_reference, 0.02),
            FlatDividendCurve(VALUATION, 0.01),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketEnvironment(
            100.0,
            conventions,
            FlatDiscountCurve(VALUATION, 0.02),
            FlatDividendCurve(VALUATION, 0.01),
            timestamp=datetime(2025, 1, 1),
        )


def test_environment_rejects_expiry_on_or_before_settlement() -> None:
    conventions = MarketConventions(
        valuation_date=VALUATION,
        settlement_lag=SettlementLag(2),
    )
    environment = MarketEnvironment(
        100.0,
        conventions,
        FlatDiscountCurve(VALUATION, 0.02),
        FlatDividendCurve(VALUATION, 0.01),
    )

    with pytest.raises(ValueError, match="after the valuation"):
        environment.resolve(_dated_contract(ExpiryDate(VALUATION.value)))
    with pytest.raises(ValueError, match="after the settlement"):
        environment.resolve(_dated_contract(ExpiryDate(date(2025, 1, 2))))


@pytest.mark.parametrize(
    ("factory", "error_type", "message"),
    [
        (
            lambda: DatedOptionContract("X", 100.0, date(2026, 1, 1), OptionType.CALL),
            TypeError,
            "ExpiryDate",
        ),
        (
            lambda: DatedOptionContract("X", 100.0, EXPIRY, "call"),
            TypeError,
            "OptionType",
        ),
        (
            lambda: MarketConventions(date(2025, 1, 1)),
            TypeError,
            "ValuationDate",
        ),
    ],
)
def test_dated_domain_type_validation(
    factory: Any, error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        factory()


def test_engine_dated_entrypoint_checks_domain_types() -> None:
    with (
        OptionsEngine(num_threads=1) as engine,
        pytest.raises(TypeError, match="DatedOptionContract"),
    ):
        engine.price_dated_option("contract", "market")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"day_count": "actual_365_fixed"}, "DayCountConvention"),
        ({"calendar": "calendar"}, "BusinessCalendar"),
        ({"settlement_lag": 2}, "SettlementLag"),
        ({"settlement_convention": "following"}, "BusinessDayConvention"),
        ({"expiry_convention": "unadjusted"}, "BusinessDayConvention"),
    ],
)
def test_market_conventions_require_typed_components(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        MarketConventions(VALUATION, **kwargs)  # type: ignore[arg-type]


def test_market_conventions_expose_canonical_payload_and_typed_expiry() -> None:
    conventions = MarketConventions(VALUATION)

    assert conventions.year_fraction(EXPIRY) == 1.0
    assert conventions.to_dict()["day_count"] == "actual_365_fixed"
    with pytest.raises(TypeError, match="ExpiryDate"):
        conventions.adjusted_expiry(EXPIRY.value)  # type: ignore[arg-type]


def test_environment_canonicalizes_equivalent_timestamp_instants_to_utc() -> None:
    eastern = timezone(timedelta(hours=2))
    first = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.03,
        timestamp=datetime(2025, 1, 1, 2, 0, tzinfo=eastern),
    )
    second = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.03,
        timestamp=datetime(2025, 1, 1, 0, 0, tzinfo=UTC),
    )

    assert first.timestamp == datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    assert first.market_id == second.market_id


def test_environment_rejects_non_datetime_timestamp_and_wrong_contract() -> None:
    with pytest.raises(TypeError, match="datetime or None"):
        MarketEnvironment.from_scalar_rates(
            spot_price=100.0,
            valuation_date=VALUATION,
            risk_free_rate=0.03,
            timestamp="2025-01-01",  # type: ignore[arg-type]
        )
    environment = MarketEnvironment.from_scalar_rates(
        spot_price=100.0,
        valuation_date=VALUATION,
        risk_free_rate=0.03,
    )
    with pytest.raises(TypeError, match="DatedOptionContract"):
        environment.resolve("contract")  # type: ignore[arg-type]


def test_forward_builder_validates_curve_roles_conventions_and_reference_dates() -> None:
    conventions = MarketConventions(VALUATION)
    other_reference = ValuationDate(date(2025, 1, 2))
    funding = FlatDiscountCurve(VALUATION, 0.02)
    carry = FlatDividendCurve(VALUATION, 0.01)

    with pytest.raises(TypeError, match="DiscountCurve"):
        ForwardBuilder(100.0, object(), carry, conventions)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CarryCurve"):
        ForwardBuilder(100.0, funding, object(), conventions)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MarketConventions"):
        ForwardBuilder(100.0, funding, carry, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="carry_curve reference"):
        ForwardBuilder(
            100.0,
            funding,
            FlatDividendCurve(other_reference, 0.01),
            conventions,
        )
    with pytest.raises(TypeError, match="ExpiryDate"):
        ForwardBuilder(100.0, funding, carry, conventions).build(EXPIRY.value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("args", "error_type", "message"),
    [
        ((None, 100.0, EXPIRY, OptionType.CALL), TypeError, "symbol"),
        ((" ", 100.0, EXPIRY, OptionType.CALL), ValueError, "symbol"),
        (("X", 0.0, EXPIRY, OptionType.CALL), ValueError, "strike"),
        (("X", 100.0, EXPIRY, OptionType.CALL, "european"), TypeError, "ExerciseStyle"),
        (
            ("X", 100.0, EXPIRY, OptionType.CALL, ExerciseStyle.EUROPEAN, 1),
            TypeError,
            "contract_id",
        ),
        (
            ("X", 100.0, EXPIRY, OptionType.CALL, ExerciseStyle.EUROPEAN, "\n"),
            ValueError,
            "contract_id",
        ),
    ],
)
def test_dated_contract_validation(
    args: tuple[Any, ...], error_type: type[Exception], message: str
) -> None:
    with pytest.raises(error_type, match=message):
        DatedOptionContract(*args)


def test_engine_dated_entrypoint_rejects_wrong_market_after_valid_contract() -> None:
    with (
        OptionsEngine(num_threads=1) as engine,
        pytest.raises(TypeError, match="MarketEnvironment"),
    ):
        engine.price_dated_option(_dated_contract(), "market")  # type: ignore[arg-type]
