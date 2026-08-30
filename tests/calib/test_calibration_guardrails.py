"""Guardrail coverage for calibration ingestion, models, and validation."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from options_engine.calib import heston, sabr, select, validators
from options_engine.calib.boards import BoardCleaner, BoardCleanerConfig, CleanBoard
from options_engine.calib.heston import HestonCalibrator, HestonConfig
from options_engine.calib.sabr import SABRCalibrator, SABRConfig, SABRTenorResult
from options_engine.calib.select import SurfaceBuilder, TenorSelection
from options_engine.calib.validators import NoArbitrageValidator, ValidationReport, Violation


def _rows(*, count: int = 7, tenor: float = 1.0) -> list[dict[str, Any]]:
    strikes = np.linspace(80.0, 120.0, count)
    return [
        {
            "tenor": tenor,
            "strike": float(strike),
            "mid_iv": 0.2 + 0.0001 * abs(strike - 100.0),
            "forward": 100.0,
            "option_type": "CALL",
        }
        for strike in strikes
    ]


def _clean(rows: list[dict[str, Any]] | None = None) -> CleanBoard:
    return CleanBoard(pd.DataFrame(rows if rows is not None else _rows()), {"counts": {}})


def test_clean_board_records_are_canonical_and_json_safe() -> None:
    empty = CleanBoard(pd.DataFrame(), {})
    assert empty.to_records() == []

    frame = pd.DataFrame(
        [
            {
                "tenor": 1.0,
                "strike": 100.0,
                "option_type": "CALL",
                "timestamp": pd.Timestamp("2025-01-01T00:00:00Z"),
                "value": math.inf,
                "missing": pd.NA,
            },
            {
                "tenor": 0.5,
                "strike": 90.0,
                "option_type": "PUT",
                "timestamp": pd.NaT,
                "value": np.float64(2.5),
                "missing": "present",
            },
        ]
    )
    records = CleanBoard(frame, {}).to_records()
    assert records[0]["timestamp"] is None
    assert records[0]["value"] == 2.5
    assert records[1]["timestamp"].startswith("2025-01-01")
    assert records[1]["value"] is None
    assert records[1]["missing"] is None


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("max_age_seconds", True, TypeError),
        ("max_age_seconds", math.nan, TypeError),
        ("max_age_seconds", -1.0, ValueError),
        ("max_age_seconds", 604_801.0, ValueError),
        ("max_future_seconds", -1.0, ValueError),
        ("max_future_seconds", 3_601.0, ValueError),
        ("mad_threshold", 0.0, ValueError),
        ("mad_threshold", 101.0, ValueError),
        ("tau_min_days", 0.0, ValueError),
        ("tau_min_days", 36_501.0, ValueError),
        ("sigma_min", 0.0, ValueError),
        ("sigma_max", 6.0, ValueError),
        ("sigma_max", 1e-5, ValueError),
        ("log_money_bounds", "bad", TypeError),
        ("log_money_bounds", (0.0,), TypeError),
        ("log_money_bounds", (False, 1.0), TypeError),
        ("log_money_bounds", (1.0, 1.0), ValueError),
        ("log_money_bounds", (math.nan, 1.0), ValueError),
        ("log_money_bins", "bad", TypeError),
        ("log_money_bins", (0.0, False), TypeError),
        ("log_money_bins", (0.0,), ValueError),
        ("log_money_bins", (0.0, math.nan), ValueError),
        ("log_money_bins", (-4.0, 0.0, 0.0, 4.0), ValueError),
        ("log_money_bins", (-3.0, 0.0, 4.0), ValueError),
    ],
)
def test_board_cleaner_configuration_rejects_unsafe_values(
    name: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        BoardCleanerConfig(**{name: value})  # type: ignore[arg-type]


def test_board_cleaner_configuration_normalises_sequences() -> None:
    config = BoardCleanerConfig(
        log_money_bounds=[-2, 2],  # type: ignore[arg-type]
        log_money_bins=[-2, 0, 2],
    )
    assert config.log_money_bounds == (-2.0, 2.0)
    assert config.log_money_bins == (-2.0, 0.0, 2.0)


@pytest.mark.parametrize(
    ("quotes", "now", "seed", "exception"),
    [
        ("bad", None, 0, TypeError),
        (1, None, 0, TypeError),
        ([], "today", 0, TypeError),
        ([], datetime.now(), 0, ValueError),
        ([], None, True, TypeError),
        ([], None, -1, ValueError),
        ([], None, 2**128, ValueError),
        ([1], None, 0, TypeError),
    ],
)
def test_board_cleaner_validates_ingestion_boundaries(
    quotes: object,
    now: object,
    seed: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        BoardCleaner().ingest(quotes, now=now, seed=seed)  # type: ignore[arg-type]


def test_board_cleaner_enforces_row_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(BoardCleaner, "MAX_QUOTES", 1)
    with pytest.raises(ValueError, match="row limit"):
        BoardCleaner().ingest(_rows(count=2))


def test_board_cleaner_empty_missing_and_default_option_type() -> None:
    assert BoardCleaner().ingest([]).qc["counts"] == {
        "total": 0,
        "retained": 0,
        "dropped": 0,
    }
    with pytest.raises(KeyError, match="missing required"):
        BoardCleaner().ingest([{"tenor": 1.0}])
    cleaned = BoardCleaner().ingest(
        [{"tenor": 1.0, "strike": 100.0, "mid_iv": 0.2, "forward": 100.0}]
    )
    assert cleaned.quotes["option_type"].tolist() == ["CALL"]


def test_board_cleaner_reports_numeric_domain_and_discount_rejections() -> None:
    rows = [
        {"tenor": "bad", "strike": 100.0, "mid_iv": 0.2, "forward": 100.0},
        {"tenor": 0.0, "strike": 100.0, "mid_iv": 0.2, "forward": 100.0},
        {
            "tenor": 1.0,
            "strike": 100.0,
            "mid_iv": 0.2,
            "forward": 100.0,
            "discount": 0.0,
        },
    ]
    result = BoardCleaner().ingest(rows)
    assert result.quotes.empty
    assert result.qc["counts"]["dropped_invalid_numeric"] == 2
    assert result.qc["counts"]["dropped_out_of_domain"] == 1


def test_board_cleaner_requires_complete_and_valid_spreads() -> None:
    incomplete = {**_rows(count=1)[0], "bid_iv": 0.19}
    with pytest.raises(KeyError, match="supplied together"):
        BoardCleaner().ingest([incomplete])

    invalid = [
        {**_rows(count=1)[0], "bid_iv": 0.21, "ask_iv": 0.19},
        {**_rows(count=1)[0], "bid_iv": "bad", "ask_iv": 0.21},
    ]
    result = BoardCleaner().ingest(invalid)
    assert result.quotes.empty
    assert result.qc["counts"]["dropped_invalid_spread"] == 2


def test_board_cleaner_rejects_stale_future_and_invalid_timestamps() -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    base = _rows(count=1)[0]
    rows = [
        {**base, "timestamp": now - timedelta(hours=1)},
        {**base, "timestamp": now + timedelta(minutes=1)},
        {**base, "timestamp": "not-a-date"},
    ]
    result = BoardCleaner().ingest(rows, now=now)
    assert result.quotes.empty
    assert result.qc["counts"]["dropped_invalid_timestamp"] == 3


def test_board_cleaner_uses_current_time_when_timestamp_is_present() -> None:
    row = {**_rows(count=1)[0], "timestamp": datetime.now(UTC)}
    assert len(BoardCleaner().ingest([row]).quotes) == 1


def test_board_cleaner_rejects_option_types_and_moneyness_bounds() -> None:
    invalid_type = {**_rows(count=1)[0], "option_type": "EXOTIC"}
    result = BoardCleaner().ingest([invalid_type])
    assert result.qc["counts"]["dropped_invalid_option_type"] == 1

    out_of_bounds = {**_rows(count=1)[0], "strike": 10_000.0}
    result = BoardCleaner().ingest([out_of_bounds])
    assert result.qc["counts"]["dropped_out_of_bounds"] == 1


@pytest.mark.parametrize("vols", [[0.2, 0.2, 0.2, 0.9], [0.19, 0.2, 0.21, 0.9]])
def test_board_cleaner_flags_zero_and_nonzero_mad_outliers(vols: list[float]) -> None:
    rows = [
        {
            "tenor": 1.0,
            "strike": 100.0 + index * 0.01,
            "mid_iv": vol,
            "forward": 100.0,
            "option_type": "call" if index else None,
        }
        for index, vol in enumerate(vols)
    ]
    result = BoardCleaner().ingest(rows)
    assert result.qc["counts"]["dropped_outlier"] == 1
    assert result.qc["residuals"]


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("beta", True, TypeError),
        ("beta", math.nan, TypeError),
        ("beta", -0.1, ValueError),
        ("beta", 1.1, ValueError),
        ("fit_beta", 1, TypeError),
        ("fit_beta", True, ValueError),
        ("seeds", (), ValueError),
        ("seeds", tuple(range(33)), ValueError),
        ("seeds", (True,), ValueError),
        ("seeds", (-1,), ValueError),
        ("seeds", (1, 1), ValueError),
        ("tolerance", True, ValueError),
        ("tolerance", math.nan, ValueError),
        ("tolerance", 1.0, ValueError),
        ("max_iterations", True, ValueError),
        ("max_iterations", 0, ValueError),
    ],
)
def test_sabr_configuration_rejects_unsafe_values(
    name: str, value: object, exception: type[Exception]
) -> None:
    kwargs: dict[str, object] = {name: value}
    if name == "fit_beta" and value is True:
        kwargs["beta"] = 0.0
    with pytest.raises(exception):
        SABRConfig(**kwargs)  # type: ignore[arg-type]


def test_sabr_fit_beta_parameter_transforms_round_trip() -> None:
    calibrator = SABRCalibrator(SABRConfig(beta=0.4, fit_beta=True, seeds=(0,)))
    params = {"alpha": 0.2, "beta": 0.4, "rho": -0.2, "nu": 0.5}
    packed = calibrator._pack_params(params)
    unpacked = calibrator._unpack_params(packed)
    assert unpacked == pytest.approx(params)
    initial = calibrator._initial_guess(
        100.0,
        np.array([90.0, 100.0, 110.0]),
        np.array([0.25, 0.2, 0.24]),
        None,
    )
    assert initial.shape == (4,)
    assert np.array_equal(
        calibrator._initial_guess(100.0, np.array([100.0]), np.array([0.2]), initial),
        initial,
    )


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("expiry", True, TypeError),
        ("alpha", math.nan, TypeError),
        ("expiry", 0.0, ValueError),
        ("alpha", 0.0, ValueError),
        ("nu", -1.0, ValueError),
        ("beta", 2.0, ValueError),
        ("rho", 1.0, ValueError),
    ],
)
def test_hagan_scalar_parameter_guardrails(
    name: str, value: object, exception: type[Exception]
) -> None:
    kwargs: dict[str, object] = {
        "expiry": 1.0,
        "alpha": 0.2,
        "beta": 0.5,
        "rho": 0.0,
        "nu": 0.5,
    }
    kwargs[name] = value
    expiry = kwargs.pop("expiry")
    with pytest.raises(exception):
        sabr.hagan_implied_volatility(
            100.0,
            100.0,
            expiry,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("forward", "strike", "exception"),
    [
        ("100", 100.0, TypeError),
        (100.0, True, TypeError),
        (math.nan, 100.0, ValueError),
        (100.0, 0.0, ValueError),
        (np.ones(2), np.ones(3), ValueError),
    ],
)
def test_hagan_input_array_guardrails(
    forward: object, strike: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        sabr.hagan_implied_volatility(
            forward,  # type: ignore[arg-type]
            strike,  # type: ignore[arg-type]
            1.0,
            alpha=0.2,
            beta=0.5,
            rho=0.0,
            nu=0.5,
        )


def test_hagan_broadcast_limit_and_nonfinite_output() -> None:
    with pytest.raises(ValueError, match="100000"):
        sabr.hagan_implied_volatility(
            np.ones(100_001),
            1.0,
            1.0,
            alpha=0.2,
            beta=0.5,
            rho=0.0,
            nu=0.5,
        )
    with pytest.raises(ValueError, match="invalid SABR"):
        sabr.hagan_implied_volatility(
            1e12,
            1e-12,
            100.0,
            alpha=1e308,
            beta=0.5,
            rho=0.99,
            nu=1e308,
        )


def test_sabr_calibrator_validates_public_inputs() -> None:
    with pytest.raises(TypeError, match="SABRConfig"):
        SABRCalibrator(object())  # type: ignore[arg-type]
    calibrator = SABRCalibrator(SABRConfig(seeds=(0,)))
    with pytest.raises(TypeError, match="CleanBoard"):
        calibrator.calibrate(pd.DataFrame())  # type: ignore[arg-type]
    assert calibrator.calibrate(_clean([])) == []
    with pytest.raises(TypeError, match="forward_curve"):
        calibrator.calibrate(_clean(), forward_curve=[])  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="required"):
        calibrator.calibrate(CleanBoard(pd.DataFrame([{"tenor": 1.0}]), {}))
    for field, value in (("mid_iv", math.nan), ("strike", 0.0)):
        rows = _rows()
        rows[0][field] = value
        with pytest.raises(ValueError):
            calibrator.calibrate(_clean(rows))
    assert calibrator.calibrate(_clean(_rows(count=4))) == []


def test_sabr_forward_resolution_is_exact_and_unambiguous() -> None:
    calibrator = SABRCalibrator()
    group = pd.DataFrame(_rows(count=5))
    assert calibrator._resolve_forward(1.0, group, {1.0: 101.0}) == 101.0
    assert calibrator._resolve_forward(1.0, group, {1.0 + 1e-10: 102.0}) == 102.0
    with pytest.raises(KeyError):
        calibrator._resolve_forward(1.0, group, {2.0: 100.0})
    with pytest.raises(ValueError, match="ambiguous"):
        calibrator._resolve_forward(
            1.0,
            group,
            {1.0 - 1e-10: 100.0, 1.0 + 1e-10: 101.0},
        )
    inconsistent = group.copy()
    inconsistent.loc[0, "forward"] = 101.0
    with pytest.raises(ValueError, match="inconsistent"):
        calibrator._resolve_forward(1.0, inconsistent, None)
    with pytest.raises(ValueError, match="finite"):
        calibrator._resolve_forward(1.0, group, {1.0: math.nan})


def test_sabr_calibration_failure_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    calibrator = SABRCalibrator(SABRConfig(seeds=(0, 1)))
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vols = np.full(5, 0.2)
    monkeypatch.setattr(
        sabr, "least_squares", lambda *_args, **_kwargs: SimpleNamespace(success=False)
    )
    with pytest.raises(RuntimeError, match="failed"):
        calibrator._calibrate_single_tenor(100.0, 1.0, strikes, vols, warm_start=None)

    def failing_fit(objective: Any, theta: np.ndarray, **_kwargs: Any) -> SimpleNamespace:
        residuals = objective(theta)
        return SimpleNamespace(success=True, fun=residuals, x=theta)

    monkeypatch.setattr(
        sabr,
        "hagan_implied_volatility",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    monkeypatch.setattr(sabr, "least_squares", failing_fit)
    with pytest.raises(RuntimeError, match="failed"):
        calibrator._calibrate_single_tenor(100.0, 1.0, strikes, vols, warm_start=None)


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("seeds", (), ValueError),
        ("seeds", tuple(range(33)), ValueError),
        ("seeds", (True,), ValueError),
        ("seeds", (-1,), ValueError),
        ("seeds", (1, 1), ValueError),
        ("tolerance", True, TypeError),
        ("tolerance", math.nan, TypeError),
        ("tolerance", 1.0, ValueError),
        ("max_iterations", True, ValueError),
        ("max_iterations", 0, ValueError),
        ("min_strikes", True, ValueError),
        ("min_strikes", 6, ValueError),
        ("weighting", "price", ValueError),
        ("spread_floor", 0.0, ValueError),
        ("holdout_fraction", 0.75, ValueError),
        ("feller_penalty", -1.0, ValueError),
        ("tenors", (), ValueError),
        ("tenors", (True,), TypeError),
        ("tenors", (0.0,), ValueError),
        ("tenors", (1.0, 1.0), ValueError),
    ],
)
def test_heston_configuration_rejects_unsafe_values(
    name: str, value: object, exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        HestonConfig(**{name: value})  # type: ignore[arg-type]


def test_heston_configuration_normalises_tenors() -> None:
    config = HestonConfig(seeds=(2, 1), tenors=[2, 1], min_strikes=7)
    assert config.seeds == (2, 1)
    assert config.tenors == (1.0, 2.0)


def test_heston_weighting_and_holdout_are_explicit() -> None:
    group = pd.DataFrame(
        {
            "strike": np.linspace(70.0, 130.0, 10),
            "mid_iv": np.linspace(0.3, 0.2, 10),
            "bid_iv": np.linspace(0.29, 0.19, 10),
            "ask_iv": np.linspace(0.31, 0.21, 10),
        }
    )
    calibrator = HestonCalibrator(HestonConfig(weighting="auto", holdout_fraction=0.2))

    weights, method = calibrator._weights(100.0, 1.0, group)
    holdout = calibrator._holdout_mask(len(group))

    assert method == "hybrid"
    assert weights.mean() == pytest.approx(1.0)
    assert np.isfinite(weights).all() and np.all(weights > 0.0)
    assert np.count_nonzero(holdout) == 2
    assert np.count_nonzero(~holdout) >= calibrator._config.min_strikes

    no_spreads = group[["strike", "mid_iv"]]
    with pytest.raises(ValueError, match="requires bid_iv"):
        HestonCalibrator(HestonConfig(weighting="bid_ask"))._weights(
            100.0,
            1.0,
            no_spreads,
        )


def _heston_price(**overrides: object) -> np.ndarray:
    values: dict[str, object] = {
        "forward": 100.0,
        "strikes": np.array([90.0, 100.0, 110.0]),
        "tenor": 1.0,
        "v0": 0.04,
        "theta": 0.04,
        "kappa": 1.5,
        "vol_of_vol": 0.4,
        "rho": -0.5,
    }
    values.update(overrides)
    return heston.heston_call_prices(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "value", "exception"),
    [
        ("strikes", ["100"], TypeError),
        ("strikes", [True], TypeError),
        ("strikes", np.ones((2, 2)), ValueError),
        ("strikes", [], ValueError),
        ("forward", True, TypeError),
        ("rho", "bad", TypeError),
        ("theta", math.nan, ValueError),
        ("forward", 0.0, ValueError),
        ("tenor", 0.0, ValueError),
        ("strikes", [0.0], ValueError),
        ("v0", 0.0, ValueError),
        ("theta", 26.0, ValueError),
        ("kappa", 101.0, ValueError),
        ("vol_of_vol", 21.0, ValueError),
        ("rho", 1.0, ValueError),
    ],
)
def test_heston_price_guardrails(name: str, value: object, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        _heston_price(**{name: value})


def test_heston_price_enforces_array_and_numerical_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="20000"):
        _heston_price(strikes=np.ones(20_001))

    monkeypatch.setattr(
        heston,
        "_heston_characteristic_function",
        lambda u, **_kwargs: np.full(np.asarray(u).shape, np.nan + 0j),
    )
    with pytest.raises(ValueError, match="unstable"):
        _heston_price()


def test_black_and_implied_volatility_boundary_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert heston._black_call_price(100.0, 90.0, 1.0, 0.0) == 10.0
    assert heston._implied_volatility(100.0, 90.0, 1.0, 10.0) == 1e-6
    assert heston._implied_volatility(100.0, 90.0, 1.0, 100.0) == 5.0
    monkeypatch.setattr(
        heston, "brentq", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("no root"))
    )
    assert math.isnan(heston._implied_volatility(100.0, 100.0, 1.0, 10.0))


def test_heston_calibrator_validates_public_inputs() -> None:
    with pytest.raises(TypeError, match="HestonConfig"):
        HestonCalibrator(object())  # type: ignore[arg-type]
    calibrator = HestonCalibrator(HestonConfig(seeds=(0,)))
    with pytest.raises(TypeError, match="CleanBoard"):
        calibrator.calibrate(pd.DataFrame())  # type: ignore[arg-type]
    assert calibrator.calibrate(_clean([])) == []
    with pytest.raises(TypeError, match="forward_curve"):
        calibrator.calibrate(_clean(), forward_curve=[])  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="required"):
        calibrator.calibrate(CleanBoard(pd.DataFrame([{"tenor": 1.0}]), {}))
    for field, value in (("mid_iv", math.inf), ("forward", 0.0)):
        rows = _rows()
        rows[0][field] = value
        with pytest.raises(ValueError):
            calibrator.calibrate(_clean(rows))
    assert HestonCalibrator(HestonConfig(tenors=(2.0,))).calibrate(_clean()) == []
    assert calibrator.calibrate(_clean(_rows(count=6))) == []


def test_heston_forward_resolution_is_exact_and_unambiguous() -> None:
    group = pd.DataFrame(_rows())
    assert HestonCalibrator._resolve_forward(1.0, group, {1.0: 101.0}) == 101.0
    with pytest.raises(KeyError):
        HestonCalibrator._resolve_forward(1.0, group, {2.0: 100.0})
    with pytest.raises(ValueError, match="ambiguous"):
        HestonCalibrator._resolve_forward(1.0, group, {1.0: 100.0, 1.0 + 1e-10: 101.0})
    inconsistent = group.copy()
    inconsistent.loc[0, "forward"] = 101.0
    with pytest.raises(ValueError, match="inconsistent"):
        HestonCalibrator._resolve_forward(1.0, inconsistent, None)
    with pytest.raises(ValueError, match="finite"):
        HestonCalibrator._resolve_forward(1.0, group, {1.0: math.nan})


def test_heston_calibration_failure_and_invalid_final_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calibrator = HestonCalibrator(HestonConfig(seeds=(0,), min_strikes=7))
    group = pd.DataFrame(_rows())[["strike", "mid_iv"]]
    monkeypatch.setattr(
        heston, "least_squares", lambda *_args, **_kwargs: SimpleNamespace(success=False)
    )
    with pytest.raises(RuntimeError, match="failed"):
        calibrator._calibrate_single_tenor(100.0, 1.0, group)

    initial = np.array([math.log(0.04), math.log(0.04), math.log(1.5), math.log(0.5), -0.3])
    monkeypatch.setattr(
        heston,
        "least_squares",
        lambda *_args, **_kwargs: SimpleNamespace(success=True, fun=np.zeros(7), x=initial),
    )
    monkeypatch.setattr(calibrator, "_model_vols", lambda *_args, **_kwargs: np.full(7, math.nan))
    with pytest.raises(RuntimeError, match="invalid values"):
        calibrator._calibrate_single_tenor(100.0, 1.0, group)


@pytest.mark.parametrize("name", ["cleaner", "sabr", "validator", "heston"])
def test_surface_builder_validates_components(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        SurfaceBuilder(**{name: object()})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("rmse", "observations", "parameters"),
    [
        (math.nan, 10, 3),
        (-1.0, 10, 3),
        (0.1, True, 3),
        (0.1, 0, 3),
        (0.1, 10, True),
        (0.1, 10, 0),
    ],
)
def test_aicc_rejects_invalid_inputs(rmse: float, observations: object, parameters: object) -> None:
    with pytest.raises(ValueError, match="AICc"):
        SurfaceBuilder._aicc(rmse, observations, parameters)  # type: ignore[arg-type]
    assert math.isinf(SurfaceBuilder._aicc(0.1, 4, 3))
    assert math.isfinite(SurfaceBuilder._aicc(0.1, 10, 3))


def test_surface_result_serialisation_and_identity() -> None:
    report = ValidationReport([])
    selection = TenorSelection(1.0, "sabr", {"alpha": 0.2}, 0.01, -10.0)
    clean = _clean()
    builder = SurfaceBuilder()
    surface_id = builder._compute_surface_id(clean, [selection])
    result = select.SurfaceBuildResult(surface_id, clean, {"ok": True}, report, [selection])
    payload = result.to_dict()
    assert payload["surface_id"] == surface_id
    assert payload["validation"]["is_ok"] is True
    assert payload["tenors"][0]["criterion"] == "AICc"


def test_surface_builder_validates_flags_and_empty_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = SurfaceBuilder()
    with pytest.raises(ValueError, match="seed"):
        builder.build([], seed=True)
    with pytest.raises(TypeError, match="enable_heston"):
        builder.build([], enable_heston=1)  # type: ignore[arg-type]

    clean = _clean()
    monkeypatch.setattr(builder._cleaner, "ingest", lambda *_args, **_kwargs: clean)
    monkeypatch.setattr(
        builder._validator, "validate", lambda *_args, **_kwargs: ValidationReport([])
    )
    monkeypatch.setattr(builder._sabr, "calibrate", lambda *_args, **_kwargs: [])
    with pytest.raises(ValueError, match="no tenors"):
        builder.build([])


def test_surface_builder_handles_heston_failure_and_disabled_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = SurfaceBuilder()
    clean = _clean()
    sabr_result = SABRTenorResult(
        1.0,
        {"alpha": 0.2, "beta": 0.5, "rho": 0.0, "nu": 0.4},
        0.01,
        np.arange(7.0),
        np.full(7, 0.2),
        np.full(7, 0.2),
        3,
    )
    monkeypatch.setattr(builder._cleaner, "ingest", lambda *_args, **_kwargs: clean)
    monkeypatch.setattr(
        builder._validator, "validate", lambda *_args, **_kwargs: ValidationReport([])
    )
    monkeypatch.setattr(builder._sabr, "calibrate", lambda *_args, **_kwargs: [sabr_result])
    monkeypatch.setattr(
        builder._heston,
        "calibrate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fit failed")),
    )
    failed = builder.build([])
    assert failed.qc["heston"] == {
        "enabled": True,
        "status": "failed",
        "error": "fit failed",
    }
    disabled = builder.build([], enable_heston=False)
    assert disabled.qc["heston"]["status"] == "disabled"


@pytest.mark.parametrize("parity_tol", [True, math.nan, -1.0, 2.0, "bad"])
def test_no_arbitrage_validator_bounds_tolerance(parity_tol: object) -> None:
    with pytest.raises(ValueError, match="parity_tol"):
        NoArbitrageValidator(parity_tol)  # type: ignore[arg-type]


def test_validation_report_helpers() -> None:
    violation = Violation("parity", 1.0, 100.0, {"parity": 1.0})
    report = ValidationReport([violation])
    assert report.is_ok is False
    assert report.reasons() == ["parity"]
    assert report.to_dict()["violations"][0]["strike"] == 100.0


def test_validator_rejects_malformed_boards() -> None:
    validator = NoArbitrageValidator()
    with pytest.raises(TypeError, match="board"):
        validator.validate([])  # type: ignore[arg-type]
    assert validator.validate(pd.DataFrame()).is_ok
    with pytest.raises(KeyError, match="missing"):
        validator.validate(pd.DataFrame([{"tenor": 1.0}]))
    for field, value in (
        ("mid_iv", math.nan),
        ("strike", 0.0),
        ("option_type", 1),
        ("option_type", "EXOTIC"),
    ):
        rows = _rows(count=1)
        rows[0][field] = value
        with pytest.raises(ValueError):
            validator.validate(pd.DataFrame(rows))


def test_validator_reports_inconsistent_curves() -> None:
    rows = _rows(count=3)
    rows[1]["forward"] = 101.0
    rows[1]["discount"] = 0.9
    rows[0]["discount"] = 1.0
    rows[2]["discount"] = 1.0
    reasons = NoArbitrageValidator().validate(pd.DataFrame(rows)).reasons()
    assert "inconsistent_forward" in reasons
    assert "inconsistent_discount" in reasons


def test_butterfly_checks_all_static_arbitrage_shapes() -> None:
    validator = NoArbitrageValidator(parity_tol=1e-8)
    bounds = validator._check_butterfly(
        1.0,
        np.array([90.0]),
        np.array([200.0]),
        forward=100.0,
    )
    assert bounds[0].kind == "price_bounds"
    with pytest.raises(ValueError, match="strictly increasing"):
        validator._check_butterfly(
            1.0,
            np.array([90.0, 90.0]),
            np.array([20.0, 19.0]),
        )
    vertical = validator._check_butterfly(
        1.0,
        np.array([90.0, 100.0]),
        np.array([20.0, 0.0]),
    )
    assert "vertical_spread" in [item.kind for item in vertical]
    shaped = validator._check_butterfly(
        1.0,
        np.array([90.0, 100.0, 110.0]),
        np.array([10.0, 20.0, 0.0]),
    )
    assert {item.kind for item in shaped} >= {"butterfly", "strike_monotonicity"}


def test_call_price_conversion_and_calendar_checks() -> None:
    validator = NoArbitrageValidator()
    rows = [
        {"tenor": 1.0, "strike": 100.0, "mid_iv": 0.2, "option_type": "PUT"},
        {"tenor": 1.0, "strike": 110.0, "mid_iv": 0.2, "option_type": "CALL"},
    ]
    strikes, prices = validator._call_prices(pd.DataFrame(rows), 100.0, 1.0)
    assert strikes.tolist() == [100.0, 110.0]
    assert np.isfinite(prices).all()

    calendar = pd.DataFrame(
        [
            {"tenor": 0.5, "strike": strike, "mid_iv": 0.5, "forward": 100.0}
            for strike in (90.0, 100.0, 110.0)
        ]
        + [
            {"tenor": 1.0, "strike": strike, "mid_iv": 0.1, "forward": 100.0}
            for strike in (90.0, 100.0, 110.0)
        ]
    )
    assert any(item.kind == "calendar" for item in validator._check_calendar(calendar))

    disjoint = pd.DataFrame(
        [
            {"tenor": 0.5, "strike": 10.0, "mid_iv": 0.2, "forward": 100.0},
            {"tenor": 1.0, "strike": 1_000.0, "mid_iv": 0.2, "forward": 100.0},
        ]
    )
    assert validator._check_calendar(disjoint) == []


def test_parity_and_black_price_helpers() -> None:
    validator = NoArbitrageValidator()
    frame = pd.DataFrame(
        [
            {"tenor": 1.0, "strike": 100.0, "mid_iv": 0.2, "option_type": "CALL"},
            {"tenor": 1.0, "strike": 100.0, "mid_iv": 0.8, "option_type": "PUT"},
        ]
    )
    assert validator._check_parity(frame, 100.0, 1.0)[0].kind == "parity"
    assert validators._black_price_single("PUT", 100.0, 90.0, 1.0, 0.0, 1.0) == 0.0
    array = validators._black_price_array(
        ["CALL", "PUT"],
        100.0,
        np.array([90.0, 110.0]),
        np.array([1.0, 1.0]),
        np.array([0.2, 0.2]),
        1.0,
    )
    assert array.shape == (2,)
