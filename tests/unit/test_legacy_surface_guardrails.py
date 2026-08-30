"""Compatibility-surface tests for the legacy SABR calibration facade."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from options_engine.calib.validators import ValidationReport, Violation
from options_engine.core import vol_surface_calibration as legacy


def _rows(count: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "tenor": 1.0,
            "strike": 90.0 + 5.0 * index,
            "forward": 100.0,
            "mid_iv": 0.2 + 0.001 * index,
        }
        for index in range(count)
    ]


def _tenor_result(rmse: float = 0.01) -> legacy.SABRTenorCalibration:
    return legacy.SABRTenorCalibration(
        tenor=1.0,
        parameters=legacy.SABRParameters(0.2, 0.5, -0.2, 0.4),
        strikes=np.array([90.0, 100.0, 110.0]),
        market_vols=np.array([0.22, 0.2, 0.21]),
        model_vols=np.array([0.22, 0.2, 0.21]),
        rmse=rmse,
    )


def test_legacy_result_objects_serialize_complete_diagnostics() -> None:
    report = legacy.QCReport(3, 1, 0, 0, 2, dropped_invalid=0)
    clean = legacy.CleanBoardResult(pd.DataFrame(_rows(2)), report)
    arbitrage = legacy.ArbitrageCheckResult([], [], [])
    result = legacy.SABRCalibrationResult(
        clean_board=clean,
        tenor_results=[_tenor_result()],
        arbitrage=arbitrage,
        fitted_surface=pd.DataFrame(),
        regime="sabr",
    )
    assert clean.to_dict()["report"]["retained_quotes"] == 2
    assert arbitrage.is_arbitrage_free is True
    assert _tenor_result().to_dict()["parameters"]["rho"] == -0.2
    assert result.qc_report["rmse"] == {
        "per_tenor": {1.0: 0.01},
        "max": 0.01,
        "mean": 0.01,
    }

    empty = legacy.SABRCalibrationResult(
        clean,
        [],
        legacy.ArbitrageCheckResult([{"tenor": 1.0}], [], []),
        pd.DataFrame(),
        "fallback",
    )
    assert empty.arbitrage.is_arbitrage_free is False
    assert empty.qc_report["rmse"]["max"] is None


@pytest.mark.parametrize("value", [True, "1", None])
def test_finite_real_rejects_non_numeric_values(value: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        legacy._finite_real("value", value, minimum=0.0, maximum=1.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -1.0, 2.0])
def test_finite_real_rejects_nonfinite_or_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        legacy._finite_real("value", value, minimum=0.0, maximum=1.0)


@pytest.mark.parametrize(
    "names",
    [
        ("", "strike"),
        (1, "strike"),
        ("x" * 129, "strike"),
        ("same", "same"),
    ],
)
def test_legacy_column_names_are_canonical(names: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="column names"):
        legacy._validate_column_names(names)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"bid_column": None, "ask_column": "ask"}, ValueError),
        ({"now": "today"}, TypeError),
        ({"now": datetime.now()}, ValueError),
        ({"max_age_seconds": True}, TypeError),
        ({"mad_threshold": 0.0}, ValueError),
    ],
)
def test_legacy_cleaner_validates_configuration(
    kwargs: dict[str, object], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        legacy.clean_option_board([], **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("board", ["bad", 1, [1]])
def test_legacy_cleaner_rejects_malformed_board_containers(board: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        legacy.clean_option_board(board)  # type: ignore[arg-type]


def test_legacy_cleaner_enforces_dataframe_and_sequence_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(legacy, "_MAX_BOARD_ROWS", 1)
    with pytest.raises(ValueError, match="row limit"):
        legacy.clean_option_board(pd.DataFrame(_rows(2)))
    with pytest.raises(ValueError, match="row limit"):
        legacy.clean_option_board(_rows(2))


def test_legacy_cleaner_handles_empty_missing_and_spread_contracts() -> None:
    assert legacy.clean_option_board([]).report.retained_quotes == 0
    with pytest.raises(KeyError, match="required"):
        legacy.clean_option_board([{"tenor": 1.0}])
    with pytest.raises(KeyError, match="supplied together"):
        legacy.clean_option_board([{**_rows(1)[0], "bid_iv": 0.19}])
    with pytest.raises(KeyError, match="midpoint"):
        legacy.clean_option_board(
            [{"tenor": 1.0, "strike": 100.0}],
            bid_column=None,
            ask_column=None,
        )


def test_legacy_cleaner_computes_midpoint_and_accounts_for_every_drop() -> None:
    now = datetime(2025, 1, 1, tzinfo=UTC)
    rows = [
        {
            "tenor": 1.0,
            "strike": 90.0,
            "forward": 100.0,
            "bid_iv": 0.19,
            "ask_iv": 0.21,
            "timestamp": now,
        },
        {
            "tenor": "bad",
            "strike": 95.0,
            "forward": 100.0,
            "bid_iv": 0.19,
            "ask_iv": 0.21,
            "timestamp": now,
        },
        {
            "tenor": 1.0,
            "strike": 100.0,
            "forward": 100.0,
            "bid_iv": "bad",
            "ask_iv": 0.21,
            "timestamp": now,
        },
        {
            "tenor": 1.0,
            "strike": 105.0,
            "forward": 100.0,
            "bid_iv": 0.25,
            "ask_iv": 0.20,
            "timestamp": now,
        },
        {
            "tenor": 1.0,
            "strike": 110.0,
            "forward": 100.0,
            "bid_iv": 0.19,
            "ask_iv": 0.21,
            "timestamp": now - timedelta(hours=1),
        },
    ]
    result = legacy.clean_option_board(
        rows,
        now=now,
        max_age_seconds=60.0,
        vol_column="computed_mid",
    )
    assert result.data["computed_mid"].tolist() == [0.2]
    assert result.report.to_dict() == {
        "total_quotes": 5,
        "dropped_invalid": 2,
        "dropped_crossed": 1,
        "dropped_stale": 1,
        "dropped_outlier": 0,
        "retained_quotes": 1,
    }


def test_legacy_cleaner_rejects_midpoints_outside_quotes_and_marks_outliers() -> None:
    invalid = [{**_rows(1)[0], "bid_iv": 0.21, "ask_iv": 0.22, "mid_iv": 0.20}]
    result = legacy.clean_option_board(invalid)
    assert result.report.dropped_invalid == 1
    assert result.data.empty

    rows = [{**row, "mid_iv": vol} for row, vol in zip(_rows(4), [0.2, 0.2, 0.2, 0.9], strict=True)]
    result = legacy.clean_option_board(rows, bid_column=None, ask_column=None)
    assert result.report.dropped_outlier == 1


def test_legacy_hagan_wrapper_delegates_to_canonical_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake(forward: object, strike: object, expiry: object, **kwargs: object) -> np.ndarray:
        captured.update({"forward": forward, "strike": strike, "expiry": expiry, **kwargs})
        return np.array([0.2])

    monkeypatch.setattr(legacy, "canonical_hagan_implied_volatility", fake)
    assert legacy.hagan_implied_volatility(100.0, 100.0, 1.0, 0.2, 0.5, 0.0, 0.4)[0] == 0.2
    assert captured["beta"] == 0.5


@pytest.mark.parametrize("vol_column", ["", "x" * 129, 1])
def test_legacy_arbitrage_validator_validates_column_name(vol_column: object) -> None:
    with pytest.raises(ValueError, match="vol_column"):
        legacy.ArbitrageValidator(vol_column=vol_column)  # type: ignore[arg-type]


def test_legacy_arbitrage_validator_validates_surface_and_maps_findings() -> None:
    validator = legacy.ArbitrageValidator()
    with pytest.raises(TypeError, match="DataFrame"):
        validator.validate([])  # type: ignore[arg-type]
    assert validator.validate(pd.DataFrame()).is_arbitrage_free
    with pytest.raises(KeyError, match="required"):
        validator.validate(pd.DataFrame([{"tenor": 1.0}]))

    validator._validator = type(
        "Stub",
        (),
        {
            "validate": staticmethod(
                lambda _frame: ValidationReport(
                    [
                        Violation("calendar", 1.0, 100.0, {"delta": -1.0}),
                        Violation("inconsistent_forward", 1.0, 0.0, {"range": 1.0}),
                        Violation("butterfly", 1.0, 100.0, {"slope": -1.0}),
                    ]
                )
            )
        },
    )()
    surface = pd.DataFrame([{"tenor": 1.0, "strike": 100.0, "model_vol": 0.2, "forward": 100.0}])
    result = validator.validate(surface)
    assert len(result.calendar_violations) == 1
    assert len(result.tenor_monotonicity_violations) == 1
    assert len(result.butterfly_violations) == 1


@pytest.mark.parametrize("max_iterations", [True, 1.5, "10"])
def test_legacy_calibrator_requires_integral_iteration_budget(max_iterations: object) -> None:
    with pytest.raises(TypeError, match="integer"):
        legacy.SABRCalibrator(max_iterations=max_iterations)  # type: ignore[arg-type]


def test_legacy_calibrator_rejects_empty_and_underidentified_boards() -> None:
    calibrator = legacy.SABRCalibrator()
    with pytest.raises(ValueError, match="removed all"):
        calibrator.calibrate([], bid_column=None, ask_column=None)
    with pytest.raises(ValueError, match="five unique strikes"):
        calibrator.calibrate(_rows(4), bid_column=None, ask_column=None)


def test_legacy_forward_resolution_supports_callable_mapping_and_board() -> None:
    group = pd.DataFrame(_rows())
    assert (
        legacy.SABRCalibrator._resolve_forward(
            1.0, group, forward_curve=lambda tenor: 100.0 + tenor, forward_column="forward"
        )
        == 101.0
    )
    assert (
        legacy.SABRCalibrator._resolve_forward(
            1.0, group, forward_curve={1.0: 102.0}, forward_column="forward"
        )
        == 102.0
    )
    assert (
        legacy.SABRCalibrator._resolve_forward(
            1.0, group, forward_curve=None, forward_column="forward"
        )
        == 100.0
    )
    with pytest.raises(KeyError, match="not found"):
        legacy.SABRCalibrator._resolve_forward(
            1.0, group, forward_curve={2.0: 100.0}, forward_column="forward"
        )
    with pytest.raises(ValueError, match="ambiguous"):
        legacy.SABRCalibrator._resolve_forward(
            1.0,
            group,
            forward_curve={1.0 - 1e-10: 100.0, 1.0 + 1e-10: 101.0},
            forward_column="forward",
        )
    with pytest.raises(KeyError, match="unavailable"):
        legacy.SABRCalibrator._resolve_forward(
            1.0, group.drop(columns="forward"), forward_curve=None, forward_column="forward"
        )
    inconsistent = group.copy()
    inconsistent.loc[0, "forward"] = 101.0
    with pytest.raises(ValueError, match="consistent"):
        legacy.SABRCalibrator._resolve_forward(
            1.0, inconsistent, forward_curve=None, forward_column="forward"
        )
    with pytest.raises(ValueError, match="within"):
        legacy.SABRCalibrator._resolve_forward(
            1.0, group, forward_curve=lambda _tenor: math.nan, forward_column="forward"
        )


def test_legacy_regime_selection_is_bounded_and_deterministic() -> None:
    results = [_tenor_result(0.1)]
    assert legacy.SABRCalibrator._select_regime(results, None) == "sabr"
    assert legacy.SABRCalibrator._select_regime(results, {"Heston": [0.01, 0.02]}) == "heston"
    assert legacy.SABRCalibrator._select_regime(results, {"empty": []}) == "sabr"
    with pytest.raises(ValueError, match="64-model"):
        legacy.SABRCalibrator._select_regime(
            results, {f"model-{index}": [0.1] for index in range(65)}
        )
    with pytest.raises(ValueError, match="names"):
        legacy.SABRCalibrator._select_regime(results, {"": [0.1]})
    with pytest.raises(ValueError, match="100000"):
        legacy.SABRCalibrator._select_regime(results, {"model": [0.1] * 100_001})
    for scores in ([math.nan], [-1.0]):
        with pytest.raises(ValueError, match="finite"):
            legacy.SABRCalibrator._select_regime(results, {"model": scores})
