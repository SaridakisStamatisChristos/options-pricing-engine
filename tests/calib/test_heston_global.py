"""Global multi-tenor Heston calibration and comparison tests."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from options_engine.calib import BoardCleaner, CleanBoard
from options_engine.calib.heston import (
    HestonCalibrationComparison,
    HestonCalibrator,
    HestonConfig,
    _HestonSlice,
)
from options_engine.calib.heston_cos import heston_cos_implied_volatilities

TRUE_PARAMS = {
    "v0": 0.04,
    "theta": 0.06,
    "kappa": 1.35,
    "vol_of_vol": 0.42,
    "rho": -0.55,
}


def _global_heston_board() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timestamp = datetime.now(UTC)
    for tenor in (0.25, 0.75, 2.0):
        forward = 100.0 * math.exp(0.01 * tenor)
        strikes = forward * np.exp(np.linspace(-0.3, 0.3, 11))
        # Generate with the independent adaptive COS family and recover with
        # Gauss-Laguerre by default, avoiding an inversion-method inverse crime.
        vols = heston_cos_implied_volatilities(
            forward,
            strikes,
            tenor,
            v0=TRUE_PARAMS["v0"],
            theta=TRUE_PARAMS["theta"],
            kappa=TRUE_PARAMS["kappa"],
            vol_of_vol=TRUE_PARAMS["vol_of_vol"],
            rho=TRUE_PARAMS["rho"],
        )
        for strike, vol in zip(strikes, vols, strict=True):
            rows.extend(
                {
                    "tenor": tenor,
                    "strike": float(strike),
                    "mid_iv": float(vol),
                    "forward": forward,
                    "option_type": option_type,
                    "timestamp": timestamp,
                }
                for option_type in ("CALL", "PUT")
            )
    return rows


@pytest.fixture(scope="module")
def coherent_board() -> CleanBoard:
    return BoardCleaner().ingest(_global_heston_board())


def test_global_calibration_recovers_one_parameter_set_across_maturities(
    coherent_board: CleanBoard,
) -> None:
    calibrator = HestonCalibrator(
        HestonConfig(
            calibration_mode="global",
            weighting="uniform",
            seeds=(0, 1),
            max_iterations=200,
            holdout_fraction=0.1,
        )
    )

    detailed = calibrator.calibrate_detailed(coherent_board)
    legacy_shape = calibrator.calibrate(coherent_board)

    assert detailed.mode == "global"
    assert detailed.shared_params is not None
    assert len(detailed.tenor_results) == 3
    assert len(legacy_shape) == 3
    for name, expected in TRUE_PARAMS.items():
        assert detailed.shared_params[name] == pytest.approx(expected, rel=2e-4, abs=2e-5)
    assert detailed.in_sample_weighted_rmse < 1e-7
    assert detailed.holdout_rmse is not None and detailed.holdout_rmse < 1e-7
    assert detailed.feller_ratio == pytest.approx(
        2.0
        * detailed.shared_params["kappa"]
        * detailed.shared_params["theta"]
        / detailed.shared_params["vol_of_vol"] ** 2
    )
    assert detailed.optimizer_diagnostics[0].success is True
    assert detailed.optimizer_diagnostics[0].evaluations > 0
    assert set(detailed.parameter_bound_proximity) == {
        "v0",
        "theta",
        "kappa",
        "vol_of_vol",
        "rho",
    }
    assert all(0.0 <= value <= 1.0 for value in detailed.parameter_bound_proximity.values())
    assert all(result.params == detailed.shared_params for result in detailed.tenor_results)
    assert all(result.calibration_mode == "global" for result in detailed.tenor_results)
    assert (
        len(detailed.residuals) == detailed.calibration_observations + detailed.holdout_observations
    )
    assert detailed.residual_summary is not None
    assert detailed.residual_summary.maximum_absolute_residual < 1e-6
    assert detailed.weight_diagnostics is not None
    assert detailed.initialization_sensitivity[0].attempts
    assert detailed.conditioning[0].effective_rank <= 5
    assert detailed.fit_quality in {"good", "acceptable", "unstable"}
    assert detailed.to_dict()["residuals"]


def test_global_calibration_can_hold_out_an_entire_tenor(coherent_board: CleanBoard) -> None:
    detailed = HestonCalibrator(
        HestonConfig(
            calibration_mode="global",
            weighting="uniform",
            seeds=(0,),
            max_iterations=200,
            holdout_fraction=0.0,
            holdout_tenors=(0.75,),
        )
    ).calibrate_detailed(coherent_board)
    held_out = next(result for result in detailed.tenor_results if result.tenor == 0.75)

    assert held_out.is_holdout_tenor is True
    assert held_out.calibration_observations == 0
    assert held_out.holdout_observations == len(held_out.strikes)
    assert held_out.holdout_rmse is not None and held_out.holdout_rmse < 1e-6
    assert detailed.holdout_rmse is not None and detailed.holdout_rmse < 1e-6
    assert detailed.tenor_holdout_rmse == pytest.approx(detailed.holdout_rmse)
    assert detailed.strike_holdout_rmse is None
    assert held_out.residual_summary is not None
    assert held_out.residual_summary.calibration_observations == 0
    assert all(observation.is_holdout for observation in held_out.residuals)


def test_compare_modes_reports_tradeoffs_without_automatic_ranking(
    coherent_board: CleanBoard,
) -> None:
    comparison = HestonCalibrator(
        HestonConfig(
            weighting="uniform",
            seeds=(0,),
            max_iterations=400,
            holdout_fraction=0.1,
        )
    ).compare_modes(coherent_board)
    payload = comparison.to_dict()

    assert isinstance(comparison, HestonCalibrationComparison)
    assert comparison.per_tenor.mode == "per_tenor"
    assert comparison.per_tenor.shared_params is None
    assert comparison.global_fit.mode == "global"
    assert comparison.global_fit.shared_params is not None
    assert math.isfinite(comparison.per_tenor.in_sample_weighted_rmse)
    assert math.isfinite(comparison.global_fit.in_sample_weighted_rmse)
    assert "global_minus_per_tenor" in payload
    assert "winner" not in payload


def test_compare_modes_uses_like_for_like_strike_holdouts(coherent_board: CleanBoard) -> None:
    comparison = HestonCalibrator(
        HestonConfig(
            calibration_mode="global",
            holdout_tenors=(0.75,),
            weighting="uniform",
            seeds=(0,),
            max_iterations=400,
            holdout_fraction=0.1,
        )
    ).compare_modes(coherent_board)

    assert all(not result.is_holdout_tenor for result in comparison.per_tenor.tenor_results)
    assert all(not result.is_holdout_tenor for result in comparison.global_fit.tenor_results)
    assert comparison.per_tenor.holdout_observations == comparison.global_fit.holdout_observations


def test_global_calibration_supports_fixed_cost_cos_objective(coherent_board: CleanBoard) -> None:
    detailed = HestonCalibrator(
        HestonConfig(
            calibration_mode="global",
            pricing_method="cos",
            cos_terms=384,
            cos_truncation=16.0,
            weighting="uniform",
            seeds=(0,),
            max_iterations=150,
            holdout_fraction=0.1,
        )
    ).calibrate_detailed(coherent_board)

    assert detailed.pricing_method == "cos"
    assert detailed.shared_params is not None
    assert detailed.in_sample_weighted_rmse < 2e-4
    assert detailed.holdout_rmse is not None and detailed.holdout_rmse < 2e-4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"calibration_mode": "joint"},
        {"pricing_method": "fft"},
        {"global_tenor_weighting": "time"},
        {"cos_terms": 16},
        {"cos_truncation": 3.0},
        {"holdout_tenors": (1.0,)},
        {
            "calibration_mode": "global",
            "tenors": (0.5, 1.0),
            "holdout_tenors": (2.0,),
        },
    ],
)
def test_global_configuration_rejects_ambiguous_or_unsafe_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        HestonConfig(**kwargs)  # type: ignore[arg-type]


def test_global_mode_requires_two_training_tenors(coherent_board: CleanBoard) -> None:
    calibrator = HestonCalibrator(
        HestonConfig(
            calibration_mode="global",
            tenors=(0.25, 0.75),
            holdout_tenors=(0.75,),
            seeds=(0,),
        )
    )
    with pytest.raises(ValueError, match="two non-holdout tenors"):
        calibrator.calibrate_detailed(coherent_board)


def test_global_mode_rejects_a_missing_holdout_tenor(coherent_board: CleanBoard) -> None:
    calibrator = HestonCalibrator(
        HestonConfig(
            calibration_mode="global",
            holdout_tenors=(3.0,),
            seeds=(0,),
        )
    )
    with pytest.raises(ValueError, match="absent from the retained"):
        calibrator.calibrate_detailed(coherent_board)


def test_global_tenor_weighting_is_explicit() -> None:
    slices = [
        _HestonSlice(
            tenor=tenor,
            forward=100.0,
            group=pd.DataFrame(),
            weights=np.ones(count),
            weighting="uniform",
            holdout=np.zeros(count, dtype=bool),
        )
        for tenor, count in ((0.5, 7), (2.0, 14))
    ]
    equal = HestonCalibrator(
        HestonConfig(calibration_mode="global", global_tenor_weighting="equal")
    )._global_weights(slices)
    observations = HestonCalibrator(
        HestonConfig(calibration_mode="global", global_tenor_weighting="observations")
    )._global_weights(slices)

    assert np.sum(equal[0]) == pytest.approx(np.sum(equal[1]))
    assert np.sum(observations[0]) == pytest.approx(7.0)
    assert np.sum(observations[1]) == pytest.approx(14.0)


def test_observation_weighting_counts_only_training_quotes() -> None:
    first_holdout = np.array([False, True, False, False, True])
    second_holdout = np.array([False, False, True, False, False, False, True])
    slices = [
        _HestonSlice(
            tenor=0.5,
            forward=100.0,
            group=pd.DataFrame(),
            weights=np.array([0.5, 3.0, 1.0, 1.5, 4.0]),
            weighting="vega",
            holdout=first_holdout,
        ),
        _HestonSlice(
            tenor=2.0,
            forward=100.0,
            group=pd.DataFrame(),
            weights=np.array([2.0, 0.5, 3.0, 1.0, 1.5, 2.0, 4.0]),
            weighting="vega",
            holdout=second_holdout,
        ),
    ]
    adjusted = HestonCalibrator(
        HestonConfig(calibration_mode="global", global_tenor_weighting="observations")
    )._global_weights(slices)

    assert np.sum(adjusted[0][~first_holdout]) == pytest.approx(3.0)
    assert np.sum(adjusted[1][~second_holdout]) == pytest.approx(5.0)


@pytest.mark.parametrize("policy", ["alternating", "wings", "centre", "fractional"])
def test_heston_strike_holdout_policies_are_deterministic_and_identified(policy: str) -> None:
    strikes = np.linspace(70.0, 130.0, 13)
    calibrator = HestonCalibrator(
        HestonConfig(holdout_policy=policy, holdout_fraction=0.25, min_strikes=7)
    )
    first = calibrator._holdout_mask(strikes, centre=100.0)
    second = calibrator._holdout_mask(strikes, centre=100.0)

    assert np.array_equal(first, second)
    assert np.any(first)
    assert np.count_nonzero(~first) >= 7
    if policy == "centre":
        assert first[int(np.argmin(np.abs(strikes - 100.0)))]
