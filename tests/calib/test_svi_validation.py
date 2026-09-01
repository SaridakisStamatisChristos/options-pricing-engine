from __future__ import annotations

import numpy as np
import pandas as pd

from options_engine.calib.boards import CleanBoard
from options_engine.calib.svi import (
    RawSVICalibrator,
    RawSVIConfig,
    SSVICalibrator,
    SSVIConfig,
    SVIParameters,
    ssvi_total_variance,
    validate_svi_slice,
)


def _board(*, decreasing_atm: bool = False) -> CleanBoard:
    rows: list[dict[str, float]] = []
    levels = (
        ((0.25, 0.020), (0.5, 0.018), (1.0, 0.050))
        if decreasing_atm
        else (
            (0.25, 0.012),
            (0.5, 0.025),
            (1.0, 0.055),
        )
    )
    for tenor, theta in levels:
        strikes = np.linspace(70.0, 130.0, 13)
        vols = np.sqrt(
            ssvi_total_variance(np.log(strikes / 100.0), theta, rho=-0.35, eta=1.1, power=0.25)
            / tenor
        )
        rows.extend(
            {
                "tenor": tenor,
                "strike": float(strike),
                "mid_iv": float(vol),
                "forward": 100.0,
            }
            for strike, vol in zip(strikes, vols, strict=True)
        )
    return CleanBoard(pd.DataFrame(rows), {})


def test_ssvi_audit_reports_deterministic_holdout_and_optimizer_evidence() -> None:
    config = SSVIConfig(seeds=(0, 1), holdout_policy="wings", holdout_fraction=0.2)
    first = SSVICalibrator(config).calibrate(_board())
    second = SSVICalibrator(config).calibrate(_board())

    assert first.holdout_observations == second.holdout_observations == 6
    assert [row.is_holdout for row in first.residuals] == [
        row.is_holdout for row in second.residuals
    ]
    assert first.holdout_rmse is not None
    assert len(first.initialization_sensitivity.attempts) == 2
    assert first.to_dict() == second.to_dict()
    assert first.wing_constraint_slack > 0.0
    assert first.curvature_constraint_slack > 0.0


def test_ssvi_reports_atm_projection_magnitude() -> None:
    result = SSVICalibrator(SSVIConfig(seeds=(0,))).calibrate(_board(decreasing_atm=True))

    assert result.atm_projection_applied
    assert result.atm_projection_absolute_adjustment > 0.0
    assert result.atm_projection_relative_adjustment > 0.0
    assert result.largest_atm_projection_tenor in {0.25, 0.5}
    assert result.total_weighted_projection_error > 0.0
    assert result.fit_quality.value in {"acceptable", "poor", "unstable"}


def test_raw_svi_butterfly_arbitrage_is_invalid_not_warning_only() -> None:
    params = SVIParameters(
        a=0.14830095944960534,
        b=0.6496433548627699,
        rho=0.8587726906484072,
        m=0.17699133467530692,
        sigma=0.01114110944658852,
    )
    diagnostics = validate_svi_slice(params)

    assert not diagnostics.admissible
    assert diagnostics.fit_quality.value == "invalid"
    assert diagnostics.butterfly_violation_count > 0
    assert diagnostics.left_wing_slope != diagnostics.right_wing_slope


def test_raw_svi_calibrator_reports_deterministic_audit_evidence() -> None:
    expected = SVIParameters(a=0.025, b=0.18, rho=-0.35, m=0.01, sigma=0.22)
    k = np.linspace(-0.45, 0.45, 17)
    variance = expected.a + expected.b * (
        expected.rho * (k - expected.m) + np.sqrt((k - expected.m) ** 2 + expected.sigma**2)
    )
    config = RawSVIConfig(seeds=(0, 1, 2), holdout_policy="wings", holdout_fraction=0.2)
    first = RawSVICalibrator(config).calibrate(k, variance, tenor=0.75, forward=100.0)
    second = RawSVICalibrator(config).calibrate(k, variance, tenor=0.75, forward=100.0)

    assert first.to_dict() == second.to_dict()
    assert first.residual_summary.calibration_observations == 14
    assert first.residual_summary.holdout_observations == 3
    assert first.residual_summary.holdout_rmse is not None
    assert len(first.initialization_sensitivity.attempts) == 3
    assert first.diagnostics.admissible
    assert first.fit_quality.value != "invalid"
    assert all(np.isfinite(tuple(first.parameter_bound_proximity.values())))


def test_raw_svi_calibrator_limits_holdout_to_identified_training_set() -> None:
    k = np.linspace(-0.2, 0.2, 6)
    variance = np.full(6, 0.04)
    config = RawSVIConfig(holdout_policy="alternating")

    result = RawSVICalibrator(config).calibrate(k, variance)

    assert result.residual_summary.calibration_observations == 5
    assert result.residual_summary.holdout_observations == 1
