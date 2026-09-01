"""Recovery and explicit expected behaviour for difficult quote boards."""

from __future__ import annotations

import pytest

from options_engine.calib.datasets import (
    AdversarialBoard,
    adversarial_board,
    sabr_recovery_board,
    ssvi_recovery_board,
)
from options_engine.calib.sabr import SABRCalibrator, SABRConfig
from options_engine.calib.svi import SSVICalibrator, SSVIConfig


@pytest.mark.parametrize("noise", [0.0, 0.001])
def test_sabr_recovers_observables_and_identifiable_fixed_beta_parameters(noise: float) -> None:
    fixture = sabr_recovery_board(noise_amplitude=noise)
    result = SABRCalibrator(
        SABRConfig(beta=fixture.parameters["beta"], seeds=(0, 1), holdout_policy="alternating")
    ).calibrate_detailed(fixture.board)
    tenor = result.tenor_results[0]

    assert result.holdout_rmse is not None and result.holdout_rmse < 0.004
    assert result.in_sample_weighted_rmse < 0.002
    assert tenor.params["alpha"] == pytest.approx(fixture.parameters["alpha"], abs=0.01)
    assert tenor.params["rho"] == pytest.approx(fixture.parameters["rho"], abs=0.04)
    assert tenor.params["nu"] == pytest.approx(fixture.parameters["nu"], abs=0.08)
    assert all(0.0 <= proximity <= 1.0 for proximity in tenor.parameter_bound_proximity.values())


@pytest.mark.parametrize("noise", [0.0, 0.001])
def test_ssvi_recovers_known_surface_and_admissibility(noise: float) -> None:
    fixture = ssvi_recovery_board(noise_amplitude=noise)
    result = SSVICalibrator(
        SSVIConfig(seeds=(0, 1), weighting="uniform", holdout_policy="alternating")
    ).calibrate(fixture.board)

    assert result.holdout_rmse is not None and result.holdout_rmse < 0.004
    assert result.weighted_rmse < 0.002
    assert result.surface.rho == pytest.approx(fixture.parameters["rho"], abs=0.03)
    assert result.surface.eta == pytest.approx(fixture.parameters["eta"], abs=0.1)
    assert result.surface.power == pytest.approx(fixture.parameters["power"], abs=0.04)
    assert result.minimum_density_factor >= 0.0
    assert result.maximum_calendar_decrease >= -1e-10
    assert result.wing_constraint_slack >= 0.0
    assert result.curvature_constraint_slack >= 0.0


@pytest.mark.parametrize(
    "case",
    [AdversarialBoard.SPARSE, AdversarialBoard.CLUSTERED_ATM],
)
def test_underidentified_ssvi_boards_are_rejected(case: AdversarialBoard) -> None:
    with pytest.raises(ValueError, match="at least 5 distinct strikes"):
        SSVICalibrator(SSVIConfig(seeds=(0,))).calibrate(adversarial_board(case))


@pytest.mark.parametrize(
    "case",
    [AdversarialBoard.GROSS_OUTLIER, AdversarialBoard.INCONSISTENT_SSVI],
)
def test_adversarial_ssvi_boards_cannot_hide_behind_average_rmse(case: AdversarialBoard) -> None:
    result = SSVICalibrator(SSVIConfig(seeds=(0,))).calibrate(adversarial_board(case))

    assert result.fit_quality.value == "poor"
    assert result.residual_summary is not None
    assert result.residual_summary.maximum_absolute_residual > 0.03
    assert result.numerical_warnings
