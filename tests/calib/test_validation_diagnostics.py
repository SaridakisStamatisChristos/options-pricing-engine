"""Fast deterministic tests for the shared calibration-validation vocabulary."""

import numpy as np
import pytest

from options_engine.calib import (
    HoldoutPolicy,
    conditioning_from_jacobian,
    deterministic_holdout_mask,
    residual_diagnostics,
)


@pytest.mark.parametrize("policy", list(HoldoutPolicy))
def test_holdouts_are_deterministic_and_preserve_training_identification(policy: str) -> None:
    strikes = np.array([110.0, 80.0, 100.0, 90.0, 120.0, 105.0, 95.0])
    first = deterministic_holdout_mask(strikes, policy, minimum_training=4)
    second = deterministic_holdout_mask(strikes, policy, minimum_training=4)
    assert np.array_equal(first, second)
    assert np.sum(~first) >= 4


def test_residual_report_keeps_wing_error_and_weight_concentration() -> None:
    rows, summary, weights = residual_diagnostics(
        tenors=np.ones(5),
        strikes=np.array([80.0, 90.0, 100.0, 110.0, 120.0]),
        forwards=np.full(5, 100.0),
        market=np.full(5, 0.2),
        fitted=np.array([0.3, 0.2, 0.2, 0.2, 0.21]),
        weights=np.array([1.0, 1.0, 10.0, 1.0, 1.0]),
        holdout=np.array([True, False, False, False, False]),
    )
    assert rows[0].is_holdout
    assert summary.maximum_absolute_residual == pytest.approx(0.1)
    assert summary.holdout_rmse == pytest.approx(0.1)
    assert weights.effective_sample_size < 3.0


def test_singular_conditioning_is_reported_not_fabricated() -> None:
    diagnostics = conditioning_from_jacobian(np.ones((6, 3)))
    assert diagnostics.weakly_identified
    assert diagnostics.effective_rank == 1
    assert diagnostics.parameter_correlation is None
