from __future__ import annotations

import numpy as np
import pytest

from options_engine.core.statistical_inference import estimate_mean


def test_student_t_uses_independent_unit_degrees_of_freedom() -> None:
    estimate = estimate_mean(np.array([1.0, 2.0]))

    assert estimate.raw_estimate == pytest.approx(1.5)
    assert estimate.degrees_of_freedom == 1
    assert estimate.critical_value is not None and estimate.critical_value > 12.0
    assert estimate.method == "student_t"


def test_bound_projection_keeps_point_inside_reported_interval() -> None:
    estimate = estimate_mean(np.array([-2.0, -1.0]), lower_bound=0.0)

    assert estimate.bounded_estimate == 0.0
    assert estimate.confidence_interval is not None
    assert estimate.confidence_interval[0] <= estimate.bounded_estimate
    assert estimate.bounded_estimate <= estimate.confidence_interval[1]
    assert estimate.projection_applied is True
    assert estimate.interval_projection_applied is True


def test_single_unit_does_not_claim_zero_sampling_error() -> None:
    estimate = estimate_mean(np.array([3.0]), lower_bound=0.0)

    assert estimate.standard_error is None
    assert estimate.confidence_interval is None
    assert estimate.degrees_of_freedom is None
    assert estimate.method is None


@pytest.mark.parametrize(
    ("observations", "kwargs", "message"),
    [
        (np.array([]), {}, "between 1"),
        (np.array([[1.0]]), {}, "one-dimensional"),
        (np.array([np.nan]), {}, "finite"),
        (np.array([1.0]), {"lower_bound": 2.0, "upper_bound": 1.0}, "must not exceed"),
        (np.array([1.0]), {"confidence_level": 1.0}, "within"),
    ],
)
def test_inference_rejects_invalid_inputs(observations, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_mean(observations, **kwargs)
