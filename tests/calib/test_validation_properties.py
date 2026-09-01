"""Bounded property tests for the model-risk validation primitives."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from options_engine.calib.validation import (
    HoldoutPolicy,
    deterministic_holdout_mask,
    residual_diagnostics,
    serializable,
)

FINITE_VALUES = st.lists(
    st.floats(min_value=1e-3, max_value=1e4, allow_nan=False, allow_infinity=False),
    min_size=6,
    max_size=40,
)


@settings(max_examples=30, deadline=None)
@given(FINITE_VALUES, st.sampled_from(list(HoldoutPolicy)), st.floats(0.0, 0.5))
def test_holdout_masks_are_deterministic_and_never_underidentify(
    values: list[float], policy: HoldoutPolicy, fraction: float
) -> None:
    first = deterministic_holdout_mask(values, policy, fraction=fraction, minimum_training=5)
    second = deterministic_holdout_mask(values, policy, fraction=fraction, minimum_training=5)

    assert np.array_equal(first, second)
    assert first.dtype == np.bool_
    assert first.shape == (len(values),)
    assert np.count_nonzero(~first) >= 5


@settings(max_examples=25, deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=25,
    )
)
def test_residual_evidence_is_finite_json_and_weights_are_normalized(vols: list[float]) -> None:
    size = len(vols)
    market = np.asarray(vols)
    fitted = market + np.linspace(-1e-3, 1e-3, size)
    weights = np.geomspace(0.1, 10.0, size)
    rows, summary, concentration = residual_diagnostics(
        tenors=np.ones(size),
        strikes=np.linspace(80.0, 120.0, size),
        forwards=np.full(size, 100.0),
        market=market,
        fitted=fitted,
        weights=weights,
    )

    payload = serializable({"rows": rows, "summary": summary, "weights": concentration})
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert all(np.isfinite(row.weighted_residual) for row in rows)
    assert 1.0 <= concentration.effective_sample_size <= size
    assert concentration.minimum_normalized_weight > 0.0
    assert concentration.maximum_normalized_weight <= 1.0


@settings(max_examples=15, deadline=None)
@given(st.sampled_from([float("nan"), float("inf"), float("-inf")]))
def test_nonfinite_residual_inputs_are_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        residual_diagnostics(
            tenors=[1.0, 1.0],
            strikes=[90.0, 110.0],
            forwards=[100.0, 100.0],
            market=[0.2, value],
            fitted=[0.2, 0.2],
        )


@settings(max_examples=20, deadline=None)
@given(st.floats(max_value=0.0, allow_nan=False, allow_infinity=False))
def test_nonpositive_weights_are_rejected(weight: float) -> None:
    with pytest.raises(ValueError, match="weights"):
        residual_diagnostics(
            tenors=[1.0, 1.0],
            strikes=[90.0, 110.0],
            forwards=[100.0, 100.0],
            market=[0.2, 0.2],
            fitted=[0.2, 0.2],
            weights=[1.0, weight],
        )
