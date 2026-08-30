"""Utility helpers exposed by :mod:`options_engine`."""

from .numerics import (
    apply_global_clamps,
    deep_itm_policy,
    deep_otm_upper_bound,
    enforce_precision_policy,
    laguerre_basis3,
    numerics_policy_hash,
    stable_regression,
)

__all__ = [
    "apply_global_clamps",
    "deep_itm_policy",
    "deep_otm_upper_bound",
    "enforce_precision_policy",
    "laguerre_basis3",
    "numerics_policy_hash",
    "stable_regression",
]
