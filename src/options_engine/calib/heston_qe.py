"""Deprecated import path for the real Heston calibrator.

The former quadratic proxy was removed in 2.0. This module remains only so
existing imports resolve to the characteristic-function implementation.
"""

from .heston import HestonCalibrator, HestonConfig, HestonTenorResult

HestonQECalibrator = HestonCalibrator

__all__ = ["HestonCalibrator", "HestonConfig", "HestonQECalibrator", "HestonTenorResult"]
