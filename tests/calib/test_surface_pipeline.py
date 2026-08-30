"""Integration tests for the volatility surface calibration pipeline."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from options_engine.calib import (
    BoardCleaner,
    HestonCalibrator,
    SABRCalibrator,
    SurfaceBuilder,
    heston_implied_volatilities,
)
from options_engine.calib.heston import HestonConfig
from options_engine.calib.heston_qe import HestonQECalibrator
from options_engine.calib.sabr import SABRConfig, hagan_implied_volatility
from options_engine.calib.select import SurfaceBuildResult


def _synthetic_board() -> list[dict[str, float]]:
    tenors = [0.25, 0.5, 1.0]
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    forward = 100.0
    base_params = {
        0.25: {"alpha": 0.25, "beta": 0.5, "rho": -0.2, "nu": 0.6},
        0.5: {"alpha": 0.22, "beta": 0.5, "rho": -0.1, "nu": 0.5},
        1.0: {"alpha": 0.2, "beta": 0.5, "rho": 0.0, "nu": 0.45},
    }
    board: list[dict[str, float]] = []
    now = datetime.now(UTC)
    for tenor in tenors:
        params = base_params[tenor]
        vols = hagan_implied_volatility(
            forward,
            strikes,
            tenor,
            alpha=params["alpha"],
            beta=params["beta"],
            rho=params["rho"],
            nu=params["nu"],
        )
        noise = np.linspace(-0.002, 0.002, len(strikes))
        for strike, vol, epsilon in zip(strikes, vols, noise, strict=False):
            board.extend(
                {
                    "tenor": tenor,
                    "strike": float(strike),
                    "mid_iv": float(vol + epsilon),
                    "forward": forward,
                    "option_type": opt_type,
                    "timestamp": now - timedelta(seconds=10),
                }
                for opt_type in ("CALL", "PUT")
            )
    return board


def _arbitrage_board() -> list[dict[str, float]]:
    board = _synthetic_board()
    for quote in board:
        if quote["tenor"] == 0.25 and quote["strike"] == 100.0:
            if quote["option_type"] == "CALL":
                quote["mid_iv"] *= 1.5
            else:
                quote["mid_iv"] *= 0.2
    return board


def _heston_friendly_board() -> list[dict[str, float]]:
    tenors = [0.5, 1.0]
    strikes = np.linspace(70.0, 130.0, 9)
    forward = 100.0
    board: list[dict[str, float]] = []
    params = {
        "v0": 0.04,
        "theta": 0.05,
        "kappa": 1.7,
        "vol_of_vol": 0.45,
        "rho": -0.6,
    }
    now = datetime.now(UTC)
    for tenor in tenors:
        vols = heston_implied_volatilities(forward, strikes, tenor, **params)
        for strike, vol in zip(strikes, vols, strict=False):
            board.extend(
                {
                    "tenor": tenor,
                    "strike": float(strike),
                    "mid_iv": float(vol),
                    "forward": forward,
                    "option_type": opt_type,
                    "timestamp": now - timedelta(seconds=5),
                }
                for opt_type in ("CALL", "PUT")
            )
    return board


@pytest.fixture()
def builder() -> SurfaceBuilder:
    return SurfaceBuilder()


def test_golden_board_calibration(builder: SurfaceBuilder) -> None:
    board = _synthetic_board()
    result = builder.build(board, seed=7)

    assert isinstance(result, SurfaceBuildResult)
    assert result.validation.is_ok
    assert len(result.selections) == 3
    for selection in result.selections:
        assert selection.model == "sabr"
        assert selection.rmse <= 0.0075


def test_arbitrage_detection_blocks(builder: SurfaceBuilder) -> None:
    board = _arbitrage_board()
    with pytest.raises(ValueError) as excinfo:
        builder.build(board, seed=1)
    assert "parity" in str(excinfo.value).lower()


def test_pipeline_determinism(builder: SurfaceBuilder) -> None:
    board = _synthetic_board()
    first = builder.build(board, seed=13)
    second = builder.build(board, seed=13)
    assert first.surface_id == second.surface_id
    assert [sel.params for sel in first.selections] == [sel.params for sel in second.selections]


def test_heston_cross_check_selection() -> None:
    board = _heston_friendly_board()
    cleaner = BoardCleaner()
    sabr = SABRCalibrator(SABRConfig(beta=0.3))
    heston = HestonCalibrator(HestonConfig(tenors=[0.5], seeds=(0,), max_iterations=150))
    builder = SurfaceBuilder(cleaner=cleaner, sabr=sabr, heston=heston)
    result = builder.build(board, seed=3)

    assert any(selection.model == "heston" for selection in result.selections)
    heston_rmses = {sel.tenor: sel.rmse for sel in result.selections if sel.model == "heston"}
    sabr_rmses = {sel.tenor: sel.rmse for sel in result.selections if sel.model == "sabr"}
    for tenor, rmse in heston_rmses.items():
        assert rmse <= sabr_rmses.get(tenor, float("inf"))


def test_deprecated_heston_qe_name_points_to_real_heston_calibrator() -> None:
    assert HestonQECalibrator is HestonCalibrator
