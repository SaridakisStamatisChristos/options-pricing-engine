"""Tests for the SABR volatility surface calibration workflow."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from options_engine.core.vol_surface_calibration import (
    SABRCalibrator,
    clean_option_board,
    hagan_implied_volatility,
)


def _build_sample_board(now: datetime) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    tenors = np.array([0.25, 0.5, 1.0], dtype=float)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0], dtype=float)

    alpha = 0.25
    beta = 0.5
    rho = -0.2
    nu = 0.4

    rows = []
    for tenor in tenors:
        forward = 100.0 * np.exp(0.01 * tenor)
        base_vols = hagan_implied_volatility(
            forward=np.full_like(strikes, forward),
            strike=strikes,
            expiry=float(tenor),
            alpha=alpha,
            beta=beta,
            rho=rho,
            nu=nu,
        )

        noise = rng.normal(scale=0.002, size=base_vols.shape)
        mid_vols = base_vols + noise

        for strike, market_vol in zip(strikes, mid_vols):
            rows.append(
                {
                    "tenor": tenor,
                    "strike": strike,
                    "forward": forward,
                    "bid_iv": market_vol - 0.002,
                    "ask_iv": market_vol + 0.002,
                    "mid_iv": market_vol,
                    "timestamp": now - timedelta(seconds=30),
                }
            )

    # Add crossed market quote
    rows.append(
        {
            "tenor": 0.5,
            "strike": 125.0,
            "forward": 100.0,
            "bid_iv": 0.30,
            "ask_iv": 0.29,
            "mid_iv": 0.295,
            "timestamp": now - timedelta(seconds=30),
        }
    )

    # Add stale quote
    rows.append(
        {
            "tenor": 0.5,
            "strike": 95.0,
            "forward": 100.0,
            "bid_iv": 0.25,
            "ask_iv": 0.26,
            "mid_iv": 0.255,
            "timestamp": now - timedelta(minutes=30),
        }
    )

    # Add outlier
    rows.append(
        {
            "tenor": 1.0,
            "strike": 150.0,
            "forward": 100.0,
            "bid_iv": 1.20,
            "ask_iv": 1.21,
            "mid_iv": 1.205,
            "timestamp": now - timedelta(seconds=30),
        }
    )

    return pd.DataFrame(rows)


def test_clean_option_board_filters_quotes() -> None:
    now = datetime.now(UTC)
    board = _build_sample_board(now)

    result = clean_option_board(
        board,
        now=now,
        max_age_seconds=60.0,
        mad_threshold=3.0,
    )

    assert result.report.total_quotes == len(board)
    assert result.report.dropped_crossed == 1
    assert result.report.dropped_stale == 1
    assert result.report.dropped_outlier == 1
    assert result.report.retained_quotes == len(result.data)
    assert not result.data.isna().any().any()


def test_full_sabr_calibration_pipeline() -> None:
    now = datetime.now(UTC)
    board = _build_sample_board(now)

    calibrator = SABRCalibrator(beta=0.5, max_age_seconds=60.0, mad_threshold=3.0)
    result = calibrator.calibrate(board, now=now)

    assert result.clean_board.report.retained_quotes == len(result.clean_board.data)
    assert result.arbitrage.is_arbitrage_free
    assert result.regime == "sabr"

    rmses = [tenor_result.rmse for tenor_result in result.tenor_results]
    assert rmses and max(rmses) < 0.01

    qc = result.qc_report
    assert qc["board"]["retained_quotes"] == len(result.fitted_surface)
    assert qc["arbitrage"]["is_arbitrage_free"] is True
