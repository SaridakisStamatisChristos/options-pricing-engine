from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from options_engine.calib.boards import CleanBoard
from options_engine.calib.svi import (
    SSVICalibrator,
    SSVIConfig,
    SSVISurface,
    SVIParameters,
    raw_svi_total_variance,
    ssvi_total_variance,
    validate_svi_slice,
)


def _ssvi_board(*, with_spreads: bool = False) -> CleanBoard:
    rows: list[dict[str, float]] = []
    rho, eta, power = -0.35, 1.1, 0.25
    for tenor, theta in ((0.25, 0.012), (0.5, 0.025), (1.0, 0.055)):
        forward = 100.0
        strikes = np.linspace(70.0, 130.0, 13)
        log_moneyness = np.log(strikes / forward)
        vols = np.sqrt(
            ssvi_total_variance(
                log_moneyness,
                theta,
                rho=rho,
                eta=eta,
                power=power,
            )
            / tenor
        )
        for strike, vol in zip(strikes, vols, strict=True):
            row = {
                "tenor": tenor,
                "strike": float(strike),
                "mid_iv": float(vol),
                "forward": forward,
            }
            if with_spreads:
                row.update({"bid_iv": float(vol - 0.005), "ask_iv": float(vol + 0.005)})
            rows.append(row)
    return CleanBoard(pd.DataFrame(rows), {})


def test_ssvi_recovers_synthetic_arbitrage_free_surface() -> None:
    result = SSVICalibrator(SSVIConfig(seeds=(0, 1), max_iterations=500)).calibrate(_ssvi_board())

    assert result.rmse < 1e-8
    assert result.surface.rho == pytest.approx(-0.35, abs=1e-6)
    assert result.surface.eta == pytest.approx(1.1, abs=1e-6)
    assert result.surface.power == pytest.approx(0.25, abs=1e-6)
    assert result.minimum_density_factor >= -1e-10
    assert result.maximum_calendar_decrease == 0.0
    assert result.maximum_wing_slope < 2.0
    assert result.atm_projection_applied is False


def test_ssvi_interpolation_preserves_calendar_order() -> None:
    result = SSVICalibrator(SSVIConfig(seeds=(0,))).calibrate(_ssvi_board(with_spreads=True))
    surface = result.surface
    log_moneyness = np.linspace(-1.0, 1.0, 101)

    short = surface.total_variance(log_moneyness, 0.25)
    middle = surface.total_variance(log_moneyness, 0.75)
    long = surface.total_variance(log_moneyness, 1.0)

    assert np.all(middle >= short - 1e-12)
    assert np.all(long >= middle - 1e-12)
    vols = surface.implied_volatility(
        np.array([80.0, 100.0, 120.0]),
        0.75,
        forward=100.0,
    )
    assert np.isfinite(vols).all()
    assert np.all(vols > 0.0)


def test_raw_svi_density_diagnostic_detects_butterfly_arbitrage() -> None:
    safe = SVIParameters(a=0.02, b=0.2, rho=-0.3, m=0.0, sigma=0.2)
    unsafe = SVIParameters(
        a=0.14830095944960534,
        b=0.6496433548627699,
        rho=0.8587726906484072,
        m=0.17699133467530692,
        sigma=0.11653644258288014,
    )

    assert validate_svi_slice(safe).butterfly_free is True
    unsafe_diagnostics = validate_svi_slice(unsafe)
    assert unsafe_diagnostics.butterfly_free is False
    assert unsafe_diagnostics.minimum_density_factor < 0.0
    variance = raw_svi_total_variance(np.array([-1.0, 0.0, 1.0]), safe)
    assert np.all(variance > 0.0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: SVIParameters(0.0, -0.1, 0.0, 0.0, 0.2), "non-negative"),
        (lambda: SVIParameters(0.0, 0.1, 1.0, 0.0, 0.2), "rho"),
        (lambda: SVIParameters(0.0, 0.1, 0.0, 0.0, 0.0), "sigma"),
        (lambda: SSVIConfig(seeds=()), "seeds"),
        (lambda: SSVIConfig(weighting="vega"), "weighting"),
        (lambda: SSVIConfig(validation_points=102), "odd"),
        (lambda: SSVIConfig(tolerance=math.nan), "finite"),
        (
            lambda: SSVISurface((1.0, 0.5), (0.04, 0.05), -0.3, 1.0, 0.25),
            "increasing",
        ),
        (
            lambda: SSVISurface((0.5, 1.0), (0.04, 0.08), 0.9, 100.0, 0.0),
            "static-arbitrage",
        ),
    ],
)
def test_svi_configuration_guardrails(factory, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_ssvi_rejects_insufficient_or_malformed_boards() -> None:
    calibrator = SSVICalibrator(SSVIConfig(seeds=(0,)))
    one_tenor = _ssvi_board().quotes.query("tenor == 0.25")
    with pytest.raises(ValueError, match="at least 2 tenors"):
        calibrator.calibrate(CleanBoard(one_tenor, {}))
    malformed = _ssvi_board().quotes.copy()
    malformed.loc[0, "strike"] = 0.0
    with pytest.raises(ValueError, match="domain"):
        calibrator.calibrate(CleanBoard(malformed, {}))
    no_spreads = CleanBoard(_ssvi_board().quotes, {})
    with pytest.raises(ValueError, match="requires bid_iv"):
        SSVICalibrator(SSVIConfig(seeds=(0,), weighting="bid_ask")).calibrate(no_spreads)
