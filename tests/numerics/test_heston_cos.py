"""Independent accuracy, convergence, and guardrail tests for Heston COS."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from options_engine.calib.heston import _heston_characteristic_function, heston_call_prices
from options_engine.calib.heston_cos import (
    HestonCOSConfig,
    _heston_log_return_cumulants,
    heston_cos_call_prices,
    heston_cos_call_prices_with_diagnostics,
    heston_cos_implied_volatilities,
)

REFERENCE = json.loads(
    (Path(__file__).resolve().parents[1] / "reference" / "quantlib_heston_v1.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["id"])
def test_cos_matches_committed_quantlib_analytic_heston(case: dict[str, float]) -> None:
    price, diagnostics = heston_cos_call_prices_with_diagnostics(
        case["forward"],
        [case["strike"]],
        case["time_to_expiry"],
        v0=case["v0"],
        theta=case["theta"],
        kappa=case["kappa"],
        vol_of_vol=case["vol_of_vol"],
        rho=case["rho"],
    )

    assert price[0] == pytest.approx(case["reference_price"], abs=3e-7)
    assert diagnostics.adaptive is True
    assert diagnostics.converged is True
    assert diagnostics.series_error_estimate is not None
    assert diagnostics.truncation_error_estimate is not None
    assert diagnostics.second_cumulant > 0.0


@pytest.mark.parametrize(
    ("tenor", "params"),
    [
        (0.25, (0.04, 0.05, 1.7, 0.45, -0.6)),
        (1.0, (0.09, 0.04, 0.8, 0.9, -0.3)),
        (3.0, (0.02, 0.12, 3.0, 0.7, 0.55)),
        (7.0, (0.16, 0.06, 0.4, 0.8, -0.75)),
    ],
)
def test_cos_agrees_with_gauss_laguerre_on_broad_parameter_grid(
    tenor: float,
    params: tuple[float, float, float, float, float],
) -> None:
    forward = 100.0
    strikes = forward * np.exp(np.linspace(-0.25, 0.25, 11))
    cos_prices = heston_cos_call_prices(
        forward,
        strikes,
        tenor,
        v0=params[0],
        theta=params[1],
        kappa=params[2],
        vol_of_vol=params[3],
        rho=params[4],
    )
    quadrature_prices = heston_call_prices(
        forward,
        strikes,
        tenor,
        v0=params[0],
        theta=params[1],
        kappa=params[2],
        vol_of_vol=params[3],
        rho=params[4],
    )

    # The long-dated/high-volatility corner reaches the independent 64-node
    # Gauss-Laguerre family's own ~2e-5 truncation floor; the tolerance is
    # fixed from that worst case, while the committed QuantLib test above is
    # the tighter absolute-accuracy criterion for COS itself.
    assert cos_prices == pytest.approx(quadrature_prices, rel=4e-6, abs=3e-5)


def test_closed_form_cumulants_match_characteristic_function_derivatives() -> None:
    params = {
        "v0": 0.07,
        "theta": 0.035,
        "kappa": 0.65,
        "vol_of_vol": 0.8,
        "rho": -0.72,
    }
    tenor = 2.3
    c1, c2 = _heston_log_return_cumulants(tenor, **params)
    step = 1e-3
    characteristic = _heston_characteristic_function(
        np.array([-step, step], dtype=np.complex128),
        forward=1.0,
        tenor=tenor,
        **params,
    )
    c1_fd = (np.angle(characteristic[1]) - np.angle(characteristic[0])) / (2.0 * step)
    c2_fd = -(math.log(abs(characteristic[1])) + math.log(abs(characteristic[0]))) / step**2

    assert c1 == pytest.approx(c1_fd, rel=2e-6, abs=1e-9)
    assert c2 == pytest.approx(c2_fd, rel=2e-6, abs=1e-9)


def test_second_cumulant_uses_exact_zero_mean_reversion_limit() -> None:
    tenor = 1.7
    v0 = 0.08
    vol_of_vol = 1.4
    rho = 0.65
    c1, c2 = _heston_log_return_cumulants(
        tenor,
        v0=v0,
        theta=0.2,
        kappa=1e-8,
        vol_of_vol=vol_of_vol,
        rho=rho,
    )
    sigma_t = vol_of_vol * tenor
    expected = v0 * tenor * (1.0 + sigma_t**2 / 12.0 - rho * sigma_t / 2.0)

    assert c1 == pytest.approx(-0.5 * v0 * tenor, rel=1e-7)
    assert c2 == pytest.approx(expected, rel=1e-12)
    assert c2 != pytest.approx(v0 * tenor, rel=1e-2)


def test_cos_preserves_bounds_shape_and_input_order() -> None:
    forward = 100.0
    strikes = np.array([140.0, 70.0, 100.0, 120.0, 85.0])
    prices = heston_cos_call_prices(
        forward,
        strikes,
        1.5,
        v0=0.05,
        theta=0.04,
        kappa=1.2,
        vol_of_vol=0.6,
        rho=-0.7,
    )
    order = np.argsort(strikes)
    sorted_prices = prices[order]
    sorted_strikes = strikes[order]
    slopes = np.diff(sorted_prices) / np.diff(sorted_strikes)

    assert prices.shape == strikes.shape
    assert np.all(prices >= np.maximum(forward - strikes, 0.0))
    assert np.all(prices <= forward)
    assert np.all(slopes >= -1.0 - 1e-9)
    assert np.all(slopes <= 1e-9)
    assert np.all(np.diff(slopes) >= -1e-9)


def test_extreme_supported_parameters_adapt_instead_of_returning_plausible_garbage() -> None:
    prices, diagnostics = heston_cos_call_prices_with_diagnostics(
        100.0,
        [70.0, 100.0, 140.0],
        1.0,
        v0=0.5,
        theta=0.1,
        kappa=0.05,
        vol_of_vol=2.0,
        rho=-0.7,
    )

    assert np.isfinite(prices).all()
    assert np.all(prices >= np.maximum(100.0 - np.array([70.0, 100.0, 140.0]), 0.0))
    assert diagnostics.terms_used > HestonCOSConfig().terms
    assert diagnostics.truncation_used > HestonCOSConfig().truncation
    assert diagnostics.converged is True


def test_adaptive_cos_rejects_an_unverifiable_work_budget() -> None:
    config = HestonCOSConfig(
        terms=64,
        truncation=8.0,
        adaptive=True,
        max_terms=64,
        max_truncation=8.0,
    )
    with pytest.raises(ValueError, match="convergence check"):
        heston_cos_call_prices(
            100.0,
            [80.0, 100.0, 120.0],
            1.0,
            v0=0.04,
            theta=0.05,
            kappa=1.7,
            vol_of_vol=0.45,
            rho=-0.6,
            config=config,
        )


def test_fixed_cost_cos_is_available_for_calibration_workloads() -> None:
    config = HestonCOSConfig(
        terms=256,
        truncation=12.0,
        adaptive=False,
        max_terms=256,
        max_truncation=12.0,
    )
    prices, diagnostics = heston_cos_call_prices_with_diagnostics(
        100.0,
        np.linspace(75.0, 125.0, 9),
        1.0,
        v0=0.04,
        theta=0.05,
        kappa=1.7,
        vol_of_vol=0.45,
        rho=-0.6,
        config=config,
    )
    vols = heston_cos_implied_volatilities(
        100.0,
        np.linspace(75.0, 125.0, 9),
        1.0,
        v0=0.04,
        theta=0.05,
        kappa=1.7,
        vol_of_vol=0.45,
        rho=-0.6,
        config=config,
    )

    assert np.isfinite(prices).all()
    assert np.isfinite(vols).all()
    assert diagnostics.adaptive is False
    assert diagnostics.converged is None
    assert diagnostics.series_error_estimate is None
    assert diagnostics.truncation_error_estimate is None


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({"terms": True}, TypeError),
        ({"terms": 16}, ValueError),
        ({"max_terms": 64, "terms": 128}, ValueError),
        ({"truncation": 3.0}, ValueError),
        ({"adaptive": 1}, TypeError),
        ({"absolute_tolerance": math.nan}, ValueError),
        ({"max_truncation": 8.0, "truncation": 10.0}, ValueError),
    ],
)
def test_cos_configuration_rejects_unsafe_values(
    kwargs: dict[str, object], exception: type[Exception]
) -> None:
    with pytest.raises(exception):
        HestonCOSConfig(**kwargs)  # type: ignore[arg-type]


def test_cos_reuses_heston_input_guardrails() -> None:
    with pytest.raises(TypeError, match="real numbers"):
        heston_cos_call_prices(
            100.0,
            ["100"],
            1.0,
            v0=0.04,
            theta=0.05,
            kappa=1.7,
            vol_of_vol=0.45,
            rho=-0.6,
        )
