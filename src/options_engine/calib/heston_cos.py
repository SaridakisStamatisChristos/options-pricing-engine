"""Independent Fang-Oosterlee COS pricing for European Heston calls.

The implementation expands the terminal log-return density in a cosine basis
on a cumulant-sized interval.  Calls are evaluated from the out-of-the-money
payoff whenever that is numerically safe and otherwise through the bounded put
payoff plus put-call parity.  This module shares the Heston characteristic
function with the Gauss-Laguerre reference implementation, but it shares
neither its inversion formula nor its quadrature nodes.

Reference
---------
Fang, F. and Oosterlee, C. W. (2008), "A Novel Pricing Method for European
Options Based on Fourier-Cosine Series Expansions", SIAM Journal on Scientific
Computing 31(2), 826-848. https://doi.org/10.1137/080718061
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .heston import (
    _heston_characteristic_function,
    _implied_volatility,
    _validated_heston_call_prices,
    _validated_heston_inputs,
)

_MIN_TERMS = 32
_MAX_TERMS = 16_384
_TAIL_STEP = 2.0
_DIRECT_CALL_MAX_LOG_BOUND = 20.0
_MAX_MATRIX_ELEMENTS = 1_000_000


@dataclass(frozen=True, slots=True)
class HestonCOSConfig:
    """Accuracy and work controls for the COS pricing family.

    ``adaptive=True`` verifies both cosine-series resolution and truncation
    stability.  Calibration can select ``adaptive=False`` for a fixed-cost
    objective after choosing ``terms`` and ``truncation`` from a convergence
    study.  A non-adaptive result is still checked against no-arbitrage bounds.
    """

    terms: int = 256
    truncation: float = 12.0
    adaptive: bool = True
    max_terms: int = 4096
    max_truncation: float = 24.0
    absolute_tolerance: float = 1e-9
    relative_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        for integer_name, integer_value, low, high in (
            ("terms", self.terms, _MIN_TERMS, _MAX_TERMS),
            ("max_terms", self.max_terms, _MIN_TERMS, _MAX_TERMS),
        ):
            if isinstance(integer_value, bool) or not isinstance(integer_value, Integral):
                raise TypeError(f"{integer_name} must be an integer")
            if not low <= int(integer_value) <= high:
                raise ValueError(f"{integer_name} must be within [{low}, {high}]")
            object.__setattr__(self, integer_name, int(integer_value))
        if self.max_terms < self.terms:
            raise ValueError("max_terms must be greater than or equal to terms")
        if not isinstance(self.adaptive, bool):
            raise TypeError("adaptive must be a boolean")
        for real_name, real_value, real_low, real_high in (
            ("truncation", self.truncation, 4.0, 40.0),
            ("max_truncation", self.max_truncation, 4.0, 40.0),
            ("absolute_tolerance", self.absolute_tolerance, 1e-14, 1e-2),
            ("relative_tolerance", self.relative_tolerance, 0.0, 1e-2),
        ):
            if isinstance(real_value, bool) or not isinstance(real_value, Real):
                raise TypeError(f"{real_name} must be a real number")
            normalised = float(real_value)
            if not math.isfinite(normalised) or not real_low <= normalised <= real_high:
                raise ValueError(f"{real_name} must be within [{real_low:g}, {real_high:g}]")
            object.__setattr__(self, real_name, normalised)
        if self.max_truncation < self.truncation:
            raise ValueError("max_truncation must be greater than or equal to truncation")


@dataclass(frozen=True, slots=True)
class HestonCOSDiagnostics:
    """Numerical evidence attached to an adaptive COS solve."""

    terms_used: int
    truncation_used: float
    interval_lower: float
    interval_upper: float
    first_cumulant: float
    second_cumulant: float
    series_error_estimate: float | None
    truncation_error_estimate: float | None
    adaptive: bool
    converged: bool | None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        return {
            "terms_used": self.terms_used,
            "truncation_used": self.truncation_used,
            "interval_lower": self.interval_lower,
            "interval_upper": self.interval_upper,
            "first_cumulant": self.first_cumulant,
            "second_cumulant": self.second_cumulant,
            "series_error_estimate": self.series_error_estimate,
            "truncation_error_estimate": self.truncation_error_estimate,
            "adaptive": self.adaptive,
            "converged": self.converged,
        }


def _heston_log_return_cumulants(
    tenor: float,
    *,
    v0: float,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho: float,
) -> tuple[float, float]:
    """Return the first two cumulants of ``log(S_T / F)``.

    The second cumulant is the closed-form derivative of the same
    characteristic function used by both pricing families.  Long-double
    intermediates and the exact zero-mean-reversion limit avoid cancellation
    when ``kappa*T`` is tiny.  The formula was independently checked against
    derivatives of ``log(phi)`` in the numerical tests.
    """

    kt = kappa * tenor
    one_minus_exp = -math.expm1(-kt)
    if kt < 1e-5:
        decay_integral = tenor * (1.0 - kt / 2.0 + kt * kt / 6.0 - kt**3 / 24.0)
    else:
        decay_integral = one_minus_exp / kappa
    c1 = 0.5 * (theta - v0) * decay_integral - 0.5 * theta * tenor
    expected_integrated_variance = theta * tenor + (v0 - theta) * decay_integral

    if kt < 1e-5:
        # As kappa -> 0, variance becomes a driftless square-root diffusion.
        # Writing the correlated log return in terms of V_T and integrated
        # variance gives this exact limit for Var[log(S_T/F)].  Retaining only
        # E[integrated variance] here would miss the vol-of-vol and leverage
        # contributions precisely where the closed form loses digits.
        sigma_t = vol_of_vol * tenor
        c2 = v0 * tenor * (1.0 + sigma_t * sigma_t / 12.0 - rho * sigma_t / 2.0)
    else:
        ld = np.longdouble
        t = ld(tenor)
        variance0 = ld(v0)
        long_variance = ld(theta)
        mean_reversion = ld(kappa)
        sigma = ld(vol_of_vol)
        correlation = ld(rho)
        exponential = np.exp(-mean_reversion * t)
        kappa2 = mean_reversion * mean_reversion
        kappa3 = kappa2 * mean_reversion
        sigma2 = sigma * sigma
        rho_sigma = correlation * sigma
        n0 = long_variance * (
            8 * kappa3 * t
            - 8 * kappa2 * rho_sigma * t
            - 8 * kappa2
            + 16 * mean_reversion * rho_sigma
            + 2 * mean_reversion * sigma2 * t
            - 5 * sigma2
        ) + variance0 * 2 * (4 * kappa2 - 4 * mean_reversion * rho_sigma + sigma2)
        n1 = long_variance * -4 * (
            2 * kappa2 * rho_sigma * t
            - 2 * kappa2
            + 4 * mean_reversion * rho_sigma
            - mean_reversion * sigma2 * t
            - sigma2
        ) + variance0 * 4 * mean_reversion * (
            2 * mean_reversion * rho_sigma * t - 2 * mean_reversion + 2 * rho_sigma - sigma2 * t
        )
        n2 = sigma2 * (long_variance - 2 * variance0)
        c2 = float((n0 + n1 * exponential + n2 * exponential * exponential) / (8 * kappa3))

    floor = max(1e-18, expected_integrated_variance * 1e-12)
    if not math.isfinite(c1) or not math.isfinite(c2) or c2 <= floor:
        raise ValueError("Heston COS cumulants are non-finite or degenerate")
    return float(c1), float(c2)


def _chi_psi(
    frequencies: NDArray[np.float64],
    basis_lower: NDArray[np.float64],
    integration_lower: NDArray[np.float64],
    integration_upper: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fang-Oosterlee payoff integrals on strike-relative intervals."""

    lower_phases = frequencies[None, :] * (integration_lower - basis_lower)
    upper_phases = frequencies[None, :] * (integration_upper - basis_lower)
    exp_lower = np.exp(np.maximum(integration_lower, -745.0))
    exp_upper = np.exp(integration_upper)
    chi = (
        exp_upper * (np.cos(upper_phases) + frequencies[None, :] * np.sin(upper_phases))
        - exp_lower * (np.cos(lower_phases) + frequencies[None, :] * np.sin(lower_phases))
    ) / (1.0 + frequencies[None, :] ** 2)
    psi = np.empty_like(chi)
    psi[:, 0] = (integration_upper - integration_lower)[:, 0]
    psi[:, 1:] = (np.sin(upper_phases[:, 1:]) - np.sin(lower_phases[:, 1:])) / frequencies[None, 1:]
    return chi, psi


def _cos_once(
    forward: float,
    strikes: NDArray[np.float64],
    tenor: float,
    *,
    v0: float,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho: float,
    terms: int,
    truncation: float,
    cumulants: tuple[float, float],
) -> tuple[NDArray[np.float64], float, float]:
    c1, c2 = cumulants
    half_width = truncation * math.sqrt(c2)
    if not math.isfinite(half_width) or half_width <= 1e-12:
        raise ValueError("Heston COS truncation interval is degenerate")
    interval_lower = c1 - half_width
    interval_upper = c1 + half_width
    width = interval_upper - interval_lower
    frequencies = np.arange(terms, dtype=float) * math.pi / width
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        characteristic = _heston_characteristic_function(
            frequencies.astype(np.complex128),
            forward=1.0,
            tenor=tenor,
            v0=v0,
            theta=theta,
            kappa=kappa,
            vol_of_vol=vol_of_vol,
            rho=rho,
        )
    if not np.isfinite(characteristic).all() or abs(characteristic[0] - 1.0) > 1e-10:
        raise ValueError("Heston COS characteristic function is numerically unstable")
    density_coefficients = np.real(characteristic * np.exp(-1j * frequencies * interval_lower))
    density_coefficients[0] *= 0.5

    prices = np.empty_like(strikes)
    log_moneyness = np.log(forward / strikes)
    chunk_size = max(1, min(strikes.size, _MAX_MATRIX_ELEMENTS // terms))
    direct_calls = interval_upper <= _DIRECT_CALL_MAX_LOG_BOUND
    for start in range(0, strikes.size, chunk_size):
        stop = min(start + chunk_size, strikes.size)
        chunk_strikes = strikes[start:stop]
        # x = log(S_T / K).  Shifting both the characteristic function and
        # interval by log(F/K) cancels from phi(u) exp(-iua), so the density
        # coefficients above are shared by every strike.
        lower = (interval_lower + log_moneyness[start:stop])[:, None]
        upper = (interval_upper + log_moneyness[start:stop])[:, None]
        chunk_prices = np.empty(stop - start, dtype=float)

        call_mask = direct_calls & (chunk_strikes >= forward) & (upper[:, 0] > 0.0)
        if np.any(call_mask):
            call_lower = np.maximum(lower[call_mask], 0.0)
            call_upper = upper[call_mask]
            chi, psi = _chi_psi(
                frequencies,
                lower[call_mask],
                call_lower,
                call_upper,
            )
            payoff_coefficients = (2.0 / width) * chunk_strikes[call_mask, None] * (chi - psi)
            chunk_prices[call_mask] = payoff_coefficients @ density_coefficients

        parity_mask = ~call_mask
        if np.any(parity_mask):
            put_lower = lower[parity_mask]
            put_upper = np.minimum(upper[parity_mask], 0.0)
            empty_put = put_lower[:, 0] >= 0.0
            put_values = np.zeros(np.count_nonzero(parity_mask), dtype=float)
            if np.any(~empty_put):
                chi, psi = _chi_psi(
                    frequencies,
                    put_lower[~empty_put],
                    put_lower[~empty_put],
                    put_upper[~empty_put],
                )
                selected_strikes = chunk_strikes[parity_mask][~empty_put]
                payoff_coefficients = (2.0 / width) * selected_strikes[:, None] * (psi - chi)
                put_values[~empty_put] = payoff_coefficients @ density_coefficients
            parity_strikes = chunk_strikes[parity_mask]
            chunk_prices[parity_mask] = put_values + forward - parity_strikes

        prices[start:stop] = chunk_prices
    return prices, interval_lower, interval_upper


def _error_threshold(
    forward: float,
    prices: NDArray[np.float64],
    config: HestonCOSConfig,
) -> float:
    scale = max(1.0, forward, float(np.max(np.abs(prices))))
    return config.absolute_tolerance * max(1.0, forward) + config.relative_tolerance * scale


def _max_error(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    return float(np.max(np.abs(first - second)))


def heston_cos_call_prices_with_diagnostics(
    forward: float,
    strikes: ArrayLike,
    tenor: float,
    *,
    v0: float,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho: float,
    config: HestonCOSConfig | None = None,
) -> tuple[NDArray[np.float64], HestonCOSDiagnostics]:
    """Return undiscounted COS call prices and convergence diagnostics."""

    if config is not None and not isinstance(config, HestonCOSConfig):
        raise TypeError("config must be a HestonCOSConfig")
    resolved = config if config is not None else HestonCOSConfig()
    strikes_array = _validated_heston_inputs(
        forward,
        strikes,
        tenor,
        v0=v0,
        theta=theta,
        kappa=kappa,
        vol_of_vol=vol_of_vol,
        rho=rho,
    )
    cumulants = _heston_log_return_cumulants(
        tenor,
        v0=v0,
        theta=theta,
        kappa=kappa,
        vol_of_vol=vol_of_vol,
        rho=rho,
    )

    terms = resolved.terms
    truncation = resolved.truncation
    prices, lower, upper = _cos_once(
        forward,
        strikes_array,
        tenor,
        v0=v0,
        theta=theta,
        kappa=kappa,
        vol_of_vol=vol_of_vol,
        rho=rho,
        terms=terms,
        truncation=truncation,
        cumulants=cumulants,
    )
    series_error: float | None = None
    truncation_error: float | None = None
    converged: bool | None = False if resolved.adaptive else None

    while resolved.adaptive:
        half_terms = max(_MIN_TERMS, terms // 2)
        half_prices, _, _ = _cos_once(
            forward,
            strikes_array,
            tenor,
            v0=v0,
            theta=theta,
            kappa=kappa,
            vol_of_vol=vol_of_vol,
            rho=rho,
            terms=half_terms,
            truncation=truncation,
            cumulants=cumulants,
        )
        series_error = _max_error(prices, half_prices)
        threshold = _error_threshold(forward, prices, resolved)
        if series_error > threshold:
            if terms >= resolved.max_terms:
                break
            terms = min(resolved.max_terms, terms * 2)
            prices, lower, upper = _cos_once(
                forward,
                strikes_array,
                tenor,
                v0=v0,
                theta=theta,
                kappa=kappa,
                vol_of_vol=vol_of_vol,
                rho=rho,
                terms=terms,
                truncation=truncation,
                cumulants=cumulants,
            )
            continue

        if truncation >= resolved.max_truncation:
            converged = truncation_error is not None and truncation_error <= _error_threshold(
                forward, prices, resolved
            )
            break
        wider = min(resolved.max_truncation, truncation + _TAIL_STEP)
        scaled_terms = math.ceil(terms * wider / truncation / 32.0) * 32
        scaled_terms = min(resolved.max_terms, max(terms, scaled_terms))
        wider_prices, wider_lower, wider_upper = _cos_once(
            forward,
            strikes_array,
            tenor,
            v0=v0,
            theta=theta,
            kappa=kappa,
            vol_of_vol=vol_of_vol,
            rho=rho,
            terms=scaled_terms,
            truncation=wider,
            cumulants=cumulants,
        )
        truncation_error = _max_error(wider_prices, prices)
        prices = wider_prices
        lower = wider_lower
        upper = wider_upper
        terms = scaled_terms
        truncation = wider
        if truncation_error <= _error_threshold(forward, prices, resolved):
            # The next loop verifies that the widened interval retained
            # sufficient frequency resolution before declaring convergence.
            half_wider, _, _ = _cos_once(
                forward,
                strikes_array,
                tenor,
                v0=v0,
                theta=theta,
                kappa=kappa,
                vol_of_vol=vol_of_vol,
                rho=rho,
                terms=max(_MIN_TERMS, terms // 2),
                truncation=truncation,
                cumulants=cumulants,
            )
            series_error = _max_error(prices, half_wider)
            if series_error <= _error_threshold(forward, prices, resolved):
                converged = True
                break

    if resolved.adaptive and not converged:
        raise ValueError(
            "Heston COS failed its adaptive series/truncation convergence check "
            f"within {resolved.max_terms} terms and L={resolved.max_truncation:g}"
        )

    validated = _validated_heston_call_prices(
        forward,
        strikes_array,
        prices,
        method="COS expansion",
    )
    diagnostics = HestonCOSDiagnostics(
        terms_used=terms,
        truncation_used=truncation,
        interval_lower=lower,
        interval_upper=upper,
        first_cumulant=cumulants[0],
        second_cumulant=cumulants[1],
        series_error_estimate=series_error,
        truncation_error_estimate=truncation_error,
        adaptive=resolved.adaptive,
        converged=converged,
    )
    return validated, diagnostics


def heston_cos_call_prices(
    forward: float,
    strikes: ArrayLike,
    tenor: float,
    *,
    v0: float,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho: float,
    config: HestonCOSConfig | None = None,
) -> NDArray[np.float64]:
    """Return undiscounted European Heston calls using the COS method."""

    prices, _ = heston_cos_call_prices_with_diagnostics(
        forward,
        strikes,
        tenor,
        v0=v0,
        theta=theta,
        kappa=kappa,
        vol_of_vol=vol_of_vol,
        rho=rho,
        config=config,
    )
    return prices


def heston_cos_implied_volatilities(
    forward: float,
    strikes: ArrayLike,
    tenor: float,
    *,
    v0: float,
    theta: float,
    kappa: float,
    vol_of_vol: float,
    rho: float,
    config: HestonCOSConfig | None = None,
) -> NDArray[np.float64]:
    """Return Black IVs generated by the independent COS pricing family."""

    strikes_array = np.atleast_1d(np.asarray(strikes, dtype=float))
    prices = heston_cos_call_prices(
        forward,
        strikes_array,
        tenor,
        v0=v0,
        theta=theta,
        kappa=kappa,
        vol_of_vol=vol_of_vol,
        rho=rho,
        config=config,
    )
    return np.array(
        [
            _implied_volatility(forward, float(strike), tenor, float(price))
            for strike, price in zip(strikes_array, prices, strict=True)
        ],
        dtype=float,
    )


__all__ = [
    "HestonCOSConfig",
    "HestonCOSDiagnostics",
    "heston_cos_call_prices",
    "heston_cos_call_prices_with_diagnostics",
    "heston_cos_implied_volatilities",
]
