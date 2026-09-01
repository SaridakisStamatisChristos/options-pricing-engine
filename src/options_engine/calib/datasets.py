"""Small deterministic calibration boards for recovery and model-risk tests.

These fixtures are deliberately generated at runtime rather than stored as large
quote files.  Noise is a fixed analytic sequence, so results do not depend on
NumPy's random-number implementation or process state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from .boards import CleanBoard
from .heston_cos import heston_cos_implied_volatilities
from .sabr import hagan_implied_volatility
from .svi import ssvi_total_variance


class AdversarialBoard(StrEnum):
    """Named, reproducible quote-board pathologies."""

    SPARSE = "sparse"
    CLUSTERED_ATM = "clustered_atm"
    NOISY_WINGS = "noisy_wings"
    GROSS_OUTLIER = "gross_outlier"
    WIDE_SPREADS = "wide_spreads"
    FLAT_SMILE = "flat_smile"
    STEEP_SKEW = "steep_skew"
    INCONSISTENT_SSVI = "inconsistent_ssvi"
    ASYMMETRIC_SPREADS = "asymmetric_spreads"
    INCONSISTENT_PUT_CALL = "inconsistent_put_call"
    TOO_FEW_TENOR_QUOTES = "too_few_tenor_quotes"
    NEAR_PARAMETER_BOUNDS = "near_parameter_bounds"


@dataclass(frozen=True, slots=True)
class SyntheticBoard:
    """A board paired with the parameters and generator used to create it."""

    board: CleanBoard
    parameters: Mapping[str, float]
    generator: str
    noise_amplitude: float


def _rows(
    tenors: Sequence[float],
    forwards: Sequence[float],
    strikes: Sequence[np.ndarray],
    vols: Sequence[np.ndarray],
) -> CleanBoard:
    records: list[dict[str, object]] = []
    for tenor, forward, tenor_strikes, tenor_vols in zip(
        tenors, forwards, strikes, vols, strict=True
    ):
        for index, (strike, vol) in enumerate(zip(tenor_strikes, tenor_vols, strict=True)):
            records.append(
                {
                    "observation_id": f"T{tenor:g}-K{index:03d}",
                    "tenor": float(tenor),
                    "strike": float(strike),
                    "forward": float(forward),
                    "mid_iv": float(vol),
                    "bid_iv": float(max(vol - 0.002, 1e-4)),
                    "ask_iv": float(vol + 0.002),
                    "option_type": "CALL",
                }
            )
    return CleanBoard(pd.DataFrame.from_records(records), {"synthetic": True})


def _noise(size: int, amplitude: float) -> np.ndarray:
    if not math.isfinite(amplitude) or not 0.0 <= amplitude <= 0.05:
        raise ValueError("noise_amplitude must be within [0, 0.05]")
    index = np.arange(size, dtype=float)
    return amplitude * (np.sin(1.61803398875 * index) + 0.5 * np.cos(0.754877666 * index))


def sabr_recovery_board(
    *,
    alpha: float = 0.22,
    beta: float = 0.65,
    rho: float = -0.35,
    nu: float = 0.55,
    tenor: float = 1.5,
    forward: float = 100.0,
    noise_amplitude: float = 0.0,
) -> SyntheticBoard:
    """Return a noise-free or mildly perturbed SABR smile with known parameters."""

    log_money = np.linspace(-0.3, 0.3, 13)
    strikes = forward * np.exp(log_money)
    vols = hagan_implied_volatility(
        forward, strikes, tenor, alpha=alpha, beta=beta, rho=rho, nu=nu
    ) + _noise(strikes.size, noise_amplitude)
    params = {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu}
    return SyntheticBoard(
        _rows((tenor,), (forward,), (strikes,), (vols,)), params, "Hagan SABR", noise_amplitude
    )


def heston_recovery_board(*, noise_amplitude: float = 0.0) -> SyntheticBoard:
    """Generate a coherent multi-tenor Heston board with the independent COS family."""

    params = {"v0": 0.04, "theta": 0.06, "kappa": 1.35, "vol_of_vol": 0.42, "rho": -0.55}
    tenors = (0.25, 0.75, 2.0)
    forwards = tuple(100.0 * math.exp(0.01 * tenor) for tenor in tenors)
    strike_rows = tuple(forward * np.exp(np.linspace(-0.3, 0.3, 11)) for forward in forwards)
    vol_rows = tuple(
        heston_cos_implied_volatilities(
            forward,
            strikes,
            tenor,
            v0=params["v0"],
            theta=params["theta"],
            kappa=params["kappa"],
            vol_of_vol=params["vol_of_vol"],
            rho=params["rho"],
        )
        + _noise(strikes.size, noise_amplitude)
        for tenor, forward, strikes in zip(tenors, forwards, strike_rows, strict=True)
    )
    return SyntheticBoard(
        _rows(tenors, forwards, strike_rows, vol_rows),
        params,
        "Fang-Oosterlee COS",
        noise_amplitude,
    )


def ssvi_recovery_board(*, noise_amplitude: float = 0.0) -> SyntheticBoard:
    """Return a globally arbitrage-admissible power-law SSVI board."""

    rho, eta, power = -0.45, 0.8, 0.25
    tenors = (0.25, 0.75, 1.5, 3.0)
    theta = (0.012, 0.027, 0.048, 0.082)
    forwards = tuple(100.0 * math.exp(0.01 * tenor) for tenor in tenors)
    k = np.linspace(-0.4, 0.4, 13)
    strike_rows = tuple(forward * np.exp(k) for forward in forwards)
    vol_rows = tuple(
        np.sqrt(ssvi_total_variance(k, value, rho=rho, eta=eta, power=power) / tenor)
        + _noise(k.size, noise_amplitude)
        for tenor, value in zip(tenors, theta, strict=True)
    )
    return SyntheticBoard(
        _rows(tenors, forwards, strike_rows, vol_rows),
        {"rho": rho, "eta": eta, "power": power},
        "power-law SSVI",
        noise_amplitude,
    )


def adversarial_board(case: AdversarialBoard | str) -> CleanBoard:
    """Construct a named difficult board without random state."""

    selected = AdversarialBoard(case)
    base = ssvi_recovery_board().board.quotes.copy()
    k = np.log(base["strike"].to_numpy() / base["forward"].to_numpy())
    if selected is AdversarialBoard.SPARSE:
        base = base.groupby("tenor", sort=True).head(3)
    elif selected is AdversarialBoard.CLUSTERED_ATM:
        base = base[np.abs(k) <= 0.07]
    elif selected is AdversarialBoard.NOISY_WINGS:
        base.loc[np.abs(k) > 0.25, "mid_iv"] += 0.04 * np.sign(k[np.abs(k) > 0.25])
    elif selected is AdversarialBoard.GROSS_OUTLIER:
        base.loc[base.index[len(base) // 2], "mid_iv"] += 0.35
    elif selected is AdversarialBoard.WIDE_SPREADS:
        base["bid_iv"] = np.maximum(base["mid_iv"] - 0.08, 1e-4)
        base["ask_iv"] = base["mid_iv"] + 0.08
    elif selected is AdversarialBoard.FLAT_SMILE:
        base["mid_iv"] = 0.2
    elif selected is AdversarialBoard.STEEP_SKEW:
        base["mid_iv"] = np.clip(0.2 - 0.5 * k, 0.02, 1.0)
    elif selected is AdversarialBoard.ASYMMETRIC_SPREADS:
        base["bid_iv"] = np.maximum(base["mid_iv"] - 0.001, 1e-4)
        base["ask_iv"] = base["mid_iv"] + np.where(k > 0.0, 0.08, 0.004)
    elif selected is AdversarialBoard.INCONSISTENT_PUT_CALL:
        duplicate = base.iloc[[len(base) // 3]].copy()
        duplicate["option_type"] = "PUT"
        duplicate["mid_iv"] += 0.12
        duplicate["observation_id"] = duplicate["observation_id"].astype(str) + "-PUT"
        base = pd.concat((base, duplicate), ignore_index=True)
    elif selected is AdversarialBoard.TOO_FEW_TENOR_QUOTES:
        shortest = float(base["tenor"].min())
        other = base[base["tenor"] != shortest]
        base = pd.concat((base[base["tenor"] == shortest].head(2), other), ignore_index=True)
    elif selected is AdversarialBoard.NEAR_PARAMETER_BOUNDS:
        base["mid_iv"] = np.clip(0.08 - 0.14 * k + 0.8 * k**2, 0.01, 2.0)
    else:
        # Reverse ATM total-variance ordering: no calendar-arbitrage-free SSVI
        # surface can reproduce this board exactly.
        decreasing_theta = 0.08 / (1.0 + base["tenor"])
        base["mid_iv"] = np.sqrt(decreasing_theta / base["tenor"]) * (1.0 + 0.05 * np.abs(k))
    return CleanBoard(
        base.reset_index(drop=True), {"synthetic": True, "adversarial": selected.value}
    )
