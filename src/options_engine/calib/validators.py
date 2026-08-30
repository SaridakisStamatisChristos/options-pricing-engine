"""No-arbitrage validation utilities."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from .boards import CleanBoard

_PARITY_TOL = 1e-8


@dataclass(slots=True)
class Violation:
    """Describes a single arbitrage violation."""

    kind: str
    tenor: float
    strike: float
    detail: dict[str, float]


@dataclass(slots=True)
class ValidationReport:
    """Outcome of the arbitrage validation."""

    violations: list[Violation]

    @property
    def is_ok(self) -> bool:
        return not self.violations

    def reasons(self) -> list[str]:
        return [violation.kind for violation in self.violations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ok": self.is_ok,
            "violations": [
                {"kind": v.kind, "tenor": v.tenor, "strike": v.strike, "detail": v.detail}
                for v in self.violations
            ],
        }


class NoArbitrageValidator:
    """Validate cleaned quote boards or model surfaces for arbitrage."""

    def __init__(self, parity_tol: float = _PARITY_TOL) -> None:
        if (
            isinstance(parity_tol, bool)
            or not isinstance(parity_tol, Real)
            or not math.isfinite(parity_tol)
            or not 0.0 <= parity_tol <= 1.0
        ):
            raise ValueError("parity_tol must be finite and within [0, 1]")
        self._parity_tol: float = float(parity_tol)

    def validate(self, board: CleanBoard | pd.DataFrame) -> ValidationReport:
        if not isinstance(board, (CleanBoard, pd.DataFrame)):
            raise TypeError("board must be a CleanBoard or DataFrame")
        data = board.quotes.copy() if isinstance(board, CleanBoard) else board.copy()
        if data.empty:
            return ValidationReport([])
        if len(data) > 100_000:
            raise ValueError("validation board exceeds the 100000-row limit")

        required = {"tenor", "strike", "mid_iv", "forward", "option_type"}
        missing = required - set(data.columns)
        if missing:
            raise KeyError(f"missing columns for validation: {sorted(missing)}")

        # Normalize and reject malformed direct DataFrame input. CleanBoard
        # instances already satisfy this contract, but the public validator
        # also accepts raw frames and must fail closed on NaN/Inf values.
        for column in ("tenor", "strike", "mid_iv", "forward"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        if "discount" not in data.columns:
            data["discount"] = 1.0
        else:
            data["discount"] = pd.to_numeric(data["discount"], errors="coerce")
        numeric = data[["tenor", "strike", "mid_iv", "forward", "discount"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("validation board contains non-finite numeric values")
        if (
            (data["tenor"] <= 0.0).any()
            or (data["tenor"] > 100.0).any()
            or (data["strike"] <= 0.0).any()
            or (data["strike"] > 1e12).any()
            or (data["forward"] <= 0.0).any()
            or (data["forward"] > 1e12).any()
            or (~data["mid_iv"].between(1e-6, 5.0, inclusive="both")).any()
            or (data["discount"] <= 0.0).any()
            or (data["discount"] > 1e6).any()
        ):
            raise ValueError("validation board contains values outside the supported domain")
        if not data["option_type"].map(lambda value: isinstance(value, str)).all():
            raise ValueError("option_type values must be strings")
        data["option_type"] = data["option_type"].str.upper()
        if not data["option_type"].isin({"CALL", "PUT"}).all():
            raise ValueError("option_type values must be CALL or PUT")
        data = data.sort_values(["tenor", "strike", "option_type"]).reset_index(drop=True)

        violations: list[Violation] = []

        for tenor, tenor_df in data.groupby("tenor", sort=True):
            forwards = tenor_df["forward"].to_numpy(dtype=float)
            discounts = np.asarray(tenor_df.get("discount", 1.0), dtype=float)
            forward = float(np.mean(forwards))
            discount = float(np.mean(discounts))
            if not np.allclose(forwards, forward, rtol=1e-10, atol=1e-12):
                violations.append(
                    Violation(
                        kind="inconsistent_forward",
                        tenor=float(tenor),
                        strike=0.0,
                        detail={"range": float(np.ptp(forwards))},
                    )
                )
            if discounts.ndim and not np.allclose(discounts, discount, rtol=1e-10, atol=1e-12):
                violations.append(
                    Violation(
                        kind="inconsistent_discount",
                        tenor=float(tenor),
                        strike=0.0,
                        detail={"range": float(np.ptp(discounts))},
                    )
                )
            strikes, call_prices = self._call_prices(tenor_df, forward, discount)
            violations.extend(
                self._check_butterfly(
                    tenor,
                    strikes,
                    call_prices,
                    forward=forward,
                    discount=discount,
                )
            )
            parity_violations = self._check_parity(tenor_df, forward, discount)
            violations.extend(parity_violations)

        violations.extend(self._check_calendar(data))

        return ValidationReport(violations)

    def _call_prices(
        self, tenor_df: pd.DataFrame, forward: float, discount: float
    ) -> tuple[np.ndarray, np.ndarray]:
        strikes: dict[float, dict[str, float]] = {}
        collapsed = (
            tenor_df.groupby(["strike", "option_type"], as_index=False)
            .agg(tenor=("tenor", "first"), mid_iv=("mid_iv", "mean"))
            .sort_values(["strike", "option_type"])
        )
        for _, row in collapsed.iterrows():
            strike = float(row["strike"])
            expiry = float(row["tenor"])
            sigma = float(row["mid_iv"])
            opt_type = row["option_type"]
            price = _black_price_single(opt_type, forward, strike, expiry, sigma, discount)
            entry = strikes.setdefault(strike, {"call": math.nan, "put": math.nan})
            if opt_type == "PUT":
                entry["put"] = price
            else:
                entry["call"] = price
        strikes_array: list[float] = []
        call_prices: list[float] = []
        for strike in sorted(strikes):
            entry = strikes[strike]
            if not math.isnan(entry["call"]):
                call_price = entry["call"]
            elif not math.isnan(entry["put"]):
                call_price = entry["put"] + discount * (forward - strike)
            else:
                continue
            strikes_array.append(strike)
            call_prices.append(call_price)
        return np.array(strikes_array, dtype=float), np.array(call_prices, dtype=float)

    def _check_butterfly(
        self,
        tenor: float,
        strikes: np.ndarray,
        call_prices: np.ndarray,
        *,
        forward: float | None = None,
        discount: float = 1.0,
    ) -> list[Violation]:
        violations: list[Violation] = []
        order = np.argsort(strikes)
        strikes = strikes[order]
        call_prices = call_prices[order]
        if forward is not None:
            price_tolerance = self._parity_tol * max(1.0, discount * forward)
            for strike, price in zip(strikes, call_prices, strict=True):
                lower = discount * max(forward - float(strike), 0.0)
                upper = discount * forward
                if price < lower - price_tolerance or price > upper + price_tolerance:
                    violations.append(
                        Violation(
                            kind="price_bounds",
                            tenor=float(tenor),
                            strike=float(strike),
                            detail={"price": float(price), "lower": lower, "upper": upper},
                        )
                    )
        if strikes.size < 2:
            return violations
        strike_gaps = np.diff(strikes)
        if np.any(strike_gaps <= 0.0):
            raise ValueError("strikes must be strictly increasing")

        slopes = np.diff(call_prices) / strike_gaps
        for idx, slope in enumerate(slopes):
            if slope < -discount - self._parity_tol:
                violations.append(
                    Violation(
                        kind="vertical_spread",
                        tenor=float(tenor),
                        strike=float(strikes[idx + 1]),
                        detail={"slope": float(slope), "lower_bound": -discount},
                    )
                )
        if strikes.size < 3:
            return violations
        for idx, slope_change in enumerate(np.diff(slopes)):
            if slope_change < -self._parity_tol:
                violations.append(
                    Violation(
                        kind="butterfly",
                        tenor=float(tenor),
                        strike=float(strikes[idx + 1]),
                        detail={
                            "left_slope": float(slopes[idx]),
                            "right_slope": float(slopes[idx + 1]),
                            "slope_change": float(slope_change),
                        },
                    )
                )
        for idx, slope in enumerate(slopes):
            if slope > self._parity_tol:
                violations.append(
                    Violation(
                        kind="strike_monotonicity",
                        tenor=float(tenor),
                        strike=float(strikes[idx + 1]),
                        detail={"slope": float(slope)},
                    )
                )
        return violations

    def _check_calendar(self, data: pd.DataFrame) -> list[Violation]:
        violations: list[Violation] = []
        curves: list[tuple[float, float, np.ndarray, np.ndarray]] = []
        for tenor, group in data.groupby("tenor", sort=True):
            collapsed = group.groupby("strike", as_index=False).agg(
                mid_iv=("mid_iv", "mean"),
                forward=("forward", "mean"),
            )
            forward = float(collapsed["forward"].mean())
            strikes = collapsed["strike"].to_numpy(dtype=float)
            log_moneyness = np.log(strikes / forward)
            total_variance = collapsed["mid_iv"].to_numpy(dtype=float) ** 2 * float(tenor)
            order = np.argsort(log_moneyness)
            curves.append(
                (
                    float(tenor),
                    forward,
                    log_moneyness[order],
                    total_variance[order],
                )
            )

        for short, long in itertools.combinations(curves, 2):
            short_tenor, _, short_k, short_w = short
            long_tenor, long_forward, long_k, long_w = long
            lower = max(float(short_k[0]), float(long_k[0]))
            upper = min(float(short_k[-1]), float(long_k[-1]))
            if lower > upper:
                continue
            grid = np.unique(
                np.concatenate(
                    [
                        short_k[(short_k >= lower) & (short_k <= upper)],
                        long_k[(long_k >= lower) & (long_k <= upper)],
                        np.array([lower, upper]),
                    ]
                )
            )
            short_interp = np.interp(grid, short_k, short_w)
            long_interp = np.interp(grid, long_k, long_w)
            for k, delta in zip(grid, long_interp - short_interp, strict=True):
                if delta < -self._parity_tol:
                    violations.append(
                        Violation(
                            kind="calendar",
                            tenor=short_tenor,
                            strike=float(long_forward * math.exp(float(k))),
                            detail={
                                "long_tenor": long_tenor,
                                "log_moneyness": float(k),
                                "total_variance_delta": float(delta),
                            },
                        )
                    )
        return violations

    def _check_parity(
        self, tenor_df: pd.DataFrame, forward: float, discount: float
    ) -> list[Violation]:
        parity_violations: list[Violation] = []
        grouped = tenor_df.groupby("strike")
        for strike, group in grouped:
            types = group["option_type"].str.upper()
            if {"CALL", "PUT"} <= set(types):
                quotes = (
                    group.groupby("option_type", as_index=False)
                    .agg(tenor=("tenor", "first"), mid_iv=("mid_iv", "mean"))
                    .sort_values("option_type")
                )
                prices = _black_price_array(
                    quotes["option_type"].to_numpy(),
                    forward,
                    np.full(len(quotes), float(strike)),
                    np.full(len(quotes), float(quotes["tenor"].iloc[0])),
                    quotes["mid_iv"].to_numpy(dtype=float),
                    discount,
                )
                parity = prices[0] - prices[1] - discount * (forward - float(strike))
                tolerance = self._parity_tol * max(1.0, discount * forward)
                if abs(float(parity)) > tolerance:
                    parity_violations.append(
                        Violation(
                            kind="parity",
                            tenor=float(quotes["tenor"].iloc[0]),
                            strike=float(strike),
                            detail={"parity": float(parity)},
                        )
                    )
        return parity_violations


def _black_price_array(
    option_types: Iterable[str],
    forward: float,
    strikes: np.ndarray,
    expiry: np.ndarray,
    vols: np.ndarray,
    discount: float,
) -> np.ndarray:
    option_types = np.asarray(list(option_types))
    strikes = np.asarray(strikes, dtype=float)
    expiry = np.asarray(expiry, dtype=float)
    vols = np.asarray(vols, dtype=float)
    discount = float(discount)
    prices = np.empty_like(strikes, dtype=float)
    for idx, (opt_type, K, T, sigma) in enumerate(
        zip(option_types, strikes, expiry, vols, strict=True)
    ):
        prices[idx] = _black_price_single(opt_type, forward, K, T, sigma, discount)
    return prices


def _black_price_single(
    option_type: str,
    forward: float,
    strike: float,
    expiry: float,
    sigma: float,
    discount: float,
) -> float:
    call_price = _black_call_price(forward, strike, expiry, sigma, discount)
    if option_type == "PUT":
        return call_price - discount * (forward - strike)
    return call_price


def _black_call_price(
    forward: float, strike: float, expiry: float, sigma: float, discount: float
) -> float:
    vol_sqrt_t = sigma * np.sqrt(expiry)
    if vol_sqrt_t < 1e-8:
        intrinsic = max(forward - strike, 0.0)
        return discount * intrinsic
    d1 = (np.log(forward / strike) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    call = discount * (forward * _norm_cdf(d1) - strike * _norm_cdf(d2))
    return float(call)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
