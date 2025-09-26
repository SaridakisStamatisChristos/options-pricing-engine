"""No-arbitrage validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import math
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
    detail: Dict[str, float]


@dataclass(slots=True)
class ValidationReport:
    """Outcome of the arbitrage validation."""

    violations: List[Violation]

    @property
    def is_ok(self) -> bool:
        return not self.violations

    def reasons(self) -> List[str]:
        return [violation.kind for violation in self.violations]

    def to_dict(self) -> Dict[str, Any]:
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
        self._parity_tol = float(parity_tol)

    def validate(self, board: CleanBoard | pd.DataFrame) -> ValidationReport:
        if isinstance(board, CleanBoard):
            data = board.quotes.copy()
        else:
            data = board.copy()
        if data.empty:
            return ValidationReport([])

        required = {"tenor", "strike", "mid_iv", "forward", "option_type"}
        missing = required - set(data.columns)
        if missing:
            raise KeyError(f"missing columns for validation: {sorted(missing)}")

        # Normalise types
        data["option_type"] = data["option_type"].str.upper()
        data = data.sort_values(["tenor", "strike", "option_type"]).reset_index(drop=True)

        violations: List[Violation] = []

        for tenor, tenor_df in data.groupby("tenor", sort=True):
            forward = float(np.mean(tenor_df["forward"].to_numpy(dtype=float)))
            discount = float(np.mean(tenor_df.get("discount", 1.0)))
            strikes, call_prices = self._call_prices(tenor_df, forward, discount)
            violations.extend(self._check_butterfly(tenor, strikes, call_prices))
            parity_violations = self._check_parity(tenor_df, forward, discount)
            violations.extend(parity_violations)

        violations.extend(self._check_calendar(data))

        return ValidationReport(violations)

    def _call_prices(
        self, tenor_df: pd.DataFrame, forward: float, discount: float
    ) -> tuple[np.ndarray, np.ndarray]:
        strikes: Dict[float, Dict[str, float]] = {}
        for _, row in tenor_df.sort_values(["strike", "option_type"]).iterrows():
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
        strikes_array: List[float] = []
        call_prices: List[float] = []
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

    def _check_butterfly(self, tenor: float, strikes: np.ndarray, call_prices: np.ndarray) -> List[Violation]:
        violations: List[Violation] = []
        order = np.argsort(strikes)
        strikes = strikes[order]
        call_prices = call_prices[order]
        if strikes.size < 3:
            return violations
        second_diff = call_prices[:-2] - 2.0 * call_prices[1:-1] + call_prices[2:]
        for idx, value in enumerate(second_diff):
            if value < -1e-8:
                violations.append(
                    Violation(
                        kind="butterfly",
                        tenor=float(tenor),
                        strike=float(strikes[idx + 1]),
                        detail={"second_diff": float(value)},
                    )
                )
        return violations

    def _check_calendar(self, data: pd.DataFrame) -> List[Violation]:
        violations: List[Violation] = []
        grouped = data.groupby(["strike", "option_type"], sort=True)
        for (strike, option_type), group in grouped:
            sorted_group = group.sort_values("tenor")
            forward = float(np.mean(sorted_group["forward"].to_numpy(dtype=float)))
            discount = float(np.mean(sorted_group.get("discount", 1.0)))
            prices = _black_price_array(
                np.full(len(sorted_group), option_type),
                forward,
                np.full(len(sorted_group), float(strike)),
                sorted_group["tenor"].to_numpy(dtype=float),
                sorted_group["mid_iv"].to_numpy(dtype=float),
                discount,
            )
            diffs = np.diff(prices)
            for tenor_low, value in zip(sorted_group["tenor"].to_numpy(dtype=float)[:-1], diffs):
                if value < -1e-8:
                    violations.append(
                        Violation(
                            kind="calendar",
                            tenor=float(tenor_low),
                            strike=float(strike),
                            detail={"delta": float(value)},
                        )
                    )
        return violations

    def _check_parity(self, tenor_df: pd.DataFrame, forward: float, discount: float) -> List[Violation]:
        parity_violations: List[Violation] = []
        grouped = tenor_df.groupby("strike")
        for strike, group in grouped:
            types = group["option_type"].str.upper()
            if {"CALL", "PUT"} <= set(types):
                quotes = group.sort_values("option_type")
                prices = _black_price_array(
                    quotes["option_type"].to_numpy(),
                    forward,
                    np.full(len(quotes), float(strike)),
                    np.full(len(quotes), float(quotes["tenor"].iloc[0])),
                    quotes["mid_iv"].to_numpy(dtype=float),
                    discount,
                )
                parity = prices[0] - prices[1] - discount * (forward - float(strike))
                if abs(float(parity)) > self._parity_tol:
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
    for idx, (opt_type, K, T, sigma) in enumerate(zip(option_types, strikes, expiry, vols)):
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
    sigma = max(float(sigma), 1e-4)
    expiry = max(float(expiry), 1e-6)
    call_price = _black_call_price(forward, strike, expiry, sigma, discount)
    if option_type == "PUT":
        return call_price - discount * (forward - strike)
    return call_price


def _black_call_price(forward: float, strike: float, expiry: float, sigma: float, discount: float) -> float:
    if sigma <= 0.0:
        sigma = 1e-4
    if expiry <= 0.0:
        expiry = 1e-6
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
