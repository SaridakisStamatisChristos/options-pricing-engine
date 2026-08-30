"""Backward-compatible SABR surface API backed by the canonical calibration stack.

The public objects in this module predate :mod:`options_engine.calib`. They are
kept for source compatibility, but formula evaluation, optimization, and
no-arbitrage checks share the canonical implementations so the package does not
maintain two conflicting definitions of the same model.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real

import numpy as np
import pandas as pd

from ..calib.boards import CleanBoard
from ..calib.sabr import SABRCalibrator as CanonicalSABRCalibrator
from ..calib.sabr import SABRConfig
from ..calib.sabr import hagan_implied_volatility as canonical_hagan_implied_volatility
from ..calib.validators import NoArbitrageValidator

_MAX_BOARD_ROWS = 100_000


@dataclass(frozen=True, slots=True)
class QCReport:
    """Summary of the quote-board cleaning stage."""

    total_quotes: int
    dropped_crossed: int
    dropped_stale: int
    dropped_outlier: int
    retained_quotes: int
    dropped_invalid: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_quotes": self.total_quotes,
            "dropped_invalid": self.dropped_invalid,
            "dropped_crossed": self.dropped_crossed,
            "dropped_stale": self.dropped_stale,
            "dropped_outlier": self.dropped_outlier,
            "retained_quotes": self.retained_quotes,
        }


@dataclass(slots=True)
class CleanBoardResult:
    """Container for the cleaned board and its QC information."""

    data: pd.DataFrame
    report: QCReport

    def to_dict(self) -> dict[str, object]:
        return {"report": self.report.to_dict(), "data": self.data.to_dict("records")}


@dataclass(slots=True)
class ArbitrageCheckResult:
    """Summary of economically meaningful static-arbitrage checks."""

    butterfly_violations: list[dict[str, float]]
    calendar_violations: list[dict[str, float]]
    tenor_monotonicity_violations: list[dict[str, float]]

    @property
    def is_arbitrage_free(self) -> bool:
        return not (
            self.butterfly_violations
            or self.calendar_violations
            or self.tenor_monotonicity_violations
        )

    def to_dict(self) -> dict[str, bool | list[dict[str, float]]]:
        return {
            "is_arbitrage_free": self.is_arbitrage_free,
            "butterfly_violations": self.butterfly_violations,
            "calendar_violations": self.calendar_violations,
            "tenor_monotonicity_violations": self.tenor_monotonicity_violations,
        }


@dataclass(frozen=True, slots=True)
class SABRParameters:
    """SABR model parameters."""

    alpha: float
    beta: float
    rho: float
    nu: float


@dataclass(slots=True)
class SABRTenorCalibration:
    """Calibration diagnostics for one tenor."""

    tenor: float
    parameters: SABRParameters
    strikes: np.ndarray
    market_vols: np.ndarray
    model_vols: np.ndarray
    rmse: float

    def to_dict(self) -> dict[str, object]:
        return {
            "tenor": self.tenor,
            "parameters": {
                "alpha": self.parameters.alpha,
                "beta": self.parameters.beta,
                "rho": self.parameters.rho,
                "nu": self.parameters.nu,
            },
            "strikes": self.strikes.tolist(),
            "market_vols": self.market_vols.tolist(),
            "model_vols": self.model_vols.tolist(),
            "rmse": self.rmse,
        }


@dataclass(slots=True)
class SABRCalibrationResult:
    """Aggregate result returned by :class:`SABRCalibrator`."""

    clean_board: CleanBoardResult
    tenor_results: list[SABRTenorCalibration]
    arbitrage: ArbitrageCheckResult
    fitted_surface: pd.DataFrame
    regime: str

    @property
    def qc_report(self) -> dict[str, object]:
        per_tenor = {result.tenor: result.rmse for result in self.tenor_results}
        rmse_values = list(per_tenor.values())
        return {
            "board": self.clean_board.report.to_dict(),
            "rmse": {
                "per_tenor": per_tenor,
                "max": float(np.max(rmse_values)) if rmse_values else None,
                "mean": float(np.mean(rmse_values)) if rmse_values else None,
            },
            "arbitrage": self.arbitrage.to_dict(),
            "regime": self.regime,
        }


def _finite_real(name: str, value: object, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not minimum <= normalized <= maximum:
        raise ValueError(f"{name} must be finite and within [{minimum}, {maximum}]")
    return normalized


def _validate_column_names(names: Sequence[str | None]) -> None:
    present = [name for name in names if name is not None]
    if any(not isinstance(name, str) or not name or len(name) > 128 for name in present):
        raise ValueError("column names must be non-empty strings of at most 128 characters")
    if len(set(present)) != len(present):
        raise ValueError("column names must be distinct")


def clean_option_board(
    board: pd.DataFrame | Sequence[Mapping[str, object]],
    *,
    now: datetime | None = None,
    max_age_seconds: float = 300.0,
    tenor_column: str = "tenor",
    strike_column: str = "strike",
    bid_column: str | None = "bid_iv",
    ask_column: str | None = "ask_iv",
    vol_column: str = "mid_iv",
    timestamp_column: str = "timestamp",
    mad_threshold: float = 4.0,
) -> CleanBoardResult:
    """Clean a legacy quote board using strict, bounded input policies."""

    max_age_seconds = _finite_real(
        "max_age_seconds", max_age_seconds, minimum=0.0, maximum=604_800.0
    )
    mad_threshold = _finite_real("mad_threshold", mad_threshold, minimum=1e-12, maximum=100.0)
    _validate_column_names(
        (tenor_column, strike_column, bid_column, ask_column, vol_column, timestamp_column)
    )
    if (bid_column is None) != (ask_column is None):
        raise ValueError("bid_column and ask_column must be configured together")
    if now is not None:
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must be timezone-aware")

    if isinstance(board, pd.DataFrame):
        if len(board) > _MAX_BOARD_ROWS:
            raise ValueError(f"quote board exceeds the {_MAX_BOARD_ROWS}-row limit")
        data = board.copy(deep=True)
    elif isinstance(board, Sequence) and not isinstance(board, (str, bytes)):
        if len(board) > _MAX_BOARD_ROWS:
            raise ValueError(f"quote board exceeds the {_MAX_BOARD_ROWS}-row limit")
        if not all(isinstance(row, Mapping) for row in board):
            raise TypeError("every quote must be a mapping")
        data = pd.DataFrame(list(board))
    else:
        raise TypeError("board must be a DataFrame or a sequence of mappings")

    total_quotes = len(data)
    if data.empty:
        return CleanBoardResult(data, QCReport(0, 0, 0, 0, 0))

    required = {tenor_column, strike_column}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")

    has_bid = bid_column is not None and bid_column in data.columns
    has_ask = ask_column is not None and ask_column in data.columns
    if has_bid != has_ask:
        raise KeyError("bid and ask columns must be supplied together")
    if vol_column not in data.columns and not (has_bid and has_ask):
        raise KeyError("vol column is missing and a bid/ask midpoint cannot be computed")

    data = data.replace([np.inf, -np.inf], np.nan)
    numeric_columns = [tenor_column, strike_column]
    if "forward" in data.columns:
        numeric_columns.append("forward")
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    invalid = data[numeric_columns].isna().any(axis=1)
    invalid |= ~data[tenor_column].between(1e-12, 100.0, inclusive="both")
    invalid |= ~data[strike_column].between(1e-12, 1e12, inclusive="both")
    if "forward" in data.columns:
        invalid |= ~data["forward"].between(1e-12, 1e12, inclusive="both")
    dropped_invalid = int(invalid.sum())
    data = data.loc[~invalid].copy()

    dropped_crossed = 0
    if has_bid and has_ask and bid_column is not None and ask_column is not None:
        data[bid_column] = pd.to_numeric(data[bid_column], errors="coerce")
        data[ask_column] = pd.to_numeric(data[ask_column], errors="coerce")
        malformed_spread = data[[bid_column, ask_column]].isna().any(axis=1)
        malformed_spread |= ~data[bid_column].between(0.0, 5.0, inclusive="both")
        malformed_spread |= ~data[ask_column].between(0.0, 5.0, inclusive="both")
        dropped_invalid += int(malformed_spread.sum())
        data = data.loc[~malformed_spread].copy()

        crossed = data[bid_column] > data[ask_column]
        dropped_crossed = int(crossed.sum())
        data = data.loc[~crossed].copy()
        if vol_column not in data.columns:
            data[vol_column] = (data[bid_column] + data[ask_column]) / 2.0

    data[vol_column] = pd.to_numeric(data[vol_column], errors="coerce")
    invalid_vol = data[vol_column].isna() | ~data[vol_column].between(1e-4, 5.0, inclusive="both")
    if has_bid and has_ask and bid_column is not None and ask_column is not None:
        invalid_vol |= (data[vol_column] < data[bid_column]) | (data[vol_column] > data[ask_column])
    dropped_invalid += int(invalid_vol.sum())
    data = data.loc[~invalid_vol].copy()

    dropped_stale = 0
    if timestamp_column in data.columns:
        current = now if now is not None else datetime.now(UTC)
        timestamps = pd.to_datetime(data[timestamp_column], utc=True, errors="coerce")
        age = (pd.Timestamp(current).tz_convert("UTC") - timestamps).dt.total_seconds()
        stale = timestamps.isna() | (age > max_age_seconds) | (age < -30.0)
        dropped_stale = int(stale.sum())
        data = data.loc[~stale].copy()

    dropped_outlier = 0
    filtered_groups: list[pd.DataFrame] = []
    for _, group in data.groupby(tenor_column, sort=True):
        values = group[vol_column].to_numpy(dtype=float)
        if values.size > 3:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            deviations = np.abs(values - median)
            if mad > 0.0:
                scaled = deviations / (1.4826 * mad)
            else:
                tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(median))
                scaled = np.where(deviations <= tolerance, 0.0, float("inf"))
            keep = scaled <= mad_threshold
            dropped_outlier += int((~keep).sum())
            group = group.loc[keep]
        filtered_groups.append(group)

    cleaned = (
        pd.concat(filtered_groups, ignore_index=True)
        if filtered_groups
        else pd.DataFrame(columns=data.columns)
    )
    cleaned = cleaned.sort_values([tenor_column, strike_column]).reset_index(drop=True)
    report = QCReport(
        total_quotes=total_quotes,
        dropped_crossed=dropped_crossed,
        dropped_stale=dropped_stale,
        dropped_outlier=dropped_outlier,
        retained_quotes=len(cleaned),
        dropped_invalid=dropped_invalid,
    )
    return CleanBoardResult(cleaned, report)


def hagan_implied_volatility(
    forward: float | np.ndarray,
    strike: float | np.ndarray,
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Return Hagan SABR volatility through the canonical implementation."""

    return canonical_hagan_implied_volatility(
        forward,
        strike,
        expiry,
        alpha=alpha,
        beta=beta,
        rho=rho,
        nu=nu,
    )


class ArbitrageValidator:
    """Compatibility adapter for canonical call-price arbitrage checks."""

    def __init__(self, *, tolerance: float = 1e-8, vol_column: str = "model_vol") -> None:
        self.tolerance = _finite_real("tolerance", tolerance, minimum=0.0, maximum=1.0)
        if not isinstance(vol_column, str) or not vol_column or len(vol_column) > 128:
            raise ValueError("vol_column must be a non-empty string of at most 128 characters")
        self.vol_column = vol_column
        self._validator = NoArbitrageValidator(parity_tol=self.tolerance)

    def validate(
        self,
        surface: pd.DataFrame,
        *,
        tenor_column: str = "tenor",
        strike_column: str = "strike",
    ) -> ArbitrageCheckResult:
        if not isinstance(surface, pd.DataFrame):
            raise TypeError("surface must be a DataFrame")
        if surface.empty:
            return ArbitrageCheckResult([], [], [])
        _validate_column_names((tenor_column, strike_column, self.vol_column))
        required = {tenor_column, strike_column, self.vol_column, "forward"}
        missing = required - set(surface.columns)
        if missing:
            raise KeyError(f"surface is missing required columns: {sorted(missing)}")

        canonical = pd.DataFrame(
            {
                "tenor": surface[tenor_column],
                "strike": surface[strike_column],
                "mid_iv": surface[self.vol_column],
                "forward": surface["forward"],
                "option_type": "CALL",
                "discount": surface["discount"] if "discount" in surface.columns else 1.0,
            }
        )
        report = self._validator.validate(canonical)
        butterfly: list[dict[str, float]] = []
        calendar: list[dict[str, float]] = []
        tenor_monotonicity: list[dict[str, float]] = []
        for violation in report.violations:
            item = {
                "tenor": float(violation.tenor),
                "strike": float(violation.strike),
                **{key: float(value) for key, value in violation.detail.items()},
            }
            if violation.kind == "calendar":
                calendar.append(item)
            elif violation.kind in {"inconsistent_forward", "inconsistent_discount"}:
                tenor_monotonicity.append(item)
            else:
                butterfly.append(item)
        return ArbitrageCheckResult(butterfly, calendar, tenor_monotonicity)


class SABRCalibrator:
    """Legacy orchestration API using canonical SABR calibration per tenor."""

    def __init__(
        self,
        *,
        beta: float = 0.5,
        max_iterations: int = 200,
        tolerance: float = 1e-8,
        max_age_seconds: float = 300.0,
        mad_threshold: float = 4.0,
        arbitrage_tolerance: float = 1e-8,
    ) -> None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral):
            raise TypeError("max_iterations must be an integer")
        self._config: SABRConfig = SABRConfig(
            beta=beta,
            max_iterations=int(max_iterations),
            tolerance=tolerance,
        )
        self.beta = float(self._config.beta)
        self.max_iterations = int(self._config.max_iterations)
        self.tolerance = float(self._config.tolerance)
        self.max_age_seconds: float = _finite_real(
            "max_age_seconds", max_age_seconds, minimum=0.0, maximum=604_800.0
        )
        self.mad_threshold: float = _finite_real(
            "mad_threshold", mad_threshold, minimum=1e-12, maximum=100.0
        )
        self.arbitrage_tolerance = _finite_real(
            "arbitrage_tolerance", arbitrage_tolerance, minimum=0.0, maximum=1.0
        )
        self._validator: ArbitrageValidator = ArbitrageValidator(
            tolerance=self.arbitrage_tolerance, vol_column="model_vol"
        )

    def calibrate(
        self,
        board: pd.DataFrame | Sequence[Mapping[str, object]],
        *,
        now: datetime | None = None,
        forward_curve: Mapping[float, float] | Callable[[float], float] | None = None,
        forward_column: str = "forward",
        tenor_column: str = "tenor",
        strike_column: str = "strike",
        bid_column: str | None = "bid_iv",
        ask_column: str | None = "ask_iv",
        vol_column: str = "mid_iv",
        timestamp_column: str = "timestamp",
        alternative_models: Mapping[str, Sequence[float]] | None = None,
    ) -> SABRCalibrationResult:
        clean_board = clean_option_board(
            board,
            now=now,
            max_age_seconds=self.max_age_seconds,
            tenor_column=tenor_column,
            strike_column=strike_column,
            bid_column=bid_column,
            ask_column=ask_column,
            vol_column=vol_column,
            timestamp_column=timestamp_column,
            mad_threshold=self.mad_threshold,
        )
        cleaned = clean_board.data
        if cleaned.empty:
            raise ValueError("board cleaning removed all quotes; calibration aborted")

        tenor_results: list[SABRTenorCalibration] = []
        surface_rows: list[dict[str, float]] = []
        canonical_calibrator = CanonicalSABRCalibrator(self._config)
        for tenor, group in cleaned.groupby(tenor_column, sort=True):
            tenor_value = float(tenor)
            forward = self._resolve_forward(
                tenor_value,
                group,
                forward_curve=forward_curve,
                forward_column=forward_column,
            )
            strikes = group[strike_column].to_numpy(dtype=float)
            market_vols = group[vol_column].to_numpy(dtype=float)
            if np.unique(strikes).size < 5:
                continue
            canonical_board = CleanBoard(
                pd.DataFrame(
                    {
                        "tenor": tenor_value,
                        "strike": strikes,
                        "mid_iv": market_vols,
                        "forward": forward,
                        "option_type": "CALL",
                    }
                ),
                {},
            )
            calibrated = canonical_calibrator.calibrate(
                canonical_board, forward_curve={tenor_value: forward}
            )
            if not calibrated:
                continue
            result = calibrated[0]
            parameters = SABRParameters(**result.params)
            model_vols = hagan_implied_volatility(
                forward,
                strikes,
                tenor_value,
                parameters.alpha,
                parameters.beta,
                parameters.rho,
                parameters.nu,
            )
            rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))
            tenor_results.append(
                SABRTenorCalibration(
                    tenor=tenor_value,
                    parameters=parameters,
                    strikes=strikes.copy(),
                    market_vols=market_vols.copy(),
                    model_vols=model_vols.copy(),
                    rmse=rmse,
                )
            )
            surface_rows.extend(
                {
                    "tenor": tenor_value,
                    "strike": float(strike),
                    "market_vol": float(market_vol),
                    "model_vol": float(model_vol),
                    "forward": forward,
                }
                for strike, market_vol, model_vol in zip(
                    strikes, market_vols, model_vols, strict=True
                )
            )

        if not tenor_results:
            raise ValueError("no tenors have the five unique strikes required for calibration")
        fitted_surface = pd.DataFrame(surface_rows)
        arbitrage = self._validator.validate(fitted_surface)
        regime = self._select_regime(tenor_results, alternative_models)
        return SABRCalibrationResult(
            clean_board=clean_board,
            tenor_results=tenor_results,
            arbitrage=arbitrage,
            fitted_surface=fitted_surface,
            regime=regime,
        )

    @staticmethod
    def _resolve_forward(
        tenor: float,
        group: pd.DataFrame,
        *,
        forward_curve: Mapping[float, float] | Callable[[float], float] | None,
        forward_column: str,
    ) -> float:
        if forward_curve is not None:
            if callable(forward_curve):
                forward = float(forward_curve(tenor))
            else:
                matches = [
                    float(value)
                    for key, value in forward_curve.items()
                    if math.isclose(float(key), tenor, rel_tol=0.0, abs_tol=1e-9)
                ]
                if not matches:
                    raise KeyError(f"tenor {tenor} not found in forward_curve")
                if len(matches) > 1:
                    raise ValueError(f"forward_curve contains ambiguous keys for tenor {tenor}")
                forward = matches[0]
        else:
            if forward_column not in group.columns:
                raise KeyError(f"forward information unavailable for tenor {tenor}")
            forwards = pd.to_numeric(group[forward_column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(forwards).all() or not np.allclose(
                forwards, forwards[0], rtol=1e-10, atol=1e-12
            ):
                raise ValueError(f"forwards for tenor {tenor} must be finite and consistent")
            forward = float(np.mean(forwards))
        if not math.isfinite(forward) or not 0.0 < forward <= 1e12:
            raise ValueError(f"forward for tenor {tenor} must be finite and within (0, 1e12]")
        return forward

    @staticmethod
    def _select_regime(
        tenor_results: Sequence[SABRTenorCalibration],
        alternative_models: Mapping[str, Sequence[float]] | None,
    ) -> str:
        scores = {"sabr": float(np.mean([result.rmse for result in tenor_results]))}
        if alternative_models:
            if len(alternative_models) > 64:
                raise ValueError("alternative_models exceeds the 64-model limit")
            for name, values in alternative_models.items():
                if not isinstance(name, str) or not name.strip() or len(name) > 128:
                    raise ValueError("alternative model names must be non-empty strings")
                if len(values) > 100_000:
                    raise ValueError("alternative model score list exceeds the 100000-value limit")
                scores_array = np.asarray(values, dtype=float)
                if scores_array.size == 0:
                    continue
                if not np.isfinite(scores_array).all() or np.any(scores_array < 0.0):
                    raise ValueError("alternative model scores must be finite and non-negative")
                scores[name.strip().lower()] = float(np.mean(scores_array))
        return min(scores.items(), key=lambda item: (item[1], item[0]))[0]


__all__ = [
    "ArbitrageCheckResult",
    "ArbitrageValidator",
    "CleanBoardResult",
    "QCReport",
    "SABRCalibrationResult",
    "SABRCalibrator",
    "SABRParameters",
    "SABRTenorCalibration",
    "clean_option_board",
    "hagan_implied_volatility",
]
