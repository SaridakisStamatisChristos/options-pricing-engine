"""Volatility surface calibration utilities with SABR support."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QCReport:
    """Quick summary of the quote board cleaning stage."""

    total_quotes: int
    dropped_crossed: int
    dropped_stale: int
    dropped_outlier: int
    retained_quotes: int

    def to_dict(self) -> Dict[str, int]:
        """Return a serialisable view of the report."""

        return {
            "total_quotes": self.total_quotes,
            "dropped_crossed": self.dropped_crossed,
            "dropped_stale": self.dropped_stale,
            "dropped_outlier": self.dropped_outlier,
            "retained_quotes": self.retained_quotes,
        }


@dataclass(slots=True)
class CleanBoardResult:
    """Container for the cleaned board and the associated QC information."""

    data: pd.DataFrame
    report: QCReport

    def to_dict(self) -> Dict[str, Union[Dict[str, int], List[Mapping[str, object]]]]:
        """Return a serialisable representation of the result."""

        return {"report": self.report.to_dict(), "data": self.data.to_dict("records")}


@dataclass(slots=True)
class ArbitrageCheckResult:
    """Summary of the arbitrage checks performed on the fitted surface."""

    butterfly_violations: List[Dict[str, float]]
    calendar_violations: List[Dict[str, float]]
    tenor_monotonicity_violations: List[Dict[str, float]]

    @property
    def is_arbitrage_free(self) -> bool:
        """Return ``True`` when no violations were detected."""

        return not (
            self.butterfly_violations
            or self.calendar_violations
            or self.tenor_monotonicity_violations
        )

    def to_dict(self) -> Dict[str, Union[bool, List[Dict[str, float]]]]:
        """Return a serialisable view of the arbitrage check result."""

        return {
            "is_arbitrage_free": self.is_arbitrage_free,
            "butterfly_violations": self.butterfly_violations,
            "calendar_violations": self.calendar_violations,
            "tenor_monotonicity_violations": self.tenor_monotonicity_violations,
        }


@dataclass(slots=True)
class SABRParameters:
    """Holds the parameters of a SABR volatility model."""

    alpha: float
    beta: float
    rho: float
    nu: float


@dataclass(slots=True)
class SABRTenorCalibration:
    """Stores SABR calibration diagnostics for a single tenor."""

    tenor: float
    parameters: SABRParameters
    strikes: np.ndarray
    market_vols: np.ndarray
    model_vols: np.ndarray
    rmse: float

    def to_dict(self) -> Dict[str, object]:
        """Return a serialisable representation of the calibration result."""

        return {
            "tenor": self.tenor,
            "parameters": self.parameters.__dict__,
            "strikes": self.strikes.tolist(),
            "market_vols": self.market_vols.tolist(),
            "model_vols": self.model_vols.tolist(),
            "rmse": self.rmse,
        }


@dataclass(slots=True)
class SABRCalibrationResult:
    """Aggregate result returned by :class:`SABRCalibrator`."""

    clean_board: CleanBoardResult
    tenor_results: List[SABRTenorCalibration]
    arbitrage: ArbitrageCheckResult
    fitted_surface: pd.DataFrame
    regime: str

    @property
    def qc_report(self) -> Dict[str, object]:
        """Build a consolidated QC report."""

        per_tenor = {result.tenor: result.rmse for result in self.tenor_results}
        rmse_values = list(per_tenor.values())
        if rmse_values:
            max_rmse = float(np.max(rmse_values))
            mean_rmse = float(np.mean(rmse_values))
        else:
            max_rmse = float("nan")
            mean_rmse = float("nan")

        return {
            "board": self.clean_board.report.to_dict(),
            "rmse": {"per_tenor": per_tenor, "max": max_rmse, "mean": mean_rmse},
            "arbitrage": self.arbitrage.to_dict(),
            "regime": self.regime,
        }


def clean_option_board(
    board: Union[pd.DataFrame, Sequence[Mapping[str, object]]],
    *,
    now: Optional[datetime] = None,
    max_age_seconds: float = 300.0,
    tenor_column: str = "tenor",
    strike_column: str = "strike",
    bid_column: Optional[str] = "bid_iv",
    ask_column: Optional[str] = "ask_iv",
    vol_column: str = "mid_iv",
    timestamp_column: str = "timestamp",
    mad_threshold: float = 4.0,
) -> CleanBoardResult:
    """Clean the option board prior to calibration.

    Parameters
    ----------
    board:
        Raw quotes either as a :class:`pandas.DataFrame` or an iterable of dictionaries.
    now:
        Timestamp of the calibration run. Defaults to ``datetime.now(UTC)``.
    max_age_seconds:
        Quotes older than this threshold are considered stale and will be dropped.
    tenor_column, strike_column, bid_column, ask_column, vol_column, timestamp_column:
        Column names used by the pipeline.
    mad_threshold:
        Threshold applied on the scaled Median Absolute Deviation to drop outliers.
    """

    if isinstance(board, pd.DataFrame):
        df = board.copy()
    else:
        df = pd.DataFrame(list(board))

    if df.empty:
        report = QCReport(0, 0, 0, 0, 0)
        return CleanBoardResult(df, report)

    required_columns = {tenor_column, strike_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"missing required columns: {sorted(missing_columns)}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[tenor_column, strike_column])

    # Ensure the primary columns are numeric
    df[tenor_column] = pd.to_numeric(df[tenor_column], errors="coerce")
    df[strike_column] = pd.to_numeric(df[strike_column], errors="coerce")
    df = df.dropna(subset=[tenor_column, strike_column])

    total_quotes = len(df)
    dropped_crossed = 0
    dropped_stale = 0
    dropped_outlier = 0

    if bid_column and bid_column in df.columns and ask_column and ask_column in df.columns:
        df[bid_column] = pd.to_numeric(df[bid_column], errors="coerce")
        df[ask_column] = pd.to_numeric(df[ask_column], errors="coerce")
        before = len(df)
        df = df[df[bid_column] <= df[ask_column]]
        dropped_crossed = before - len(df)

    # Determine the mid implied volatility column
    if vol_column not in df.columns:
        if bid_column and bid_column in df.columns and ask_column and ask_column in df.columns:
            df[vol_column] = (df[bid_column] + df[ask_column]) / 2.0
        else:
            raise KeyError(
                "vol_column missing and bid/ask columns not provided; cannot compute mid vol"
            )

    df[vol_column] = pd.to_numeric(df[vol_column], errors="coerce")
    df = df.dropna(subset=[vol_column])

    if timestamp_column in df.columns:
        if now is None:
            now = datetime.now(UTC)

        timestamps = pd.to_datetime(df[timestamp_column], utc=True, errors="coerce")
        age = (now - timestamps).dt.total_seconds()
        before = len(df)
        df = df[age <= max_age_seconds]
        dropped_stale = before - len(df)

    if df.empty:
        report = QCReport(total_quotes, dropped_crossed, dropped_stale, total_quotes, 0)
        return CleanBoardResult(df, report)

    # Median-MAD outlier filter per tenor
    filtered_groups: List[pd.DataFrame] = []

    for _, group in df.groupby(tenor_column):
        values = group[vol_column].to_numpy(dtype=float)
        if values.size > 3:
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            if mad > 0.0:
                scaled = np.abs(values - median) / (1.4826 * mad)
                mask = scaled <= mad_threshold
                dropped_outlier += int((~mask).sum())
                group = group.loc[mask]

        filtered_groups.append(group)

    if filtered_groups:
        df = pd.concat(filtered_groups, ignore_index=True)
    else:
        df = pd.DataFrame(columns=df.columns)

    retained_quotes = len(df)
    report = QCReport(total_quotes, dropped_crossed, dropped_stale, dropped_outlier, retained_quotes)
    return CleanBoardResult(df.reset_index(drop=True), report)


def hagan_implied_volatility(
    forward: Union[float, np.ndarray],
    strike: Union[float, np.ndarray],
    expiry: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> np.ndarray:
    """Return Hagan's SABR implied volatility approximation."""

    if expiry <= 0:
        raise ValueError("expiry must be positive")

    F = np.asarray(forward, dtype=float)
    K = np.asarray(strike, dtype=float)
    if F.shape != K.shape and F.size not in (1,) and K.size not in (1,):
        raise ValueError("forward and strike must be broadcastable")

    F = np.broadcast_to(F, np.shape(K))
    K = np.broadcast_to(K, np.shape(F))

    beta = float(np.clip(beta, 0.0, 1.0))
    rho = float(np.clip(rho, -0.999, 0.999))
    alpha = float(max(alpha, 1e-12))
    nu = float(max(nu, 1e-12))

    with np.errstate(divide="ignore", invalid="ignore"):
        fk_beta = np.power(F * K, (1.0 - beta) / 2.0)
        fk_beta = np.where(fk_beta <= 0.0, 1e-12, fk_beta)
        log_fk = np.log(np.where(K == 0.0, 1.0, F / K))
        z = (nu / alpha) * fk_beta * log_fk

        sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z**2)
        numerator = sqrt_term + z - rho
        denominator = 1.0 - rho
        x_z = np.log(np.where(numerator <= 0.0, 1.0, numerator / denominator))

        small_z = np.abs(z) < 1e-12
        z_over_x = np.where(small_z, 1.0, z / x_z)

        one_minus_beta = 1.0 - beta
        one_minus_beta_sq = one_minus_beta**2

        log_fk_sq = log_fk**2
        log_fk_quartic = log_fk_sq**2

        term1 = alpha / (
            fk_beta
            * (
                1.0
                + (one_minus_beta_sq / 24.0) * log_fk_sq
                + (one_minus_beta_sq**2 / 1920.0) * log_fk_quartic
            )
        )

        term2 = z_over_x

        term3 = 1.0 + (
            ((one_minus_beta_sq / 24.0) * (alpha**2) / (fk_beta**2))
            + (rho * beta * nu * alpha) / (4.0 * fk_beta)
            + ((2.0 - 3.0 * rho**2) * nu**2 / 24.0)
        ) * expiry

        implied = term1 * term2 * term3

    implied = np.where(np.isnan(implied) | (implied <= 0.0), alpha / fk_beta, implied)
    return implied.astype(float)


class ArbitrageValidator:
    """Run no-arbitrage checks on a fitted volatility surface."""

    def __init__(self, *, tolerance: float = 1e-8, vol_column: str = "model_vol") -> None:
        self.tolerance = tolerance
        self.vol_column = vol_column

    def validate(
        self,
        surface: pd.DataFrame,
        *,
        tenor_column: str = "tenor",
        strike_column: str = "strike",
    ) -> ArbitrageCheckResult:
        """Validate the provided surface and collect arbitrage diagnostics."""

        if surface.empty:
            return ArbitrageCheckResult([], [], [])

        if self.vol_column not in surface.columns:
            raise KeyError(f"surface is missing required column '{self.vol_column}'")

        butterfly_violations: List[Dict[str, float]] = []
        calendar_violations: List[Dict[str, float]] = []
        tenor_monotonicity_violations: List[Dict[str, float]] = []

        for tenor, group in surface.groupby(tenor_column):
            ordered = group.sort_values(strike_column)
            vols = ordered[self.vol_column].to_numpy(dtype=float)
            if vols.size < 3:
                continue

            total_var = (vols**2) * float(tenor)
            second_diff = total_var[:-2] - 2.0 * total_var[1:-1] + total_var[2:]
            for idx, value in enumerate(second_diff, start=1):
                if value < -self.tolerance:
                    butterfly_violations.append(
                        {
                            "tenor": float(tenor),
                            "strike_left": float(ordered.iloc[idx - 1][strike_column]),
                            "strike_mid": float(ordered.iloc[idx][strike_column]),
                            "strike_right": float(ordered.iloc[idx + 1][strike_column]),
                            "violation": float(value),
                        }
                    )

        for strike, group in surface.groupby(strike_column):
            ordered = group.sort_values(tenor_column)
            vols = ordered[self.vol_column].to_numpy(dtype=float)
            tenors = ordered[tenor_column].to_numpy(dtype=float)
            if vols.size < 2:
                continue

            total_var = (vols**2) * tenors
            diff = np.diff(total_var)
            for idx, value in enumerate(diff):
                if value < -self.tolerance:
                    calendar_violations.append(
                        {
                            "strike": float(strike),
                            "tenor_start": float(tenors[idx]),
                            "tenor_end": float(tenors[idx + 1]),
                            "violation": float(value),
                        }
                    )

        base_tolerance = max(self.tolerance, 1e-4)
        tenor_summary = (
            surface.groupby(tenor_column)[self.vol_column]
            .mean()
            .sort_index()
        )
        tenor_vols = tenor_summary.to_numpy(dtype=float)
        tenor_values = tenor_summary.index.to_numpy(dtype=float)
        if tenor_vols.size > 1:
            total_var = (tenor_vols**2) * tenor_values
            drops = np.diff(total_var)
            for idx, value in enumerate(drops):
                if value < -base_tolerance:
                    tenor_monotonicity_violations.append(
                        {
                            "tenor_start": float(tenor_values[idx]),
                            "tenor_end": float(tenor_values[idx + 1]),
                            "violation": float(value),
                        }
                    )

        return ArbitrageCheckResult(
            butterfly_violations, calendar_violations, tenor_monotonicity_violations
        )


class SABRCalibrator:
    """Pipeline performing board cleaning, SABR fitting and arbitrage validation."""

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
        self.beta = float(np.clip(beta, 0.0, 1.0))
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.max_age_seconds = max_age_seconds
        self.mad_threshold = mad_threshold
        self.arbitrage_tolerance = arbitrage_tolerance
        self._validator = ArbitrageValidator(
            tolerance=arbitrage_tolerance, vol_column="model_vol"
        )

    def calibrate(
        self,
        board: Union[pd.DataFrame, Sequence[Mapping[str, object]]],
        *,
        now: Optional[datetime] = None,
        forward_curve: Optional[Union[Mapping[float, float], Callable[[float], float]]] = None,
        forward_column: str = "forward",
        tenor_column: str = "tenor",
        strike_column: str = "strike",
        bid_column: Optional[str] = "bid_iv",
        ask_column: Optional[str] = "ask_iv",
        vol_column: str = "mid_iv",
        timestamp_column: str = "timestamp",
        alternative_models: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> SABRCalibrationResult:
        """Run the full SABR calibration workflow."""

        clean_board_result = clean_option_board(
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

        cleaned = clean_board_result.data
        if cleaned.empty:
            raise ValueError("board cleaning removed all quotes; calibration aborted")

        tenor_results: List[SABRTenorCalibration] = []
        surface_rows: List[Dict[str, float]] = []

        grouped = cleaned.groupby(tenor_column)
        for tenor, group in grouped:
            tenor_value = float(tenor)
            strikes = group[strike_column].to_numpy(dtype=float)
            market_vols = group[vol_column].to_numpy(dtype=float)
            if strikes.size < 3:
                LOGGER.warning("Skipping tenor %.6f due to insufficient quotes", tenor_value)
                continue

            forward = self._resolve_forward(
                tenor_value, group, forward_curve=forward_curve, forward_column=forward_column
            )

            params, model_vols, rmse = self._calibrate_single_tenor(
                forward, tenor_value, strikes, market_vols
            )

            tenor_results.append(
                SABRTenorCalibration(
                    tenor=tenor_value,
                    parameters=params,
                    strikes=strikes,
                    market_vols=market_vols,
                    model_vols=model_vols,
                    rmse=rmse,
                )
            )

            for strike, market_vol, model_vol in zip(strikes, market_vols, model_vols):
                surface_rows.append(
                    {
                        "tenor": tenor_value,
                        "strike": float(strike),
                        "market_vol": float(market_vol),
                        "model_vol": float(model_vol),
                        "forward": float(forward),
                    }
                )

        if not tenor_results:
            raise ValueError("no tenors available for calibration")

        fitted_surface = pd.DataFrame(surface_rows)
        arbitrage = self._validator.validate(fitted_surface)

        regime = self._select_regime(tenor_results, alternative_models)

        return SABRCalibrationResult(
            clean_board=clean_board_result,
            tenor_results=tenor_results,
            arbitrage=arbitrage,
            fitted_surface=fitted_surface,
            regime=regime,
        )

    def _resolve_forward(
        self,
        tenor: float,
        group: pd.DataFrame,
        *,
        forward_curve: Optional[Union[Mapping[float, float], Callable[[float], float]]],
        forward_column: str,
    ) -> float:
        if forward_curve is not None:
            if callable(forward_curve):
                return float(forward_curve(tenor))

            if tenor in forward_curve:
                return float(forward_curve[tenor])
            # Attempt tolerant lookup if tenor slightly different
            for key, value in forward_curve.items():
                if math.isclose(key, tenor, rel_tol=0.0, abs_tol=1e-9):
                    return float(value)
            raise KeyError(f"tenor {tenor} not found in forward_curve")

        if forward_column in group.columns:
            forward_values = group[forward_column].to_numpy(dtype=float)
            return float(np.mean(forward_values))

        raise KeyError(f"forward information unavailable for tenor {tenor}")

    def _calibrate_single_tenor(
        self,
        forward: float,
        tenor: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,
    ) -> Tuple[SABRParameters, np.ndarray, float]:
        strikes = strikes.astype(float)
        market_vols = market_vols.astype(float)

        def residuals(params: np.ndarray) -> np.ndarray:
            alpha, rho, nu = params
            model_vols = hagan_implied_volatility(
                forward, strikes, tenor, alpha=abs(alpha), beta=self.beta, rho=np.tanh(rho), nu=abs(nu)
            )
            return model_vols - market_vols

        # Initial guesses
        atm_index = np.argmin(np.abs(strikes - forward))
        atm_vol = market_vols[atm_index]
        alpha0 = max(atm_vol * (forward ** (1 - self.beta)), 1e-3)
        rho0 = 0.0
        nu0 = 0.5

        x0 = np.array([alpha0, rho0, nu0], dtype=float)

        result = least_squares(
            residuals,
            x0,
            bounds=(np.array([1e-6, -5.0, 1e-6]), np.array([5.0, 5.0, 5.0])),
            max_nfev=self.max_iterations,
            ftol=self.tolerance,
            xtol=self.tolerance,
        )

        if not result.success:
            LOGGER.warning(
                "SABR calibration failed to converge for tenor %.6f: %s",
                tenor,
                result.message,
            )

        alpha = abs(result.x[0])
        rho = float(np.tanh(result.x[1]))
        nu = abs(result.x[2])

        model_vols = hagan_implied_volatility(
            forward, strikes, tenor, alpha=alpha, beta=self.beta, rho=rho, nu=nu
        )
        rmse = float(np.sqrt(np.mean((model_vols - market_vols) ** 2)))

        parameters = SABRParameters(alpha=alpha, beta=self.beta, rho=rho, nu=nu)
        return parameters, model_vols, rmse

    def _select_regime(
        self,
        tenor_results: Sequence[SABRTenorCalibration],
        alternative_models: Optional[Mapping[str, Sequence[float]]],
    ) -> str:
        sabr_rmse = [result.rmse for result in tenor_results]
        sabr_score = float(np.mean(sabr_rmse)) if sabr_rmse else float("nan")

        scores = {"sabr": sabr_score}
        if alternative_models:
            for name, values in alternative_models.items():
                array = np.asarray(list(values), dtype=float)
                if array.size == 0:
                    continue
                scores[name.lower()] = float(np.mean(array))

        best_model = min(scores.items(), key=lambda item: item[1])[0]
        return best_model

