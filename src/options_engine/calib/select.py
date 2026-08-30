"""Model selection and volatility surface construction."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from numbers import Integral
from typing import Any

from .boards import BoardCleaner, CleanBoard
from .heston import HestonCalibrator, HestonTenorResult
from .sabr import SABRCalibrator, SABRConfig
from .svi import SSVICalibrationResult, SSVICalibrator
from .validators import NoArbitrageValidator, ValidationReport


@dataclass(slots=True)
class TenorSelection:
    tenor: float
    model: str
    params: dict[str, float]
    rmse: float
    selection_score: float
    criterion: str = "AICc"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenor": self.tenor,
            "model": self.model,
            "params": self.params,
            "rmse": self.rmse,
            "selection_score": self.selection_score,
            "criterion": self.criterion,
        }


@dataclass(slots=True)
class SurfaceBuildResult:
    surface_id: str
    clean_board: CleanBoard
    qc: dict[str, Any]
    validation: ValidationReport
    selections: list[TenorSelection]
    ssvi: SSVICalibrationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "qc": self.qc,
            "validation": self.validation.to_dict(),
            "tenors": [selection.to_dict() for selection in self.selections],
            "ssvi": self.ssvi.to_dict() if self.ssvi is not None else None,
        }


class SurfaceBuilder:
    """End-to-end orchestration of the volatility surface calibration workflow."""

    def __init__(
        self,
        *,
        cleaner: BoardCleaner | None = None,
        sabr: SABRCalibrator | None = None,
        validator: NoArbitrageValidator | None = None,
        heston: HestonCalibrator | None = None,
        ssvi: SSVICalibrator | None = None,
    ) -> None:
        if cleaner is not None and not isinstance(cleaner, BoardCleaner):
            raise TypeError("cleaner must be a BoardCleaner")
        if sabr is not None and not isinstance(sabr, SABRCalibrator):
            raise TypeError("sabr must be a SABRCalibrator")
        if validator is not None and not isinstance(validator, NoArbitrageValidator):
            raise TypeError("validator must be a NoArbitrageValidator")
        if heston is not None and not isinstance(heston, HestonCalibrator):
            raise TypeError("heston must be a HestonCalibrator")
        if ssvi is not None and not isinstance(ssvi, SSVICalibrator):
            raise TypeError("ssvi must be an SSVICalibrator")
        self._cleaner = cleaner if cleaner is not None else BoardCleaner()
        self._sabr = sabr if sabr is not None else SABRCalibrator(SABRConfig())
        self._validator = validator if validator is not None else NoArbitrageValidator()
        self._heston = heston if heston is not None else HestonCalibrator()
        self._ssvi = ssvi if ssvi is not None else SSVICalibrator()

    def build(
        self,
        quotes: Iterable[Mapping[str, Any]],
        *,
        forward_curve: Mapping[float, float] | None = None,
        now: datetime | None = None,
        seed: int = 0,
        enable_heston: bool = True,
        enable_ssvi: bool = True,
    ) -> SurfaceBuildResult:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 2**128 - 1:
            raise ValueError("seed must be an integer within [0, 2**128 - 1]")
        if not isinstance(enable_heston, bool):
            raise TypeError("enable_heston must be a boolean")
        if not isinstance(enable_ssvi, bool):
            raise TypeError("enable_ssvi must be a boolean")
        clean = self._cleaner.ingest(quotes, now=now, seed=seed)
        validation = self._validator.validate(clean)
        if not validation.is_ok:
            raise ValueError("arbitrage violations detected", validation.to_dict())

        sabr_results = self._sabr.calibrate(clean, forward_curve=forward_curve)
        if not sabr_results:
            raise ValueError("no tenors available for calibration")

        heston_results: list[HestonTenorResult] = []
        heston_error: str | None = None
        if enable_heston:
            try:
                heston_results = self._heston.calibrate(clean, forward_curve=forward_curve)
            except (RuntimeError, ValueError) as exc:
                heston_results = []
                heston_error = str(exc)
        heston_map = {result.tenor: result for result in heston_results}

        ssvi_result: SSVICalibrationResult | None = None
        ssvi_error: str | None = None
        if enable_ssvi:
            try:
                ssvi_result = self._ssvi.calibrate(clean, forward_curve=forward_curve)
            except (KeyError, RuntimeError, ValueError) as exc:
                ssvi_error = str(exc)

        selections: list[TenorSelection] = []
        for result in sabr_results:
            heston_result = heston_map.get(result.tenor)
            best_model = "sabr"
            best_params = result.params
            best_rmse = result.rmse
            observations = len(result.market_vols)
            best_score = self._aicc(
                result.rmse,
                observations,
                result.parameter_count,
            )
            if heston_result is not None:
                heston_score = self._aicc(
                    heston_result.rmse,
                    len(heston_result.market_vols),
                    heston_result.parameter_count,
                )
            else:
                heston_score = float("inf")
            if heston_result is not None and heston_score < best_score:
                best_model = "heston"
                best_params = heston_result.params
                best_rmse = heston_result.rmse
                best_score = heston_score
            selections.append(
                TenorSelection(
                    tenor=result.tenor,
                    model=best_model,
                    params={key: float(val) for key, val in best_params.items()},
                    rmse=float(best_rmse),
                    selection_score=float(best_score),
                )
            )

        surface_id = self._compute_surface_id(clean, selections, ssvi_result)
        qc = {
            "board": clean.qc,
            "rmse": {selection.tenor: selection.rmse for selection in selections},
            "models": {selection.tenor: selection.model for selection in selections},
            "heston": {
                "enabled": enable_heston,
                "status": "failed"
                if heston_error
                else "completed"
                if enable_heston
                else "disabled",
                "error": heston_error,
            },
            "heston_diagnostics": {
                result.tenor: {
                    "weighted_rmse": result.weighted_rmse,
                    "holdout_rmse": result.holdout_rmse,
                    "feller_ratio": result.feller_ratio,
                    "feller_satisfied": result.feller_satisfied,
                    "weighting": result.weighting,
                    "parameter_change_l2": result.parameter_change_l2,
                }
                for result in heston_results
            },
            "ssvi": {
                "enabled": enable_ssvi,
                "status": "failed" if ssvi_error else "completed" if enable_ssvi else "disabled",
                "error": ssvi_error,
                "minimum_density_factor": (
                    ssvi_result.minimum_density_factor if ssvi_result is not None else None
                ),
                "maximum_calendar_decrease": (
                    ssvi_result.maximum_calendar_decrease if ssvi_result is not None else None
                ),
            },
        }
        return SurfaceBuildResult(surface_id, clean, qc, validation, selections, ssvi_result)

    @staticmethod
    def _aicc(rmse: float, observations: int, parameters: int) -> float:
        """Small-sample Akaike score used instead of raw in-sample RMSE."""

        if (
            not math.isfinite(rmse)
            or rmse < 0.0
            or isinstance(observations, bool)
            or not isinstance(observations, Integral)
            or isinstance(parameters, bool)
            or not isinstance(parameters, Integral)
            or observations < 1
            or parameters < 1
        ):
            raise ValueError("AICc inputs are outside the supported domain")
        if observations <= parameters + 1:
            return float("inf")
        mean_squared_error = max(float(rmse) ** 2, 1e-16)
        aic = observations * math.log(mean_squared_error) + 2.0 * parameters
        correction = 2.0 * parameters * (parameters + 1) / (observations - parameters - 1)
        return float(aic + correction)

    def _compute_surface_id(
        self,
        clean: CleanBoard,
        selections: Sequence[TenorSelection],
        ssvi: SSVICalibrationResult | None = None,
    ) -> str:
        identity_columns = [
            column
            for column in ("tenor", "strike", "mid_iv", "forward", "option_type", "discount")
            if column in clean.quotes.columns
        ]
        identity_quotes = (
            clean.quotes[identity_columns]
            .sort_values(["tenor", "strike", "option_type"])
            .to_dict("records")
        )
        payload = {
            "quotes": identity_quotes,
            "selections": [selection.to_dict() for selection in selections],
            "ssvi": ssvi.to_dict() if ssvi is not None else None,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest
