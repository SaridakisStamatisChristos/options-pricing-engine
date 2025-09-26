"""Model selection and volatility surface construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from .boards import BoardCleaner, CleanBoard
from .heston_qe import HestonQECalibrator, HestonTenorResult
from .sabr import SABRCalibrator, SABRConfig, SABRTenorResult
from .validators import NoArbitrageValidator, ValidationReport


@dataclass(slots=True)
class TenorSelection:
    tenor: float
    model: str
    params: Dict[str, float]
    rmse: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenor": self.tenor,
            "model": self.model,
            "params": self.params,
            "rmse": self.rmse,
        }


@dataclass(slots=True)
class SurfaceBuildResult:
    surface_id: str
    clean_board: CleanBoard
    qc: Dict[str, Any]
    validation: ValidationReport
    selections: List[TenorSelection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surface_id": self.surface_id,
            "qc": self.qc,
            "validation": self.validation.to_dict(),
            "tenors": [selection.to_dict() for selection in self.selections],
        }


class SurfaceBuilder:
    """End-to-end orchestration of the volatility surface calibration workflow."""

    def __init__(
        self,
        *,
        cleaner: Optional[BoardCleaner] = None,
        sabr: Optional[SABRCalibrator] = None,
        validator: Optional[NoArbitrageValidator] = None,
        heston: Optional[HestonQECalibrator] = None,
    ) -> None:
        self._cleaner = cleaner or BoardCleaner()
        self._sabr = sabr or SABRCalibrator(SABRConfig())
        self._validator = validator or NoArbitrageValidator()
        self._heston = heston or HestonQECalibrator()

    def build(
        self,
        quotes: Iterable[Mapping[str, Any]],
        *,
        forward_curve: Mapping[float, float] | None = None,
        now: Optional[pd.Timestamp] = None,
        seed: int = 0,
        enable_heston: bool = True,
    ) -> SurfaceBuildResult:
        clean = self._cleaner.ingest(quotes, now=now, seed=seed)
        validation = self._validator.validate(clean)
        if not validation.is_ok:
            raise ValueError("arbitrage violations detected", validation.to_dict())

        sabr_results = self._sabr.calibrate(clean, forward_curve=forward_curve)
        if not sabr_results:
            raise ValueError("no tenors available for calibration")

        heston_results: List[HestonTenorResult] = []
        if enable_heston:
            try:
                heston_results = self._heston.calibrate(clean, forward_curve=forward_curve)
            except RuntimeError:
                heston_results = []
        heston_map = {result.tenor: result for result in heston_results}

        selections: List[TenorSelection] = []
        for result in sabr_results:
            heston_result = heston_map.get(result.tenor)
            best_model = "sabr"
            best_params = result.params
            best_rmse = result.rmse
            if heston_result is not None and heston_result.rmse < best_rmse:
                best_model = "heston_qe"
                best_params = heston_result.params
                best_rmse = heston_result.rmse
            selections.append(
                TenorSelection(
                    tenor=result.tenor,
                    model=best_model,
                    params={key: float(val) for key, val in best_params.items()},
                    rmse=float(best_rmse),
                )
            )

        surface_id = self._compute_surface_id(clean, selections)
        qc = {
            "board": clean.qc,
            "rmse": {selection.tenor: selection.rmse for selection in selections},
            "models": {selection.tenor: selection.model for selection in selections},
        }
        return SurfaceBuildResult(surface_id, clean, qc, validation, selections)

    def _compute_surface_id(self, clean: CleanBoard, selections: Sequence[TenorSelection]) -> str:
        payload = {
            "quotes": clean.to_records(),
            "selections": [selection.to_dict() for selection in selections],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return digest[:32]
