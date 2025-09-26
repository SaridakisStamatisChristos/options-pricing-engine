"""Quote board ingestion and cleaning utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CleanBoard:
    """Container holding the cleaned quotes and QC metadata."""

    quotes: pd.DataFrame
    qc: Dict[str, Any]

    def to_records(self) -> List[Dict[str, Any]]:
        """Return the cleaned quotes as JSON serialisable records."""

        if self.quotes.empty:
            return []
        ordered = self.quotes.sort_values(["tenor", "strike", "option_type"]).reset_index(drop=True)
        records = ordered.to_dict("records")
        for record in records:
            for key, value in list(record.items()):
                if isinstance(value, (np.floating, np.integer)):
                    record[key] = float(value)
                elif isinstance(value, (datetime, np.datetime64, pd.Timestamp)):
                    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
                    if pd.isna(timestamp):
                        record[key] = None
                    else:
                        record[key] = timestamp.isoformat()
        return records


@dataclass(slots=True)
class BoardCleanerConfig:
    """Configuration for :class:`BoardCleaner`."""

    max_age_seconds: float = 5 * 60.0
    mad_threshold: float = 4.0
    tau_min_days: float = 1e-4
    sigma_min: float = 1e-4
    log_money_bounds: Tuple[float, float] = (-4.0, 4.0)
    log_money_bins: Sequence[float] = tuple(np.linspace(-4.0, 4.0, 17))


class BoardCleaner:
    """Ingest raw quotes and apply deterministic cleaning rules."""

    REQUIRED_COLUMNS = {"tenor", "strike", "mid_iv", "forward"}

    def __init__(self, config: Optional[BoardCleanerConfig] = None) -> None:
        self._config = config or BoardCleanerConfig()

    def ingest(
        self,
        quotes: Iterable[Mapping[str, Any]],
        *,
        now: Optional[datetime] = None,
        seed: int = 0,
    ) -> CleanBoard:
        """Clean the provided quotes and return a :class:`CleanBoard`.

        Parameters
        ----------
        quotes:
            Iterable of mappings representing raw quotes. Each mapping must include
            at least ``tenor`` (in years), ``strike``, ``mid_iv`` (Black implied vol),
            and ``forward``. Optional columns include ``option_type`` (``CALL`` or
            ``PUT``), ``bid_iv``, ``ask_iv`` and ``timestamp``. Missing option types
            default to calls. Any additional fields are preserved during cleaning.
        now:
            Timestamp used to evaluate quote staleness. Defaults to ``datetime.now(UTC)``.
        seed:
            RNG seed used when breaking ties deterministically. The cleaning pipeline
            itself is deterministic but we honour the seed to make any potential
            floating point ties reproducible across platforms.
        """

        df = pd.DataFrame(list(quotes))
        if df.empty:
            return CleanBoard(df, self._empty_report())

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise KeyError(f"missing required columns: {missing_str}")

        df = df.replace([np.nan, np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["tenor", "strike", "mid_iv", "forward"])

        df["tenor"] = pd.to_numeric(df["tenor"], errors="coerce")
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["mid_iv"] = pd.to_numeric(df["mid_iv"], errors="coerce")
        df["forward"] = pd.to_numeric(df["forward"], errors="coerce")

        df = df.dropna(subset=["tenor", "strike", "mid_iv", "forward"])

        cfg = self._config
        df = df[df["tenor"] >= max(cfg.tau_min_days / 365.0, 0.0)]
        df = df[df["mid_iv"] >= cfg.sigma_min]

        if df.empty:
            return CleanBoard(df, self._empty_report())

        total = len(df)
        removals: List[Dict[str, Any]] = []

        if "bid_iv" in df.columns and "ask_iv" in df.columns:
            df["bid_iv"] = pd.to_numeric(df["bid_iv"], errors="coerce")
            df["ask_iv"] = pd.to_numeric(df["ask_iv"], errors="coerce")
            crossed = df["bid_iv"] > df["ask_iv"]
            if crossed.any():
                removals.extend(
                    {
                        "index": int(idx),
                        "tenor": float(row.tenor),
                        "strike": float(row.strike),
                        "reason": "crossed",
                    }
                    for idx, row in df.loc[crossed].iterrows()
                )
            df = df.loc[~crossed]

        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if now is None:
                now = datetime.now(UTC)
            age = (now - timestamps).dt.total_seconds()
            stale_mask = age > cfg.max_age_seconds
            if stale_mask.any():
                removals.extend(
                    {
                        "index": int(idx),
                        "tenor": float(row.tenor),
                        "strike": float(row.strike),
                        "reason": "stale",
                    }
                    for idx, row in df.loc[stale_mask].iterrows()
                )
            df = df.loc[~stale_mask]

        df = df.dropna(subset=["tenor", "strike", "mid_iv", "forward"])
        if df.empty:
            return CleanBoard(df, self._report(total, removals, retained=0))

        option_type = df.get("option_type")
        if option_type is None:
            df["option_type"] = "CALL"
        else:
            df["option_type"] = option_type.fillna("CALL").str.upper()
            df.loc[~df["option_type"].isin({"CALL", "PUT"}), "option_type"] = "CALL"

        log_money = np.log(np.clip(df["strike"] / df["forward"], 1e-12, None))
        bounds_low, bounds_high = cfg.log_money_bounds
        out_of_bounds = (log_money < bounds_low) | (log_money > bounds_high)
        if out_of_bounds.any():
            removals.extend(
                {
                    "index": int(idx),
                    "tenor": float(row.tenor),
                    "strike": float(row.strike),
                    "reason": "out_of_bounds",
                }
                for idx, row in df.loc[out_of_bounds].iterrows()
            )
        df = df.loc[~out_of_bounds].copy()

        if df.empty:
            return CleanBoard(df, self._report(total, removals, retained=0))

        df["log_moneyness"] = log_money.loc[df.index]

        rng = np.random.default_rng(seed)

        residuals: List[Dict[str, Any]] = []
        filtered_groups: List[pd.DataFrame] = []

        for tenor, tenor_df in df.groupby("tenor", sort=True):
            tenor_df = tenor_df.sort_values("log_moneyness").reset_index(drop=True)
            bucket_ids = np.digitize(tenor_df["log_moneyness"], cfg.log_money_bins, right=True)
            keep_mask = np.ones(len(tenor_df), dtype=bool)
            for bucket in np.unique(bucket_ids):
                bucket_mask = bucket_ids == bucket
                bucket_values = tenor_df.loc[bucket_mask, "mid_iv"].to_numpy(dtype=float)
                if bucket_values.size == 0:
                    continue
                median = float(np.median(bucket_values))
                mad = float(np.median(np.abs(bucket_values - median)))
                scaled = np.zeros_like(bucket_values)
                if mad > 0.0:
                    scaled = np.abs(bucket_values - median) / (1.4826 * mad)
                else:
                    scaled = np.abs(bucket_values - median)
                residuals.append(
                    {
                        "tenor": float(tenor),
                        "bucket": int(bucket),
                        "median": median,
                        "mad": mad,
                    }
                )
                mask = scaled <= cfg.mad_threshold
                if not np.all(mask):
                    drop_indices = np.where(~mask)[0]
                    rng_order = np.argsort(drop_indices)
                    for local_idx in drop_indices[rng_order]:
                        row = tenor_df.iloc[int(np.flatnonzero(bucket_mask)[local_idx])]
                        removals.append(
                            {
                                "index": int(row.name),
                                "tenor": float(row.tenor),
                                "strike": float(row.strike),
                                "reason": "outlier",
                            }
                        )
                keep_mask[bucket_mask] &= mask
            filtered_groups.append(tenor_df.loc[keep_mask])

        if filtered_groups:
            cleaned = pd.concat(filtered_groups, ignore_index=True)
        else:
            cleaned = pd.DataFrame(columns=df.columns)

        cleaned = cleaned.reset_index(drop=True)
        cleaned = cleaned.sort_values(["tenor", "strike", "option_type"]).reset_index(drop=True)

        qc = self._report(total, removals, retained=len(cleaned))
        qc["residuals"] = residuals

        return CleanBoard(cleaned, qc)

    def _empty_report(self) -> Dict[str, Any]:
        return self._report(0, [], retained=0)

    def _report(
        self,
        total: int,
        removals: Sequence[Mapping[str, Any]],
        *,
        retained: int,
    ) -> Dict[str, Any]:
        counts: MutableMapping[str, int] = {
            "total": int(total),
            "retained": int(retained),
            "dropped": int(total - retained),
        }
        reason_counts: Dict[str, int] = {}
        for removal in removals:
            reason = str(removal.get("reason", "other"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        counts.update({f"dropped_{reason}": count for reason, count in sorted(reason_counts.items())})
        return {"counts": dict(counts), "removals": list(removals)}
