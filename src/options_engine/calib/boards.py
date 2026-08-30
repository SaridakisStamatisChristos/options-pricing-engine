"""Quote board ingestion and cleaning utilities."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from numbers import Integral, Real
from typing import Any, ClassVar, cast

import numpy as np
import pandas as pd


@dataclass(slots=True)
class CleanBoard:
    """Container holding the cleaned quotes and QC metadata."""

    quotes: pd.DataFrame
    qc: dict[str, Any]

    def to_records(self) -> list[dict[str, Any]]:
        """Return the cleaned quotes as JSON serialisable records."""

        if self.quotes.empty:
            return []
        ordered = self.quotes.sort_values(["tenor", "strike", "option_type"]).reset_index(drop=True)
        records = cast(list[dict[str, Any]], ordered.to_dict("records"))
        for record in records:
            for key, value in list(record.items()):
                if isinstance(value, (datetime, np.datetime64, pd.Timestamp)):
                    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
                    if pd.isna(timestamp):
                        record[key] = None
                    else:
                        record[key] = timestamp.isoformat()
                elif isinstance(value, Real) and not isinstance(value, bool):
                    numeric = float(value)
                    record[key] = numeric if math.isfinite(numeric) else None
                elif value is pd.NA or value is pd.NaT:
                    record[key] = None
        return records


@dataclass(frozen=True, slots=True)
class BoardCleanerConfig:
    """Configuration for :class:`BoardCleaner`."""

    max_age_seconds: float = 5 * 60.0
    max_future_seconds: float = 30.0
    mad_threshold: float = 4.0
    tau_min_days: float = 1e-4
    sigma_min: float = 1e-4
    sigma_max: float = 5.0
    log_money_bounds: tuple[float, float] = (-4.0, 4.0)
    log_money_bins: Sequence[float] = tuple(np.linspace(-4.0, 4.0, 17))

    def __post_init__(self) -> None:
        for name, value in (
            ("max_age_seconds", self.max_age_seconds),
            ("max_future_seconds", self.max_future_seconds),
            ("mad_threshold", self.mad_threshold),
            ("tau_min_days", self.tau_min_days),
            ("sigma_min", self.sigma_min),
            ("sigma_max", self.sigma_max),
        ):
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise TypeError(f"{name} must be a finite real number")
        if not 0.0 <= self.max_age_seconds <= 604_800.0:
            raise ValueError("max_age_seconds must be within [0, 604800]")
        if not 0.0 <= self.max_future_seconds <= 3_600.0:
            raise ValueError("max_future_seconds must be within [0, 3600]")
        if not 0.0 < self.mad_threshold <= 100.0:
            raise ValueError("mad_threshold must be within (0, 100]")
        if not 0.0 < self.tau_min_days <= 36_500.0:
            raise ValueError("tau_min_days must be within (0, 36500]")
        if not 0.0 < self.sigma_min < self.sigma_max <= 5.0:
            raise ValueError("sigma bounds must satisfy 0 < sigma_min < sigma_max <= 5")
        if (
            isinstance(self.log_money_bounds, (str, bytes))
            or not isinstance(self.log_money_bounds, Sequence)
            or len(self.log_money_bounds) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, Real)
                for value in self.log_money_bounds
            )
        ):
            raise TypeError("log_money_bounds must contain two real numbers")
        low, high = (float(value) for value in self.log_money_bounds)
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("log_money_bounds must be finite and strictly increasing")
        object.__setattr__(self, "log_money_bounds", (low, high))
        if (
            isinstance(self.log_money_bins, (str, bytes))
            or not isinstance(self.log_money_bins, Sequence)
            or len(self.log_money_bins) > 10_000
            or any(
                isinstance(value, bool) or not isinstance(value, Real)
                for value in self.log_money_bins
            )
        ):
            raise TypeError("log_money_bins must be a bounded sequence of real numbers")
        bins = tuple(float(value) for value in self.log_money_bins)
        if len(bins) < 2 or not all(math.isfinite(value) for value in bins):
            raise ValueError("log_money_bins must contain at least two finite values")
        if not all(left < right for left, right in itertools.pairwise(bins)):
            raise ValueError("log_money_bins must be strictly increasing")
        if bins[0] > low or bins[-1] < high:
            raise ValueError("log_money_bins must span log_money_bounds")
        object.__setattr__(self, "log_money_bins", bins)


class BoardCleaner:
    """Ingest raw quotes and apply deterministic cleaning rules."""

    REQUIRED_COLUMNS: ClassVar[frozenset[str]] = frozenset({"tenor", "strike", "mid_iv", "forward"})
    MAX_QUOTES: ClassVar[int] = 100_000

    def __init__(self, config: BoardCleanerConfig | None = None) -> None:
        self._config = config or BoardCleanerConfig()

    def ingest(
        self,
        quotes: Iterable[Mapping[str, Any]],
        *,
        now: datetime | None = None,
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

        if now is not None:
            if not isinstance(now, datetime):
                raise TypeError("now must be a datetime")
            if now.tzinfo is None or now.utcoffset() is None:
                raise ValueError("now must be timezone-aware")
        if isinstance(quotes, (str, bytes)) or not isinstance(quotes, Iterable):
            raise TypeError("quotes must be an iterable of mappings")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        if not 0 <= seed <= 2**128 - 1:
            raise ValueError("seed must be an integer within [0, 2**128 - 1]")
        quote_rows = list(itertools.islice(quotes, self.MAX_QUOTES + 1))
        if len(quote_rows) > self.MAX_QUOTES:
            raise ValueError(f"quote board exceeds the {self.MAX_QUOTES}-row limit")
        if not all(isinstance(row, Mapping) for row in quote_rows):
            raise TypeError("every quote must be a mapping")
        df = pd.DataFrame(quote_rows)
        if df.empty:
            return CleanBoard(df, self._empty_report())

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise KeyError(f"missing required columns: {missing_str}")

        total = len(df)
        df["_source_index"] = np.arange(total, dtype=int)
        removals: list[dict[str, Any]] = []
        df = df.replace([np.inf, -np.inf], np.nan)

        numeric_columns = ["tenor", "strike", "mid_iv", "forward"]
        if "discount" in df.columns:
            numeric_columns.append("discount")
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        invalid_numeric = df[numeric_columns].isna().any(axis=1)
        removals.extend(
            {"index": int(index), "reason": "invalid_numeric"}
            for index in df.loc[invalid_numeric, "_source_index"]
        )
        df = df.loc[~invalid_numeric].copy()

        cfg = self._config
        domain_valid = (
            (df["tenor"] >= cfg.tau_min_days / 365.0)
            & (df["tenor"] <= 100.0)
            & (df["strike"] > 0.0)
            & (df["strike"] <= 1e12)
            & (df["forward"] > 0.0)
            & (df["forward"] <= 1e12)
            & df["mid_iv"].between(cfg.sigma_min, cfg.sigma_max, inclusive="both")
        )
        if "discount" in df.columns:
            domain_valid &= df["discount"].between(1e-12, 1e6, inclusive="both")
        removals.extend(
            {"index": int(index), "reason": "out_of_domain"}
            for index in df.loc[~domain_valid, "_source_index"]
        )
        df = df.loc[domain_valid].copy()

        if df.empty:
            df = df.drop(columns=["_source_index"], errors="ignore")
            return CleanBoard(df, self._report(total, removals, retained=0))

        has_bid = "bid_iv" in df.columns
        has_ask = "ask_iv" in df.columns
        if has_bid != has_ask:
            raise KeyError("bid_iv and ask_iv must be supplied together")
        if has_bid and has_ask:
            df["bid_iv"] = pd.to_numeric(df["bid_iv"], errors="coerce")
            df["ask_iv"] = pd.to_numeric(df["ask_iv"], errors="coerce")
            invalid_spread = (
                df[["bid_iv", "ask_iv"]].isna().any(axis=1)
                | (df["bid_iv"] < 0.0)
                | (df["ask_iv"] < 0.0)
                | (df["bid_iv"] > cfg.sigma_max)
                | (df["ask_iv"] > cfg.sigma_max)
                | (df["bid_iv"] > df["ask_iv"])
                | (df["mid_iv"] < df["bid_iv"])
                | (df["mid_iv"] > df["ask_iv"])
            )
            if invalid_spread.any():
                removals.extend(
                    {
                        "index": int(row["_source_index"]),
                        "tenor": float(row.tenor),
                        "strike": float(row.strike),
                        "reason": "invalid_spread",
                    }
                    for _idx, row in df.loc[invalid_spread].iterrows()
                )
            df = df.loc[~invalid_spread].copy()

        if "timestamp" in df.columns:
            timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if now is None:
                now = datetime.now(UTC)
            age = (pd.Timestamp(now).tz_convert("UTC") - timestamps).dt.total_seconds()
            stale_mask = (
                timestamps.isna() | (age > cfg.max_age_seconds) | (age < -cfg.max_future_seconds)
            )
            if stale_mask.any():
                removals.extend(
                    {
                        "index": int(row["_source_index"]),
                        "tenor": float(row.tenor),
                        "strike": float(row.strike),
                        "reason": "invalid_timestamp",
                    }
                    for _idx, row in df.loc[stale_mask].iterrows()
                )
            df = df.loc[~stale_mask].copy()

        if df.empty:
            df = df.drop(columns=["_source_index"], errors="ignore")
            return CleanBoard(df, self._report(total, removals, retained=0))

        option_type = df.get("option_type")
        if option_type is None:
            df["option_type"] = "CALL"
        else:
            df["option_type"] = option_type.fillna("CALL").str.upper()
            invalid_option = ~df["option_type"].isin({"CALL", "PUT"})
            if invalid_option.any():
                removals.extend(
                    {
                        "index": int(row["_source_index"]),
                        "tenor": float(row.tenor),
                        "strike": float(row.strike),
                        "reason": "invalid_option_type",
                    }
                    for _idx, row in df.loc[invalid_option].iterrows()
                )
            df = df.loc[~invalid_option]

        log_money = np.log(np.clip(df["strike"] / df["forward"], 1e-12, None))
        bounds_low, bounds_high = cfg.log_money_bounds
        out_of_bounds = (log_money < bounds_low) | (log_money > bounds_high)
        if out_of_bounds.any():
            removals.extend(
                {
                    "index": int(row["_source_index"]),
                    "tenor": float(row.tenor),
                    "strike": float(row.strike),
                    "reason": "out_of_bounds",
                }
                for _idx, row in df.loc[out_of_bounds].iterrows()
            )
        df = df.loc[~out_of_bounds].copy()

        if df.empty:
            df = df.drop(columns=["_source_index"], errors="ignore")
            return CleanBoard(df, self._report(total, removals, retained=0))

        df["log_moneyness"] = log_money.loc[df.index]

        residuals: list[dict[str, Any]] = []
        filtered_groups: list[pd.DataFrame] = []

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
                    tolerance = 8.0 * np.finfo(float).eps * max(1.0, abs(median))
                    scaled = np.where(
                        np.abs(bucket_values - median) <= tolerance,
                        0.0,
                        float("inf"),
                    )
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
                                "index": int(row["_source_index"]),
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
        cleaned = cleaned.drop(columns=["_source_index"], errors="ignore")
        cleaned = cleaned.sort_values(["tenor", "strike", "option_type"]).reset_index(drop=True)

        qc = self._report(total, removals, retained=len(cleaned))
        qc["residuals"] = residuals

        return CleanBoard(cleaned, qc)

    def _empty_report(self) -> dict[str, Any]:
        return self._report(0, [], retained=0)

    def _report(
        self,
        total: int,
        removals: Sequence[Mapping[str, Any]],
        *,
        retained: int,
    ) -> dict[str, Any]:
        counts: MutableMapping[str, int] = {
            "total": int(total),
            "retained": int(retained),
            "dropped": int(total - retained),
        }
        reason_counts: dict[str, int] = {}
        for removal in removals:
            reason = str(removal.get("reason", "other"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        counts.update(
            {f"dropped_{reason}": count for reason, count in sorted(reason_counts.items())}
        )
        return {"counts": dict(counts), "removals": list(removals)}
