#!/usr/bin/env python3
"""Generate concise, deterministic Step 11 calibration evidence.

JSON and Markdown generation require only runtime dependencies.  ``--plots`` is
strictly optional and imports matplotlib only after it is requested.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from options_engine.calib import (
    HestonCalibrator,
    HestonConfig,
    SABRCalibrator,
    SABRConfig,
    SSVICalibrator,
    SSVIConfig,
    adversarial_board,
    heston_recovery_board,
    sabr_recovery_board,
    ssvi_recovery_board,
)


def generate() -> dict[str, Any]:
    """Run bounded recovery and adverse cases and return JSON-native evidence."""

    sabr_clean_fixture = sabr_recovery_board()
    sabr_noisy_fixture = sabr_recovery_board(noise_amplitude=0.0005)
    sabr_config = SABRConfig(
        beta=sabr_clean_fixture.parameters["beta"],
        seeds=(0, 1, 2),
        holdout_policy="alternating",
    )
    sabr_clean = SABRCalibrator(sabr_config).calibrate_detailed(sabr_clean_fixture.board)
    sabr_noisy = SABRCalibrator(sabr_config).calibrate_detailed(sabr_noisy_fixture.board)

    heston_fixture = heston_recovery_board()
    heston_config = HestonConfig(
        calibration_mode="global",
        pricing_method="gauss_laguerre",
        weighting="uniform",
        seeds=(0,),
        holdout_fraction=0.1,
        holdout_tenors=(0.75,),
        max_iterations=250,
    )
    heston_comparison = HestonCalibrator(heston_config).compare_modes(heston_fixture.board)

    ssvi_clean_fixture = ssvi_recovery_board()
    ssvi_noisy_fixture = ssvi_recovery_board(noise_amplitude=0.0005)
    ssvi_config = SSVIConfig(seeds=(0, 1, 2), weighting="uniform", holdout_policy="alternating")
    ssvi_clean = SSVICalibrator(ssvi_config).calibrate(ssvi_clean_fixture.board)
    ssvi_noisy = SSVICalibrator(ssvi_config).calibrate(ssvi_noisy_fixture.board)
    outlier = SSVICalibrator(SSVIConfig(seeds=(0,), weighting="uniform")).calibrate(
        adversarial_board("gross_outlier")
    )

    return {
        "schema_version": 2,
        "determinism": "No wall-clock timestamps or unseeded random state are used.",
        "independence": {
            "heston_generator": heston_fixture.generator,
            "heston_recovery": "Gauss-Laguerre",
            "sabr_inverse_crime": "Hagan formula generates and recovers; observable recovery only.",
            "ssvi_inverse_crime": "Power-law SSVI generates and recovers; observable recovery only.",
        },
        "sabr_noise_free": sabr_clean.to_dict(),
        "sabr_noisy": sabr_noisy.to_dict(),
        "heston_comparison": heston_comparison.to_dict(),
        "ssvi_noise_free": ssvi_clean.to_dict(),
        "ssvi_noisy": ssvi_noisy.to_dict(),
        "adversarial_gross_outlier": outlier.to_dict(),
    }


def _metric(result: dict[str, Any]) -> float:
    if "in_sample_weighted_rmse" in result:
        return float(result["in_sample_weighted_rmse"])
    if "weighted_rmse" in result:
        return float(result["weighted_rmse"])
    return float(result["rmse"])


def _detail(result: dict[str, Any]) -> dict[str, Any]:
    """Select slice evidence when an aggregate intentionally contains only rollups."""

    tenor_results = result.get("tenor_results")
    if isinstance(tenor_results, list) and tenor_results:
        first = tenor_results[0]
        if isinstance(first, dict):
            return first
    return result


def markdown(report: dict[str, Any]) -> str:
    """Render the important decisions without copying the observation tables."""

    comparison = report["heston_comparison"]
    cases = (
        ("SABR noise-free", report["sabr_noise_free"]),
        ("SABR mildly noisy", report["sabr_noisy"]),
        ("Heston per-tenor COS→GL", comparison["per_tenor"]),
        ("Heston global COS→GL", comparison["global"]),
        ("SSVI noise-free", report["ssvi_noise_free"]),
        ("SSVI mildly noisy", report["ssvi_noisy"]),
        ("SSVI gross outlier", report["adversarial_gross_outlier"]),
    )
    rows = []
    for name, result in cases:
        detail = _detail(result)
        holdout = result.get("holdout_rmse")
        holdout_text = "—" if holdout is None else f"{float(holdout):.8g}"
        sensitivity = detail.get("initialization_sensitivity")
        if isinstance(sensitivity, list):
            starts = ", ".join(str(item.get("classification", "n/a")) for item in sensitivity)
        elif isinstance(sensitivity, dict):
            starts = str(sensitivity.get("classification", "n/a"))
        else:
            starts = "n/a"
        rows.append(
            f"| {name} | {_metric(result):.8g} | {holdout_text} | "
            f"{result.get('fit_quality', 'n/a')} | {starts} |"
        )
    body = "\n".join(rows)
    delta = comparison["global_minus_per_tenor"]
    return f"""# Calibration validation snapshot

Generated deterministically by `reports/generate_calibration_validation.py`.
This is evidence, not a claim of parameter confidence or universal model validity.

| Case | training weighted RMSE | holdout RMSE | classification | start sensitivity |
|---|---:|---:|---|---|
{body}

The global-minus-per-tenor training RMSE is
`{float(delta["in_sample_weighted_rmse"]):.8g}`. No winner is selected: the JSON
retains residuals, parameter stability, strike holdouts, and the global fit's
whole-tenor holdout separately.

Heston quotes are generated with the independent COS family and recovered with
Gauss-Laguerre. SABR and SSVI cases are unavoidable inverse-crime examples and
are used for observable recovery and diagnostic regression only.
"""


def concise(report: dict[str, Any]) -> dict[str, Any]:
    """Return reviewed headline evidence; detailed JSON remains opt-in."""

    comparison = report["heston_comparison"]
    source_cases = {
        "sabr_noise_free": report["sabr_noise_free"],
        "sabr_noisy": report["sabr_noisy"],
        "heston_per_tenor": comparison["per_tenor"],
        "heston_global": comparison["global"],
        "ssvi_noise_free": report["ssvi_noise_free"],
        "ssvi_noisy": report["ssvi_noisy"],
        "adversarial_gross_outlier": report["adversarial_gross_outlier"],
    }
    cases: dict[str, dict[str, Any]] = {}
    for name, result in source_cases.items():
        detail = _detail(result)
        cases[name] = {
            key: detail.get(key)
            for key in (
                "rmse",
                "weighted_rmse",
                "in_sample_weighted_rmse",
                "holdout_rmse",
                "strike_holdout_rmse",
                "tenor_holdout_rmse",
                "fit_quality",
                "minimum_density_factor",
                "maximum_calendar_decrease",
            )
            if key in detail
        }
    return {
        "schema_version": report["schema_version"],
        "determinism": report["determinism"],
        "independence": report["independence"],
        "cases": cases,
    }


def plots(report: dict[str, Any], directory: Path) -> None:
    """Write optional diagnostic plots; matplotlib is not a runtime dependency."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional local tooling
        raise SystemExit("--plots requires the optional matplotlib package") from exc

    directory.mkdir(parents=True, exist_ok=True)
    sources = {
        "sabr_noisy": report["sabr_noisy"],
        "heston_global": report["heston_comparison"]["global"],
        "ssvi_noisy": report["ssvi_noisy"],
        "ssvi_outlier": report["adversarial_gross_outlier"],
    }
    for name, result in sources.items():
        residuals = _detail(result).get("residuals", [])
        if not residuals:
            continue
        x = [float(row["log_moneyness"]) for row in residuals]
        y = [float(row["residual"]) for row in residuals]
        colors = ["tab:orange" if row["is_holdout"] else "tab:blue" for row in residuals]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(
            x,
            [float(row["market_volatility"]) for row in residuals],
            c=colors,
            marker="o",
            label="market",
        )
        axes[0].scatter(
            x,
            [float(row["fitted_volatility"]) for row in residuals],
            c=colors,
            marker="x",
            label="fitted",
        )
        axes[0].set(xlabel="log-moneyness", ylabel="volatility", title=f"{name}: smile")
        axes[0].legend()
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].scatter(x, y, c=colors, s=18)
        axes[1].set(
            xlabel="log-moneyness",
            ylabel="fitted - market volatility",
            title=f"{name}: residuals",
        )
        figure.tight_layout()
        figure.savefig(directory / f"{name}_smile_and_residuals.png", dpi=150)
        plt.close(figure)

        sensitivity = _detail(result).get("initialization_sensitivity")
        if not isinstance(sensitivity, dict):
            continue
        attempts = sensitivity.get("attempts", [])
        successful = [row for row in attempts if row.get("success") and row.get("parameters")]
        if len(successful) < 2:
            continue
        figure, axis = plt.subplots(figsize=(7, 4))
        for attempt in successful:
            axis.plot(attempt["parameters"], marker="o", alpha=0.7, label=str(attempt["seed"]))
        axis.set(
            xlabel="parameter index",
            ylabel="converged parameter value",
            title=f"{name}: successful starts",
        )
        axis.legend(title="seed")
        figure.tight_layout()
        figure.savefig(directory / f"{name}_start_stability.png", dpi=150)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json", type=Path, default=Path("reports/calibration_validation_snapshot.json")
    )
    parser.add_argument("--markdown", type=Path, default=Path("reports/CALIBRATION_VALIDATION.md"))
    parser.add_argument("--plots", type=Path, help="optional directory for PNG residual plots")
    parser.add_argument(
        "--detailed-json",
        type=Path,
        help="optional full observation-level JSON output (not committed)",
    )
    args = parser.parse_args()
    report = generate()
    args.json.write_text(
        json.dumps(concise(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.detailed_json is not None:
        args.detailed_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    args.markdown.write_text(markdown(report), encoding="utf-8")
    if args.plots is not None:
        plots(report, args.plots)


if __name__ == "__main__":
    main()
