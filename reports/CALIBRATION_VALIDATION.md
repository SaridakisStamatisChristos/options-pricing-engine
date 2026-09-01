# Calibration validation snapshot

Generated deterministically by `reports/generate_calibration_validation.py`.
This is evidence, not a claim of parameter confidence or universal model validity.

| Case | training weighted RMSE | holdout RMSE | classification | start sensitivity |
|---|---:|---:|---|---|
| SABR noise-free | 1.9631675e-15 | 2.0385966e-15 | good | stable |
| SABR mildly noisy | 0.00020880743 | 0.00055277126 | good | stable |
| Heston per-tenor COS→GL | 6.4513283e-08 | 4.2112425e-07 | acceptable | stable, stable, stable |
| Heston global COS→GL | 5.0223268e-10 | 1.0276896e-09 | acceptable | stable |
| SSVI noise-free | 5.1701738e-15 | 4.3249816e-15 | good | stable |
| SSVI mildly noisy | 0.00024941292 | 0.00056960739 | good | stable |
| SSVI gross outlier | 0.04585923 | — | poor | stable |

The global-minus-per-tenor training RMSE is
`-6.401105e-08`. No winner is selected: the JSON
retains residuals, parameter stability, strike holdouts, and the global fit's
whole-tenor holdout separately.

Heston quotes are generated with the independent COS family and recovered with
Gauss-Laguerre. SABR and SSVI cases are unavoidable inverse-crime examples and
are used for observable recovery and diagnostic regression only.
