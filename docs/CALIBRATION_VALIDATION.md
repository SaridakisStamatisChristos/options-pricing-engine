# Calibration validation and model selection

Calibration minimizes a numerical objective; it does **not** prove that a model
is economically valid, identified, or suitable outside its quote range. Review
observation-level residuals and diagnostics before using calibrated output.

## Evidence hierarchy

1. **Admissibility first.** Invalid parameters, failed optimization, butterfly or
   calendar arbitrage classify a fit as invalid regardless of RMSE.
2. **Out-of-sample evidence.** Training and deterministic holdout errors remain
   separate. Wing holdouts test extrapolation more severely than interleaved
   holdouts; an ATM holdout tests whether wings determine the smile centre.
3. **Residual structure.** Inspect bias, MAE, maximum error and percentiles as well
   as RMSE. A small average must not conceal a large wing miss or single outlier.
4. **Numerical stability.** Failed-start fraction, objective and parameter spread,
   and materially different near-optimal solutions expose initialization risk.
5. **Local identification.** Jacobian singular values, effective rank and condition
   number are local numerical diagnostics. Approximate covariance/correlation is
   not a confidence interval and depends on local linearization and residual scale.
6. **Boundary and weight risk.** Review economic and transformed bound proximity,
   arbitrage-constraint slack, effective sample size and top-weight concentration.

`good`, `acceptable`, `poor`, `unstable`, and `invalid` are conservative summaries,
not marketing labels. The reason codes and underlying metrics are authoritative.

## Model-specific interpretation

### SABR

Fixed beta removes one poorly identified direction but does not identify alpha,
rho and nu automatically. Short expiries, sparse/ATM-clustered strikes, extreme
rho or nu, and use beyond the calibrated strike interval deserve warnings. With
fitted beta, inspect alpha/beta correlation and conditioning rather than treating
the calibrated beta as an independently measured economic parameter.

### Heston

Per-tenor calibration can fit local smiles closely but permits economically large
parameter jumps. Global calibration is coherent and supports whole-tenor holdout,
but is not automatically superior: it may underfit genuine term structure. Compare
strike and tenor holdout errors, Feller status, bounds, cross-tenor changes, and the
local identification of `v0`, `theta`, `kappa`, `vol_of_vol`, and `rho`.

### raw SVI and SSVI

Raw SVI is a slice parameterization, **not** a globally arbitrage-safe surface. A
raw-SVI slice with negative density factor or dense-grid butterfly violations is
invalid even at negligible RMSE. Global SSVI applies explicit sufficient shape
constraints and dense calendar/butterfly validation. Still inspect wing/curvature
slack and isotonic ATM projection: a large projection means quotes themselves were
economically inconsistent. Interpolated and extrapolated points must be labelled.

## Reproducible evidence

`options_engine.calib.datasets` supplies deterministic noise-free, mildly noisy,
and adversarial boards. Heston data use COS generation for Gauss–Laguerre recovery;
the SABR and SSVI fixtures necessarily use their calibration formula and document
that inverse crime. Run `python reports/generate_calibration_validation.py` to
write sorted JSON and concise Markdown. There is no plotting dependency; residual
records can be plotted offline with optional tooling without affecting runtime.

The report includes noise-free and noisy SABR/SSVI recovery, Heston per-tenor and
global recovery side by side, strike and whole-tenor holdouts, optimizer-start
sensitivity, and an outlier board. It deliberately makes no automatic model
selection. Pass `--plots reports/plots` to create residual-by-log-moneyness PNGs
when matplotlib is installed; matplotlib is neither imported nor required by the
core package.

## Performance discipline

Detailed audits intentionally cost more than consuming an already calibrated
parameter vector: multi-start optimization, holdouts, independent Heston pricing,
and Jacobian diagnostics perform real additional work. The regular calibrator
entry points remain backward compatible, while the report is an offline workflow.
Run `python reports/benchmark_calibration_validation.py` on a controlled runner to
measure the bounded representative workload. Timings are not committed because
they are hardware- and load-dependent; performance regressions belong in the
opt-in benchmark suite, not in deterministic validation evidence.
