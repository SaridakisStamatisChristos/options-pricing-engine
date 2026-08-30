# Numerical methods and validation

## Conventions

The engine uses continuously compounded risk-free rate `r`, continuous dividend
yield `q`, calendar time in years, and volatility in annualized decimal units.
Vanilla payoffs are denominated in the same currency units as spot and strike.
The core models do not model discrete dividends, borrow constraints,
transaction costs, exercise fees, settlement lags, or stochastic rates.

## European options

### Black–Scholes

European calls and puts use the standard continuous-dividend formulas:

\[
d_1 = \frac{\log(S/K) + (r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},
\qquad d_2=d_1-\sigma\sqrt{T}.
\]

Analytic price and Greeks are returned. Theta is calendar-time decay per day,
so its sign is the negative derivative with respect to remaining maturity.
American contracts are rejected.

### CRR tree

The Cox–Ross–Rubinstein tree supports European and American exercise. If the
risk-neutral probability is outside `(0, 1)`, the implementation doubles the
requested step count until a valid tree is obtained or raises an explicit
error. It never clips an invalid probability into the interval. The resolved
step count appears in `model_used`.

### Finite-difference PDE

The independent PDE solver supports European and American vanilla contracts on
a bounded uniform spot grid. It uses Crank–Nicolson time stepping, replaces the
first step with two implicit-Euler half-steps for Rannacher smoothing, solves
European tridiagonal systems with the Thomas algorithm, and solves the American
linear-complementarity problem with projected SOR.

The response includes the resolved spatial and time grids, truncation boundary,
PSOR convergence and iteration counts, projection status, and finite-difference
delta, gamma, and calendar theta. Work is rejected before allocation when grid
limits are exceeded. This numerical family is tested against committed
high-resolution QuantLib finite-difference references rather than against only
another implementation inside this package.

### Terminal Monte Carlo

Risk-neutral terminal lognormal draws price European vanilla payoffs. Available
variance reduction includes:

- antithetic pairs, whose pair averages are treated as independent sampling
  units for uncertainty estimation;
- a discounted-terminal-underlying martingale control with known expectation
  \(S_0 e^{-qT}\), using coefficients fitted on another fold;
- randomized quasi-Monte Carlo using 16 independently scrambled Sobol
  replicates and a Student-t interval over replicate means;
- randomized stratification using independent stratified replicates and the
  same replicate-level interval construction.

Whenever variance is estimated from ordinary samples, antithetic-pair means,
or independent randomized replicates, the engine uses a Student-t critical
value with `independent_units - 1` degrees of freedom. A single independent
unit has no estimable sampling variance and therefore does not receive a
zero-width interval. Point estimates, raw intervals, bounded intervals,
critical values, degrees of freedom, and independent-unit counts are retained
as separate audit fields.

The control deliberately does not contain an analytically priced option payoff
component. Such a component would reconstruct the Black–Scholes answer and make
the reported Monte Carlo error circular.

The compatibility `/quote` and `/batch` APIs optionally apply an adaptive
precision budget to terminal Monte Carlo. Starting from the requested path
count, the engine doubles paths until the measured 95% confidence-interval
half-width is within `precision.target_ci_bps` or `precision.max_paths` is
reached. It reports the resolved path count and whether the target was met; it
does not clip the measured interval. Quasi-Monte Carlo remains a terminal
European estimator and is not accepted for American exercise.

## American options

Longstaff–Schwartz simulates full risk-neutral paths and regresses discounted
continuation cashflows on several candidate bases. Candidate diagnostics include
AIC, BIC, and cross-validation RMSE; stopping predictions are cross-fitted so a
path is not exercised using coefficients trained on that same sampling fold.

The raw stopping-policy estimate is a statistical policy lower bound, not an
exact American price. Uncertainty is estimated over independent
antithetic-pair averages and an exact European value is used as a cross-fitted
control. The response separately exposes `raw_policy_estimate`,
`bounded_estimate`, `projection_applied`, the raw confidence interval, and the
bounded confidence interval. Tests compare independent references using
statistical tolerances derived from the reported standard error.

For a call with no positive continuous dividend and a non-negative interest
rate, early exercise is never optimal under the model assumptions. The
implementation returns the exact European value and records the theorem in
model metadata. Negative-rate calls remain on the numerical American path.

## Bounds and input policy

Invalid finite-domain inputs are rejected. The engine does not silently replace
spot, strike, maturity, volatility, or rates with nearby values.

Returned American point estimates are projected only onto model-independent
bounds:

- lower bound: maximum of immediate exercise and the corresponding European
  value;
- call upper bound: \(S_0\exp(\max(-qT,0))\);
- put upper bound: \(K\exp(\max(-rT,0))\).

The same monotone no-arbitrage projection is applied to both the point estimate
and the endpoints of its interval, so they describe the same bounded statistic.
The unprojected estimate, interval, and measured half-width remain available;
separate flags identify point and interval projection.

## Volatility calibration

### SABR

SABR calibration fits a smile per tenor using Hagan's approximation with
bounded multi-start least squares. Duplicate call/put observations at a strike
are collapsed before fitting so they do not double-weight the same implied-vol
point.

### Heston

Heston pricing evaluates the characteristic function with 64-point
Gauss–Laguerre quadrature and inverts call prices to Black implied volatility.
The calibrated parameters are initial variance, long-run variance, mean
reversion, volatility of variance, and spot/variance correlation. Multi-start
bounded least squares is deterministic for fixed seeds.

Calibration can use uniform, Black-vega, inverse bid/ask-width, or hybrid
weights. A deterministic strike holdout reports out-of-sample IV RMSE without
making the result depend on random splitting. Each tenor reports the Feller
ratio (2\kappa\theta/\xi^2), whether it is at least one, and the parameter
change from the previous tenor. The Feller condition is diagnostic rather than
a hard existence constraint; optional regularization can penalize violations.

This is not a time-discretized QE simulator. The old `HestonQECalibrator` name
is only a deprecated compatibility alias. Each maturity is still calibrated
independently, so these diagnostics do not turn the tenor collection into one
globally coherent Heston process.

### SVI and SSVI

Raw SVI support evaluates total variance, Lee wing slopes, and the
Gatheral–Jacquier density factor for slice diagnostics. The surface builder can
also fit one global power-law SSVI shape to all tenors. ATM total variance is
projected onto a non-decreasing sequence with weighted pool-adjacent-violators;
shape parameters are bounded by sufficient wing and curvature constraints.

The fitted surface is accepted only after a dense density-factor check and a
calendar check on a common log-moneyness grid, including intermediate
maturities. Extrapolation beyond the calibrated tenor range is rejected. These
constraints make the construction static-arbitrage-aware; they do not supply
missing market conventions or prove dynamic consistency with a stochastic
process.

### Selection and arbitrage validation

SABR and Heston candidates are compared with corrected Akaike information
criterion (AICc), which penalizes extra parameters and small samples. A raw
in-sample RMSE comparison would systematically favor the more flexible model.

Validation checks include:

- option-type validity and consistent forwards/discount factors;
- call/put parity;
- price monotonicity in strike;
- convexity through nondecreasing secant slopes on non-uniform strike grids;
- nondecreasing total variance at fixed log-moneyness across maturities.

Validation findings are reported rather than silently modifying the quote
board.

Committed reference fixtures under `tests/reference` were produced by
QuantLib 1.43 using analytic European, high-resolution finite-difference
American, and analytic Heston engines. QuantLib is deliberately not a project
dependency; fixture regeneration is an explicit reviewed migration.

## Reproducibility

Explicit seeds are represented by NumPy `SeedSequence` lineage in replay
capsules. Cache keys include exact floating-point representations of contract
and market terms, exercise/option style, model configuration, and seed identity.
Unseeded stochastic requests bypass cache to preserve their stochastic meaning.

Deterministic replay covers terminal Monte Carlo capsules. Replay payloads reject
NaN and infinity and are hashed using canonical JSON. Process-local capsule
retention is bounded and is not a substitute for durable model-risk records.

## Independent validation checklist

Before adopting a release:

1. compare Black–Scholes prices/Greeks with an independent library;
2. verify parity, strike monotonicity, convexity, and limiting cases over the
   intended market domain;
3. compare CRR convergence across step sequences rather than a single tree;
4. measure Monte Carlo confidence coverage over many independent seeds;
5. compare American estimates with the committed independent high-resolution
   PDE fixtures and additional market-domain cases;
6. calibrate synthetic SABR/Heston surfaces with known parameters and then use
   held-out strikes/tenors;
7. validate day-count, dividend, settlement, and quote conventions outside this
   generic engine.
