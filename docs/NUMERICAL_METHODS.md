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

The reported estimate is a policy lower bound, not an exact American price.
Uncertainty is estimated over independent antithetic-pair averages and an exact
European value is used as a cross-fitted control. Tests compare tree references
using statistical tolerances derived from the reported standard error.

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

Reported confidence half-widths are measured, classified, and returned without
artificial tightening. A projection flag identifies a constrained point
estimate.

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

This is not a time-discretized QE simulator. The old `HestonQECalibrator` name
is only a deprecated compatibility alias.

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
5. compare American estimates with an independent high-resolution tree or PDE;
6. calibrate synthetic SABR/Heston surfaces with known parameters and then use
   held-out strikes/tenors;
7. validate day-count, dividend, settlement, and quote conventions outside this
   generic engine.
