# Numerical methods and validation

## Conventions

The numerical models use continuously compounded risk-free rate `r`, continuous
dividend yield `q`, time in years, and volatility in annualized decimal units.
Vanilla payoffs are denominated in the same currency units as spot and strike.
The separate `options_engine.market` layer resolves typed dates, day counts,
explicit holiday calendars, T+n settlement, funding/discount curves,
continuous dividend/carry curves, and spot-settlement forwards into equivalent
scalar `T`, `r`, and `q`. See [Market conventions](MARKET_CONVENTIONS.md).

CRR and finite differences support deterministic discrete cash dividends as
explicit limited-liability spot jumps. Black–Scholes, terminal MC/QMC, and
Longstaff–Schwartz reject such schedules rather than converting cash to a
continuous yield. See [Discrete dividends](DISCRETE_DIVIDENDS.md) for the exact
economic model, event ordering, limitations, and independent fixtures. The
core models do not model borrow constraints, transaction costs, exercise fees,
stochastic rates, or corporate-action uncertainty. Their Greeks retain
scalar-input semantics; curve-node risk is outside this generic engine.

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

With deterministic cash dividends, the tree applies
`S(t_i+)=max(S(t_i-)-D_i,0)` by piecewise-linear interpolation of the
post-event value function. Ex-times are snapped to the nearest tree layer and
the alignment error is reported. Vega then uses a central volatility bump
because differentiating interpolation weights analytically would not preserve
the existing vega semantics.

### Finite-difference PDE

The independent PDE solver supports European and American vanilla contracts on
a bounded spot domain. The default mesh applies a sinh transform and contains
both current spot and strike exactly. This concentrates resolution around the
valuation point and payoff kink without removing the degenerate (S=0)
boundary. A uniform spot mesh remains available through `grid_type="uniform"`
for reproduction of earlier studies. First- and second-derivative weights use
the full unequal-spacing formulas; index-space uniform-grid coefficients are
never reused on a transformed mesh.

Time integration is Crank–Nicolson. The first time interval is replaced by two
implicit-Euler half-steps by default (Rannacher smoothing) to damp oscillations
from the non-smooth terminal payoff. European tridiagonal systems use the
Thomas algorithm. American linear-complementarity problems expose two
independent solution families:

- `exercise_solver="psor"` uses projected successive over-relaxation and
  requires both an update tolerance and a normalized complementarity residual;
- `exercise_solver="penalty"` solves a penalized piecewise-linear equation
  with active-set semismooth Newton steps, direct tridiagonal solves, and a
  merit-function line search. It does not call or fall back to PSOR. The raw
  obstacle violation before the final projection, normalized penalized-equation
  residual, and post-projection LCP residual are reported separately so finite
  penalty bias and floating-point cancellation remain visible. A stable active
  set affected by cancellation is accepted only when the independently
  evaluated projected LCP residual also satisfies the parameter-scaled limit.

At (S=0) and (S=S_{max}), the solver applies the corresponding discounted
vanilla asymptotes and their American intrinsic maxima. These formulas remain
valid for continuous dividends and supported negative rates. One-sided delta
residuals at both boundaries and the risk-neutral lognormal probability beyond
(S_{max}) are reported as truncation diagnostics. `s_max_override` can hold a
reviewed domain fixed across separate experiments; it must exceed spot and
strike.

`refinement_levels` systematically multiplies both spatial and time resolution
while keeping the same (S_{max}). The returned value is always the direct
finest-grid solution. For a Rannacher-smoothed European solve, the diagnostics
may use the formal second order to report a two-grid Richardson error estimate.
For an American/free-boundary solve, no formal order is assumed: three levels,
same-sign changes, and a credible observed order are required before an error
estimate is emitted. Otherwise only raw level prices and differences are
reported. Richardson extrapolation is diagnostic and never silently replaces
the published price.

Cash ex-events are exact backward-time anchors. The solver applies the spot
jump by interpolation, enforces American exercise on both sides, resets the
cash-aware asymptotic boundary, and performs a fresh pair of Rannacher implicit
half-steps after every event. Formal second-order Richardson diagnostics are
disabled for cash-jump runs; observed-order diagnostics require three credible
grids.

The response retains the v2.1.0 grid, PSOR, bounds, and projection keys and adds
mesh-spacing extrema, exact-anchor flags, every refinement level and price,
per-level iteration/LCP residuals, observed order, error estimate, extrapolated
diagnostic value, boundary residuals, and penalty-specific convergence fields.
Work is rejected before allocation when the finest requested grid exceeds the
step or work limits.

Regression evidence includes exact polynomial consistency of the non-uniform
operator, European convergence against analytic Black–Scholes, independent
PSOR/penalty agreement, and committed QuantLib 1.43 cases covering deep ITM and
OTM contracts, seven-day maturity, high volatility, continuous dividends, and
negative rates. See `reports/PDE_CONVERGENCE.md` for the fixed convergence
snapshot and reproduction commands.

Cash-dividend regression evidence additionally uses committed QuantLib 1.43
`Spot` dividend-model fixtures over European/American calls and puts, multiple
events, deep ITM/OTM strikes, short maturity, high volatility, and negative
rates. Both American PDE solvers and the separately implemented CRR event
lattice are checked against those fixtures.

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

Two independent transform families price European Heston calls:

- `heston_call_prices` retains the 64-point Gauss–Laguerre inversion of the two
  Heston probabilities. It is the backward-compatible default and remains the
  fastest implementation on the committed NumPy benchmark.
- `heston_cos_call_prices` implements the Fang–Oosterlee cosine expansion. Its
  interval uses the first two closed-form cumulants of `log(S_T/F)`. The second
  cumulant is evaluated with long-double intermediates, with its exact
  zero-mean-reversion limit used to avoid cancellation when `kappa*T` is tiny,
  and checked in tests against derivatives of `log(phi)`. Strike-relative
  payoff coefficients avoid unsafe exponentials; calls use an OTM expansion
  when safe and otherwise the bounded put payoff plus parity.

Adaptive COS mode separately checks series resolution (full versus half term
count) and tail truncation (successively wider cumulant intervals). A solve
that exhausts `max_terms`/`max_truncation` raises instead of returning a
finite-looking unconverged price. Fixed-cost COS mode is available for
calibration after a user-selected convergence study. Both families enforce
intrinsic/forward bounds, call-spread limits, and convexity. The committed
QuantLib 1.43 fixture is checked against COS directly, and a broader parameter
grid cross-checks COS against Gauss–Laguerre.

The calibrated parameters are initial variance, long-run variance, mean
reversion, volatility of variance, and spot/variance correlation. Multi-start
bounded least squares is deterministic for fixed seeds. The default
`calibration_mode="per_tenor"` preserves the previous independent-smile fit.
`calibration_mode="global"` fits one parameter vector jointly to at least two
training maturities, producing one coherent time-homogeneous Heston process
over those maturities.

Strike weighting can be uniform, squared Black-vega, inverse squared bid/ask
width, or the product of vega and spread weights (`hybrid`). Each slice is
normalized before fitting. In global mode, `global_tenor_weighting="equal"`
gives every training tenor the same total objective weight;
`"observations"` lets tenors contribute in proportion to their retained quote
count.

A deterministic strike holdout reports out-of-sample IV RMSE without random
splitting. Global mode can additionally exclude complete maturities through
`holdout_tenors`, which tests the shared process out of tenor rather than only
out of strike. Detailed results report:

- in-sample weighted IV RMSE and unweighted holdout IV RMSE;
- Feller ratio `2*kappa*theta/vol_of_vol**2` and satisfaction flag;
- parameter-bound proximity on `[0, 1]`, where zero is the center of the
  transformed optimizer interval and one is an active boundary;
- termination status/message, function/Jacobian evaluations, cost, optimality,
  active mask, attempted/successful starts, and selected seed.

The Feller condition is diagnostic rather than a hard existence constraint;
optional regularization can penalize violations.

This is not a time-discretized QE simulator. The old `HestonQECalibrator` name
is only a deprecated compatibility alias. Per-tenor fitting has more local
flexibility and normally cannot have worse training fit; it may imply mutually
inconsistent processes across expiry. Global fitting enforces one process and
supports whole-tenor validation; it may fit real surfaces worse when Heston's
time-homogeneous assumptions are inadequate. `compare_modes()` reports both
sets of metrics on identical strike holdouts and their deltas and intentionally
emits no automatic winner. Complete-tenor holdout diagnostics are run through
the global detailed-calibration API because an independent per-tenor fit cannot
train on a maturity whose quotes are all excluded.

Reproduce the accuracy/runtime measurements with
`python reports/benchmark_heston_pricing.py`; committed observations and the
test environment are recorded in `reports/HESTON_PRICING_BENCHMARK.md`.

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
