# Market conventions and curve resolution

The scalar pricing classes retain the compact state
`(time_to_expiry, risk_free_rate, dividend_yield)` plus an optional immutable
cash-dividend schedule. The `options_engine.market` layer owns civil dates,
calendars, settlement, and curve interpolation. For dated valuation it can
either produce the historical endpoint-equivalent scalars or an immutable,
date-free interval-factor context consumed by models with true deterministic
term-structure support. Existing scalar constructors and `price_option()` are
unchanged.

## Dates and day counts

`ValuationDate` and `ExpiryDate` wrap civil `datetime.date` values. They reject
`datetime` inputs so time-zone truncation cannot happen implicitly. Supported
day counts are:

- Actual/365 Fixed;
- Actual/360;
- Actual/Actual ISDA, split at calendar-year boundaries;
- 30/360 US (NASD end-of-month rules);
- 30E/360.

Year fractions are deterministic and antisymmetric. The dated option adapter
requires a strictly positive year fraction no greater than the core model
limit of 100 years.

## Calendars and settlement

`BusinessCalendar` is immutable and accepts an explicit holiday set and
weekend weekdays. It supports unadjusted, following, modified-following,
preceding, and modified-preceding adjustment. `SettlementLag` advances T+n in
business days, excluding the valuation date. Zero-lag settlement applies the
configured settlement adjustment to the valuation date.

The library includes only two deterministic generic calendars:

- `WEEKEND_CALENDAR`, with Saturday and Sunday closed;
- `ALL_DAYS_CALENDAR`, with every civil day open.

It does not claim to know exchange, currency, or jurisdictional holidays.
Production callers must supply and version the applicable holiday set. The
calendar fingerprint includes its name, weekend definition, and every holiday.

## Curves

All rates are annual continuously compounded decimals. Curves are anchored to
an explicit `ValuationDate` and own their day-count convention.

`ContinuousZeroCurve` linearly interpolates zero rates. Its first rate is flat
back to the reference date. `DiscountFactorCurve` log-linearly interpolates
strictly positive discount factors, including the synthetic reference node
`D(t,t)=1`. `FlatDiscountCurve` provides the scalar-rate special case.

`ContinuousDividendCurve`, `DividendFactorCurve`, and `FlatDividendCurve`
provide the corresponding continuous dividend/carry factors. Typed dated cash
dividends are resolved separately and are never converted implicitly into a
continuous yield.

Extrapolation beyond the last node is never accidental. Each interpolated
curve selects one of:

- `raise` (the default);
- `flat_zero`;
- `flat_forward`, using the final integrated-rate slope.

Node dates must be strictly increasing. Rates, factors, derived zero rates,
queries before the reference date, and numerically unrepresentable factors are
validated. Canonical SHA-256 curve identifiers include the curve family,
reference date, day count, extrapolation policy, dates, and exact hexadecimal
floating-point values.

## Three pricing representations

The library distinguishes three paths explicitly:

1. **Scalar flat-rate pricing.** `OptionContract`, `MarketData`, and
   `OptionsEngine.price_option()` use the supplied constant `r` and `q`.
2. **Endpoint-equivalent dated compatibility pricing.**
   `MarketEnvironment.resolve()` and
   `price_dated_option(..., curve_aware=False)` reduce dated curves to one
   `(r_eq,q_eq)` while preserving expiry discount and forward endpoints.
3. **True curve-aware dated pricing.** `resolve_curve_aware()` and the default
   `price_dated_option(..., curve_aware=True)` retain exact interval factors for
   supported models. No expiry-equivalent rate is used in their dynamic
   evolution.

### Forward and endpoint-compatible resolution

Let `t` be valuation, `t_s` spot settlement, and `T` adjusted expiry. Funding
and carry factors between arbitrary dates are ratios of their reference-date
curves. Without fixed cash events, the forward builder uses

\[
F_c(t_s,T)=S_t\frac{Q(t_s,T)}{D(t_s,T)}.
\]

The option present value still uses the valuation-to-expiry funding factor
`D(t,T)`. For a dated contract with year fraction `tau`, the adapter constructs

\[
r_{eq}=-\frac{\log D(t,T)}{\tau},\qquad
q_{eq}=r_{eq}-\frac{\log(F_c(t_s,T)/S_t)}{\tau}.
\]

Consequently, the unchanged scalar kernels reproduce both endpoints:

\[
e^{-r_{eq}\tau}=D(t,T),\qquad
S_t e^{(r_{eq}-q_{eq})\tau}=F_c(t_s,T).
\]

For fixed cash amounts `D_i` after spot settlement and before expiry, the
forward report additionally provides the conventional curve deduction

\[
A(T)=\sum_i D_i\frac{Q(t_i,T)}{D(t_i,T)},\qquad
F_{cash}(t_s,T)=F_c(t_s,T)-A(T).
\]

The equivalent `q_eq` is still derived from `F_c`, not `F_cash`; the cash
events are then supplied explicitly to CRR/PDE. This avoids double counting.
For non-flat curves, one endpoint-equivalent `(r_eq,q_eq)` cannot reproduce
every event-to-expiry accrual. Compatibility diagnostics therefore report the
curve deduction, scalar-kernel deduction, and their mismatch. The numerical
spot-jump model also floors the equity at zero, so `F_cash` is a transparent
curve-bookkeeping forward rather than a volatility-dependent claim that
overrides the documented limited-liability process.

### True interval-factor representation

Let model time run from `0` to `T`. The curve-aware adapter exposes

\[
D(t_0,t_1)=\frac{D(0,t_1)}{D(0,t_0)},\qquad
Q(t_0,t_1)=\frac{Q(0,t_1)}{Q(0,t_0)}.
\]

For a numerical interval of length `Delta t`, the coefficients are obtained
without differentiating the curves:

\[
r_{step}=-\frac{\log D(t_0,t_1)}{\Delta t},\qquad
q_{step}=-\frac{\log Q(t_0,t_1)}{\Delta t}.
\]

The adapter represents built-in flat, linear-continuous-zero, and log-linear
factor curves as immutable piecewise log-factor segments. Civil curve-node
dates map to exact model-time anchors. If curve and contract day counts differ,
native curve time is affinely re-parameterized between civil anchors; supplied
factors at every anchor and at expiry remain exact. A date mapping that does
not yield strictly increasing model/source times is rejected.

Quoted spot settles on `t_s`. For a non-zero settlement lag, the basis
`log(D(0,t_s)/Q(0,t_s))` is accrued only over `[0,t_s]` and then remains
constant. This preserves the settlement-aware forward while leaving every
post-settlement interval ratio unchanged. Because valid cash ex-dates must be
after settlement, their event-to-expiry funding and carry factors are exactly
the original curve ratios.

European Black–Scholes depends only on `D(0,T)`, `Q(0,T)`, and integrated
variance, so terminal-equivalent curves give the same value. American exercise
compares continuation and intrinsic value at intermediate times; different
paths of `D(t_i,t_j)` and `Q(t_i,t_j)` can therefore produce different prices
despite identical expiry endpoints.

`MarketEnvironment.resolve()` returns the scalar `OptionContract` and
`MarketData` plus the original and adjusted dates, settlement date, all four
funding/carry factors, forward, curve IDs, conventions ID, and equivalent
rates. `resolve_curve_aware()` adds the immutable factor context and verifies
that it reproduces the supplied expiry discount and settlement-aware forward.
`OptionsEngine.price_dated_option()` attaches the applicable evidence under
`market_conventions`.

The original scalar rate fields and `OptionsEngine.price_option()` call shape
remain backward compatible; `MarketData` adds an optional empty-by-default
schedule. `MarketEnvironment.from_scalar_rates()` is also
available when callers want explicit dates while retaining flat `r` and `q`.

## Example

```python
from datetime import date

from options_engine.core.models import OptionType
from options_engine.core.pricing_engine import OptionsEngine
from options_engine.market import (
    ContinuousDividendCurve,
    ContinuousZeroCurve,
    DatedOptionContract,
    ExpiryDate,
    MarketConventions,
    MarketEnvironment,
    SettlementLag,
    ValuationDate,
    ZeroRateNode,
)

valuation = ValuationDate(date(2026, 1, 2))
expiry = ExpiryDate(date(2026, 12, 18))
conventions = MarketConventions(valuation, settlement_lag=SettlementLag(2))
market = MarketEnvironment(
    spot_price=205.0,
    conventions=conventions,
    discount_curve=ContinuousZeroCurve(
        valuation,
        (ZeroRateNode(expiry.value, 0.04),),
    ),
    carry_curve=ContinuousDividendCurve(
        valuation,
        (ZeroRateNode(expiry.value, 0.005),),
    ),
)
contract = DatedOptionContract("AAPL", 200.0, expiry, OptionType.CALL)

with OptionsEngine(num_threads=1) as engine:
    result = engine.price_dated_option(
        contract,
        market,
        override_volatility=0.24,
    )
```

## Boundaries

True curve-aware model support is explicit: finite difference and CRR support
European/American exercise; analytic Black–Scholes supports European claims
without cash dividends. Terminal Monte Carlo and Longstaff–Schwartz currently
reject true-curve requests. Callers may deliberately select endpoint
compatibility, but no unsupported request is flattened implicitly.

This layer does not bootstrap curves, source market data, invent holiday
calendars, infer ex-dates/entitlement, model uncertain dividends, or calculate
bucketed curve risk.
Scalar-model rho retains its established parallel scalar-rate meaning.
Curve-aware Black–Scholes and CRR do not report rho without an explicit curve
bump convention; analytic curve-aware theta is likewise omitted without a
curve-roll convention. Those responsibilities require product- and
desk-specific policies above this generic resolver.
