# Deterministic discrete cash dividends

## Economic model

The feature models a vanilla equity whose ex-dividend cash amounts and ex-times
are known at valuation. Let `t=0` be valuation, `T` expiry, `r` the continuously
compounded funding rate, `q` the continuous proportional carry/dividend yield,
and `D_i > 0` a fixed same-currency cash amount at `0 < t_i < T`. Between
events, under the risk-neutral measure,

\[
dS_t=(r-q)S_t\,dt+\sigma S_t\,dW_t.
\]

At an ex-event the equity follows the limited-liability spot jump

\[
S_{t_i+}=\max(S_{t_i-}-D_i,0).
\]

The option holder does not receive `D_i`. A continuous `q` and the fixed cash
schedule may coexist: `q` remains proportional carry between events and every
`D_i` remains an explicit jump. The implementation never replaces the cash
schedule with an implied continuous yield.

For a European claim, the backward jump condition is

\[
V(t_i-,S)=V(t_i+,\max(S-D_i,0)).
\]

For an American claim, exercise is available on each side of the event and

\[
V(t_i-,S)=\max\left(\Phi(S),
V(t_i+,\max(S-D_i,0))\right),
\]

where `Phi` is intrinsic value. This ordering is material for calls immediately
before a large dividend. Events exactly at valuation or expiry are rejected
because entitlement/exercise ordering would otherwise require an additional
product convention.

## Supported numerical families

| Model | Cash-dividend support | Event treatment |
| --- | --- | --- |
| Adaptive CRR | European and American | Event snapped to the nearest tree time; post-jump value is monotone piecewise-linear interpolation at `max(S-D,0)`; the maximum time-alignment error is reported |
| Crank–Nicolson PDE | European and American | Every event is an exact time-mesh anchor; piecewise-linear spot interpolation applies the jump; Rannacher smoothing restarts after each jump |
| PDE American PSOR | Supported | LCP diagnostics remain solver-specific |
| PDE American penalty | Supported | Independent active-set penalty diagnostics remain solver-specific |
| Black–Scholes | Rejected | No closed-form cash-jump reduction is used |
| Terminal Monte Carlo / randomized QMC | Rejected | Terminal lognormal sampling does not represent intervening cash jumps |
| Longstaff–Schwartz | Rejected | Its current path generator has no cash-jump state transition |
| Heston/SABR/SVI/SSVI calibration | Not part of this input contract | Calibration APIs retain their existing forward/quote conventions |

CRR remains a recombining interpolation lattice between events. Cash jumps
reduce its event accuracy to first order locally, so convergence should be
checked by increasing `steps`; diagnostics expose requested and aligned times.
The PDE time grid is event-aligned exactly, but the spot jump still introduces
interpolation error. Cash-dividend PDE runs therefore do not claim a formal
second-order Richardson estimate. With three credible grids, the engine may
report an observed order and corresponding diagnostic extrapolation.

The call boundary at a backward time `tau=T-t` includes all still-future cash
events. If `tau_i=T-t_i`, its large-spot European asymptote is

\[
S_{max}e^{-q\tau}-Ke^{-r\tau}
-\sum_{\tau_i\leq\tau}D_i e^{-r(\tau-\tau_i)}e^{-q\tau_i}.
\]

The equality is included only after applying the event jump. Immediately
before the jump the event is future; immediately after it is not. American
boundaries additionally take the maximum with intrinsic value.

## Scalar and dated inputs

The backward-compatible scalar API accepts a `CashDividendSchedule` on
`MarketData`; every event uses the same year-fraction clock as
`OptionContract.time_to_expiry`. Schedules are immutable, strictly increasing,
unique, finite, positive, limited to 64 events, and identified by a SHA-256
fingerprint over exact hexadecimal floats.

The market-conventions layer accepts `DatedCashDividendSchedule` with explicit
`ExDividendDate` values. It converts active events to scalar ex-times with the
configured day count. Ex-dates are not silently business-day-adjusted. An
active ex-date must be a business day in the supplied calendar, must be after
spot settlement, and must be strictly before adjusted expiry. Events after a
particular contract's expiry are retained in the market snapshot but filtered
from that contract's resolved schedule.

The curve forward reports the continuous-carry forward separately from the
fixed-cash future deduction. Equivalent scalar `q` is derived from the former,
then the cash schedule is passed explicitly. This prevents double counting.
For shaped curves, diagnostics report any difference between the curve-based
cash future deduction and the endpoint-equivalent scalar kernel's deduction.

## Limitations

The implementation intentionally does not infer or model:

- uncertain, stochastic, optional, proportional, or issuer-cancellable dividends;
- record dates, payment dates, due-bill periods, or entitlement across an
  unsettled trade (an event on or before spot settlement is rejected);
- tax, withholding, currency conversion, dividend default, stock loan, or
  manufactured payments;
- an escrowed-dividend process or a rule allowing the equity to become
  negative; the supported convention is the limited-liability `max(S-D,0)`
  spot model;
- intraday event/exercise ordering;
- curve bootstrapping or market-data sourcing.

The cash amounts are deterministic nominal amounts in the equity/option
currency. Users remain responsible for supplying the correct corporate-action
schedule and calendar as of the valuation timestamp.

## Independent validation

`tests/reference/quantlib_discrete_dividends_v1.json` contains committed
QuantLib 1.43 `FdBlackScholesVanillaEngine` fixtures using its independent
`Spot` cash-dividend model on a 2,400 x 1,200 grid. Cases cover European and
American calls and puts, deep ITM/OTM strikes, short maturity, high volatility,
negative rates, continuous carry together with cash events, and multiple/large
dividends. The generator is
`reports/generate_quantlib_references.py`; QuantLib remains a generator-only
dependency. Tests also require CRR grid convergence, PDE observed convergence,
exact event alignment, renewed Rannacher smoothing, and PSOR/penalty agreement.
