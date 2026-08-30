# PDE convergence evidence

This snapshot records the convergence cases enforced by
`tests/numerics/test_finite_difference.py`. Values use the default anchored
sinh grid, Crank–Nicolson time stepping, and two Rannacher half-steps. Every
level in a row uses the same truncated domain.

## European ATM call

Inputs: (S=K=100), (T=1), (r=5\%\), (q=2\%\),
\(\sigma=20\%\). The independent analytic Black–Scholes value is
`9.227005508154036`.

| Space × time | PDE price | Absolute analytic error |
| ---: | ---: | ---: |
| 50 × 50 | 9.212649544314 | 1.435596e-2 |
| 100 × 100 | 9.223381279772 | 3.624228e-3 |
| 200 × 200 | 9.226103932637 | 9.015755e-4 |

The observed order from the last three direct values is `1.9788`. The formal
second-order Richardson estimate on the finest direct value is `9.075510e-4`;
the actual error is `9.015755e-4`. The diagnostic extrapolation is
`9.227011483591`, whose analytic error is `5.975437e-6`. The engine still
returns the direct 200 × 200 value.

## American ATM put

Inputs are the same except for an American put. The committed QuantLib 1.43
`FdBlackScholesVanillaEngine(tGrid=2000,xGrid=800,dampingSteps=2)` reference is
`6.660430122800409`.

| Space × time | PSOR price | Penalty price |
| ---: | ---: | ---: |
| 60 × 60 | 6.648526476515 | 6.648526475362 |
| 120 × 120 | 6.657463896167 | 6.657464052754 |
| 240 × 240 | 6.659773235954 | 6.659773627891 |

The observed orders are `1.9524` (PSOR) and `1.9523` (penalty). At the finest
level the two independent LCP families differ by `3.919374e-7`. Because an
American free boundary can reduce or destabilize formal order, the engine uses
the observed order only after three same-direction refinements.

## External difficult-regime matrix

`tests/reference/quantlib_vanilla_v1.json` adds independent American cases for
deep ITM, deep OTM, seven-day/high-volatility, high-volatility dividends, and
negative rates. Both exercise solvers are checked on a 240 × 320 mesh against
the QuantLib 2,000 × 800 values with an absolute tolerance of `0.002`. Separate
representative in-repository cross-validation cases require PSOR/penalty
agreement within `1e-5` and independently inspect each solver's residuals.

Reproduce the evidence with:

```bash
uv run pytest tests/numerics/test_finite_difference.py
uv run pytest tests/numerics/test_quantlib_references.py
```

Regenerate the independent fixture only as a reviewed reference migration:

```bash
uv run --with 'QuantLib==1.43' reports/generate_quantlib_references.py
```
