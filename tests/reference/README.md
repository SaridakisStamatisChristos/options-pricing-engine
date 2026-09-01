# Numerical reference fixtures

These fixtures are committed outputs from QuantLib 1.43, not values generated
by the implementation under test. They provide a stable independent numerical
family for regression testing:

- `quantlib_vanilla_v1.json`: analytic European and high-resolution finite-
  difference American values, including labeled deep-ITM/OTM, short-maturity,
  high-volatility, dividend, and negative-rate regimes.
- `quantlib_heston_v1.json`: analytic Heston values using integration order
  192. Both the Gauss–Laguerre and independent Fang–Oosterlee COS families are
  checked against these committed values.
- `quantlib_curve_aware_v1.json`: high-resolution finite-difference American
  values built from independent dated linear-zero funding and carry curves,
  including negative portions and shaped curves with fixed cash dividends.

The files include the valuation date, day-count convention, engine, and grid
settings required to interpret the numbers. QuantLib is deliberately not a
runtime or development dependency of this project. To regenerate them in an
isolated environment:

```bash
uv run --with 'QuantLib==1.43' reports/generate_quantlib_references.py
uv run pytest tests/numerics/test_quantlib_references.py
```

The American fixture is checked with both the PSOR and penalty implementations.
QuantLib's `FdBlackScholesVanillaEngine` is an external reference family, not a
source of coefficients or exercise decisions for either solver.

Treat a QuantLib version change as a reference-data migration: review every
numeric diff and update the recorded source version in the same commit.
