# Numerical reference fixtures

These fixtures are committed outputs from QuantLib 1.43, not values generated
by the implementation under test. They provide a stable independent numerical
family for regression testing:

- `quantlib_vanilla_v1.json`: analytic European and high-resolution finite-
  difference American values.
- `quantlib_heston_v1.json`: analytic Heston values using integration order
  192.

The files include the valuation date, day-count convention, engine, and grid
settings required to interpret the numbers. QuantLib is deliberately not a
runtime or development dependency of this project. To regenerate them in an
isolated environment:

```bash
uv run --with 'QuantLib==1.43' reports/generate_quantlib_references.py
uv run pytest tests/numerics/test_quantlib_references.py
```

Treat a QuantLib version change as a reference-data migration: review every
numeric diff and update the recorded source version in the same commit.
