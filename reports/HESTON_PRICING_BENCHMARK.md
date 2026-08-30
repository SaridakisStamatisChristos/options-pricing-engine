# Heston pricing accuracy and runtime

This report is a reproducible engineering measurement, not a portable speed
claim. Run `python reports/benchmark_heston_pricing.py` on the target machine.
The numbers below were measured on 2026-08-30 with Python 3.12.13, NumPy 2.5.2,
SciPy 1.18.1, Linux 6.18.35 x86-64. Values are medians of 50 warmed calls.

| Batch | Gauss–Laguerre 64 | COS fixed 256/L12 | COS adaptive |
| ---: | ---: | ---: | ---: |
| 1 | 156.8 µs | 192.3 µs | 979.2 µs |
| 9 | 169.7 µs | 363.2 µs | 1,908.2 µs |
| 64 | 279.0 µs | 1,225.1 µs | 7,610.4 µs |
| 256 | 730.7 µs | 5,420.0 µs | 29,419.5 µs |

| Method | Maximum absolute error against committed QuantLib 1.43 cases |
| --- | ---: |
| Gauss–Laguerre 64 | `1.599e-07` |
| COS fixed 256/L12 | `9.139e-06` |
| COS adaptive | `1.879e-08` |

The result is deliberately not marketed as “COS is faster.” In this vectorized
NumPy implementation, the existing 64-node quadrature is faster for the tested
arbitrary-strike batches and already highly accurate. Fixed COS provides a
bounded-work independent family with a visible accuracy tradeoff. Adaptive COS
is the strongest reference calculation on these fixtures, but its independent
series and truncation checks cost more. This separation is useful in practice:
Gauss–Laguerre remains the production default, fixed COS can be selected for a
calibration convergence study, and adaptive COS serves as a cross-validation
family.
