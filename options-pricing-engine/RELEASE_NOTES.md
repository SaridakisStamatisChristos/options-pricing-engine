## v0.9.0
**Highlights:** LSMC American pricer with CI, MC Greek CIs + fallbacks, numerics policies (clamps, tails, precision buckets), performance & precision gates, idempotent capsules with policy hash.
**API:** /quote accepts Monte Carlo VR flags (antithetic, use_qmc, use_cv) and American mode; responses include `ci`, `greeks`, `ci_greeks`, `greeks_meta`, `seed_lineage`, `model_used`.
**Perf gates:** MC p50/p95/p99 ~1.55/1.90/2.22 ms on golden cell; precision buckets enforced.
**Notes:** Latency test uses CV=off; precision covered in precision test.
