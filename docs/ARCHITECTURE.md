# Architecture

The package keeps financial contracts and result types separate from numerical
algorithms, calibration, API transport, and operational controls. Public model
classes live in focused modules; `core.pricing_models` remains a compatibility
facade so the 2.0 import surface is not broken by the split.

| Layer | Main modules | Responsibility |
| --- | --- | --- |
| Domain | `core.models` | Contracts, markets, exercise style, and auditable results |
| Analytic and lattice | `core.black_scholes`, `core.crr` | Closed-form European pricing and adaptive binomial pricing |
| Stochastic | `core.monte_carlo`, `core.lsmc`, `core.statistical_inference` | Terminal simulation, stopping policies, variance reduction, and uncertainty |
| PDE | `core.finite_difference` | Crank–Nicolson/Rannacher valuation and American PSOR exercise |
| Orchestration | `core.pricing_engine`, `core.replay_pricing` | Model selection, compatibility checks, caching, and deterministic replay |
| Calibration | `calib.sabr`, `calib.heston`, `calib.svi`, `calib.select` | Smile/surface fit, model selection, and static-arbitrage diagnostics |
| Service | `api`, `security`, `observability` | HTTP schemas, identity, admission controls, telemetry, and replay endpoints |

Shared stochastic helpers collapse antithetic paths into independent sampling
units before inference. This keeps the meaning of standard errors independent
of the particular model class. Calibration results similarly carry weighting,
holdout, constraint, and cross-tenor diagnostics rather than returning only an
optimizer parameter vector.

## Compatibility boundaries

- `options_engine.core.pricing_models` re-exports the established 2.0 names,
  including private aliases used by older tests and integrations. New code
  should import the focused module that owns a model.
- The versioned `/api/v1` routes are canonical. Legacy `/quote`, `/batch`,
  `/greeks`, and replay routes share the same model and security layer.
- `HestonQECalibrator` remains a deprecated alias; pricing is characteristic-
  function Heston, not a QE time-stepping proxy.

## State and scale

Pricing workers are thread-bounded within a process. Rate limits, idempotency,
replay capsules, and result caches are also process-local, so a deployment uses
one Uvicorn worker per container. Horizontal replicas require external durable
state when consistent cross-replica replay or idempotency is a requirement.
