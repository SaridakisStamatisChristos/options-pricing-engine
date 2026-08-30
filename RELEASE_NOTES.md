# Release notes

## 2.0.0 — repository and numerical correctness repair

This release intentionally breaks several unsafe 1.x behaviors.

### Numerical correctness

- Corrected Black–Scholes calendar-theta signs.
- Added model/exercise compatibility checks instead of pricing American options
  with European-only models.
- Replaced CRR probability clipping with adaptive step refinement.
- Replaced circular option-payoff control variates with a cross-fitted
  discounted-underlying martingale control.
- Reworked Longstaff–Schwartz stopping regressions with cross-fitting, honest
  standard errors over antithetic pairs, a European control, and exact
  no-early-exercise handling for non-dividend calls.
- Replaced confidence-interval clipping and heuristic tail caps with reported
  sampling uncertainty and model-independent no-arbitrage bounds.
- Implemented a real five-parameter Heston characteristic-function pricer and
  calibration, replacing the polynomial proxy previously called Heston QE.
- Changed smile-model selection from raw training RMSE to AICc.
- Bounded public SABR parameters to the optimizer-supported domain so extreme
  finite values fail closed instead of leaking numerical overflows.
- Corrected convexity checks on uneven strike grids and calendar checks at fixed
  log-moneyness.

### Reproducibility and API

- Cache keys now include exact contract/market floats, option and exercise
  styles, model configuration, and seed identity. Unseeded stochastic requests
  bypass cache.
- Unified the versioned and compatibility routes in one FastAPI application.
- Enforced request byte limits incrementally rather than buffering arbitrary
  bodies.
- Hardened OIDC/JWKS transport, key selection, algorithm matching, refresh, and
  event-loop isolation. Replaced the vulnerable `python-jose`/`ecdsa` chain
  with PyJWT and `cryptography`, including real JWK verification and minimum
  RSA key strength.
- Replaced middleware rate limiting that broke on modern lazy FastAPI routers
  with one exact per-client moving-window quota shared across protected routes.
- Bounded replay/idempotency stores and made the single-process state contract
  explicit.

### Build and operations

- Flattened the accidental nested repository so GitHub recognizes workflows.
- Moved tests out of the installed package.
- Added a typed marker, locked dependency graph, Python 3.11–3.13 CI, CodeQL,
  locked and minimum-supported dependency auditing, patched Starlette/PyJWT/
  pytest floors, package inspection, and container scanning.
- Replaced the development image with a locked, multi-stage, non-root runtime
  and one worker per container.
- Replaced fixed machine-specific benchmark gates with opt-in smoke budgets.

### Migration notes

- Python 3.10 is no longer supported.
- Invalid or unsupported inputs now raise/return explicit errors; they are no
  longer silently clamped or converted to zero-valued prices.
- The canonical Heston model name is `heston`. `HestonQECalibrator` and the
  `calib.heston_qe` module remain deprecated import aliases for one transition
  cycle.
- Deployments must use one Uvicorn worker per container unless process-local
  rate-limit, cache, idempotency, and replay state are replaced externally.
