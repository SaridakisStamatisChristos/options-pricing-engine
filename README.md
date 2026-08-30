# Options Pricing Engine

[![CI](https://github.com/SaridakisStamatisChristos/options-pricing-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/SaridakisStamatisChristos/options-pricing-engine/actions/workflows/ci.yml)
[![CodeQL](https://github.com/SaridakisStamatisChristos/options-pricing-engine/actions/workflows/codeql.yml/badge.svg)](https://github.com/SaridakisStamatisChristos/options-pricing-engine/actions/workflows/codeql.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11–3.13-3776AB.svg)](pyproject.toml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

An auditable Python library and FastAPI service for vanilla option valuation,
risk sensitivities, implied-volatility calibration, and deterministic Monte
Carlo replay.

The project is beta software. Validate models and market conventions against
your own independent implementation before using results in a financial
decision or production risk process.

## What is implemented

| Capability | Implementation | Important boundary |
| --- | --- | --- |
| European valuation | Black–Scholes with continuous dividends; adaptive CRR; terminal Monte Carlo; Crank–Nicolson PDE | Black–Scholes and terminal Monte Carlo reject American contracts |
| American valuation | Adaptive CRR; cross-fitted Longstaff–Schwartz; Crank–Nicolson/Rannacher PDE with PSOR | LSMC distinguishes its raw policy estimate from its bounded reported estimate |
| Monte Carlo inference | Antithetic pairs, cross-fitted discounted-underlying control variate, independently scrambled Sobol replicates, Student-t intervals over independent units | Raw and no-arbitrage-projected estimates and intervals are both retained; unseeded requests bypass cache |
| Greeks | Analytic Black–Scholes; tree/finite-difference CRR; pathwise and likelihood-ratio Monte Carlo estimators with fallbacks | Monte Carlo Greeks include estimator metadata |
| Smile calibration | SABR and five-parameter Heston characteristic-function calibration with liquidity-aware weights and deterministic holdouts | Model selection uses AICc; Heston reports Feller and cross-tenor stability diagnostics |
| Volatility surfaces | Raw-SVI slice diagnostics; global power-law SSVI with monotone ATM variance; strike, parity, convexity, and calendar checks | SSVI enforces sufficient wing/curvature constraints and validates density on a dense grid |
| Service controls | OIDC/JWKS authentication, scopes, bounded bodies, rate limits, back-pressure, metrics, replay capsules | Rate-limit/idempotency/replay state is process-local; run one worker per container |

See [Numerical methods](docs/NUMERICAL_METHODS.md) for assumptions, formulas,
validation policy, and known limitations.

## Install

Python 3.11–3.13 is supported. The committed lock file is the reproducible
development path:

```bash
uv sync --locked --extra dev
```

For library consumers:

```bash
python -m pip install .
```

## Library quick start

```python
from options_engine.core.models import MarketData, OptionContract, OptionType
from options_engine.core.black_scholes import BlackScholesModel

contract = OptionContract(
    symbol="AAPL",
    strike_price=200.0,
    time_to_expiry=0.5,
    option_type=OptionType.CALL,
)
market = MarketData(
    spot_price=205.0,
    risk_free_rate=0.04,
    dividend_yield=0.005,
)

result = BlackScholesModel().calculate_price(contract, market, volatility=0.24)
print(result.theoretical_price, result.delta, result.vega)
```

`FiniteDifferenceModel` in `options_engine.core.finite_difference` provides an
independent PDE family for both European and American vanilla contracts. The
legacy `options_engine.core.pricing_models` import path remains a compatibility
facade.

Contract identifiers hash the exact economic terms, including option and
exercise style. Reusing the same symbol does not alias nearby strikes or
maturities.

## Run the API

Every pricing route is authenticated. For local development, configure a
development-only HMAC key plus the issuer and audience claims expected in the
token:

```bash
export OPE_ENVIRONMENT=development
export OIDC_ISSUER=https://issuer.example.test
export OIDC_AUDIENCE=options-pricing-engine
export DEV_JWT_SECRET=local-development-key-at-least-32-bytes
uv run uvicorn options_engine.api.fastapi_app:app --host 127.0.0.1 --port 8000
```

Production requires `OIDC_ISSUER`, `OIDC_AUDIENCE`, `OIDC_JWKS_URL`, and an
explicit host allow-list. Development secrets are rejected in production.

Primary routes are under `/api/v1`:

- `POST /api/v1/pricing/single`
- `POST /api/v1/pricing/batch`
- `POST /api/v1/risk/aggregate-greeks`
- `GET|POST /api/v1/market-data/volatility`
- `GET /healthz`
- `GET /metrics`

Compatibility routes (`/quote`, `/batch`, `/greeks`, `/replay/{capsule_id}`)
remain available for 1.x clients and use the same middleware and authentication.
For terminal Monte Carlo, `/quote` and `/batch` may set
`precision.target_ci_bps` and `precision.max_paths`. The engine doubles the
path count until the reported 95% confidence-interval half-width meets the
target or the cap is reached. Responses report the actual path count and
`model_used.precision.target_met`; measured raw intervals are never clipped to
the target. Responses also expose the bounded interval and its projection
diagnostics.
Interactive OpenAPI documentation is at `/docs` outside production.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPE_ENVIRONMENT` | `development` | `production` enables mandatory identity and host checks |
| `OIDC_ISSUER` | unset | Required token issuer |
| `OIDC_AUDIENCE` | unset | Required token audience |
| `OIDC_JWKS_URL` | unset | HTTPS JWKS URL; HTTP is accepted only for loopback development |
| `DEV_JWT_SECRET` | unset | Non-production HMAC key: plaintext UTF-8 or explicit `base64:`/`hex:`, 32–4096 decoded bytes |
| `DEV_JWT_ADDITIONAL_SECRETS` | unset | Comma-separated previous keys using the same explicit encoding rules |
| `OPE_ALLOWED_HOSTS` | loopback in development | Comma-separated trusted `Host` values; required in production |
| `OPE_ALLOWED_ORIGINS` | local origins in development | Explicit CORS origin list |
| `OPE_THREADS` | `8` | Pricing thread-pool size inside one process |
| `OPE_THREAD_QUEUE_MAX` | `32` | Bounded waiting jobs before back-pressure |
| `OPE_THREAD_QUEUE_TIMEOUT_SECONDS` | `0.5` | Maximum queue admission wait |
| `OPE_THREAD_TASK_TIMEOUT_SECONDS` | `30` | Portfolio/request wait bound; running Python work cannot be forcibly killed |
| `OPE_MAX_CONTRACTS` | `1000` | Pricing batch contract limit |
| `OPE_MAX_RISK_CONTRACTS` | pricing limit | Risk aggregation contract limit |
| `OPE_MONTE_CARLO_SEED` | unset | Root seed; explicit request seeds take precedence |
| `RATE_LIMIT_DEFAULT` | `60/minute` | Per-process moving-window limit shared by each client across protected routes |
| `MAX_BODY_BYTES` | `1048576` | Maximum request bytes, enforced while streaming |

## Containers

The image is multi-stage, uses the locked dependency graph, runs as UID 10001,
and contains no compilers or development dependencies. The Compose example adds
a read-only root filesystem, drops all Linux capabilities, and enables
`no-new-privileges`:

```bash
docker compose -f docker/docker-compose.yml up --build
```

Use one Uvicorn worker per container. Scale horizontally with multiple
containers. Multiple workers in one container would create divergent rate
limits, idempotency records, replay capsules, and caches.

## Verification

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run bandit -q -r src -ll
uv run pytest -m "not performance" --cov=options_engine
RUN_PERFORMANCE_TESTS=1 uv run pytest -m performance
uv build
```

The performance test is an opt-in smoke budget, not a portable benchmark. CI
also runs dependency auditing, CodeQL, distribution inspection, and a Trivy
container scan.

## Project docs

- [Numerical methods and validation](docs/NUMERICAL_METHODS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Operations runbook](OPERATIONS.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Release notes](RELEASE_NOTES.md)

Licensed under the Apache License 2.0.
