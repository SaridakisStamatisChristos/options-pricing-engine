# Options Pricing Engine

The Options Pricing Engine (OPE) is a FastAPI service that exposes real-time options
valuation and risk analytics powered by Black-Scholes, binomial and Monte Carlo models.
It is designed for production use with hardened authentication, observability and
operational guardrails.

## Quick start

1. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install .[dev]
   ```

2. **Provide mandatory secrets**

   Configure the OpenID Connect issuer that signs access tokens. The JWKS endpoint can be
   hosted locally using any static file server or an identity provider in your environment.

   ```bash
   export OIDC_ISSUER="https://auth.local"
   export OIDC_AUDIENCE="options-pricing-engine"
   export OIDC_JWKS_URL="https://auth.local/.well-known/jwks.json"
   ```

   For local development you may optionally set `DEV_JWT_SECRET` (and rotation values in
   `DEV_JWT_ADDITIONAL_SECRETS`) to use the built-in HMAC fallback while no OIDC provider is
   available. Secrets must decode to at least 32 bytes (base64url or hexadecimal values are
   accepted). The variables are ignored – and forbidden – when `OPE_ENVIRONMENT=production`.

3. **Launch the API**

   ```bash
   uvicorn options_engine.api.fastapi_app:app --host 0.0.0.0 --port 8000
   ```

4. **Smoke test**

   ```bash
   curl -H "Authorization: Bearer ${OIDC_BEARER_TOKEN}" \
        -H "Content-Type: application/json" \
        -d @examples/price_single.json \
        http://localhost:8000/api/v1/pricing/single
   ```

The Dockerfile ships with the same defaults; ensure you inject the environment variables
above when building or running the container.

## Environment configuration

All knobs are controlled via environment variables. Omitted values fall back to safe
defaults in non-production environments but must be explicitly configured in production.

| Variable | Description | Default (non-prod) |
| --- | --- | --- |
| `OPE_ENVIRONMENT` | Deployment environment label (`development`, `staging`, `production`). | `development` |
| `OIDC_ISSUER` | **Required in production.** Expected issuer for JWTs. | unset |
| `OIDC_AUDIENCE` | **Required in production.** Audience claim required on JWTs. | unset |
| `OIDC_JWKS_URL` | **Required in production.** HTTPS endpoint serving the JWKS document. | unset |
| `RATE_LIMIT_DEFAULT` | Default SlowAPI rate limit applied to authenticated routes. | `60/minute` |
| `MAX_BODY_BYTES` | Maximum accepted request payload size. | `1_048_576` |
| `DEV_JWT_SECRET` | Optional symmetric secret for non-production environments. Must be ≥32 bytes after decoding (base64url or hex). | unset |
| `DEV_JWT_ADDITIONAL_SECRETS` | Additional development secrets accepted during rotation. Must satisfy the same length/encoding requirements. | empty |
| `OPE_ALLOWED_HOSTS` | Comma-separated host allow-list for the TrustedHost middleware. **Required in production.** | `localhost,127.0.0.1` |
| `OPE_ALLOWED_ORIGINS` | CORS allow list. | `http://localhost,http://localhost:3000,http://localhost:8000` |
| `OPE_CORS_ALLOW_CREDENTIALS` | Whether CORS responses include credentials. | `true` |
| `OPE_THREADS` | Worker threads backing the pricing engine. | `8` |
| `OPE_THREAD_QUEUE_MAX` | Maximum queued pricing jobs before rejecting new work. | `32` |
| `OPE_THREAD_QUEUE_TIMEOUT_SECONDS` | How long to wait when enqueuing work before returning `503`. `0` disables waiting. | `0.5` |
| `OPE_THREAD_TASK_TIMEOUT_SECONDS` | Hard timeout for pricing jobs once scheduled. `0` disables the limit. | `30.0` |
| `OPE_MAX_CONTRACTS` | Maximum contracts accepted by pricing endpoints. | `1000` |
| `OPE_MAX_RISK_CONTRACTS` | Maximum contracts accepted by risk aggregation endpoint. | `OPE_MAX_CONTRACTS` |
| `OPE_MONTE_CARLO_SEED` | Optional global seed for Monte Carlo reproducibility. | unset |

## Authentication and security

* Access tokens are validated against the configured OIDC issuer and JWKS. Claims `iss`, `aud`,
  `exp`, `nbf`, `iat` and `kid` are required with a ±60s clock skew tolerance.
* Tokens must include the appropriate scopes (`pricing:read`, `market-data:write`, …). The
  `scope` or `scp` claim is parsed as a space-delimited list.
* JWKS documents are cached for 5 minutes. During key rotation both the previous and current
  keys remain valid automatically.
* In non-production environments the API can validate tokens signed with `DEV_JWT_SECRET` when an
  OIDC provider is unavailable. Multiple secrets may be provided (via
  `DEV_JWT_ADDITIONAL_SECRETS`) to smooth rotations. Development tokens must carry the same
  `iss`, `aud`, `sub`, `iat`, `nbf` and `exp` claims as production OIDC tokens and are rejected in
  production deployments.
* API endpoints attach strict transport security, request IDs and JSON structured logs for every
  call. Headers such as `X-Content-Type-Options` and `Strict-Transport-Security` are enforced by
  middleware.
* Trusted hosts and CORS defaults are restrictive. In production you must configure explicit
  allow-lists.

### Example authenticated request

```bash
TOKEN="$(your_token_issuer_flow)"
curl -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d @examples/price_single.json \
     http://localhost:8000/api/v1/pricing/single
```

## Deterministic Monte Carlo runs

Monte Carlo simulations can be made reproducible by providing a seed either globally or per
request:

* `OPE_MONTE_CARLO_SEED` seeds all Monte Carlo runs with independent child sequences. When
  debugging stochastic behaviour, pin this in the environment for deterministic responses.
* Requests may include an optional `seed` field. For batch requests each contract receives a
  deterministic child seed ensuring results are stable across retries.

The response includes the computed statistics (price, standard error, 95% confidence interval)
for auditability.

## Operational guardrails

* **Thread pool saturation:** queue length, queue wait time and rejection counters are exposed
  via Prometheus (`/metrics`). Requests are rejected with HTTP 503 when capacity is exhausted.
* **Input validation:** Batch endpoints reject requests larger than `OPE_MAX_CONTRACTS`/`OPE_MAX_RISK_CONTRACTS`.
* **Health checks:** `/health` returns service status and environment metadata.

## Observability

* `/metrics` exposes Prometheus counters, gauges and histograms for request rates, latency,
  rate-limit or payload rejections, authentication failures, model timings and thread pool
  pressure. Look for metrics prefixed with `ope_request_latency_seconds`, `ope_rate_limit_…` and
  `ope_auth_failures_…`.
* Alerting starter rules live under `monitoring/prometheus/rules.yml` with SLOs for error rate,
  latency and thread pool saturation.

## Testing & linting

Run the suite locally before committing:

```bash
pytest
ruff check src
mypy src
```

## Further reading

* [`OPERATIONS.md`](OPERATIONS.md) – runbook with on-call workflows and scaling guidance.
* [`SECURITY.md`](SECURITY.md) – token management, rotation and hardening checklist.
