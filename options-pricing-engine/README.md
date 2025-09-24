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

   The service refuses to start without a signing key for JWTs:

   ```bash
   export OPE_JWT_SECRET="$(openssl rand -hex 32)"
   ```

3. **Launch the API**

   ```bash
   uvicorn options_engine.api.fastapi_app:app --host 0.0.0.0 --port 8000
   ```

4. **Smoke test**

   ```bash
   curl -H "Authorization: Bearer <token>" \
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
| `OPE_JWT_SECRET` | **Required.** Primary HMAC secret for JWT signing. | – |
| `OPE_JWT_ADDITIONAL_SECRETS` | Comma-separated list of previous secrets accepted for token rotation. | empty |
| `OPE_JWT_AUDIENCE` / `OPE_JWT_ISSUER` | Expected JWT audience and issuer claims. | `options-pricing-engine` |
| `OPE_JWT_EXP_MINUTES` | Access token lifetime. | `30` |
| `OPE_JWT_LEEWAY_SECONDS` | Clock skew tolerance when validating tokens. | `30` |
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

* JWTs must be issued using the configured `OPE_JWT_SECRET`; there is no insecure fallback.
* Tokens include `exp`, `nbf`, `iat`, `aud`, `iss` and `jti` claims by default. You can rotate
  secrets without downtime by setting `OPE_JWT_ADDITIONAL_SECRETS` to the previous keys.
* API endpoints attach strict transport and anti-MIME-sniffing headers via middleware.
* Trusted hosts and CORS defaults are restrictive. In production you must configure explicit
  allow-lists.

## Deterministic Monte Carlo runs

Monte Carlo simulations can be made reproducible by providing a seed either globally or per
request:

* `OPE_MONTE_CARLO_SEED` seeds all Monte Carlo runs with independent child sequences.
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
  error counts, model timings and thread pool pressure.
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
