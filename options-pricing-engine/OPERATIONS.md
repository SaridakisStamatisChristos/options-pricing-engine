# Operations Runbook

This document captures the common operational tasks and incident response
playbooks for the Options Pricing Engine (OPE).

## Service topology

* **API:** FastAPI application served by Uvicorn (recommended 2–4 workers).
* **Background workers:** CPU-bound pricing models executed via an internal
  thread pool sized by `OPE_THREADS`.
* **Caches:** In-memory LRU cache for recent pricing results; no external
  dependencies.

## Startup checklist

1. Export required secrets (see [`README.md`](README.md)). Ensure `OIDC_ISSUER`,
   `OIDC_AUDIENCE` and `OIDC_JWKS_URL` are present in production.
2. `uvicorn options_engine.api.fastapi_app:app --host 0.0.0.0 --port 8000`.
3. Verify `/healthz` returns `status=ok`, the expected `environment` and
   non-null uptime.
4. Confirm Prometheus is scraping `/metrics` and the `ope_request_total`
   counter is incrementing.

## Health probes

* **Liveness/readiness:** `GET /healthz`
* **Metrics:** `GET /metrics` (Prometheus exposition format)

## Alert response

Alerts are defined in `monitoring/prometheus/rules.yml` and should be wired into
PagerDuty or your preferred alert manager.

### SLO-aligned alerts

* **`OPEHighHttpErrorRate`** – investigate FastAPI JSON logs for 5xx bursts
  (each entry includes a `request_id` and the authenticated `user` where
  available). Check upstream dependencies, especially the OIDC issuer, for
  outages. Watch `ope_auth_failures_total{reason="jwt_error"}` for spikes that
  may signal clock drift or invalid tokens.
* **`OPEHighLatencyP95`** – examine `ope_request_latency_seconds` per route and
  `ope_model_latency_seconds` to determine the bottlenecked pricing model.
  Inspect CPU utilisation; increase `OPE_THREADS` or scale horizontally if
  saturation persists. For Monte Carlo workloads ensure callers respect
  batching guidance.
* **`OPEThreadPoolRejections`** – monitor both `ope_threadpool_queue_depth` and
  `ope_threadpool_queue_wait_seconds`. Sustained saturation should trigger
  capacity planning. Temporary relief can be achieved by increasing
  `OPE_THREAD_QUEUE_MAX` and/or `OPE_THREADS`. `ope_threadpool_saturation_total`
  provides a counter for hard rejections.
* **`OPERateLimitSpike` / `OPEPayloadRejections`** – high rates on
  `ope_rate_limit_rejections_total` or `ope_payload_too_large_total` indicate
  abusive clients or misconfigured integrations. Consider tuning
  `RATE_LIMIT_DEFAULT` or `MAX_BODY_BYTES` while coordinating with consumers.

## Scaling guidance

* **Vertical:** Increase `OPE_THREADS` to saturate CPU, but benchmark to ensure
  diminishing returns do not degrade latency.
* **Horizontal:** Deploy additional instances behind a load balancer. Because
  pricing state is in-memory only, no coordination is required.
* **Queue tuning:** `OPE_THREAD_QUEUE_MAX` controls back-pressure. For latency
  sensitive environments, prefer smaller queues and graceful 503s over long
  waits.

## Token rotation

OIDC signing keys are delivered via JWKS. The authenticator caches the latest
document for five minutes while retaining the previously active key set.

1. Publish the new key to the identity provider. Keep the old key in the JWKS
   during the rollout window.
2. Wait for clients to refresh their tokens (recommended overlap ≥10 minutes).
   The service accepts both the previous and current `kid` values during this
   time.
3. Remove the legacy key from the JWKS once monitoring confirms all tokens
   carry the new `kid`. The cache refresh interval is five minutes; trigger a
   manual refresh by clearing the FastAPI pod if necessary.

## Troubleshooting tips

* `ope_model_errors_total{model="monte_carlo_20k"}` increments when pricing
  models throw. Inspect logs (search by `request_id`) for stack traces.
* `ope_threadpool_tasks_in_flight` near `OPE_THREADS` indicates full utilisation;
  combine with queue depth to spot saturation early.
* `ope_rate_limit_rejections_total` and `ope_payload_too_large_total` help spot
  abusive clients before they manifest as latency problems.
* Use the optional `seed` request field in staging to reproduce Monte Carlo
  scenarios before deploying fixes.
