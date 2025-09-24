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

1. Export required secrets (see [`README.md`](README.md)).
2. `uvicorn options_engine.api.fastapi_app:app --host 0.0.0.0 --port 8000`.
3. Verify `/health` returns `status=healthy` and the expected `environment`.
4. Confirm Prometheus is scraping `/metrics` and the `ope_request_total`
   counter is incrementing.

## Health probes

* **Liveness/readiness:** `GET /health`
* **Metrics:** `GET /metrics` (Prometheus exposition format)

## Alert response

Alerts are defined in `monitoring/prometheus/rules.yml` and should be wired into
PagerDuty or your preferred alert manager.

### `OPEHighHttpErrorRate`

1. Inspect the FastAPI logs for 5xx bursts.
2. Check upstream dependencies (e.g. authentication provider) for outages.
3. If the issue is caused by malformed client requests, consider tightening
   validation or request limits.

### `OPEHighLatencyP95`

1. Look at `ope_model_latency_seconds` histogram to identify the slow model.
2. Inspect CPU utilisation on the instance; consider increasing `OPE_THREADS`
   or scaling horizontally.
3. For Monte Carlo workloads, ensure callers are not requesting excessively
   large batches; the API enforces `OPE_MAX_CONTRACTS` but clients may need to
   batch responsibly.

### `OPEThreadPoolRejections`

1. Check `ope_threadpool_queue_depth` and `ope_threadpool_queue_wait_seconds`
   histograms.
2. If the queue depth is pegged, increase `OPE_THREAD_QUEUE_MAX` and/or
   `OPE_THREADS` temporarily, and file an issue to revisit capacity planning.
3. Review logs for repeated `Pricing task timed out` errors; tune
   `OPE_THREAD_TASK_TIMEOUT_SECONDS` judiciously.

### `OPEPersistentQueueBacklog`

1. Inspect the offending route (`ope_request_latency_seconds` labelled by
   route) to determine which endpoint is causing pressure.
2. Evaluate whether cache hit rate is low; if so consider increasing the LRU
   cache size (`cache_size` parameter when initialising `OptionsEngine`).

## Scaling guidance

* **Vertical:** Increase `OPE_THREADS` to saturate CPU, but benchmark to ensure
  diminishing returns do not degrade latency.
* **Horizontal:** Deploy additional instances behind a load balancer. Because
  pricing state is in-memory only, no coordination is required.
* **Queue tuning:** `OPE_THREAD_QUEUE_MAX` controls back-pressure. For latency
  sensitive environments, prefer smaller queues and graceful 503s over long
  waits.

## Token rotation

1. Generate the new secret.
2. Deploy with `OPE_JWT_SECRET=<new>` and
   `OPE_JWT_ADDITIONAL_SECRETS=<old_1>,<old_2>` to accept legacy tokens.
3. After clients have rotated, remove the old secrets from
   `OPE_JWT_ADDITIONAL_SECRETS` and redeploy.

## Troubleshooting tips

* `ope_model_errors_total{model="monte_carlo_20k"}` increments when pricing
  models throw. Inspect logs for stack traces.
* `ope_threadpool_tasks_in_flight` near `OPE_THREADS` indicates full utilisation;
  combine with queue depth to spot saturation early.
* Use the optional `seed` request field in staging to reproduce Monte Carlo
  scenarios before deploying fixes.
