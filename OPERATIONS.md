# Operations Runbook

## Deployment contract

Run exactly one Uvicorn process per container and scale with container replicas.
The response cache, rate-limit counters, idempotency records, and replay capsule
store are bounded but process-local. A multi-worker command would split that
state and produce inconsistent client behavior.

The pricing engine uses a bounded thread pool inside the process. Threads keep
request handling responsive, but CPU-bound NumPy/BLAS behavior depends on the
linked numerical libraries. Benchmark the complete container on its target CPU
before changing `OPE_THREADS` or BLAS thread settings.

The in-process rate limiter applies one moving-window quota per visible client
across all protected routes. Enforce global or identity-aware quotas at the
trusted ingress, and configure that ingress to overwrite forwarding headers so
clients cannot spoof the address visible to the application.

## Startup checklist

1. Supply `OPE_ENVIRONMENT=production`, `OIDC_ISSUER`, `OIDC_AUDIENCE`,
   `OIDC_JWKS_URL`, and `OPE_ALLOWED_HOSTS`.
2. Confirm no `DEV_JWT_*` variable is present. Startup intentionally fails if a
   development secret is supplied in production.
3. Start `uvicorn options_engine.api.fastapi_app:app --host 0.0.0.0 --port 8000 --workers 1`.
4. Wait for `GET /healthz` to return `200` and the expected package version and
   environment.
5. Restrict `/metrics` and health endpoints at the ingress if they must not be
   public in your environment.
6. Confirm Prometheus records `ope_request_total` after a synthetic request.

The image health check selects its `Host` header from `ALLOWED_HOSTS` or
`OPE_ALLOWED_HOSTS`; ensure the service accepts the first configured value.

## Probes and shutdown

- Liveness/readiness: `GET /healthz`
- Compatibility health alias: `GET /health`
- Metrics: `GET /metrics`
- Container stop grace: at least 30 seconds

The configured task timeout bounds how long a request waits. Python threads
cannot safely be killed after they begin; timed-out computations may continue
until their numerical routine returns. Use the queue/in-flight metrics when
choosing pod termination and autoscaling behavior.

## Capacity and back-pressure

| Signal | Interpretation | First response |
| --- | --- | --- |
| `ope_threadpool_tasks_in_flight` near `OPE_THREADS` | Workers are occupied | Inspect model mix and CPU saturation |
| `ope_threadpool_queue_depth` above zero | Requests are waiting | Scale replicas or lower per-request cost |
| `ope_threadpool_rejections_total` increasing | Admission capacity exhausted | Scale out; do not mask sustained saturation with a large queue |
| `ope_threadpool_queue_wait_seconds` rising | Latency is queue-driven | Reduce queue length for fail-fast behavior or add capacity |
| Model latency histogram regression | Computation changed | Reproduce with a pinned seed and compare model/version metadata |

Prefer horizontal scaling. Increasing `OPE_THREAD_QUEUE_MAX` trades fewer 503s
for higher tail latency and does not create compute capacity.

## Alert response

Rules are in `monitoring/prometheus/rules.yml`.

### High server-error rate

1. Group `ope_request_errors_total` 5xx responses by route/status and compare
   expected 4xx traffic through `ope_request_total`.
2. Correlate logs with `X-Request-ID`.
3. Check OIDC/JWKS reachability and clock synchronization.
4. Check queue rejection and model error counters.
5. Compare the deployed image digest and package version with the last healthy
   deployment.

### High p95 latency

1. Compare request latency with model latency and queue wait histograms.
2. Check CPU throttling, memory pressure, and BLAS oversubscription.
3. Identify Monte Carlo path counts and batch sizes from request/model metadata.
4. Scale replicas before increasing in-process threads.

### Authentication failures

- `unknown_kid`: verify the issuer published the new key and that `alg`, `use`,
  and `key_ops` are valid.
- `jwks_unavailable`: the service uses a previously cached valid key set during
  a bounded grace period; restore the identity provider before it expires.
- `aud`, `iss`, or `expired`: inspect client configuration and clock drift.
- `dev_bad_sig`: a non-production client is using the wrong rotation key.

## OIDC key rotation

1. Publish the new signing key alongside the old key.
2. Begin issuing tokens with the new `kid`.
3. Keep both keys present for at least the maximum token lifetime plus the JWKS
   cache TTL.
4. Monitor authentication failures, then remove the old key.

The decoder verifies that the token header algorithm exactly matches the
selected JWK algorithm before and after a forced refresh.

## Rollback and replay

Monte Carlo responses may include a replay capsule and seed lineage. Capsules
are local to one process and have bounded TTL/capacity; they are diagnostic aids,
not durable audit storage. Persist response payloads and image/package digests
in your own audit system when retention is required.

For a rollback:

1. Restore the previous immutable image digest.
2. Verify `/healthz` version/build metadata.
3. Replay a pinned set of independent golden scenarios.
4. Compare prices, Greeks, confidence intervals, and model identifiers—not only
   HTTP status.

## Backup and recovery

The service has no durable internal database. Volatility surface updates,
caches, replay capsules, and idempotency records disappear on restart. Clients
must be able to republish market state, and durable request/audit retention must
live outside this process.
