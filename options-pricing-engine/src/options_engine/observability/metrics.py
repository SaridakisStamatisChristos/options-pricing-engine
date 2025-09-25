"""Prometheus metrics used across the pricing engine."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


REQUEST_COUNT = Counter(
    "ope_request_total",
    "Total number of HTTP requests processed",
    labelnames=("method", "route", "status_code"),
)

REQUEST_ERRORS = Counter(
    "ope_request_errors_total",
    "Total number of error responses emitted",
    labelnames=("method", "route", "status_code"),
)

REQUEST_LATENCY = Histogram(
    "ope_request_latency_seconds",
    "Distribution of HTTP request latency",
    labelnames=("method", "route"),
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ),
)

MODEL_LATENCY = Histogram(
    "ope_model_latency_seconds",
    "Time spent executing pricing models",
    labelnames=("model",),
    buckets=(
        0.0005,
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
    ),
)

MODEL_ERRORS = Counter(
    "ope_model_errors_total",
    "Number of failures encountered while executing pricing models",
    labelnames=("model",),
)

AUTH_FAILURES = Counter(
    "ope_auth_failures_total",
    "Number of authentication or authorization failures",
    labelnames=("reason",),
)

RATE_LIMIT_REJECTIONS = Counter(
    "ope_rate_limit_rejections_total",
    "Number of requests rejected due to rate limiting",
    labelnames=("route",),
)

PAYLOAD_TOO_LARGE = Counter(
    "ope_payload_too_large_total",
    "Number of requests rejected because the payload exceeded limits",
    labelnames=("route",),
)

THREADPOOL_QUEUE_DEPTH = Gauge(
    "ope_threadpool_queue_depth",
    "Tasks waiting in the pricing engine queue",
    labelnames=("engine",),
)

THREADPOOL_IN_FLIGHT = Gauge(
    "ope_threadpool_tasks_in_flight",
    "Currently executing pricing tasks",
    labelnames=("engine",),
)

THREADPOOL_WORKERS = Gauge(
    "ope_threadpool_workers",
    "Configured worker threads for the pricing engine",
    labelnames=("engine",),
)

THREADPOOL_QUEUE_WAIT = Histogram(
    "ope_threadpool_queue_wait_seconds",
    "Time spent waiting to submit work to the pricing engine",
    labelnames=("engine",),
    buckets=(
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
    ),
)

THREADPOOL_REJECTIONS = Counter(
    "ope_threadpool_rejections_total",
    "Number of requests rejected because the pricing engine queue was full",
    labelnames=("engine",),
)

THREADPOOL_SATURATION = Counter(
    "ope_threadpool_saturation_total",
    "Number of times the pricing engine queue was saturated",
    labelnames=("engine",),
)

