# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OPE_ENVIRONMENT=production

RUN groupadd --gid 10001 options-engine \
    && useradd --uid 10001 --gid options-engine --no-create-home --shell /usr/sbin/nologin options-engine

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv

WORKDIR /app
USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import os, urllib.request; raw = os.getenv('ALLOWED_HOSTS') or os.getenv('OPE_ALLOWED_HOSTS') or '127.0.0.1'; host = next(part.strip() for part in raw.split(',') if part.strip()); request = urllib.request.Request('http://127.0.0.1:8000/healthz', headers={'Host': host}); urllib.request.urlopen(request, timeout=3).read()"]

# The rate limiter, replay capsules, and idempotency cache are process-local.
# Scale with one Uvicorn worker per container and multiple container replicas.
CMD ["uvicorn", "options_engine.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
