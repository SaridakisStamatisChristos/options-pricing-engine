FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .[dev]
COPY src/ ./src/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "options_engine.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
