# ---- build stage: install the package into a venv
FROM python:3.12-slim AS build
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .

# ---- runtime stage: non-root, no build tooling
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 app
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PROPINTEL_HOME=/app MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY docs/knowledge ./docs/knowledge
RUN mkdir -p data models reports mlruns && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "propintel.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
