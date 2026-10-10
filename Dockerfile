# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Stage 1: build the React frontend
# ---------------------------------------------------------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2: Python API that also serves the built frontend
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LOG_FORMAT=json \
    ENVIRONMENT=production

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

COPY backend/app backend/app
COPY sample_data sample_data
COPY --from=web /web/dist frontend/dist

RUN useradd --create-home --uid 10001 analyst && chown -R analyst /app
USER analyst

WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# One worker: sessions live in memory (see README > Scaling for the multi-instance design).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--timeout-graceful-shutdown", "10"]
