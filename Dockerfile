# syntax=docker/dockerfile:1.7

# --- pickable base--------------
  ARG BASE_IMAGE=python:3.11-slim

  # =========================
  # Builder (wheels cache)
  # =========================
  FROM ${BASE_IMAGE} AS builder
  
  ENV DEBIAN_FRONTEND=noninteractive \
      PIP_DISABLE_PIP_VERSION_CHECK=1 
  
  SHELL ["/bin/bash", "-o", "pipefail", "-c"]
  WORKDIR /wheels
  
  # build deps for native wheels
  RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
      && rm -rf /var/lib/apt/lists/*
  
  COPY requirements.txt ./requirements.txt
  COPY constraints.txt ./constraints.txt
  
  # Use BuildKit cache for pip
  RUN --mount=type=cache,target=/root/.cache/pip \
    if [[ -f constraints.txt ]]; then \
      python -m pip wheel --wheel-dir=/wheels -r requirements.txt -c constraints.txt; \
    else \
      python -m pip wheel --wheel-dir=/wheels -r requirements.txt; \
    fi
  
  
  # =========================
  # Runtime (slim + non-root)
  # =========================
  FROM ${BASE_IMAGE} AS runtime
  
  ENV DEBIAN_FRONTEND=noninteractive \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      PIP_NO_CACHE_DIR=1 \
      PORT=8000 \
      PYTHONPATH=/app
  
  SHELL ["/bin/bash", "-o", "pipefail", "-c"]
  WORKDIR /app
  
  # Non-root user
  RUN useradd -u 1001 -ms /bin/bash appuser
  
  # Install from prebuilt wheels
  COPY --from=builder /wheels /wheels
  COPY requirements.txt ./requirements.txt
  COPY constraints.txt ./constraints.txt
  
  RUN --mount=type=cache,target=/root/.cache/pip \
      if [[ -f constraints.txt ]]; then \
        python -m pip install --no-index --find-links=/wheels -r requirements.txt -c constraints.txt; \
      else \
        python -m pip install --no-index --find-links=/wheels -r requirements.txt; \
      fi \
      && rm -rf /wheels
  
  # App sources
  COPY . .
  
  # Ensure runtime dirs exist & are writable (important)
  RUN mkdir -p \
        /app/data \
        /app/workshops_ai/data/outbox \
        /app/workshops_ai/data/state \
        /app/workshops_ai/data/chroma \
        /app/logs \
   && chown -R appuser:appuser /app
  
  # Entrypoint + healthcheck
  COPY docker/entrypoint.sh /entrypoint.sh
  COPY docker/healthcheck.py /healthcheck.py
  RUN chmod +x /entrypoint.sh
  
  USER appuser
  EXPOSE 8000
  
  HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD ["python", "/healthcheck.py"]
  
  ENTRYPOINT ["/entrypoint.sh"]
  CMD ["uvicorn", "workshops_ai.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
  