# File: Dockerfile
# =========================
# Builder (wheels cache)
# =========================
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /wheels

# Build deps – most libs come as manylinux wheels, but keep toolchain for safety
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Dependency manifests
COPY requirements.txt ./requirements.txt
COPY constraints.txt ./constraints.txt

# Reproducible wheels (with constraints if present)
RUN /bin/sh -c 'if [ -f constraints.txt ]; then \
      pip wheel --wheel-dir=/wheels -r requirements.txt -c constraints.txt; \
    else \
      pip wheel --wheel-dir=/wheels -r requirements.txt; \
    fi'


# =========================
# Runtime (slim + non-root)
# =========================
FROM python:3.11-slim AS runtime

# NOTE: The app reads config from .env at runtime via docker-compose env_file.
# Keep only minimal runtime env here.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Non-root user (UID 1001 for volume compatibility)
RUN useradd -u 1001 -ms /bin/bash appuser

# Install from prebuilt wheels
COPY --from=builder /wheels /wheels
COPY requirements.txt ./requirements.txt
COPY constraints.txt ./constraints.txt
RUN /bin/sh -c 'if [ -f constraints.txt ]; then \
      pip install --no-index --find-links=/wheels -r requirements.txt -c constraints.txt; \
    else \
      pip install --no-index --find-links=/wheels -r requirements.txt; \
    fi' && \
    rm -rf /wheels

# App sources
COPY . .

# Prepare data/log directories and permissions
RUN mkdir -p /app/data/outbox /app/logs && \
    chown -R appuser:appuser /app

# Entrypoint + healthcheck
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/healthcheck.py /healthcheck.py
RUN chmod +x /entrypoint.sh

USER appuser

EXPOSE 8000

# Healthcheck hits /livez on $PORT
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD ["python", "/healthcheck.py"]

# Launch ASGI (port mapping handled by compose)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "workshops_ai.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
