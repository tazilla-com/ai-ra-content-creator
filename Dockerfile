# =========================
# Builder (wheels cache)
# =========================
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /wheels

# System deps pro případné buildy (většina balíků přijde jako manylinux wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

# Dependency manifesty
COPY requirements.txt ./requirements.txt
COPY constraints.txt ./constraints.txt

# Build/pull wheels (s constraints, pokud jsou k dispozici)
RUN /bin/sh -c 'if [ -f constraints.txt ]; then \
      pip wheel --wheel-dir=/wheels -r requirements.txt -c constraints.txt; \
    else \
      pip wheel --wheel-dir=/wheels -r requirements.txt; \
    fi'


# =========================
# Runtime (slim + non-root)
# =========================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    DATA_ROOT=/app/data \
    ENABLE_DOCS=true \
    WORKSHOPS_AI_CORS_ORIGINS="*"

WORKDIR /app

# Non-root uživatel
RUN useradd -u 1001 -ms /bin/bash appuser

# Instalace z vyrobených wheels (repro build)
COPY --from=builder /wheels /wheels
COPY requirements.txt ./requirements.txt
COPY constraints.txt ./constraints.txt
RUN /bin/sh -c 'if [ -f constraints.txt ]; then \
      pip install --no-index --find-links=/wheels -r requirements.txt -c constraints.txt; \
    else \
      pip install --no-index --find-links=/wheels -r requirements.txt; \
    fi' && \
    rm -rf /wheels

# Aplikace
COPY . .

# Připrav adresáře a práva
RUN mkdir -p /app/data/outbox /app/logs && \
    chown -R appuser:appuser /app

# Entrypoint + healthcheck (musí existovat v repo: docker/entrypoint.sh, docker/healthcheck.py)
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/healthcheck.py /healthcheck.py
RUN chmod +x /entrypoint.sh

USER appuser

EXPOSE 8000

# Healthcheck (čte PORT z env a volá /livez)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD ["python", "/healthcheck.py"]

# Spuštění ASGI aplikace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "workshops_ai.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
