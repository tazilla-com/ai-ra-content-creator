#!/usr/bin/env bash
set -euo pipefail

# Ensure runtime dirs exist (for mounted volumes)
mkdir -p /app/data/outbox /app/logs

# Best-effort permissions (mounted volumes may already be owned correctly)
if [ -w /app/data ] && [ -w /app/logs ]; then
  :
else
  echo "[entrypoint] Warning: /app/data or /app/logs are not writable by current user." >&2
fi

exec "$@"
