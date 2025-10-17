from __future__ import annotations

from typing import Optional
from workshops_ai.backend.port import BackendPort
from workshops_ai.backend.csv_backend import CsvBackend

_backend: Optional[BackendPort] = None

def get_backend_port() -> BackendPort:
    """
    Return process-wide BackendPort singleton.
    Today exposes CsvBackend; later you can swap to ApiClientBackend
    without touching agents/manager.
    """
    global _backend
    if _backend is None:
        _backend = CsvBackend()  # CsvBackend reads paths from settings()
    return _backend
