# File: workshops_ai/tools/memory.py
# Purpose:
#   Lightweight in-process memory for Workshop Agents.
#   - Per-org snapshot store + per-tab buckets (assets/impacts/threats/controls).
#   - Thread-safe (RLock), non-blocking API (all functions are sync).
#   - TTL support (expires_at); lazy pruning on access.
#   - Size guards: max keys per org, max value size; LRU eviction beyond limits.
#   - JSON-safe dumps for size accounting (falls back to repr()).
#
# Notes:
#   - This is *process-local* only. For multi-process/deploy use a proper KV store.
#   - Errors never escape in getters; setters validate and may raise ValueError/MemoryError.
#   - Keep it dependency-light; telemetry is optional (best-effort).
#
# Usage:
#   memory_set("acme", "assets.last_selected_asset_id", "a-123", tab="assets", ttl_sec=3600)
#   aid = memory_get("acme", "assets.last_selected_asset_id", tab="assets")
#   memory_delete("acme", "assets.last_selected_asset_id", tab="assets")
#   memory_list("acme", tab="assets", prefix="assets.")
#   memory_clear_org("acme")

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Tuple

from workshops_ai.settings import get_settings

# Optional, best-effort telemetry
try:  # pragma: no cover
    from workshops_ai.tools.telemetry import telemetry_log  # type: ignore
except Exception:  # pragma: no cover
    def telemetry_log(*args, **kwargs):  # type: ignore
        return None

# ------------------------------------------------------------------------------
# Config (centralized via settings)
# ------------------------------------------------------------------------------

_SETTINGS = get_settings()
MAX_ORG_KEYS: int = int(getattr(_SETTINGS.memory, "max_org_keys", 1000))
MAX_VALUE_BYTES: int = int(getattr(_SETTINGS.memory, "max_value_bytes", 262_144))  # ~256 KiB
SOFT_TOTAL_BYTES_PER_ORG: int = int(
    getattr(_SETTINGS.memory, "soft_total_bytes_per_org", 8_388_608)
)  # ~8 MiB

# ------------------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------------------

@dataclass
class _Entry:
    key: str
    value: Any
    size_bytes: int
    created_at: float
    updated_at: float
    expires_at: Optional[float] = None


@dataclass
class _OrgStore:
    """Holds global keys and per-tab buckets for a single org."""
    global_kv: Dict[str, _Entry] = field(default_factory=dict)
    tabs: Dict[str, Dict[str, _Entry]] = field(default_factory=dict)
    total_bytes: int = 0  # approximate (sum of entry.size_bytes)


# ------------------------------------------------------------------------------
# In-memory store & lock
# ------------------------------------------------------------------------------

_STORE: Dict[str, _OrgStore] = {}
_LOCK = RLock()

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------

def memory_set(
    org_id: str,
    key: str,
    value: Any,
    *,
    tab: Optional[str] = None,
    ttl_sec: Optional[float] = None,
) -> None:
    """
    Set a value for `org_id` under `key`. If `tab` is provided, store it under that tab bucket.
    Enforces value size limit and LRU eviction when org exceeds caps.
    Raises:
        ValueError   – on invalid org_id/key or oversize value (MAX_VALUE_BYTES).
        MemoryError  – if eviction fails to free enough space (very rare).
    """
    now = time.time()
    if not org_id or not isinstance(org_id, str):
        raise ValueError("org_id must be a non-empty string")
    if not key or not isinstance(key, str):
        raise ValueError("key must be a non-empty string")

    size = _approx_size_bytes(value)
    if size > MAX_VALUE_BYTES:
        raise ValueError(f"value too large ({size} bytes > {MAX_VALUE_BYTES})")

    with _LOCK:
        store = _STORE.setdefault(org_id, _OrgStore())
        bucket = _get_bucket(store, tab)
        _prune_expired(store)  # lazy prune

        # Adjust total for replacement
        prev = bucket.get(key)
        if prev:
            store.total_bytes -= max(prev.size_bytes, 0)

        expires_at = (now + float(ttl_sec)) if ttl_sec and ttl_sec > 0 else None
        entry = _Entry(
            key=key,
            value=value,
            size_bytes=size,
            created_at=prev.created_at if prev else now,
            updated_at=now,
            expires_at=expires_at,
        )
        bucket[key] = entry
        store.total_bytes += size

        # Enforce caps
        _enforce_caps(org_id, store)

        # Best-effort telemetry
        try:
            telemetry_log(
                "memory.set",
                {"org_id": org_id, "tab": tab, "key": key, "size": size, "ttl_sec": ttl_sec},
                level="DEBUG",
                org_id=org_id,
                source="AI",
            )
        except Exception:
            pass


def memory_get(org_id: str, key: str, *, tab: Optional[str] = None, default: Any = None) -> Any:
    """
    Get a value or `default` if not found/expired. Never raises on read paths.
    Updates the entry's `updated_at` for fair LRU behavior.
    """
    now = time.time()
    if not org_id or not key:
        return default

    with _LOCK:
        store = _STORE.get(org_id)
        if not store:
            return default

        bucket = _get_bucket(store, tab, create=False)
        if bucket is None:
            return default

        entry = bucket.get(key)
        if not entry:
            return default

        if entry.expires_at and entry.expires_at <= now:
            # Expired – remove
            try:
                store.total_bytes -= max(entry.size_bytes, 0)
                del bucket[key]
            except Exception:
                pass
            return default

        # Touch for LRU
        entry.updated_at = now
        return entry.value


def memory_delete(org_id: str, key: str, *, tab: Optional[str] = None) -> bool:
    """
    Delete a value; returns True if existed and was removed.
    """
    if not org_id or not key:
        return False

    with _LOCK:
        store = _STORE.get(org_id)
        if not store:
            return False
        bucket = _get_bucket(store, tab, create=False)
        if bucket is None:
            return False
        entry = bucket.pop(key, None)
        if entry:
            store.total_bytes -= max(entry.size_bytes, 0)
            return True
        return False


def memory_list(org_id: str, *, tab: Optional[str] = None, prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a *copy* of keys → values for the given org (and tab if specified).
    Expired entries are omitted. Intended for diagnostics/UI previews.
    """
    now = time.time()
    out: Dict[str, Any] = {}
    if not org_id:
        return out

    with _LOCK:
        store = _STORE.get(org_id)
        if not store:
            return out

        buckets: Iterable[Tuple[str, Dict[str, _Entry]]]
        if tab:
            b = _get_bucket(store, tab, create=False)
            buckets = [(tab, b or {})]
        else:
            buckets = [("global", store.global_kv), *[(t, b) for t, b in store.tabs.items()]]

        for _, bkt in buckets:
            for k, e in list(bkt.items()):
                if e.expires_at and e.expires_at <= now:
                    # prune on the fly
                    try:
                        store.total_bytes -= max(e.size_bytes, 0)
                        del bkt[k]
                    except Exception:
                        pass
                    continue
                if prefix and not k.startswith(prefix):
                    continue
                out[k] = e.value
    return out


def memory_clear_org(org_id: str) -> None:
    """Remove all memory for an org (idempotent)."""
    if not org_id:
        return
    with _LOCK:
        _STORE.pop(org_id, None)
    try:
        telemetry_log("memory.clear_org", {"org_id": org_id}, level="INFO", org_id=org_id)
    except Exception:
        pass


# ------------------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------------------

def _get_bucket(store: _OrgStore, tab: Optional[str], *, create: bool = True) -> Dict[str, _Entry] | None:
    if not tab:
        return store.global_kv
    if create:
        return store.tabs.setdefault(tab, {})
    return store.tabs.get(tab)


def _approx_size_bytes(value: Any) -> int:
    """
    Approximate size of a value for budgeting:
      1) try JSON (ensure_ascii=False) → len of resulting UTF-8 bytes
      2) fallback to repr() encoded to UTF-8
    """
    try:
        s = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return len(s.encode("utf-8"))
    except Exception:
        try:
            r = repr(value)
            return len(r.encode("utf-8"))
        except Exception:
            return 64  # arbitrary tiny fallback


def _prune_expired(store: _OrgStore) -> None:
    now = time.time()
    # Global
    for k in list(store.global_kv.keys()):
        e = store.global_kv.get(k)
        if e and e.expires_at and e.expires_at <= now:
            store.total_bytes -= max(e.size_bytes, 0)
            store.global_kv.pop(k, None)
    # Tabs
    for t, b in list(store.tabs.items()):
        for k in list(b.keys()):
            e = b.get(k)
            if e and e.expires_at and e.expires_at <= now:
                store.total_bytes -= max(e.size_bytes, 0)
                b.pop(k, None)
        if not b:
            store.tabs.pop(t, None)


def _enforce_caps(org_id: str, store: _OrgStore) -> None:
    """
    Enforce MAX_ORG_KEYS and SOFT_TOTAL_BYTES_PER_ORG via LRU eviction.
    Eviction order: oldest updated_at first, across global + all tabs.
    """
    # Build LRU list
    items: list[tuple[float, str, Optional[str]]] = []  # (updated_at, key, tab)
    for k, e in store.global_kv.items():
        items.append((e.updated_at, k, None))
    for tab, b in store.tabs.items():
        for k, e in b.items():
            items.append((e.updated_at, k, tab))

    # Nothing to do
    if not items:
        return

    # Conditions
    def _over_key_limit() -> bool:
        total_keys = len(store.global_kv) + sum(len(b) for b in store.tabs.values())
        return total_keys > MAX_ORG_KEYS

    def _over_byte_limit() -> bool:
        return store.total_bytes > SOFT_TOTAL_BYTES_PER_ORG

    if not _over_key_limit() and not _over_byte_limit():
        return

    # Sort oldest first
    items.sort(key=lambda t: t[0])

    idx = 0
    evicted = 0
    while (idx < len(items)) and (_over_key_limit() or _over_byte_limit()):
        _, key, tab = items[idx]
        idx += 1
        bucket = store.global_kv if tab is None else store.tabs.get(tab, {})
        ent = bucket.pop(key, None)
        if ent:
            store.total_bytes -= max(ent.size_bytes, 0)
            evicted += 1

    if _over_key_limit() or _over_byte_limit():
        # This would be extremely rare (e.g., negative sizes or miscount)
        raise MemoryError("Unable to enforce memory limits after eviction.")

    if evicted:
        try:
            telemetry_log(
                "memory.eviction",
                {"org_id": org_id, "evicted": evicted, "total_bytes": store.total_bytes},
                level="INFO",
                org_id=org_id,
            )
        except Exception:
            pass


__all__ = [
    "memory_set",
    "memory_get",
    "memory_delete",
    "memory_list",
    "memory_clear_org",
]
