# File: workshops_ai/tools/memory.py
# Purpose:
#   Lightweight memory for Workshop Agents with a Redis backend and in-process fallback.
#   - Per-org snapshot store + per-tab buckets (assets/impacts/threats/controls).
#   - Thread-safe (RLock), non-blocking public API (all functions are sync).
#   - TTL support (per key); lazy pruning (in-proc); native TTL in Redis.
#   - Guards: max keys per org, max value size; LRU eviction when limits are exceeded.
#   - JSON-safe serialization for size accounting; falls back to repr() if needed.
#
# Notes:
#   - Redis is preferred if configured; otherwise in-process store is used.
#   - Read paths never raise; write paths validate and may raise ValueError/MemoryError.
#   - Telemetry is best-effort (optional).
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
from typing import Any, Dict, Iterable, Optional, Tuple

from workshops_ai.settings import get_settings

# Redis (sync client via redis_pool helper)
try:  # pragma: no cover
    from workshops_ai.app.services.redis_pool import get_redis_sync  # type: ignore
except Exception:  # pragma: no cover
    get_redis_sync = None  # type: ignore

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

def _get(obj, path: str, default: Any) -> Any:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return default
    return cur

MAX_ORG_KEYS: int = int(_get(_SETTINGS, "memory.max_org_keys", 1000))
MAX_VALUE_BYTES: int = int(_get(_SETTINGS, "memory.max_value_bytes", 262_144))  # ~256 KiB
SOFT_TOTAL_BYTES_PER_ORG: int = int(_get(_SETTINGS, "memory.soft_total_bytes_per_org", 8_388_608))  # ~8 MiB

REDIS_NS: str = str(_get(_SETTINGS, "memory.redis_namespace", "wsai:mem"))
USE_REDIS: bool = bool(_get(_SETTINGS, "redis.url", "")) and bool(_get(_SETTINGS, "memory.use_redis", True))


# ------------------------------------------------------------------------------
# In-proc data model
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
# In-proc store & lock
# ------------------------------------------------------------------------------

_STORE: Dict[str, _OrgStore] = {}
_LOCK = RLock()


# ------------------------------------------------------------------------------
# Redis helpers (sync client) — values are JSON envelopes {v, s, c, u}
# v: value, s: size_bytes, c: created_at, u: updated_at
# ------------------------------------------------------------------------------

def _r_key(org_id: str, tab: Optional[str], key: str) -> str:
    t = tab or "global"
    return f"{REDIS_NS}:{org_id}:{t}:{key}"


def _r_idx(org_id: str) -> str:
    # ZSET LRU index per org
    return f"{REDIS_NS}:{org_id}:idx"


def _r_bytes(org_id: str) -> str:
    # Approx total bytes per org
    return f"{REDIS_NS}:{org_id}:bytes"


def _r_scan_pattern(org_id: str, tab: Optional[str], prefix: Optional[str]) -> str:
    t = tab or "*"
    if prefix:
        return f"{REDIS_NS}:{org_id}:{t}:{prefix}*"
    return f"{REDIS_NS}:{org_id}:{t}:*"


def _approx_size_bytes(value: Any) -> int:
    """Approximate size as UTF-8; prefer JSON, fallback to repr()."""
    try:
        s = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return len(s.encode("utf-8"))
    except Exception:
        try:
            r = repr(value)
            return len(r.encode("utf-8"))
        except Exception:
            return 64


# ------------------------------------------------------------------------------
# Public API (Redis preferred + in-proc fallback)
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
    Set a value with optional TTL. In Redis mode, performs LRU enforcement via ZSET + bytes counter.
    Raises:
        ValueError   – invalid inputs / oversize value
        MemoryError  – if enforcement cannot reclaim enough space (extremely rare)
    """
    now = time.time()
    if not org_id or not isinstance(org_id, str):
        raise ValueError("org_id must be a non-empty string")
    if not key or not isinstance(key, str):
        raise ValueError("key must be a non-empty string")

    size = _approx_size_bytes(value)
    if size > MAX_VALUE_BYTES:
        raise ValueError(f"value too large ({size} bytes > {MAX_VALUE_BYTES})")

    # 1) Redis backend
    if USE_REDIS and get_redis_sync is not None:
        r = None
        try:
            r = get_redis_sync()
        except Exception:
            r = None

        if r is not None:
            rkey = _r_key(org_id, tab, key)
            zkey = _r_idx(org_id)
            bkey = _r_bytes(org_id)
            try:
                prev = r.get(rkey)
                prev_size = 0
                created_at = now
                if prev:
                    try:
                        pobj = json.loads(prev)
                        prev_size = int(pobj.get("s") or 0)
                        created_at = float(pobj.get("c") or now)
                    except Exception:
                        prev_size = 0
                        created_at = now

                payload = {"v": value, "s": size, "c": created_at, "u": now}
                blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

                # Preserve TTL when ttl_sec is None; use KEEPTTL if available, otherwise simulate.
                if ttl_sec and ttl_sec > 0:
                    r.set(rkey, blob, ex=int(ttl_sec))
                else:
                    ttl = r.ttl(rkey)
                    try:
                        r.set(rkey, blob, keepttl=True)  # redis>=6.0
                    except TypeError:
                        r.set(rkey, blob)
                        if ttl and ttl > 0:
                            r.expire(rkey, ttl)

                # Update LRU & bytes
                delta = size - prev_size
                pipe = r.pipeline()
                pipe.zadd(zkey, {rkey: now})
                if delta != 0:
                    pipe.incrby(bkey, delta)
                pipe.execute()

                # Enforce caps (LRU)
                _redis_enforce_caps(r, org_id, max_keys=MAX_ORG_KEYS, soft_bytes=SOFT_TOTAL_BYTES_PER_ORG)

                try:
                    telemetry_log(
                        "memory.set.redis",
                        {"org_id": org_id, "tab": tab, "key": key, "size": size, "ttl_sec": ttl_sec},
                        level="DEBUG",
                        org_id=org_id,
                        source="AI",
                    )
                except Exception:
                    pass
                return  # SUCCESS
            except Exception as e:
                try:
                    telemetry_log("memory.set.redis_error", {"org_id": org_id, "err": str(e)}, level="WARNING", org_id=org_id)
                except Exception:
                    pass
                # fall through to in-proc

    # 2) In-proc fallback
    with _LOCK:
        store = _STORE.setdefault(org_id, _OrgStore())
        bucket = _get_bucket(store, tab)
        _prune_expired(store)

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

        _enforce_caps_inproc(org_id, store)

        try:
            telemetry_log(
                "memory.set.inproc",
                {"org_id": org_id, "tab": tab, "key": key, "size": size, "ttl_sec": ttl_sec},
                level="DEBUG",
                org_id=org_id,
                source="AI",
            )
        except Exception:
            pass


def memory_get(org_id: str, key: str, *, tab: Optional[str] = None, default: Any = None) -> Any:
    """
    Get a value or default. In Redis mode, performs a 'touch' (updates updated_at + ZSET score)
    and preserves TTL.
    """
    now = time.time()
    if not org_id or not key:
        return default

    # Redis
    if USE_REDIS and get_redis_sync is not None:
        r = None
        try:
            r = get_redis_sync()
        except Exception:
            r = None

        if r is not None:
            rkey = _r_key(org_id, tab, key)
            zkey = _r_idx(org_id)
            try:
                raw = r.get(rkey)
                if not raw:
                    return default
                try:
                    obj = json.loads(raw)
                except Exception:
                    return default

                # touch while preserving TTL
                obj["u"] = now
                blob = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
                ttl = r.ttl(rkey)
                try:
                    r.set(rkey, blob, keepttl=True)
                except TypeError:
                    r.set(rkey, blob)
                    if ttl and ttl > 0:
                        r.expire(rkey, ttl)
                r.zadd(zkey, {rkey: now})

                return obj.get("v")
            except Exception:
                return default

    # In-proc
    with _LOCK:
        store = _STORE.get(org_id)
        if not store:
            return default
        bucket = _get_bucket(store, tab, create=False)
        if not bucket:
            return default

        entry = bucket.get(key)
        if not entry:
            return default

        if entry.expires_at and entry.expires_at <= now:
            try:
                store.total_bytes -= max(entry.size_bytes, 0)
                del bucket[key]
            except Exception:
                pass
            return default

        entry.updated_at = now
        return entry.value


def memory_delete(org_id: str, key: str, *, tab: Optional[str] = None) -> bool:
    """Delete a value; returns True if removed."""
    if not org_id or not key:
        return False

    # Redis
    if USE_REDIS and get_redis_sync is not None:
        r = None
        try:
            r = get_redis_sync()
        except Exception:
            r = None

        if r is not None:
            rkey = _r_key(org_id, tab, key)
            zkey = _r_idx(org_id)
            bkey = _r_bytes(org_id)
            try:
                prev = r.get(rkey)
                prev_size = 0
                if prev:
                    try:
                        pobj = json.loads(prev)
                        prev_size = int(pobj.get("s") or 0)
                    except Exception:
                        prev_size = 0
                pipe = r.pipeline()
                pipe.delete(rkey)
                pipe.zrem(zkey, rkey)
                if prev_size:
                    pipe.decrby(bkey, prev_size)
                res = pipe.execute()
                return bool(res and res[0])
            except Exception:
                return False

    # In-proc
    with _LOCK:
        store = _STORE.get(org_id)
        if not store:
            return False
        bucket = _get_bucket(store, tab, create=False)
        if not bucket:
            return False
        entry = bucket.pop(key, None)
        if entry:
            store.total_bytes -= max(entry.size_bytes, 0)
            return True
        return False


def memory_list(org_id: str, *, tab: Optional[str] = None, prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a copy of key → value for the given org (and tab if specified).
    Skips expired entries. In Redis mode, performs SCAN with a pattern.
    """
    now = time.time()
    out: Dict[str, Any] = {}
    if not org_id:
        return out

    # Redis
    if USE_REDIS and get_redis_sync is not None:
        r = None
        try:
            r = get_redis_sync()
        except Exception:
            r = None

        if r is not None:
            pattern = _r_scan_pattern(org_id, tab, prefix)
            try:
                cursor = 0
                while True:
                    cursor, keys = r.scan(cursor=cursor, match=pattern, count=200)
                    if keys:
                        vals = r.mget(keys)
                        for kname, raw in zip(keys, vals):
                            if not raw:
                                continue
                            try:
                                obj = json.loads(raw)
                            except Exception:
                                continue
                            try:
                                orig_key = kname.split(":", 3)[-1]
                            except Exception:
                                orig_key = kname
                            out[orig_key] = obj.get("v")
                    if cursor == 0:
                        break
            except Exception:
                return out
            return out

    # In-proc
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

    # Redis
    if USE_REDIS and get_redis_sync is not None:
        r = None
        try:
            r = get_redis_sync()
        except Exception:
            r = None

        if r is not None:
            try:
                pattern = f"{REDIS_NS}:{org_id}:*"
                cursor = 0
                keys: list[str] = []
                while True:
                    cursor, ks = r.scan(cursor=cursor, match=pattern, count=500)
                    if ks:
                        keys.extend(ks)
                    if cursor == 0:
                        break
                if keys:
                    r.delete(*keys)
            except Exception:
                pass
            try:
                telemetry_log("memory.clear_org.redis", {"org_id": org_id}, level="INFO", org_id=org_id)
            except Exception:
                pass
            return

    # In-proc
    with _LOCK:
        _STORE.pop(org_id, None)
    try:
        telemetry_log("memory.clear_org.inproc", {"org_id": org_id}, level="INFO", org_id=org_id)
    except Exception:
        pass


# ------------------------------------------------------------------------------
# In-proc internals
# ------------------------------------------------------------------------------

def _get_bucket(store: _OrgStore, tab: Optional[str], *, create: bool = True) -> Dict[str, _Entry] | None:
    if not tab:
        return store.global_kv
    if create:
        return store.tabs.setdefault(tab, {})
    return store.tabs.get(tab)


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


def _enforce_caps_inproc(org_id: str, store: _OrgStore) -> None:
    """LRU eviction across global + tab buckets using updated_at."""
    items: list[tuple[float, str, Optional[str]]] = []  # (updated_at, key, tab)
    for k, e in store.global_kv.items():
        items.append((e.updated_at, k, None))
    for tab, b in store.tabs.items():
        for k, e in b.items():
            items.append((e.updated_at, k, tab))

    if not items:
        return

    def _over_key_limit() -> bool:
        total_keys = len(store.global_kv) + sum(len(b) for b in store.tabs.values())
        return total_keys > MAX_ORG_KEYS

    def _over_byte_limit() -> bool:
        return store.total_bytes > SOFT_TOTAL_BYTES_PER_ORG

    if not _over_key_limit() and not _over_byte_limit():
        return

    items.sort(key=lambda t: t[0])  # oldest first

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
        raise MemoryError("Unable to enforce memory limits after eviction.")

    if evicted:
        try:
            telemetry_log(
                "memory.eviction.inproc",
                {"org_id": org_id, "evicted": evicted, "total_bytes": store.total_bytes},
                level="INFO",
                org_id=org_id,
            )
        except Exception:
            pass


# ------------------------------------------------------------------------------
# Redis enforcement helpers (LRU & bytes limit)
# ------------------------------------------------------------------------------

def _redis_enforce_caps(r, org_id: str, *, max_keys: int, soft_bytes: int) -> None:
    """
    Enforce MAX_ORG_KEYS and SOFT_TOTAL_BYTES_PER_ORG in Redis using:
    - ZSET LRU index (updated_at as score)
    - total bytes counter
    """
    zkey = _r_idx(org_id)
    bkey = _r_bytes(org_id)

    try:
        total_keys = int(r.zcard(zkey) or 0)
        total_bytes = int(r.get(bkey) or 0)
    except Exception:
        return

    if total_keys <= max_keys and total_bytes <= soft_bytes:
        return

    evicted = 0
    while total_keys > max_keys or total_bytes > soft_bytes:
        try:
            victims = r.zrange(zkey, 0, 0)
            if not victims:
                break
            victim = victims[0]
            raw = r.get(victim)
            vsize = 0
            if raw:
                try:
                    obj = json.loads(raw)
                    vsize = int(obj.get("s") or 0)
                except Exception:
                    vsize = 0
            pipe = r.pipeline()
            pipe.delete(victim)
            pipe.zrem(zkey, victim)
            if vsize:
                pipe.decrby(bkey, vsize)
            pipe.execute()
            evicted += 1
            total_keys = int(r.zcard(zkey) or 0)
            total_bytes = int(r.get(bkey) or 0)
        except Exception:
            break

    if evicted:
        try:
            telemetry_log(
                "memory.eviction.redis",
                {"org_id": org_id, "evicted": evicted, "total_bytes": total_bytes, "total_keys": total_keys},
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
