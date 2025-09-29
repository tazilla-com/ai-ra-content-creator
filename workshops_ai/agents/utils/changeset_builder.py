# File: workshops_ai/agents/utils/changeset_builder.py
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel

from workshops_ai.models.changeset import ChangeSet, Audit

__all__ = [
    "build_changeset",
    "make_idempotency_key",
    "build_audit_envelope",
]

# --------------------------------------------------------------------------- #
# Normalization helpers for stable hashing / serialization
# --------------------------------------------------------------------------- #

def _to_plain(obj: Any) -> Any:
    """
    Convert input (BaseModel/dict/sequence/primitives) to a JSON-serializable structure.
    - Pydantic model -> model_dump(mode="json", by_alias=False)
    - datetime -> ISO string
    - set/tuple -> list
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json", by_alias=False, serialize_as_any=True)
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _hash_actions(actions: Iterable[Any]) -> str:
    """
    Compute a short, stable hash over an *ordered* list of actions.
    JSON keys are sorted to produce the same fingerprint for equivalent structures.
    """
    plain = [_to_plain(a) for a in actions]
    data = json.dumps(plain, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Idempotency helpers
# --------------------------------------------------------------------------- #

def make_idempotency_key(org_id: str, actions: Sequence[Any], *, window_seconds: int = 300) -> str:
    """
    Create a UUIDv5 key with time-bucketing (default 5 minutes).
    Allows safe re-submits within a short window without duplicates.
    """
    bucket = int(time.time() // window_seconds)
    seed = f"https://workshops.ai/idem/{org_id}:{bucket}:{_hash_actions(actions)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


# --------------------------------------------------------------------------- #
# Audit envelope
# --------------------------------------------------------------------------- #

def build_audit_envelope(
    *,
    user_id: str,
    org_id: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    batch_id: str | None = None,
    source: str = "AI",
    extra_tags: list[str] | None = None,
) -> Audit:
    """
    Unified audit envelope (snake_case everywhere).
    - Always includes 'AI' in tags
    - If batch_id is present, adds 'batch-<batch_id>'
    - extra_tags are de-duplicated while preserving order
    """
    tags: list[str] = ["AI"]
    if batch_id:
        tags.append(f"batch-{batch_id}")
    if extra_tags:
        seen: set[str] = set()
        merged: list[str] = []
        for t in tags + extra_tags:
            if t not in seen:
                merged.append(t)
                seen.add(t)
        tags = merged

    return Audit(
        user_id=user_id,
        ts=datetime.now(timezone.utc),
        source=source,
        tags=tags,
        org_id=org_id,
        session_id=session_id,
        request_id=request_id,
        batch_id=batch_id,
    )


# --------------------------------------------------------------------------- #
# ChangeSet builder (single or multi-action)
# --------------------------------------------------------------------------- #

def build_changeset(
    org_id: str,
    user_id: str,
    actions: Sequence[Any],
    *,
    intent: str = "apply",
    session_id: str | None = None,
    request_id: str | None = None,
    batch_id: str | None = None,
    extra_tags: list[str] | None = None,
) -> ChangeSet:
    """
    Build a ChangeSet with stable idempotency and a unified audit envelope.
    `actions` may be dicts or Pydantic models (with 'op' as the discriminator).
    """
    normalized_intent = "preview" if intent == "preview" else "apply"

    # Ensure the batch tag is visible in audit tags via extra_tags as well.
    # The audit builder also adds it, and will de-duplicate tags.
    if batch_id:
        extra_tags = list(extra_tags or [])
        tag = f"batch-{batch_id}"
        if tag not in extra_tags:
            extra_tags.append(tag)

    idemp_key = make_idempotency_key(org_id, actions)

    audit = build_audit_envelope(
        user_id=user_id,
        org_id=org_id,
        session_id=session_id,
        request_id=request_id,
        batch_id=batch_id,
        extra_tags=extra_tags,
    )

    return ChangeSet(
        idempotency_key=idemp_key,
        org_id=org_id,
        intent=normalized_intent,
        audit=audit,
        actions=list(actions),
    )
