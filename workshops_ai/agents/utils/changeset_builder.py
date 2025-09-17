# File: workshops_ai/agents/utils/changeset_builder.py
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Iterable

from workshops_ai.models.changeset import ChangeSet

__all__ = [
    "build_changeset",
    "make_idempotency_key",
]

# --------------------------------------------------------------------------- #
# Idempotency helpers
# --------------------------------------------------------------------------- #

def _hash_actions(actions: Iterable[dict[str, Any]]) -> str:
    """
    Produce a short, stable hash over the ordered list of actions.
    Sorting keys ensures deterministic JSON for equal structures.
    """
    data = json.dumps(list(actions), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def make_idempotency_key(org_id: str, actions: list[dict[str, Any]]) -> str:
    """
    Create a 5-minute bucketed UUIDv5 key to avoid duplicate applies while allowing
    quick re-submissions within a short window during demos.
    """
    bucket = int(time.time() // 300)  # 5-minute window
    seed = f"tazila:{org_id}:{bucket}:{_hash_actions(actions)}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


# --------------------------------------------------------------------------- #
# ChangeSet builder (single or multi-action)
# --------------------------------------------------------------------------- #

def build_changeset(
    org_id: str,
    user_id: str,
    actions: list[dict[str, Any]],
    intent: str = "apply",
) -> ChangeSet:
    """
    Build a ChangeSet with stable idempotency and a consistent audit envelope.
    `intent` is normalized to 'apply' or 'preview'.
    """
    normalized_intent = "preview" if intent == "preview" else "apply"

    return ChangeSet(
        idempotency_key=make_idempotency_key(org_id, actions),
        org_id=org_id,
        intent=normalized_intent,
        audit={
            "actor": "AI",
            "user_id": user_id,
            "source": "WorkshopsAI",  # keep canonical audit source
        },
        actions=actions,
    )
