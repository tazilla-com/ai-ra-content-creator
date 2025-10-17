# workshops_ai/graph/conversation_memory.py
# -----------------------------------------------------------------------------
# rolling conversation window over Redis-like KV; async-first
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr


Role = Literal["user", "assistant", "tool", "system"]


class ChatTurn(BaseModel):
    """
    Canonical schema for one chat turn stored in memory.
    - No extra fields allowed.
    - Strings are trimmed.
    - Timestamps are UTC ISO strings on dump.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    role: Role
    content: StrictStr = Field(min_length=1)
    # Tool call metadata (optional; kept compact for memory)
    tool_name: Optional[StrictStr] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_output: Optional[StrictStr] = None

    ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class MemoryConfig:
    """
    Configuration for rolling memory.
    """
    key_prefix: str = "chat:history"
    max_messages: StrictInt = 8            # default rolling window size
    ttl_seconds: Optional[StrictInt] = 7200  # 2h; None disables TTL
    namespace: StrictStr = "default"       # logical partition (e.g., org env)


class _InMemoryKV:
    """
    Fallback KV if no external memory is provided.
    Not for production; survives only within process.
    """
    __slots__ = ("_store", "_exp")

    def __init__(self) -> None:
        self._store: dict[str, str] = {}
        self._exp: dict[str, float] = {}

    async def get(self, key: str) -> Optional[str]:
        now = time.time()
        exp = self._exp.get(key)
        if exp is not None and now > exp:
            self._store.pop(key, None)
            self._exp.pop(key, None)
            return None
        return self._store.get(key)

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        self._store[key] = value
        if ttl is not None:
            self._exp[key] = time.time() + ttl
        else:
            self._exp.pop(key, None)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)
        self._exp.pop(key, None)


class _KVAdapter:
    """
    Uniform async adapter around user-provided memory callables or objects.
    Supports:
      - async callables: await func(key), await func(key, value, ttl)
      - sync callables: func(key), func(key, value, ttl)  (wrapped with to_thread)
      - object with 'get/set/delete' async or sync methods (e.g., Redis client)
    """
    def __init__(
        self,
        get: Optional[Callable[..., Any]] = None,
        set_: Optional[Callable[..., Any]] = None,
        delete: Optional[Callable[..., Any]] = None,
        obj: Optional[Any] = None,
    ) -> None:
        if not any([get, set_, delete, obj]):
            # No external memory supplied → use in-process fallback
            self._backend = _InMemoryKV()
            self._mode = "inmem"
            return

        self._backend = obj
        self._get = get
        self._set = set_
        self._del = delete
        self._mode = "callables" if any([get, set_, delete]) else "object"

    async def _to_async(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            result = fn(*args, **kwargs)
        except TypeError:
            # Some 'set' functions might not accept ttl -> try without it.
            if "ttl" in kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k != "ttl"}
                result = fn(*args, **kwargs)
            else:
                raise
        if asyncio.iscoroutine(result):
            return await result
        return await asyncio.to_thread(lambda: result)

    async def get(self, key: str) -> Optional[str]:
        if self._mode == "inmem":
            return await self._backend.get(key)  # type: ignore[no-any-return]
        if self._mode == "callables":
            if self._get is None:
                return None
            return await self._to_async(self._get, key)  # type: ignore[arg-type]
        # object mode
        getter = getattr(self._backend, "get", None)
        if getter is None:
            return None
        result = getter(key)
        if asyncio.iscoroutine(result):
            return await result
        return await asyncio.to_thread(lambda: result)  # type: ignore[no-any-return]

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if self._mode == "inmem":
            await self._backend.set(key, value, ttl)  # type: ignore[misc]
            return
        if self._mode == "callables":
            if self._set is None:
                return
            await self._to_async(self._set, key, value, ttl=ttl)  # type: ignore[arg-type]
            return
        # object mode
        setter = getattr(self._backend, "set", None)
        if setter is None:
            return
        result = setter(key, value, ttl=ttl)
        if asyncio.iscoroutine(result):
            await result
        else:
            await asyncio.to_thread(lambda: result)

    async def delete(self, key: str) -> None:
        if self._mode == "inmem":
            await self._backend.delete(key)  # type: ignore[misc]
            return
        if self._mode == "callables":
            if self._del is None:
                return
            await self._to_async(self._del, key)  # type: ignore[arg-type]
            return
        # object mode
        deleter = getattr(self._backend, "delete", None)
        if deleter is None:
            return
        result = deleter(key)
        if asyncio.iscoroutine(result):
            await result
        else:
            await asyncio.to_thread(lambda: result)


class RollingWindowMemory:
    """
    Async-first rolling conversation memory.
    - Stores a compact JSON array of ChatTurn dicts under a single key.
    - Trims to 'max_messages' (keep most recent).
    - Applies TTL per write (best-effort; ignored if backend doesn't support it).
    - Safe to use even if no KV is wired (falls back to in-process).
    """

    def __init__(
        self,
        memory_get: Optional[Callable[..., Any]] = None,
        memory_set: Optional[Callable[..., Any]] = None,
        memory_delete: Optional[Callable[..., Any]] = None,
        memory_obj: Optional[Any] = None,
        config: Optional[MemoryConfig] = None,
    ) -> None:
        self.cfg = config or MemoryConfig()
        self.kv = _KVAdapter(get=memory_get, set_=memory_set, delete=memory_delete, obj=memory_obj)

    # ------------------------------- Keying -------------------------------- #

    def make_key(
        self,
        *,
        org_id: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """
        Build a deterministic key: <prefix>:<ns>:<org>:<user or - >:<thread or - >
        """
        ns = (namespace or self.cfg.namespace).strip() or "default"
        org = (org_id or "-").strip() or "-"
        user = (user_id or "-").strip() or "-"
        thread = (thread_id or "-").strip() or "-"
        return f"{self.cfg.key_prefix}:{ns}:{org}:{user}:{thread}"

    # ------------------------------ IO helpers ----------------------------- #

    async def _load_raw(self, key: str) -> list[dict[str, Any]]:
        raw = await self.kv.get(key)
        if not raw:
            return []
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                # Filter only dict-like items (forward compatibility)
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            # Corrupted payload → drop
            return []
        return []

    async def _save_raw(self, key: str, items: list[dict[str, Any]]) -> None:
        payload = json.dumps(items, separators=(",", ":"), ensure_ascii=False)
        await self.kv.set(key, payload, ttl=self.cfg.ttl_seconds)

    # ------------------------------ Public API ----------------------------- #

    async def read(
        self,
        *,
        org_id: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> list[ChatTurn]:
        """
        Load and parse the current rolling window for the given scope.
        """
        key = self.make_key(org_id=org_id, user_id=user_id, thread_id=thread_id, namespace=namespace)
        items = await self._load_raw(key)
        turns: list[ChatTurn] = []
        for it in items[-self.cfg.max_messages :]:
            try:
                turns.append(ChatTurn.model_validate(it))
            except Exception:
                # Skip malformed entries silently
                continue
        return turns

    async def append(
        self,
        *,
        org_id: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
        new_messages: Sequence[ChatTurn | dict[str, Any] | tuple[Role, str]] = (),
        trim_to: Optional[int] = None,
    ) -> list[ChatTurn]:
        """
        Append one or more messages and persist trimmed window.
        Returns the final list of ChatTurns actually stored.
        """
        key = self.make_key(org_id=org_id, user_id=user_id, thread_id=thread_id, namespace=namespace)
        existing = await self.read(org_id=org_id, user_id=user_id, thread_id=thread_id, namespace=namespace)

        # Normalize incoming
        normalized: list[ChatTurn] = []
        for m in new_messages:
            if isinstance(m, ChatTurn):
                normalized.append(m)
            elif isinstance(m, tuple) and len(m) == 2:
                r, c = m
                normalized.append(ChatTurn(role=r, content=c))
            elif isinstance(m, dict):
                try:
                    normalized.append(ChatTurn.model_validate(m))
                except Exception:
                    # skip invalid inputs
                    continue

        merged = existing + normalized
        limit = trim_to if trim_to is not None else self.cfg.max_messages
        # keep last 'limit'
        merged = merged[-limit:]

        # Persist
        await self._save_raw(key, [t.model_dump(mode="json") for t in merged])
        return merged

    async def replace(
        self,
        *,
        org_id: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
        messages: Sequence[ChatTurn | dict[str, Any] | tuple[Role, str]] = (),
    ) -> list[ChatTurn]:
        """
        Replace the whole window with provided messages (trimmed).
        """
        key = self.make_key(org_id=org_id, user_id=user_id, thread_id=thread_id, namespace=namespace)
        normalized: list[ChatTurn] = []
        for m in messages:
            if isinstance(m, ChatTurn):
                normalized.append(m)
            elif isinstance(m, tuple) and len(m) == 2:
                r, c = m
                normalized.append(ChatTurn(role=r, content=c))
            elif isinstance(m, dict):
                try:
                    normalized.append(ChatTurn.model_validate(m))
                except Exception:
                    continue
        trimmed = normalized[-self.cfg.max_messages :]
        await self._save_raw(key, [t.model_dump(mode="json") for t in trimmed])
        return trimmed

    async def clear(
        self,
        *,
        org_id: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        key = self.make_key(org_id=org_id, user_id=user_id, thread_id=thread_id, namespace=namespace)
        await self.kv.delete(key)

    # ------------------------- LLM interop helpers ------------------------- #

    @staticmethod
    def to_llm_messages(turns: Sequence[ChatTurn], *, drop_tool_output: bool = True) -> list[dict[str, str]]:
        """
        Convert ChatTurn list to vanilla chat messages for LLMs.
        - Keeps (role, content) only; tool output can be dropped for brevity.
        - Keeps chronological order (old → new).
        """
        msgs: list[dict[str, str]] = []
        for t in turns:
            if t.role == "tool" and drop_tool_output:
                # Keep just a short note to preserve context
                content = t.tool_output or t.content
                if content:
                    msgs.append({"role": "assistant", "content": f"(tool) {content}"})
                continue
            msgs.append({"role": t.role, "content": t.content})
        return msgs
