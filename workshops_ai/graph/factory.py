# File: workshops_ai/graph/factory.py
# -----------------------------------------------------------------------------
# Purpose
# Central LangGraph factory + lightweight DI for all tabs (assets | impacts | threats).
#
# Exposes:
#   - GraphDeps: dependency bundle injected into each graph
#   - build_graph(tab, deps) -> RunnableGraph
#   - ainvoke_with_timeout(graph, inputs, timeout=...)  # async
#   - invoke_with_timeout(graph, inputs, timeout=...)   # sync
#   - make_default_deps() -> GraphDeps                  # default deps for routers
#   - build_checkpointer(settings)                      # checkpointer constructor (alias)
#
# Notes
# - Imports are localized to avoid heavy module loads and circular deps.
# - Dependencies are minimal; graphs must gracefully handle missing optional services.
# - Checkpointer is selected via the centralized builder in graph/checkpointer.py.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import atexit
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Final

# LangGraph type (soft-optional at import-time)
try:  # pragma: no cover
    from langgraph.types import RunnableGraph  # type: ignore
except Exception:  # pragma: no cover
    RunnableGraph = Any  # type: ignore[misc,assignment]

# Optional redis client (ephemeral memory)
try:  # pragma: no cover
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:  # pragma: no cover
    AsyncRedis = None  # type: ignore

from workshops_ai.settings import get_settings
from workshops_ai.graph.checkpointer import make_checkpointer as _central_make_checkpointer

# -----------------------------------------------------------------------------
# Minimal retriever protocol + factory type
# -----------------------------------------------------------------------------

class Retriever(Protocol):
    """Minimal retriever interface used by 'explain' nodes inside graphs."""
    async def asearch(self, query: str, *, k: int = 4) -> list[dict]: ...
    def search(self, query: str, *, k: int = 4) -> list[dict]: ...

MakeRetriever = Callable[[str, str], Optional[Retriever]]

# Redis-backed ephemeral memory accessors (per-org/session)
class MemoryGet(Protocol):
    async def __call__(self, key: str) -> Optional[str]: ...

class MemorySet(Protocol):
    async def __call__(self, key: str, value: str, *, ttl_sec: Optional[int] = None) -> None: ...

# -----------------------------------------------------------------------------
# Dependency bundle injected into each graph
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphDeps:
    """
    Dependency injection container for graphs.
    """
    backend_port: Any
    ws_hub: Any
    make_retriever: MakeRetriever
    checkpointer: Any
    settings: Any
    tab: str
    llm: Any | None = None
    memory_get: MemoryGet | None = None
    memory_set: MemorySet | None = None

# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------

_SUPPORTED_TABS: Final[set[str]] = {"assets", "impacts", "threats"}

def build_graph(tab: str, deps: GraphDeps) -> RunnableGraph:
    """
    Return a compiled LangGraph for the given tab.

    Each graph module must expose:
        create_graph(deps: GraphDeps) -> RunnableGraph
    """
    key = (tab or "").strip().lower()
    if key not in _SUPPORTED_TABS:
        raise ValueError(f"Unsupported tab '{tab}'. Must be one of: assets, impacts, threats.")

    # Localized imports to avoid import-time side effects and cycles.
    if key == "assets":
        from workshops_ai.graph.asset_graph import create_graph  # type: ignore
    elif key == "impacts":
        from workshops_ai.graph.impact_graph import create_graph  # type: ignore
    else:  # key == "threats"
        from workshops_ai.graph.threat_graph import create_graph  # type: ignore

    graph = create_graph(deps)
    return graph

# -----------------------------------------------------------------------------
# Ephemeral memory (Redis), safe no-op fallback if Redis missing
# -----------------------------------------------------------------------------

def make_redis_memory(settings: Any):
    """
    Return (async) memory_get, memory_set functions backed by Redis if available,
    else safe no-ops. Keys are stored under f"{checkpoint_ns}:mem:{key}".
    """
    redis_url = getattr(settings, "redis_url", None) or (getattr(getattr(settings, "redis", None), "url", None))
    ns = getattr(settings, "checkpoint_ns", None) or "lg:ckpt"

    if not redis_url or AsyncRedis is None:
        async def _get(_key: str) -> Optional[str]:
            return None
        async def _set(_key: str, _value: str, *, ttl_sec: Optional[int] = None) -> None:
            return None
        return _get, _set

    client = AsyncRedis.from_url(redis_url, decode_responses=True)

    async def memory_get(key: str) -> Optional[str]:
        val = await client.get(f"{ns}:mem:{key}")
        return val

    async def memory_set(key: str, value: str, *, ttl_sec: Optional[int] = None) -> None:
        k = f"{ns}:mem:{key}"
        if ttl_sec and ttl_sec > 0:
            await client.setex(k, ttl_sec, value)
        else:
            await client.set(k, value)

    # Close client on interpreter exit (best-effort)
    def _finalizer():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(client.close())
            else:
                loop.run_until_complete(client.close())
        except Exception:
            pass
    atexit.register(_finalizer)

    return memory_get, memory_set

# -----------------------------------------------------------------------------
# Default dependencies factory
# -----------------------------------------------------------------------------

def make_default_deps(
    tab: str | None = None,
    *,
    settings: Any | None = None,
    backend_port: Any | None = None,
    ws_hub: Any | None = None,
    make_retriever: Optional[MakeRetriever] = None,
    llm: Any | None = None,
    memory_get: MemoryGet | None = None,
    memory_set: MemorySet | None = None,
) -> GraphDeps:
    """
    Build a minimal, safe set of dependencies for graph compilation/execution.

    Provided:
      - settings           : loaded via get_settings() if not provided
      - checkpointer       : Redis/SQLite/Memory per settings
      - backend_port       : CSV/API backend if available (can be None; graphs should degrade)
      - ws_hub             : None (graphs shouldn't rely on it hard)
      - make_retriever     : RAG retriever factory (optional; graphs degrade to LLM fallback)
      - llm                : None by default
      - memory_get/set     : Redis-based ephemeral helpers when redis_url is set; else no-op
    """
    s = settings or get_settings()

    # Checkpointer from the centralized builder (robust across versions)
    saver = _central_make_checkpointer(settings=s)

    # Backend (allow external injection; otherwise try CSV provider)
    if backend_port is None:
        try:
            from workshops_ai.backend.provider import get_backend_port  # type: ignore
            backend_port = get_backend_port()
        except Exception:
            try:
                from workshops_ai.backend.csv_backend import get_backend_port  # type: ignore
                backend_port = get_backend_port()
            except Exception:
                backend_port = None

    # WS hub: optional; graphs should not hard-depend on it
    ws = ws_hub

    # Retriever factory (optional)
    if make_retriever is None:
        def _none_retriever(_org_id: str, _tab: str) -> Optional[Retriever]:
            return None
        make_retriever = _none_retriever

    # Ephemeral memory (Redis if configured; otherwise no-op)
    if memory_get is None or memory_set is None:
        mget, mset = make_redis_memory(s)
        memory_get = memory_get or mget
        memory_set = memory_set or mset

    return GraphDeps(
        backend_port=backend_port,
        ws_hub=ws,
        make_retriever=make_retriever,
        checkpointer=saver,
        settings=s,
        tab=(tab or "").lower() if tab else "",
        llm=llm,
        memory_get=memory_get,
        memory_set=memory_set,
    )

# -----------------------------------------------------------------------------
# Invoke helpers with timeout (support optional 'config'), with robust async→sync fallback
# -----------------------------------------------------------------------------

async def _invoke_async(graph: RunnableGraph, inputs: Any, config: Optional[dict]) -> Any:
    """
    Always prefer async. Do NOT fallback to sync for graphs with async-only nodes.
    """
    if hasattr(graph, "ainvoke"):
        if config is not None:
            return await graph.ainvoke(inputs, config=config)  # type: ignore[arg-type]
        return await graph.ainvoke(inputs)  # type: ignore[misc]

    # Rare legacy case: graph has no ainvoke at all -> use sync in a thread
    loop = asyncio.get_running_loop()
    if config is not None and hasattr(graph, "invoke"):
        return await loop.run_in_executor(None, lambda: graph.invoke(inputs, config=config))  # type: ignore[misc]
    if hasattr(graph, "invoke"):
        return await loop.run_in_executor(None, lambda: graph.invoke(inputs))  # type: ignore[misc]
    return await loop.run_in_executor(None, lambda: graph(inputs))  # type: ignore[operator]

async def ainvoke_with_timeout(graph: RunnableGraph, inputs: Any, *, timeout: int, config: Optional[dict] = None) -> Any:
    return await asyncio.wait_for(_invoke_async(graph, inputs, config=config), timeout=timeout)

def invoke_with_timeout(graph: RunnableGraph, inputs: Any, *, timeout: int, config: Optional[dict] = None) -> Any:
    return asyncio.run(ainvoke_with_timeout(graph, inputs, timeout=timeout, config=config))

# Alias for compatibility with places that import build_checkpointer from factory
def build_checkpointer(*, settings: Any | None = None) -> Any:
    return _central_make_checkpointer(settings=settings)

__all__ = [
    "GraphDeps",
    "build_graph",
    "ainvoke_with_timeout",
    "invoke_with_timeout",
    "make_redis_memory",
    "make_default_deps",
    "build_checkpointer",
]
