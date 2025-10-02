# workshops_ai/graph/factory.py
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
#   - make_langgraph_config(...) -> dict                # builds required 'configurable' keys
#   - build_checkpointer(settings=None)                 # compatibility alias
#
# Notes
# - Imports are localized to avoid heavy module loads and circular deps.
# - Dependencies are minimal; graphs must gracefully handle missing optional services.
# - Checkpointer is selected via centralized builder in graph/checkpointer.py.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import atexit
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Final, Awaitable

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

# Submitter/outbox (online submit preferred, fallback to OUTBOX)
from workshops_ai.app.services.outbox_writer import (
    submit_or_fallback,
    SubmitOrFallbackResult,
)

# Changeset model for type hints in submit callable
from workshops_ai.models.changeset import ChangeSet


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


# Submit callable protocol (graphs call this to persist mutations)
class SubmitChangeSet(Protocol):
    async def __call__(
        self,
        changeset: ChangeSet,
        *,
        require_success: bool = False
    ) -> SubmitOrFallbackResult: ...


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
    # LLM used by planner/LLM nodes (can be None; graphs should degrade gracefully)
    llm: Any | None = None
    # Ephemeral memory helpers (Redis if configured, no-ops otherwise)
    memory_get: MemoryGet | None = None
    memory_set: MemorySet | None = None
    # Mutations: submit changeset (online preferred, fallback to OUTBOX)
    submit_changeset: SubmitChangeSet | None = None
    # NEW: expose checkpoint namespace so routers/tests can pass it to LangGraph
    checkpoint_ns: str | None = None


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
        raise ValueError(
            "Unsupported tab '{tab}'. Must be one of: assets, impacts, threats.".format(tab=tab)
        )

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
    redis_url = getattr(settings, "redis_url", None) or getattr(
        getattr(settings, "redis", None), "url", None
    )
    ns = getattr(settings, "checkpoint_ns", None) or "lg:ckpt"

    if not redis_url or AsyncRedis is None:
        async def _get(_key: str) -> Optional[str]:
            return None

        async def _set(_key: str, _value: str, *, ttl_sec: Optional[int] = None) -> None:
            return None

        return _get, _set

    client = AsyncRedis.from_url(redis_url, decode_responses=True)

    async def memory_get(key: str) -> Optional[str]:
        return await client.get(f"{ns}:mem:{key}")

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
# Helper: build submit_changeset callable
# -----------------------------------------------------------------------------

def _make_submit_callable(settings: Any) -> SubmitChangeSet:
    """
    Returns a callable that tries to submit to the primary backend
    (if an API client is available), otherwise falls back to writing
    into the OUTBOX directory via submit_or_fallback().
    """
    # Try to locate an online submitter client (optional)
    submitter: Any = None
    # Prefer a dedicated API client if present (soft dependency)
    try:
        from workshops_ai.app.services.tazila_client import get_client  # type: ignore
        submitter = get_client()  # expected to implement .submit_changeset(...)
    except Exception:
        # No online client available; outbox fallback will be used.
        submitter = None

    async def _submit(
        changeset: ChangeSet,
        *,
        require_success: bool = False,
    ) -> SubmitOrFallbackResult:
        # submit_or_fallback handles both async and sync submitters transparently
        return await submit_or_fallback(
            changeset,
            submitter=submitter,
            require_success=require_success,
        )

    return _submit


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
    submit_changeset: SubmitChangeSet | None = None,
) -> GraphDeps:
    """
    Build a minimal, safe set of dependencies for graph compilation/execution.

    Provided:
      - settings           : loaded via get_settings() if not provided
      - checkpointer       : Redis/SQLite/Memory per settings (built internally)
      - backend_port       : CSV/API backend if available (can be None; graphs should degrade)
      - ws_hub             : optional; graphs should not hard-depend on it
      - make_retriever     : RAG retriever factory (optional; graphs degrade to LLM fallback)
      - llm                : optional LLM instance
      - memory_get/set     : Redis-based ephemeral helpers when redis_url is set; else no-op
      - submit_changeset   : async callable performing online submit or OUTBOX fallback
    """
    s = settings or get_settings()

    # Checkpointer from the centralized builder (robust across versions)
    saver = _central_make_checkpointer(settings=s)

    # Backend (allow external injection; otherwise try provider → CSV fallback)
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

    # Submit callable (online preferred, fallback to OUTBOX)
    if submit_changeset is None:
        submit_changeset = _make_submit_callable(s)

    # Surface checkpoint namespace (from flat settings first; allow nested fallback)
    ns = getattr(s, "checkpoint_ns", None)
    if not ns:
        # If your AppSettings nests checkpointer config, try to read it here
        ns = getattr(getattr(s, "checkpointer", object()), "namespace", None)

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
        submit_changeset=submit_changeset,
        checkpoint_ns=ns,
    )


# -----------------------------------------------------------------------------
# Invoke helpers with timeout (support optional 'config'), robust async→sync fallback
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
        return await loop.run_in_executor(
            None, lambda: graph.invoke(inputs, config=config)  # type: ignore[misc]
        )
    if hasattr(graph, "invoke"):
        return await loop.run_in_executor(None, lambda: graph.invoke(inputs))  # type: ignore[misc]
    return await loop.run_in_executor(None, lambda: graph(inputs))  # type: ignore[operator]


async def ainvoke_with_timeout(
    graph: RunnableGraph,
    inputs: Any,
    *,
    timeout: int,
    config: Optional[dict] = None
) -> Any:
    return await asyncio.wait_for(_invoke_async(graph, inputs, config=config), timeout=timeout)


def invoke_with_timeout(
    graph: RunnableGraph,
    inputs: Any,
    *,
    timeout: int,
    config: Optional[dict] = None
) -> Any:
    return asyncio.run(ainvoke_with_timeout(graph, inputs, timeout=timeout, config=config))


# -----------------------------------------------------------------------------
# Utility: build the LangGraph 'configurable' dict (thread_id/namespace)
# -----------------------------------------------------------------------------

def make_langgraph_config(
    *,
    org_id: str,
    tab: str,
    user_id: str | None,
    deps: GraphDeps
) -> dict:
    """
    Compose a LangGraph config with required 'configurable' keys for the checkpointer.
    - thread_id      : required by LangGraph checkpointers
    - checkpoint_ns  : optional but recommended (taken from deps.checkpoint_ns)
    """
    cfg: dict[str, Any] = {"configurable": {}}
    user = (user_id or "anon").strip() or "anon"
    cfg["configurable"]["thread_id"] = f"{org_id}:{tab}:{user}"
    if deps.checkpoint_ns:
        cfg["configurable"]["checkpoint_ns"] = deps.checkpoint_ns
    return cfg


# -----------------------------------------------------------------------------
# Compatibility alias (accepts optional argument)
# -----------------------------------------------------------------------------

def build_checkpointer(settings: Any | None = None) -> Any:
    """
    Compatibility wrapper.
    - If called as build_checkpointer() -> uses get_settings() internally.
    - If called as build_checkpointer(settings) or build_checkpointer(settings=...)
      -> forwards to the centralized builder.
    """
    if settings is None:
        settings = get_settings()
    return _central_make_checkpointer(settings=settings)


__all__ = [
    "GraphDeps",
    "Retriever",
    "MakeRetriever",
    "SubmitChangeSet",
    "build_graph",
    "ainvoke_with_timeout",
    "invoke_with_timeout",
    "make_redis_memory",
    "make_default_deps",
    "make_langgraph_config",
    "build_checkpointer",
]
