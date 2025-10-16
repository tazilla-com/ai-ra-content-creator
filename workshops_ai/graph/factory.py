# File: workshops_ai/graph/factory.py
# =============================================================================
# Purpose: Central LangGraph factory + lightweight DI for all tabs (assets | impacts | threats).
# - Single source of truth for building GraphDeps (used by manager + routers)
# - Injects a centralized llm_client (forced to LangChain ChatOpenAI) and explicit llm_temperature
# - Preserves checkpointer, Redis-backed ephemeral memory
# - Exposes a unified submit_changeset wrapper (online submit → OUTBOX fallback) used by all graphs
# - Exposes helpers to build per-turn LangGraph config (thread_id/checkpoint_ns/lang)
# - Ensures RAG retriever is wired consistently via the retriever factory
#
# Language stability:
# - `_normalize_lang_iso(...)` normalizes Accept-Language-like values to ISO primaries.
# - `build_graph_config(req, deps=None)` produces a LangGraph `configurable` with stable `lang`
#   (and legacy `user_lang`) so agents keep replying in the user’s language (no drift).
#
# Thread identity (single source of truth):
# - `_derive_thread_id(req, ...)` computes a stable thread id:
#     1) prefer req.session_id (sticky across reconnects),
#     2) else uuid5(org, conn_id|user_id), always scoped by tab,
#   and GraphDeps.thread_id_for(...) delegates to it.
# =============================================================================

from __future__ import annotations

import logging
import asyncio
import atexit
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Optional, Protocol, Final

# LangGraph type (soft-optional at import-time to avoid hard dependency in tests)
try:  # pragma: no cover
    from langgraph.types import RunnableGraph  # type: ignore
except Exception:  # pragma: no cover
    RunnableGraph = Any  # type: ignore[misc,assignment]

# Optional redis client (for ephemeral memory)
try:  # pragma: no cover
    from redis.asyncio import Redis as AsyncRedis  # type: ignore
except Exception:  # pragma: no cover
    AsyncRedis = None  # type: ignore

from workshops_ai.settings import get_settings, AppSettings
from workshops_ai.graph.checkpointer import make_checkpointer as _central_make_checkpointer

# Central LLM client + RAG retriever
from workshops_ai.llm.client import make_llm_client_from_settings  # noqa: F401
from workshops_ai.rag.retriever import make_retriever as _rag_make_retriever

# Enforce LangChain OpenAI chat model
from langchain_openai import ChatOpenAI  # type: ignore

# Unified submit wrapper (online → OUTBOX fallback)
from workshops_ai.app.services.outbox_writer import (
    submit_or_fallback,
    SubmitOrFallbackResult,
)

# Changeset model (type hints for submit callable)
from workshops_ai.models.changeset import ChangeSet

# Optional: used by GraphDeps.thread_id_for(...) helper
from workshops_ai.schemas.ui_contract import UIRequest

logger = logging.getLogger(__name__)

# =============================================================================
# Minimal retriever protocol + factory type
# =============================================================================

class Retriever(Protocol):
    """Minimal retriever interface used by 'explain' nodes inside graphs."""
    async def asearch(self, query: str, *, k: int = 4) -> list[dict]: ...
    def search(self, query: str, *, k: int = 4) -> list[dict]: ...


MakeRetriever = Callable[[str, str], Optional[Retriever]]

# =============================================================================
# LRU-cached builders (LLM client, Checkpointer)
# =============================================================================

@lru_cache(maxsize=8)
def _get_llm_client_cached(
    *,
    model: str,
    temperature: float,
    http_timeout_sec: Optional[int],
    max_retries: Optional[int],
    api_key: str,
    base_url: Optional[str],
) -> Any:
    """
    Build a LangChain ChatOpenAI client and cache it by its immutable config.
    This avoids re-instantiation per request/graph build.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": float(temperature),
    }
    if base_url:
        kwargs["base_url"] = base_url
    if http_timeout_sec is not None:
        kwargs["timeout"] = int(http_timeout_sec)
    if max_retries is not None:
        kwargs["max_retries"] = int(max_retries)

    return ChatOpenAI(**kwargs)  # type: ignore[call-arg]


@lru_cache(maxsize=8)
def _get_checkpointer_cached(
    *,
    kind: str | None,
    redis_url: str | None,
    sqlite_path: str | None,
    namespace: str | None,
    ttl_sec: int | None,
) -> Any:
    """
    Build the LangGraph checkpointer and cache it by its effective backend config.
    Supports two signatures of the central builder:
      - make_checkpointer(settings=AppSettings)
      - make_checkpointer(kind=..., redis_url=..., sqlite_path=..., namespace=..., ttl_sec=...)
    """
    try:
        return _central_make_checkpointer(
            kind=kind,
            redis_url=redis_url,
            sqlite_path=sqlite_path,
            namespace=namespace,
            ttl_sec=ttl_sec,
        )
    except TypeError:
        # Fallback if the project's builder expects settings only.
        return _central_make_checkpointer(settings=get_settings())

# =============================================================================
# Redis-backed ephemeral memory accessors (per-org/session)
# =============================================================================
# NOTE: Graph memory callers in our codebase are of two kinds:
#   1) Rolling window KV adapter → calls get(key) / set(key, value, ttl=...)
#   2) Manager/routers rich access → call with kwargs (org_id=..., thread_id=..., key=..., ...)
# We therefore provide an adapter that supports BOTH forms.

class MemoryGet(Protocol):
    async def __call__(self, key: str) -> Optional[str]: ...
    # Rich form supported via kwargs:
    #   memory_get(org_id=..., user_id=..., thread_id=..., namespace=..., key=...)


class MemorySet(Protocol):
    async def __call__(self, key: str, value: str, *, ttl_sec: Optional[int] = None) -> None: ...
    # Rich form supported via kwargs:
    #   memory_set(org_id=..., user_id=..., thread_id=..., namespace=..., key=..., value=..., ttl_sec=...)


class MemoryDelete(Protocol):
    async def __call__(self, key: str) -> None: ...
    # Rich form supported via kwargs:
    #   memory_delete(org_id=..., user_id=..., thread_id=..., namespace=..., key=...)


# Submit callable protocol (graphs call this to persist mutations)
class SubmitChangeSet(Protocol):
    async def __call__(
        self,
        changeset: ChangeSet,
        *,
        require_success: bool = False
    ) -> SubmitOrFallbackResult: ...


# =============================================================================
# Thread id derivation (single source of truth)
# =============================================================================

def _derive_thread_id(req: UIRequest, fallback_tab: str | None = None) -> str:
    """
    Build a stable thread_id for LangGraph/checkpointers and per-thread memory.

    Priority:
      1) If `req.session_id` is present → use it (sticky across reconnects).
      2) Else compute uuid5(namespace=org_id, name=(conn_id | user_id)).
    Always scope by tab to avoid cross-talk between tabs.

    Shape:
      "<org_id>:<tab>:<session_id | uuid5_hex>"
    """
    tab = (getattr(req, "active_tab", None) or fallback_tab or "assets").strip() or "assets"
    org = (getattr(req, "org_id", None) or "org").strip() or "org"

    # 1) Session-based thread id (preferred)
    sid = getattr(req, "session_id", None)
    if isinstance(sid, str) and sid.strip():
        return f"{org}:{tab}:{sid.strip()}"

    # 2) Deterministic uuid5(org, seed) where seed = conn_id | user_id | "anon"
    seed = getattr(req, "conn_id", None) or getattr(req, "user_id", None) or "anon"
    ns = uuid.uuid5(uuid.NAMESPACE_URL, f"org:{org}")
    th = uuid.uuid5(ns, str(seed))
    return f"{org}:{tab}:{th.hex}"


# =============================================================================
# Dependency bundle injected into each graph
# =============================================================================

@dataclass(frozen=True)
class GraphDeps:
    """
    Dependency injection container for graphs.

    LLM DI:
    - Prefer `llm_client` (primary handle for tool-choice / chat completion),
      forced to LangChain ChatOpenAI by the factory.
    - `llm_temperature` is provided separately to avoid coupling to any client type.
    - `llm` is kept for backward compatibility with older agent nodes (avoid in new code).

    IMPORTANT: Use `submit_changeset` for ALL mutations across tabs.
               It is the unified wrapper that performs online submit and falls back to OUTBOX.
    """
    backend_port: Any
    ws_hub: Any
    make_retriever: MakeRetriever
    checkpointer: Any
    settings: AppSettings
    tab: str

    # --- LLM interfaces ---
    llm_client: Any | None = None          # preferred (LangChain ChatOpenAI)
    llm_temperature: float | None = None   # explicit scalar
    llm: Any | None = None                 # legacy (avoid in new code)

    # Ephemeral memory helpers (Redis if configured, no-ops otherwise)
    memory_get: MemoryGet | None = None
    memory_set: MemorySet | None = None
    memory_delete: MemoryDelete | None = None

    # Mutations: unified submit wrapper (online preferred, fallback to OUTBOX)
    submit_changeset: SubmitChangeSet | None = None

    # Back-compat alias (legacy routers/tools). Prefer submit_changeset everywhere.
    submit_or_outbox: SubmitChangeSet | None = None

    # Expose checkpoint namespace so routers/tests can pass it to LangGraph
    checkpoint_ns: str | None = None

    # Optional request-scoped policy snapshot (e.g., TTLs for memory)
    effective_policy: Any | None = None

    # Stable per-thread key builder (used by graphs for memory key composition)
    def thread_id_for(self, req: Optional[UIRequest] = None) -> str:
        """
        Build a stable thread_id using the single source of truth:
          session_id → uuid5(org, conn_id|user_id), always scoped by tab.
        """
        if req is not None:
            try:
                return _derive_thread_id(req, fallback_tab=self.tab)
            except Exception:
                pass
        # Last-resort fallback (very rare)
        tab = (self.tab or "assets").strip() or "assets"
        return f"org:{tab}:fallback"


# =============================================================================
# Graph construction (tab-aware dispatcher)
# =============================================================================

_SUPPORTED_TABS: Final[set[str]] = {"assets", "impacts", "threats"}


def build_graph(tab: str, deps: GraphDeps) -> RunnableGraph:
    """
    Return a compiled LangGraph for the given tab.

    Each graph module must expose:
        create_graph(deps: GraphDeps) -> RunnableGraph
    """
    key = (tab or "").strip().lower()
    if key not in _SUPPORTED_TABS:
        raise ValueError(f"Unsupported tab '{key}'. Must be one of: assets, impacts, threats.")

    # Lazy imports avoid import-time side effects and circular deps.
    if key == "assets":
        from workshops_ai.graph.asset_graph import create_graph  # type: ignore
    elif key == "impacts":
        from workshops_ai.graph.impact_graph import create_graph  # type: ignore
    else:  # "threats"
        from workshops_ai.graph.threat_graph import create_graph  # type: ignore

    graph = create_graph(deps)
    return graph


# =============================================================================
# Helpers for memory adapters
# =============================================================================

def _coerce_mem_args(*args, **kwargs) -> dict[str, Any]:
    """
    Normalize arguments for memory_get/set/delete adapter to support both calling styles.

    Returns a dict:
      {
        "mode": "simple" | "rich",
        "key": str,
        "value": str,
        "ttl": Optional[int],
        "org_id": str, "user_id": Optional[str], "thread_id": str, "namespace": str
      }
    """
    # SIMPLE positional: get(key) / set(key, value, ttl=?)
    if args and isinstance(args[0], str) and not {"org_id", "thread_id", "namespace", "key"} & set(kwargs):
        key = args[0]
        value = args[1] if len(args) > 1 else kwargs.get("value", "")
        ttl = kwargs.get("ttl", kwargs.get("ttl_sec", None))
        return {
            "mode": "simple",
            "key": key,
            "value": value,
            "ttl": ttl,
            "org_id": "",
            "user_id": None,
            "thread_id": "",
            "namespace": "mem",
        }

    # SIMPLE keyword: get(key="...") / set(key="...", value="...", ttl=?)
    if "key" in kwargs and not {"org_id", "thread_id", "namespace"} & set(kwargs):
        return {
            "mode": "simple",
            "key": kwargs.get("key", ""),
            "value": kwargs.get("value", ""),
            "ttl": kwargs.get("ttl", kwargs.get("ttl_sec", None)),
            "org_id": "",
            "user_id": None,
            "thread_id": "",
            "namespace": "mem",
        }

    # RICH keyword form
    return {
        "mode": "rich",
        "key": kwargs.get("key", ""),
        "value": kwargs.get("value", ""),
        "ttl": kwargs.get("ttl_sec", kwargs.get("ttl", None)),
        "org_id": kwargs.get("org_id", ""),
        "user_id": kwargs.get("user_id"),
        "thread_id": kwargs.get("thread_id", ""),
        "namespace": kwargs.get("namespace", "mem"),
    }


# =============================================================================
# Ephemeral memory (Redis), safe no-op fallback if Redis missing
# =============================================================================

def make_redis_memory(settings: AppSettings):
    """
    Return (async) memory_get, memory_set, memory_delete functions.

    Manager/routers may call with rich kwargs:
      memory_get(org_id=..., user_id=..., thread_id=..., namespace=..., key=...)
      memory_set(org_id=..., user_id=..., thread_id=..., namespace=..., key=..., value=..., ttl_sec=...)
      memory_delete(org_id=..., user_id=..., thread_id=..., namespace=..., key=...)

    Rolling window adapters call with simple positional/keyword:
      get(key), set(key, value, ttl=?), delete(key)

    We support both. If Redis is unavailable, return no-op coroutines with the same API.
    """
    # Resolve Redis URL from settings
    redis_url = getattr(settings, "redis_url", None) or getattr(
        getattr(settings, "redis", None), "url", None
    )
    # Use the checkpoint namespace as a stable prefix to avoid collisions across envs
    ns = getattr(settings, "checkpoint_ns", None) or "lg:ckpt"

    # ------------------------------
    # No Redis available → no-op API
    # ------------------------------
    if not redis_url or AsyncRedis is None:
        async def memory_get_noop(*args, **kwargs) -> Optional[str]:
            _ = _coerce_mem_args(*args, **kwargs)
            return None

        async def memory_set_noop(*args, **kwargs) -> None:
            _ = _coerce_mem_args(*args, **kwargs)
            return None

        async def memory_delete_noop(*args, **kwargs) -> None:
            _ = _coerce_mem_args(*args, **kwargs)
            return None

        return memory_get_noop, memory_set_noop, memory_delete_noop

    # ------------------------------
    # Redis-backed implementation
    # ------------------------------
    client = AsyncRedis.from_url(redis_url, decode_responses=True)

    def _compose_key(*, org_id: str, user_id: str | None, thread_id: str, namespace: str, key: str) -> str:
        uid = (user_id or "anon").strip() or "anon"
        # Structure: <ckpt-ns>:mem:<org>:<thread>:<user>:<namespace>:<key>
        return f"{ns}:mem:{org_id}:{thread_id}:{uid}:{namespace}:{key}"

    async def memory_get(*args, **kwargs) -> Optional[str]:
        fields = _coerce_mem_args(*args, **kwargs)
        if fields["mode"] == "simple":
            return await client.get(fields["key"])
        k = _compose_key(
            org_id=fields["org_id"],
            user_id=fields["user_id"],
            thread_id=fields["thread_id"],
            namespace=fields["namespace"],
            key=fields["key"],
        )
        return await client.get(k)

    async def memory_set(*args, **kwargs) -> None:
        fields = _coerce_mem_args(*args, **kwargs)
        if fields["mode"] == "simple":
            ttl = fields["ttl"]
            if ttl and int(ttl) > 0:
                await client.setex(fields["key"], int(ttl), fields["value"])
            else:
                await client.set(fields["key"], fields["value"])
            return

        k = _compose_key(
            org_id=fields["org_id"],
            user_id=fields["user_id"],
            thread_id=fields["thread_id"],
            namespace=fields["namespace"],
            key=fields["key"],
        )
        ttl = fields["ttl"]
        if ttl and int(ttl) > 0:
            await client.setex(k, int(ttl), fields["value"])
        else:
            await client.set(k, fields["value"])

    async def memory_delete(*args, **kwargs) -> None:
        fields = _coerce_mem_args(*args, **kwargs)
        if fields["mode"] == "simple":
            await client.delete(fields["key"])
            return
        k = _compose_key(
            org_id=fields["org_id"],
            user_id=fields["user_id"],
            thread_id=fields["thread_id"],
            namespace=fields["namespace"],
            key=fields["key"],
        )
        await client.delete(k)

    # Close client on interpreter exit (best-effort)
    def _finalizer() -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(client.close())
            else:
                loop.run_until_complete(client.close())
        except Exception:
            pass

    atexit.register(_finalizer)
    return memory_get, memory_set, memory_delete


# =============================================================================
# Helper: build submit_changeset callable (online → fallback OUTBOX)
# =============================================================================

def _make_submit_callable(settings: AppSettings) -> SubmitChangeSet:
    """
    Returns a callable that tries to submit to the primary backend
    (if an API client is available), otherwise falls back to writing
    into the OUTBOX directory via submit_or_fallback().

    This is the canonical submit used by ALL graphs via deps.submit_changeset.
    """
    submitter: Any = None
    try:
        from workshops_ai.app.services.tazila_client import get_client  # type: ignore
        submitter = get_client()  # expected to implement .submit_changeset(...)
    except Exception:
        submitter = None

    outbox_dir = getattr(getattr(settings, "tazila", object()), "outbox_path", None)

    async def _submit(
        changeset: ChangeSet,
        *,
        require_success: bool = False,
    ) -> SubmitOrFallbackResult:
        return await submit_or_fallback(
            changeset,
            submitter=submitter,
            require_success=require_success,
            outbox_dir=outbox_dir,
        )

    return _submit


# =============================================================================
# Default dependencies factory (used by manager and routers)
# =============================================================================

def make_default_deps(
    tab: str | None = None,
    *,
    settings: AppSettings | None = None,
    backend_port: Any | None = None,
    ws_hub: Any | None = None,
    make_retriever: Optional[MakeRetriever] = None,
    # --- LLM DI (preferred) ---
    llm_client: Any | None = None,        # ignored on purpose; we enforce ChatOpenAI below
    llm_temperature: float | None = None,
    # legacy LLM handle (kept to avoid breaking older graphs)
    llm: Any | None = None,
    # Memory + submit
    memory_get: MemoryGet | None = None,
    memory_set: MemorySet | None = None,
    submit_changeset: SubmitChangeSet | None = None,
) -> GraphDeps:
    """
    Build a minimal, safe set of dependencies for graph compilation/execution.

    CHANGE:
    - We *always* construct a LangChain `ChatOpenAI` from settings and inject it as `llm_client`.
      Any ad-hoc/legacy client is ignored to keep graphs clean and consistent.
    - We expose `submit_changeset` as the unified submit wrapper. Prefer it everywhere.
    """
    # Settings first (manager may pass settings; routers often pass app.state.settings)
    s = settings or get_settings()

    # --- Checkpointer (CACHED) ---
    kind = getattr(getattr(s, "checkpointer", object()), "kind", None) or getattr(s, "checkpointer_kind", None)
    redis_url = getattr(getattr(s, "redis", object()), "url", None)
    sqlite_path = getattr(getattr(getattr(s, "sqlite", object()), "ckpt_path", object()), "__str__", lambda: None)()
    ns = getattr(s, "checkpoint_ns", None)
    ttl_sec = getattr(s, "checkpoint_ttl_sec", None)
    saver = _get_checkpointer_cached(
        kind=kind,
        redis_url=redis_url,
        sqlite_path=sqlite_path,
        namespace=ns,
        ttl_sec=ttl_sec,
    )

    # Backend port (prefer provider, fallback to CSV, else None)
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

    # WS hub passthrough (optional)
    ws = ws_hub

    # Retriever factory (default to our RAG retriever)
    if make_retriever is None:
        def _default_make_retriever(org_id: str, tab: str) -> Optional[Retriever]:
            return _rag_make_retriever(org_id=org_id, tab=tab)
        make_retriever = _default_make_retriever

    # Ephemeral memory (Redis if configured; else no-op)
    if memory_get is None or memory_set is None:
        mget, mset, mdel = make_redis_memory(s)
        memory_get = memory_get or mget
        memory_set = memory_set or mset
        memory_delete = mdel
    else:
        memory_delete = None

    # Submit callable (online preferred, fallback to OUTBOX) — CANONICAL
    if submit_changeset is None:
        submit_changeset = _make_submit_callable(s)

    # Back-compat alias for legacy callers still using deps.submit_or_outbox(...).
    # Internally this just delegates to submit_changeset to keep lifecycle unified.
    async def _submit_or_outbox(
        changeset: ChangeSet,
        *,
        require_success: bool = False
    ) -> SubmitOrFallbackResult:
        return await submit_changeset(changeset, require_success=require_success)

    # --- LLM client (CACHED, forced to ChatOpenAI) ---
    if llm_temperature is None:
        llm_temperature = float(getattr(getattr(s, "llm", object()), "temperature", 0.2))

    model = getattr(getattr(s, "llm", object()), "model", None)
    api_key = getattr(getattr(s, "llm", object()), "api_key", None) or getattr(s, "openai_api_key", None)
    base_url = getattr(getattr(s, "llm", object()), "base_url", None) or getattr(s, "openai_base_url", None)
    http_timeout = getattr(getattr(s, "llm", object()), "http_timeout_sec", None)
    max_retries = getattr(getattr(s, "llm", object()), "openai_max_retries", None)

    if not model or not api_key:
        raise RuntimeError("ChatOpenAI cannot be constructed: missing llm.model or API key in settings.")

    llm_client = _get_llm_client_cached(
        model=model,
        api_key=api_key,
        temperature=llm_temperature,
        http_timeout_sec=http_timeout,
        max_retries=max_retries,
        base_url=base_url,
    )

    try:
        logger.info(
            "LLM client wired (LangChain ChatOpenAI): model=%s, base_url=%s, temperature=%s",
            model,
            base_url,
            llm_temperature,
        )
    except Exception:
        pass

    # Expose checkpoint namespace (helps routers/tests pass it to LangGraph)
    ckpt_ns = ns or getattr(getattr(s, "checkpointer", object()), "namespace", None)

    return GraphDeps(
        backend_port=backend_port,
        ws_hub=ws,
        make_retriever=make_retriever,
        checkpointer=saver,
        settings=s,
        tab=(tab or "").lower() if tab else "",
        # --- LLM DI (preferred + legacy) ---
        llm_client=llm_client,             # guaranteed ChatOpenAI
        llm_temperature=llm_temperature,
        llm=llm,                           # legacy passthrough (avoid using in new code)
        # --- Memory + submit ---
        memory_get=memory_get,
        memory_set=memory_set,
        memory_delete=memory_delete,
        submit_changeset=submit_changeset,          # canonical submit for ALL graphs
        submit_or_outbox=_submit_or_outbox,         # legacy alias, delegates to submit_changeset
        checkpoint_ns=ckpt_ns,
        effective_policy=None,
    )


def get_graph_deps(
    *,
    tab: str | None = None,
    settings: AppSettings | None = None,
    backend_port: Any | None = None,
    ws_hub: Any | None = None,
) -> GraphDeps:
    """
    Router/manager-facing name expected across the codebase.
    Thin wrapper over make_default_deps(...) to keep one source of truth.
    """
    return make_default_deps(
        tab=tab,
        settings=settings,
        backend_port=backend_port,
        ws_hub=ws_hub,
    )


# =============================================================================
# Invoke helpers with timeout (support optional 'config'), robust async→sync fallback
# =============================================================================

async def _invoke_async(graph: RunnableGraph, inputs: Any, config: Optional[dict]) -> Any:
    """Always prefer async. Do NOT fallback to sync for graphs with async-only nodes."""
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
        return await loop.run_in_executor(None, lambda: graph.invoke(inputs))  # type: ignore/misc
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


# =============================================================================
# Language normalization + LangGraph config builders
# =============================================================================

def _normalize_lang_iso(tag: Optional[str]) -> str:
    """
    Normalize Accept-Language-ish values into a primary ISO language code.

    Examples:
      "cs-CZ" -> "cs"
      "cs-CZ, cs" -> "cs"
      "en" -> "en"
      None -> "cs" (conservative default to keep UI Czech-first unless overridden)
    """
    if not tag:
        return "cs"
    head = str(tag).split(",")[0].split(";")[0].strip()
    primary = (head.split("-")[0] or "cs").lower()
    return primary


def make_langgraph_config(
    *,
    org_id: str,
    tab: str,
    user_id: str | None,
    deps: GraphDeps,
    user_lang: Optional[str] = None,
) -> dict:
    """
    Compose a LangGraph config with required 'configurable' keys for the checkpointer.
    - thread_id      : required by LangGraph checkpointers
    - checkpoint_ns  : optional but recommended (taken from deps.checkpoint_ns)
    - lang/user_lang : stable per-request language hint for ALL agent prompts

    NOTE:
    - `user_lang` is expected to be a primary ISO code (e.g., "cs"|"sk"|"en").
      This will be written to BOTH `configurable.lang` (new) and `configurable.user_lang`
      (legacy), so older nodes/templates can keep working unchanged.
    """
    cfg: dict[str, Any] = {"configurable": {}}
    user = (user_id or "anon").strip() or "anon"
    cfg["configurable"]["thread_id"] = f"{org_id}:{tab}:{user}"
    if deps.checkpoint_ns:
        cfg["configurable"]["checkpoint_ns"] = deps.checkpoint_ns

    # Language (stable; set both keys for back-compat)
    if user_lang:
        iso = _normalize_lang_iso(user_lang)
        cfg["configurable"]["lang"] = iso
        cfg["configurable"]["user_lang"] = iso  # back-compat
    return cfg


def build_graph_config(req: UIRequest, deps: Optional[GraphDeps] = None) -> dict[str, Any]:
    """
    Convenience wrapper to build the per-turn LangGraph config from a UIRequest.

    - Extracts and normalizes language from `req.language` (already ISO by ws_router).
    - Computes a stable thread_id via the single source of truth:
        * If deps is provided → deps.thread_id_for(req) → _derive_thread_id(...)
        * Else → _derive_thread_id(req) directly
    - Sets both `configurable.lang` (new) and `configurable.user_lang` (legacy).
    """
    # Normalize language (defensive; ws_router already sends ISO like "cs")
    iso = _normalize_lang_iso(getattr(req, "language", None))

    # Thread id (single source of truth)
    if deps is not None:
        thread_id = deps.thread_id_for(req)
        ckpt_ns = deps.checkpoint_ns
    else:
        thread_id = _derive_thread_id(req)
        ckpt_ns = None

    cfg: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if ckpt_ns:
        cfg["configurable"]["checkpoint_ns"] = ckpt_ns

    cfg["configurable"]["lang"] = iso
    cfg["configurable"]["user_lang"] = iso  # back-compat

    return cfg


# =============================================================================
# Compatibility alias (accepts optional argument)
# =============================================================================

def build_checkpointer(settings: AppSettings | None = None) -> Any:
    """
    Compatibility wrapper.
    - If called as build_checkpointer() -> uses get_settings() internally.
    - If called as build_checkpointer(settings) or build_checkpointer(settings=...),
      forwards to the centralized (CACHED) builder.
    """
    s = settings or get_settings()
    kind = getattr(getattr(s, "checkpointer", object()), "kind", None) or getattr(s, "checkpointer_kind", None)
    redis_url = getattr(getattr(s, "redis", object()), "url", None)
    sqlite_path = getattr(getattr(s, "sqlite", object()), "ckpt_path", None)
    # Some settings may wrap the path; ensure a plain string (best-effort)
    if hasattr(sqlite_path, "__str__"):
        sqlite_path = str(sqlite_path)
    ns = getattr(s, "checkpoint_ns", None)
    ttl_sec = getattr(s, "checkpoint_ttl_sec", None)
    return _get_checkpointer_cached(
        kind=kind,
        redis_url=redis_url,
        sqlite_path=sqlite_path,
        namespace=ns,
        ttl_sec=ttl_sec,
    )


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
    "get_graph_deps",
    "make_langgraph_config",
    "build_graph_config",
    "build_checkpointer",
    "MemoryGet",
    "MemorySet",
    "MemoryDelete",
]
