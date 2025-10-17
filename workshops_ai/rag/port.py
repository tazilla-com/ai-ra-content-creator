# File: workshops_ai/rag/port.py
"""
RAG ports (interfaces) for vector stores and retrievers.

Goals
- Stable, store-agnostic contract for RAG components.
- Explicit scoping by (org_id, tab) so each tenant/tab has isolated context.
- Minimal, production-friendly surface: upsert, delete, similarity search.
- Async-first with sync fallbacks (matching how graphs may call retrievers).

You can implement these protocols for Chroma, Qdrant, Weaviate, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class Scope:
    """Index/collection/namespace selection."""
    org_id: str
    tab: str  # e.g., "impacts" | "threats" | "assets"


@dataclass(frozen=True)
class TextDoc:
    """
    A single chunk of text to index.
    - `doc_id` is caller-provided (stable id); the store may derive an internal id.
    - `metadata` should contain only non-PII, non-sensitive values safe for logging.
    """
    doc_id: str
    text: str
    scope: Scope
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """Result row returned by similarity search."""
    text: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ----------------------------
# RetrieverPort
# ----------------------------

class RetrieverPort(Protocol):
    """
    Contract for a retriever that performs similarity search.
    Implementations should be *read-only* and fast; indexing is done via VectorStorePort.
    """

    def search(self, query: str, *, k: int = 5, where: Optional[Mapping[str, Any]] = None) -> list[dict]:
        """
        Sync search. Returns a list of dicts with at least:
          - 'text': str
          - 'score': float
          - 'metadata': dict (should include 'source' or similar when available)
        """
        ...

    async def asearch(self, query: str, *, k: int = 5, where: Optional[Mapping[str, Any]] = None) -> list[dict]:
        """Async search. Same shape as `search`."""
        ...


# ----------------------------
# VectorStorePort
# ----------------------------

class VectorStorePort(Protocol):
    """
    Contract for a vector store driver (Chroma/Qdrant/…).

    Scope model
    - Each (org_id, tab) maps to a collection/namespace. Implementations decide
      how to name/route it, but MUST isolate data across scopes.
    """

    # --- Upsert ---

    def upsert_texts(self, docs: Sequence[TextDoc]) -> list[str]:
        """
        Insert or update chunks. Returns a list of store-generated IDs (same length as `docs`).
        Must be idempotent w.r.t. (scope, doc_id).
        """
        ...

    async def aupsert_texts(self, docs: Sequence[TextDoc]) -> list[str]:
        """Async version of `upsert_texts`."""
        ...

    # --- Delete ---

    def delete_scope(self, scope: Scope) -> int:
        """
        Remove all chunks for a given scope.
        Returns number of deleted records (best-effort).
        """
        ...

    async def adelete_scope(self, scope: Scope) -> int:
        """Async version of `delete_scope`."""
        ...

    def delete_ids(self, scope: Scope, *, doc_ids: Iterable[str]) -> int:
        """
        Remove specific logical documents by their caller-provided `doc_id`s.
        Returns number of deleted records (best-effort).
        """
        ...

    async def adelete_ids(self, scope: Scope, *, doc_ids: Iterable[str]) -> int:
        """Async version of `delete_ids`."""
        ...

    # --- Search (convenience; retrievers usually wrap this) ---

    def similarity_search(
        self,
        scope: Scope,
        query: str,
        *,
        k: int = 5,
        where: Optional[Mapping[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """
        Perform a similarity search within the given scope and optional metadata filter.
        Returns top-k results ordered by relevance (higher is better).
        """
        ...

    async def asimilarity_search(
        self,
        scope: Scope,
        query: str,
        *,
        k: int = 5,
        where: Optional[Mapping[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """Async version of `similarity_search`."""
        ...

    # --- Retriever factory ---

    def make_retriever(self, scope: Scope) -> RetrieverPort:
        """
        Return a lightweight retriever bound to `scope`. Implementations may capture
        store connection, embedding model reference, and retrieval params.
        """
        ...

    # --- Health / admin (optional) ---

    def ping(self) -> bool:
        """Return True if the store is reachable and ready."""
        ...

    async def aping(self) -> bool:
        """Async version of `ping`."""
        ...
