# File: workshops_ai/rag/chroma_store.py
"""
Chroma-backed VectorStorePort implementation.

Design goals
- On-disk persistence (persist_directory) with one Chroma collection per (org_id, tab).
- Stable ids: store-level id = f"{org_id}:{tab}:{doc_id}" for idempotent upserts/deletes.
- Pluggable embeddings: FastEmbed (default) or OpenAI (via env/settings).
- Read/write API compatible with VectorStorePort + a lightweight RetrieverPort.

Dependencies
- chromadb
- fastembed (default) OR OpenAI (when provider == "openai")

Notes
- This module is intentionally light: ingest/chunking happens elsewhere.
- `where` filters are forwarded directly to Chroma (must be JSON-like).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    import chromadb  # type: ignore
    from chromadb import Client
    from chromadb.config import Settings as ChromaSettings  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("chromadb must be installed to use ChromaVectorStore") from e

from workshops_ai.rag.port import (
    RetrieverPort,
    RetrievedChunk,
    Scope,
    TextDoc,
    VectorStorePort,
)

logger = logging.getLogger(__name__)


# ----------------------------
# Embedding selection
# ----------------------------

def _make_embedding_function(settings: Any):
    """
    Choose an embedding function based on settings/env:

    WORKSHOPS_AI_RAG__EMBED_PROVIDER = "fastembed" | "openai"
    WORKSHOPS_AI_RAG__EMBED_MODEL    = model id (e.g., "BAAI/bge-small-en-v1.5" or "text-embedding-3-small")

    Defaults:
      provider=fastembed, model="BAAI/bge-small-en-v1.5"
    """
    provider = (
        getattr(settings, "RAG_EMBED_PROVIDER", None)
        or os.getenv("WORKSHOPS_AI_RAG__EMBED_PROVIDER")
        or "fastembed"
    ).strip().lower()

    model = (
        getattr(settings, "RAG_EMBED_MODEL", None)
        or os.getenv("WORKSHOPS_AI_RAG__EMBED_MODEL")
        or "BAAI/bge-small-en-v1.5"
    )

    if provider == "openai":
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("OpenAI embedding function not available in chromadb utils") from e

        api_key = (
            getattr(settings, "OPENAI_API_KEY", None)
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set; OpenAI embeddings will fail at runtime.")
        return OpenAIEmbeddingFunction(api_key=api_key, model_name=model)

    # Default: fastembed
    try:
        from chromadb.utils.embedding_functions import FastEmbedEmbeddingFunction  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("FastEmbedEmbeddingFunction not available; install fastembed") from e

    return FastEmbedEmbeddingFunction(model_name=model)


# ----------------------------
# Utilities
# ----------------------------

def _sanitize(s: str) -> str:
    """Make a string safe for Chroma collection names."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9:_\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "default"


def _coll_name(scope: Scope, ns: str) -> str:
    return _sanitize(f"{ns}:{scope.org_id}:{scope.tab}")


def _store_id(scope: Scope, doc_id: str) -> str:
    return f"{scope.org_id}:{scope.tab}:{doc_id}"


# ----------------------------
# Retriever implementation
# ----------------------------

class _ChromaRetriever(RetrieverPort):
    def __init__(self, store: "ChromaVectorStore", scope: Scope):
        self._store = store
        self._scope = scope

    def search(self, query: str, *, k: int = 5, where: Optional[Mapping[str, Any]] = None) -> list[dict]:
        results = self._store.similarity_search(self._scope, query, k=k, where=where)
        out: list[dict] = []
        for r in results:
            out.append(
                {
                    "text": r.text,
                    "score": r.score,
                    "metadata": dict(r.metadata or {}),
                }
            )
        return out

    async def asearch(self, query: str, *, k: int = 5, where: Optional[Mapping[str, Any]] = None) -> list[dict]:
        results = await self._store.asimilarity_search(self._scope, query, k=k, where=where)
        out: list[dict] = []
        for r in results:
            out.append(
                {
                    "text": r.text,
                    "score": r.score,
                    "metadata": dict(r.metadata or {}),
                }
            )
        return out


# ----------------------------
# Vector store implementation
# ----------------------------

class ChromaVectorStore(VectorStorePort):
    def __init__(self, *, client: Client, persist_directory: str, namespace: str, embedding_function: Any):
        self._client = client
        self._persist_dir = persist_directory
        self._ns = namespace
        self._ef = embedding_function

    # --- core plumbing ---

    def _collection(self, scope: Scope):
        name = _coll_name(scope, self._ns)
        try:
            return self._client.get_or_create_collection(name=name, embedding_function=self._ef)
        except Exception:
            # In some chroma versions, changing EF on existing collection is not allowed.
            # Fallback: get as-is (assumes compatible EF).
            return self._client.get_or_create_collection(name=name)

    def _persist(self):
        try:
            self._client.persist()
        except Exception:
            # No-op if client does not support persist() in certain backends.
            pass

    # --- upsert ---

    def upsert_texts(self, docs: Sequence[TextDoc]) -> list[str]:
        if not docs:
            return []
        # Group by scope: fewer round-trips
        by_scope: dict[str, list[TextDoc]] = {}
        for d in docs:
            key = _coll_name(d.scope, self._ns)
            by_scope.setdefault(key, []).append(d)

        out_ids: list[str] = []
        for coll_name, group in by_scope.items():
            scope = group[0].scope
            coll = self._collection(scope)
            ids = [_store_id(scope, d.doc_id) for d in group]
            metadatas = [dict(d.metadata or {}, source=d.metadata.get("source") if d.metadata else None) for d in group]
            documents = [d.text for d in group]
            coll.upsert(ids=ids, documents=documents, metadatas=metadatas)
            out_ids.extend(ids)
        self._persist()
        return out_ids

    async def aupsert_texts(self, docs: Sequence[TextDoc]) -> list[str]:
        return await asyncio.to_thread(self.upsert_texts, docs)

    # --- delete ---

    def delete_scope(self, scope: Scope) -> int:
        coll = self._collection(scope)
        try:
            # Chroma has no "delete all in collection" API; workaround: delete_collection then recreate.
            self._client.delete_collection(coll.name)
            # Recreate empty collection for future use
            self._client.get_or_create_collection(name=coll.name, embedding_function=self._ef)
            self._persist()
            return 1  # best-effort count
        except Exception as e:
            logger.warning("delete_scope failed for %s: %s", coll.name, e)
            return 0

    async def adelete_scope(self, scope: Scope) -> int:
        return await asyncio.to_thread(self.delete_scope, scope)

    def delete_ids(self, scope: Scope, *, doc_ids: Iterable[str]) -> int:
        ids = [_store_id(scope, d) for d in doc_ids]
        if not ids:
            return 0
        coll = self._collection(scope)
        try:
            coll.delete(ids=ids)
            self._persist()
            return len(ids)
        except Exception as e:
            logger.warning("delete_ids failed: %s", e)
            return 0

    async def adelete_ids(self, scope: Scope, *, doc_ids: Iterable[str]) -> int:
        return await asyncio.to_thread(self.delete_ids, scope, doc_ids=doc_ids)

    # --- search ---

    def similarity_search(
        self,
        scope: Scope,
        query: str,
        *,
        k: int = 5,
        where: Optional[Mapping[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        coll = self._collection(scope)
        # Chroma returns lists aligned across documents/ids/distances
        try:
            res = coll.query(query_texts=[query], n_results=int(k), where=where or {})
        except Exception as e:
            logger.warning("similarity_search failed: %s", e)
            return []

        docs = (res.get("documents") or [[]])[0]
        metadatas = (res.get("metadatas") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        # Convert distances to a "score" where higher is better. Chroma typically returns smaller distance => better.
        out: list[RetrievedChunk] = []
        for text, meta, dist in zip(docs, metadatas, distances):
            try:
                score = 1.0 / (1.0 + float(dist))
            except Exception:
                score = 0.0
            out.append(RetrievedChunk(text=text, score=score, metadata=meta or {}))
        return out

    async def asimilarity_search(
        self,
        scope: Scope,
        query: str,
        *,
        k: int = 5,
        where: Optional[Mapping[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        return await asyncio.to_thread(self.similarity_search, scope, query, k=k, where=where)

    # --- retriever ---

    def make_retriever(self, scope: Scope) -> RetrieverPort:
        return _ChromaRetriever(self, scope)

    # --- health ---

    def ping(self) -> bool:
        try:
            # Simple operation: list collections
            _ = self._client.list_collections()
            return True
        except Exception as e:
            logger.warning("Chroma ping failed: %s", e)
            return False

    async def aping(self) -> bool:
        return await asyncio.to_thread(self.ping)


# ----------------------------
# Factory
# ----------------------------

def make_chroma_store(settings: Any) -> ChromaVectorStore:
    """
    Create a ChromaVectorStore from app settings / env.

    Env/Settings:
      WORKSHOPS_AI_RAG__DATA_DIR   (default: ./data/chroma)
      WORKSHOPS_AI_RAG__NAMESPACE  (default: rag)
      WORKSHOPS_AI_RAG__EMBED_PROVIDER (default: fastembed)
      WORKSHOPS_AI_RAG__EMBED_MODEL    (default: BAAI/bge-small-en-v1.5)
    """
    persist_dir = (
        getattr(settings, "RAG_DATA_DIR", None)
        or os.getenv("WORKSHOPS_AI_RAG__DATA_DIR")
        or "./data/chroma"
    )
    namespace = (
        getattr(settings, "RAG_NAMESPACE", None)
        or os.getenv("WORKSHOPS_AI_RAG__NAMESPACE")
        or "rag"
    )
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    ef = _make_embedding_function(settings)
    client = chromadb.Client(
        ChromaSettings(
            is_persistent=True,
            persist_directory=str(persist_dir),
        )
    )
    logger.info("Chroma store initialized at %s (ns=%s)", persist_dir, namespace)
    return ChromaVectorStore(client=client, persist_directory=str(persist_dir), namespace=namespace, embedding_function=ef)
