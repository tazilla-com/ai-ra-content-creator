# File: workshops_ai/rag/retriever.py
"""
Scoped RAG retriever constructor.

What this module does
- Builds a retriever bound to (org_id, tab) scope.
- Picks the backing VectorStore driver (default: Chroma).
- Applies unified defaults for `top_k`, optional MMR re-ranking, and metadata filters.

Design
- `make_retriever(org_id, tab)` returns an object with `.search()` and `.asearch()`.
- Under the hood we get a VectorStorePort, create a scope, and wrap its retriever.
- Optional MMR (Maximal Marginal Relevance) is applied as a post-processing step:
  we re-rank the top-N (N >= top_k) results using embeddings; if embedding backend
  is unavailable, we fall back to base ranking.

Env/Settings (read from central settings with env fallbacks)
- WORKSHOPS_AI_RAG__STORE            : "chroma" (default) | "qdrant" (future)
- WORKSHOPS_AI_RAG__TOP_K            : default top-k (int, default 5)
- WORKSHOPS_AI_RAG__MMR_ENABLED      : "true"/"false" (default "false")
- WORKSHOPS_AI_RAG__MMR_LAMBDA       : float in [0,1], trade-off relevance vs diversity (default 0.5)
- WORKSHOPS_AI_RAG__MMR_POOL         : candidate pool size before MMR (default max(top_k*3, 12))

Notes
- This module stays store-agnostic via VectorStorePort / RetrieverPort. Currently
  we ship a `ChromaVectorStore` factory, but the selector is ready for more backends.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from workshops_ai.rag.port import RetrieverPort, Scope, VectorStorePort

logger = logging.getLogger(__name__)


# ----------------------------
# Settings helpers
# ----------------------------

def _get_settings():
    try:
        from workshops_ai.settings import get_settings  # type: ignore
        return get_settings()
    except Exception:
        return None


def _env_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _cfg(key: str, default: Any) -> Any:
    s = _get_settings()
    if s and hasattr(s, key):
        return getattr(s, key)
    return os.getenv(key, default)


# ----------------------------
# Vector store selector
# ----------------------------

def _make_store() -> VectorStorePort:
    store = (_cfg("WORKSHOPS_AI_RAG__STORE", "chroma") or "chroma").strip().lower()
    if store == "chroma":
        from workshops_ai.rag.chroma_store import make_chroma_store
        return make_chroma_store(_get_settings())
    # Future: add qdrant/weaviate here
    raise ValueError(f"Unsupported RAG store '{store}'")


# ----------------------------
# Lightweight embeddings for MMR
# ----------------------------

class _Embedder:
    """
    Small embedding helper.
    Tries FastEmbed first (zero-config); if unavailable, tries OpenAI (needs OPENAI_API_KEY).
    If neither is available, MMR will be disabled gracefully.
    """
    def __init__(self):
        self._mode = None
        # Try FastEmbed
        try:
            from fastembed import TextEmbedding  # type: ignore
            self._fe_cls = TextEmbedding
            self._fe_model = "BAAI/bge-small-en-v1.5"
            self._mode = "fastembed"
        except Exception:
            self._fe_cls = None

        # Try OpenAI as fallback
        if self._mode is None:
            try:
                import openai  # type: ignore
                self._openai = openai
                self._openai.api_key = os.getenv("OPENAI_API_KEY")
                # Default model can be overridden by env; keep conservative embedding model
                self._oai_model = os.getenv("WORKSHOPS_AI_RAG__EMBED_MODEL", "text-embedding-3-small")
                if not self._openai.api_key:
                    raise RuntimeError("OPENAI_API_KEY not set")
                self._mode = "openai"
            except Exception:
                self._openai = None

    def available(self) -> bool:
        return self._mode is not None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._mode == "fastembed":
            emb = self._fe_cls(model_name=self._fe_model)  # type: ignore
            # TextEmbedding returns a generator of [float] vectors
            return [vec for vec in emb.embed(texts)]
        if self._mode == "openai":
            # OpenAI responses depend on SDK; use a generic schema
            resp = self._openai.embeddings.create(model=self._oai_model, input=texts)  # type: ignore
            return [d.embedding for d in resp.data]  # type: ignore[attr-defined]
        raise RuntimeError("No embedding backend available")


def _cosine(a: list[float], b: list[float]) -> float:
    # Basic cosine similarity (no numpy dependency).
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _mmr_rerank(query: str, docs: list[dict], *, lambda_: float, top_k: int) -> list[dict]:
    """
    Simple MMR: greedily pick items maximizing λ * sim(query, doc) - (1-λ) * max_sim(doc, selected).
    Returns up to top_k docs in new order.
    """
    texts = [d.get("text") or "" for d in docs]
    emb = _Embedder()
    if not emb.available():
        logger.info("MMR disabled (no embedding backend available); using base ranking.")
        return docs[:top_k]

    try:
        qv = emb.embed([query])[0]
        dvs = emb.embed(texts)
    except Exception as e:
        logger.info("MMR embedding failed: %s; using base ranking.", e)
        return docs[:top_k]

    selected: list[int] = []
    candidates = list(range(len(docs)))
    while candidates and len(selected) < top_k:
        best_i = None
        best_score = -1.0
        for i in candidates:
            rel = _cosine(qv, dvs[i])
            div = 0.0
            if selected:
                div = max(_cosine(dvs[i], dvs[j]) for j in selected)
            score = lambda_ * rel - (1.0 - lambda_) * div
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)  # type: ignore[arg-type]
        candidates.remove(best_i)  # type: ignore[arg-type]
    return [docs[i] for i in selected]


# ----------------------------
# Wrapper retriever with defaults + MMR
# ----------------------------

@dataclass
class _ScopedRetriever(RetrieverPort):
    base: RetrieverPort
    org_id: str
    tab: str
    top_k: int
    mmr_enabled: bool
    mmr_lambda: float
    mmr_pool: int

    def _where(self) -> dict:
        # Scope-level filters are handled by the collection itself.
        # If you add per-doc filters, pass them here.
        return {}

    def search(self, query: str, *, k: int = 5, where: Optional[Mapping[str, Any]] = None) -> list[dict]:
        k_final = int(k or self.top_k)
        pool = max(int(self.mmr_pool), k_final) if self.mmr_enabled else k_final
        docs = self.base.search(query, k=pool, where=where or self._where())
        if self.mmr_enabled and len(docs) > k_final:
            return _mmr_rerank(query, docs, lambda_=float(self.mmr_lambda), top_k=k_final)
        return docs[:k_final]

    async def asearch(self, query: str, *, k: int = 5, where: Optional[Mapping[str, Any]] = None) -> list[dict]:
        k_final = int(k or self.top_k)
        pool = max(int(self.mmr_pool), k_final) if self.mmr_enabled else k_final
        docs = await self.base.asearch(query, k=pool, where=where or self._where())
        if self.mmr_enabled and len(docs) > k_final:
            # Run MMR sync; it does small CPU work only
            return _mmr_rerank(query, docs, lambda_=float(self.mmr_lambda), top_k=k_final)
        return docs[:k_final]


# ----------------------------
# Public factory
# ----------------------------

def make_retriever(*, org_id: str, tab: str) -> RetrieverPort:
    """
    Return a retriever bound to (org_id, tab), with unified defaults and optional MMR.
    """
    store = _make_store()
    scope = Scope(org_id=org_id, tab=tab)

    # Base retriever from the store
    base = store.make_retriever(scope)

    # Config
    top_k = int(_cfg("WORKSHOPS_AI_RAG__TOP_K", 5))
    mmr_enabled = _env_bool(str(_cfg("WORKSHOPS_AI_RAG__MMR_ENABLED", "false")))
    mmr_lambda = float(_cfg("WORKSHOPS_AI_RAG__MMR_LAMBDA", 0.5))
    mmr_pool = int(_cfg("WORKSHOPS_AI_RAG__MMR_POOL", max(top_k * 3, 12)))

    return _ScopedRetriever(
        base=base,
        org_id=org_id,
        tab=tab,
        top_k=top_k,
        mmr_enabled=mmr_enabled,
        mmr_lambda=mmr_lambda,
        mmr_pool=mmr_pool,
    )
