# File: workshops_ai/rag/retriever.py
"""
Scoped RAG retriever constructor (settings-first, no ENV fallbacks).

What this module does:
- Builds a retriever bound to (org_id, tab) scope.
- Picks the backing VectorStore driver (default: Chroma) from settings.
- Applies unified defaults for `top_k`, optional MMR re-ranking, and metadata filters.

Design:
- `make_retriever(org_id, tab)` returns an object with `.search()` and `.asearch()`.
- Under the hood we get a VectorStorePort, create a scope, and wrap its retriever.
- MMR (Maximal Marginal Relevance) is optional, controlled by settings.rag.*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from workshops_ai.rag.port import RetrieverPort, Scope, VectorStorePort
from workshops_ai.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Settings helpers (settings-first)
# ------------------------------------------------------------------------------

class _RagCfg:
    """Materialized subset of RAG config derived solely from AppSettings."""

    def __init__(self, settings: AppSettings):
        rag = settings.rag
        self.enabled: bool = bool(rag.enabled)
        self.store: str = rag.store

        # Embeddings backend selection
        self.embed_provider: str = rag.embed_provider
        self.embed_model: str = rag.embed_model
        self.openai_api_key: Optional[str] = getattr(settings.llm, "api_key", None)

        # Retriever params
        self.top_k: int = int(rag.top_k)
        self.mmr_enabled: bool = bool(rag.effective_use_mmr())
        self.mmr_lambda: float = float(rag.mmr_lambda)
        self.mmr_pool: int = int(rag.mmr_pool)

        # Store paths/namespacing (consumed by store factory)
        self.namespace: str = rag.namespace
        self.data_dir: str = rag.effective_index_dir()


def _make_store(settings: AppSettings) -> VectorStorePort:
    """Select and build the vector store according to settings."""
    if settings.rag.store == "chroma":
        from workshops_ai.rag.chroma_store import make_chroma_store
        return make_chroma_store(settings)
    raise ValueError(f"Unsupported RAG store '{settings.rag.store}'")


# ------------------------------------------------------------------------------
# Lightweight embeddings for MMR
# ------------------------------------------------------------------------------

class _Embedder:
    """
    Minimal embedding helper controlled by settings.rag.embed_provider/model.

    Modes:
      - "fastembed": local, zero-config
      - "openai"   : requires settings.llm.api_key (provided externally via settings)
    Falls back to 'disabled' if unavailable → MMR off.
    """

    def __init__(self, provider: str, model: str, *, openai_api_key: Optional[str]):
        self._mode: Optional[str] = None
        self._model = model
        self._openai = None
        self._fe_cls = None

        p = (provider or "").strip().lower()
        if p == "fastembed":
            try:
                from fastembed import TextEmbedding  # type: ignore
                self._fe_cls = TextEmbedding
                self._mode = "fastembed"
            except Exception as e:
                logger.info("FastEmbed unavailable (%s); will try OpenAI if configured.", e)

        if self._mode is None and p in ("openai", "fastembed") and openai_api_key:
            try:
                import openai  # type: ignore
                self._openai = openai
                self._openai.api_key = openai_api_key  # settings-provided (no env read)
                self._mode = "openai"
            except Exception as e:
                logger.info("OpenAI embeddings unavailable (%s). MMR will be disabled.", e)

    def available(self) -> bool:
        return self._mode is not None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._mode == "fastembed":
            emb = self._fe_cls(model_name=self._model)  # type: ignore
            return [vec for vec in emb.embed(texts)]
        if self._mode == "openai":
            resp = self._openai.embeddings.create(model=self._model, input=texts)  # type: ignore
            return [d.embedding for d in resp.data]  # type: ignore[attr-defined]
        raise RuntimeError("No embedding backend available")


def _cosine(a: list[float], b: list[float]) -> float:
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


def _mmr_rerank(query: str, docs: list[dict], *, lambda_: float, top_k: int, embedder: _Embedder) -> list[dict]:
    """
    Greedy MMR: λ * sim(q, d) - (1-λ) * max_sim(d, S)
    Returns up to top_k docs in new order; falls back to base ranking on any failure.
    """
    if not embedder.available():
        logger.info("MMR disabled (no embedding backend available); using base ranking.")
        return docs[:top_k]

    texts = [d.get("text") or "" for d in docs]
    try:
        qv = embedder.embed([query])[0]
        dvs = embedder.embed(texts)
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
            div = max((_cosine(dvs[i], dvs[j]) for j in selected), default=0.0)
            score = lambda_ * rel - (1.0 - lambda_) * div
            if score > best_score:
                best_score = score
                best_i = i
        selected.append(best_i)  # type: ignore[arg-type]
        candidates.remove(best_i)  # type: ignore[arg-type]
    return [docs[i] for i in selected]


# ------------------------------------------------------------------------------
# Wrapper retriever with defaults + MMR
# ------------------------------------------------------------------------------

@dataclass
class _ScopedRetriever(RetrieverPort):
    base: RetrieverPort
    org_id: str
    tab: str
    enabled: bool
    top_k: int
    mmr_enabled: bool
    mmr_lambda: float
    mmr_pool: int
    embedder: Optional[_Embedder]  # reused across calls

    def _where(self) -> dict:
        # Scope-level filters are handled by the collection itself.
        return {}

    def search(self, query: str, *, k: int = 5, where: Optional[Mapping[str, Any]] = None) -> list[dict]:
        if not self.enabled:
            return []
        k_final = int(k or self.top_k)
        pool = max(int(self.mmr_pool), k_final) if self.mmr_enabled else k_final
        docs = self.base.search(query, k=pool, where=where or self._where())
        if self.mmr_enabled and len(docs) > k_final and self.embedder:
            return _mmr_rerank(query, docs, lambda_=float(self.mmr_lambda), top_k=k_final, embedder=self.embedder)
        return docs[:k_final]

    async def asearch(self, query: str, *, k: int = 5, where: Optional[Mapping[str, Any]] = None) -> list[dict]:
        if not self.enabled:
            return []
        k_final = int(k or self.top_k)
        pool = max(int(self.mmr_pool), k_final) if self.mmr_enabled else k_final
        docs = await self.base.asearch(query, k=pool, where=where or self._where())
        if self.mmr_enabled and len(docs) > k_final and self.embedder:
            return _mmr_rerank(query, docs, lambda_=float(self.mmr_lambda), top_k=k_final, embedder=self.embedder)
        return docs[:k_final]


# ------------------------------------------------------------------------------
# Public factory
# ------------------------------------------------------------------------------

def make_retriever(*, org_id: str, tab: str) -> RetrieverPort:
    """
    Return a retriever bound to (org_id, tab), with unified defaults and optional MMR.
    Respects settings.rag (enabled, top_k, MMR, embeddings). No ENV read.
    """
    settings = get_settings()
    cfg = _RagCfg(settings)
    store = _make_store(settings)
    scope = Scope(org_id=org_id, tab=tab)

    base = store.make_retriever(scope)

    embedder = _Embedder(
        cfg.embed_provider,
        cfg.embed_model,
        openai_api_key=cfg.openai_api_key,
    ) if cfg.mmr_enabled else None

    return _ScopedRetriever(
        base=base,
        org_id=org_id,
        tab=tab,
        enabled=cfg.enabled,
        top_k=cfg.top_k,
        mmr_enabled=cfg.mmr_enabled,
        mmr_lambda=cfg.mmr_lambda,
        mmr_pool=cfg.mmr_pool,
        embedder=embedder,
    )
