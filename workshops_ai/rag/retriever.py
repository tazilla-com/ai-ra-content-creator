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

Env/Settings (central settings with ENV fallbacks)
- settings.rag.* is preferred; WORKSHOPS_AI_RAG__* acts as a fallback.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from workshops_ai.rag.port import RetrieverPort, Scope, VectorStorePort
from workshops_ai.settings import get_settings

logger = logging.getLogger(__name__)


# ----------------------------
# Settings helpers
# ----------------------------

def _env_bool(val: Optional[str], default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _effective_rag_cfg():
    """
    Prefer settings().rag; fall back to ENV if needed.
    """
    s = get_settings()
    rag = getattr(s, "rag", None)

    # Store
    store = (rag.store if rag and getattr(rag, "store", None) else
             (os.getenv("WORKSHOPS_AI_RAG__STORE") or "chroma")).strip().lower()

    # Embeddings
    embed_provider = (rag.embed_provider if rag and getattr(rag, "embed_provider", None) else
                      (os.getenv("WORKSHOPS_AI_RAG__EMBED_PROVIDER") or "fastembed")).strip().lower()
    embed_model = (rag.embed_model if rag and getattr(rag, "embed_model", None) else
                   (os.getenv("WORKSHOPS_AI_RAG__EMBED_MODEL") or
                    ("BAAI/bge-small-en-v1.5" if embed_provider == "fastembed" else "text-embedding-3-small")))

    # Retriever params
    top_k = int(rag.top_k if rag and getattr(rag, "top_k", None) is not None
                else os.getenv("WORKSHOPS_AI_RAG__TOP_K", 5))
    # mmr_enabled respects new flag, otherwise legacy use_mmr; as ENV fallback keep the original var
    mmr_enabled = (rag.effective_use_mmr() if rag else
                   _env_bool(os.getenv("WORKSHOPS_AI_RAG__MMR_ENABLED", os.getenv("WORKSHOPS_AI_RAG__USE_MMR", "false"))))
    mmr_lambda = float(rag.mmr_lambda if rag and getattr(rag, "mmr_lambda", None) is not None
                       else os.getenv("WORKSHOPS_AI_RAG__MMR_LAMBDA", 0.5))
    mmr_pool = int(rag.mmr_pool if rag and getattr(rag, "mmr_pool", None) is not None
                   else os.getenv("WORKSHOPS_AI_RAG__MMR_POOL", max(top_k * 3, 12)))

    # Global enable switch
    enabled = bool(rag.enabled) if rag and getattr(rag, "enabled", None) is not None \
        else _env_bool(os.getenv("WORKSHOPS_AI_RAG__ENABLED", "true"), True)

    # Namespace/data_dir are consumed by the store factory; leave to make_chroma_store via settings
    return {
        "enabled": enabled,
        "store": store,
        "top_k": top_k,
        "mmr_enabled": mmr_enabled,
        "mmr_lambda": mmr_lambda,
        "mmr_pool": mmr_pool,
        "embed_provider": embed_provider,
        "embed_model": embed_model,
        "settings": s,  # pass through for store factory
    }


# ----------------------------
# Vector store selector
# ----------------------------

def _make_store() -> VectorStorePort:
    cfg = _effective_rag_cfg()
    if cfg["store"] == "chroma":
        from workshops_ai.rag.chroma_store import make_chroma_store
        return make_chroma_store(cfg["settings"])  # settings carries rag.data_dir/namespace
    # Future: qdrant/weaviate
    raise ValueError(f"Unsupported RAG store '{cfg['store']}'")


# ----------------------------
# Lightweight embeddings for MMR
# ----------------------------

class _Embedder:
    """
    Minimal embedding helper controlled by settings.rag.embed_provider/model.
    - "fastembed": local, zero-config
    - "openai"   : needs OPENAI_API_KEY
    Falls back to 'disabled' if unavailable → MMR off.
    """
    def __init__(self, provider: str, model: str):
        self._mode: Optional[str] = None
        self._model = model
        self._openai = None
        self._fe_cls = None

        provider = (provider or "").strip().lower()
        if provider == "fastembed":
            try:
                from fastembed import TextEmbedding  # type: ignore
                self._fe_cls = TextEmbedding
                self._mode = "fastembed"
            except Exception as e:
                logger.info("FastEmbed unavailable (%s); falling back to OpenAI if configured.", e)

        if self._mode is None and provider in ("openai", "fastembed"):
            try:
                import openai  # type: ignore
                self._openai = openai
                self._openai.api_key = os.getenv("OPENAI_API_KEY")
                if not self._openai.api_key:
                    raise RuntimeError("OPENAI_API_KEY not set")
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


# ----------------------------
# Wrapper retriever with defaults + MMR
# ----------------------------

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
            # RAG disabled → empty list -> graphs will fall back to LLM-only explain.
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


# ----------------------------
# Public factory
# ----------------------------

def make_retriever(*, org_id: str, tab: str) -> RetrieverPort:
    """
    Return a retriever bound to (org_id, tab), with unified defaults and optional MMR.
    Respects settings.rag (enabled, top_k, MMR, embeddings).
    """
    cfg = _effective_rag_cfg()
    store = _make_store()
    scope = Scope(org_id=org_id, tab=tab)

    # Base retriever from the store
    base = store.make_retriever(scope)

    # Embedder according to config (reused on the retriever)
    embedder = _Embedder(cfg["embed_provider"], cfg["embed_model"]) if cfg["mmr_enabled"] else None

    return _ScopedRetriever(
        base=base,
        org_id=org_id,
        tab=tab,
        enabled=bool(cfg["enabled"]),
        top_k=int(cfg["top_k"]),
        mmr_enabled=bool(cfg["mmr_enabled"]),
        mmr_lambda=float(cfg["mmr_lambda"]),
        mmr_pool=int(cfg["mmr_pool"]),
        embedder=embedder,
    )
