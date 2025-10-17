# File: workshops_ai/rag/ingest.py
"""
PDF ingest → chunk → upsert to VectorStore.

What this module does
- Extracts text from a PDF (path or bytes-like).
- Splits text into overlapping chunks.
- Builds scoped TextDoc objects (org_id, tab) with stable UUIDv5 doc_ids.
- Upserts chunks into the configured VectorStorePort (default: Chroma).
- Provides utilities to purge an entire (org_id, tab) scope or specific doc_ids.

Design choices
- Deterministic IDs: uuid5(namespace, f"{source}|p{page}|c{idx}") to allow idempotent re-ingest.
- Metadata kept minimal and non-PII: {source, page, chunk_index, title?}.
- Async-first helpers with sync wrappers.
"""

from __future__ import annotations

import asyncio
import io
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from workshops_ai.rag.port import Scope, TextDoc, VectorStorePort, RetrievedChunk

# Optional imports guarded for clear error messages
try:
    from pypdf import PdfReader  # lightweight and reliable text extractor
except Exception as e:  # pragma: no cover
    raise RuntimeError("pypdf must be installed for PDF ingest (pip install pypdf).") from e


# ----------------------------
# Settings helpers
# ----------------------------

def _get_settings():
    try:
        from workshops_ai.settings import get_settings  # type: ignore
        return get_settings()
    except Exception:
        return None


def _cfg_str(key: str, default: str) -> str:
    s = _get_settings()
    if s and hasattr(s, key):
        v = getattr(s, key)
        if isinstance(v, str) and v.strip():
            return v
    return os.getenv(key, default)


def _cfg_int(key: str, default: int) -> int:
    s = _get_settings()
    cand = None
    if s and hasattr(s, key):
        cand = getattr(s, key)
    else:
        ev = os.getenv(key)
        cand = int(ev) if (ev and ev.isdigit()) else None
    try:
        return int(cand) if cand is not None else default
    except Exception:
        return default


def _cfg_bool(key: str, default: bool) -> bool:
    s = _get_settings()
    if s and hasattr(s, key):
        v = getattr(s, key)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
    ev = os.getenv(key)
    if ev is None:
        return default
    return ev.strip().lower() in ("1", "true", "yes", "on")


# ----------------------------
# Vector store selector
# ----------------------------

def _make_store() -> VectorStorePort:
    store = (_cfg_str("WORKSHOPS_AI_RAG__STORE", "chroma") or "chroma").strip().lower()
    if store == "chroma":
        from workshops_ai.rag.chroma_store import make_chroma_store
        return make_chroma_store(_get_settings())
    # Future: add qdrant/weaviate here as needed.
    raise ValueError(f"Unsupported RAG store '{store}'")


# ----------------------------
# Chunking
# ----------------------------

def _chunk_text(text: str, *, size: int, overlap: int) -> list[str]:
    """
    Simple sliding-window chunker operating on characters.
    For higher quality retrieval you may replace with a token-aware splitter later.
    """
    text = (text or "").strip()
    if not text:
        return []
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


# ----------------------------
# PDF extraction
# ----------------------------

@dataclass(frozen=True)
class PdfExtract:
    pages: list[str]
    title: Optional[str]


def _extract_pdf_text(source: bytes | bytearray | memoryview | str | Path) -> PdfExtract:
    """
    Extract per-page text from a PDF. Accepts:
    - file path (str/Path)
    - bytes / bytearray / memoryview
    """
    if isinstance(source, (str, Path)):
        f = open(source, "rb")
        close = True
        stream = f
    else:
        stream = io.BytesIO(bytes(source))
        close = False

    try:
        reader = PdfReader(stream)
        title = None
        try:
            # pypdf returns a DocumentInformation object; values may be None
            info = reader.metadata or {}
            title = getattr(info, "title", None) or info.get("/Title") if isinstance(info, dict) else None
        except Exception:
            title = None

        pages: list[str] = []
        max_pages = _cfg_int("WORKSHOPS_AI_RAG__MAX_PAGES", 0)  # 0 = no cap
        limit = len(reader.pages) if max_pages <= 0 else min(max_pages, len(reader.pages))
        for i in range(limit):
            try:
                txt = reader.pages[i].extract_text() or ""
            except Exception:
                txt = ""
            pages.append(txt)
        return PdfExtract(pages=pages, title=title)
    finally:
        if close:
            stream.close()


# ----------------------------
# ID generation (deterministic)
# ----------------------------

# Namespace seed allows reproducible uuid5 across deployments
_UUID5_NS = uuid.UUID(_cfg_str("WORKSHOPS_AI_RAG__UUID5_NAMESPACE", "00000000-0000-5000-a000-000000000000"))

def _make_doc_id(*, source: str, page: int, chunk_index: int) -> str:
    name = f"{source}|p{page}|c{chunk_index}"
    return str(uuid.uuid5(_UUID5_NS, name))


# ----------------------------
# Public API
# ----------------------------

def ingest_pdf(
    *,
    org_id: str,
    tab: str,
    pdf: bytes | bytearray | memoryview | str | Path,
    source: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
    """
    Sync ingest: extract text → chunk → upsert to store.
    Returns a summary: {"scope": {...}, "chunks": N, "ids": [..]}.
    """
    return asyncio.run(
        aingest_pdf(
            org_id=org_id,
            tab=tab,
            pdf=pdf,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )


async def aingest_pdf(
    *,
    org_id: str,
    tab: str,
    pdf: bytes | bytearray | memoryview | str | Path,
    source: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> dict:
    """
    Async ingest: extract text → chunk → upsert to store.
    """
    store = _make_store()
    scope = Scope(org_id=org_id, tab=tab)

    cfg_size = chunk_size if chunk_size is not None else _cfg_int("WORKSHOPS_AI_RAG__CHUNK_SIZE", 1200)
    cfg_overlap = chunk_overlap if chunk_overlap is not None else _cfg_int("WORKSHOPS_AI_RAG__CHUNK_OVERLAP", 200)

    # Extract
    pdfx = await asyncio.to_thread(_extract_pdf_text, pdf)
    src_label = (source or "").strip() or (str(pdf) if isinstance(pdf, (str, Path)) else "uploaded.pdf")
    title = pdfx.title

    # Build docs
    docs: list[TextDoc] = []
    for page_idx, page_text in enumerate(pdfx.pages, start=1):
        chunks = _chunk_text(page_text, size=cfg_size, overlap=cfg_overlap)
        for ci, chunk in enumerate(chunks):
            doc_id = _make_doc_id(source=src_label, page=page_idx, chunk_index=ci)
            meta = {"source": src_label, "page": page_idx, "chunk_index": ci}
            if title:
                meta["title"] = title
            docs.append(TextDoc(doc_id=doc_id, text=chunk, scope=scope, metadata=meta))

    # Upsert (async)
    ids = await store.aupsert_texts(docs)  # type: ignore[attr-defined]
    return {
        "scope": {"org_id": org_id, "tab": tab},
        "title": title,
        "chunks": len(docs),
        "ids": ids,
    }


def purge_scope(*, org_id: str, tab: str) -> int:
    """
    Sync purge: removes all chunks under (org_id, tab).
    Returns number of deleted records (best-effort).
    """
    return asyncio.run(apurge_scope(org_id=org_id, tab=tab))


async def apurge_scope(*, org_id: str, tab: str) -> int:
    store = _make_store()
    scope = Scope(org_id=org_id, tab=tab)
    return await store.adelete_scope(scope)  # type: ignore[attr-defined]


def purge_ids(*, org_id: str, tab: str, doc_ids: Iterable[str]) -> int:
    """
    Sync purge of specific doc_ids (logical ids).
    """
    return asyncio.run(apurge_ids(org_id=org_id, tab=tab, doc_ids=doc_ids))


async def apurge_ids(*, org_id: str, tab: str, doc_ids: Iterable[str]) -> int:
    store = _make_store()
    scope = Scope(org_id=org_id, tab=tab)
    return await store.adelete_ids(scope, doc_ids=doc_ids)  # type: ignore[attr-defined]
