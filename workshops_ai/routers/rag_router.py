# File: workshops_ai/routers/rag_router.py
"""
RAG ingest & purge API.

Endpoints
- POST   /rag/{org_id}/ingest?tab=impacts
         Multipart PDF upload → extract → chunk → upsert to vector store.
- DELETE /rag/{org_id}?tab=impacts[&doc_id=...&doc_id=...]
         Purge all chunks for the (org_id, tab) scope, or only selected doc_ids.

Auth
- Uses existing `require_auth` dependency (API key / JWT), aligned with the rest of the app.

Notes
- Chunking, deterministic IDs, and store selection are handled in rag.ingest.
- Side-effect free outside the vector store (no application state changes).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status, Query

# Security dependency — prefer root-level placement; fallback to app/ if used there.
try:
    from workshops_ai.security import require_auth  # type: ignore
except Exception:
    try:
        from workshops_ai.app.security import require_auth  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing require_auth dependency (workshops_ai.security).") from e

from workshops_ai.rag.ingest import aingest_pdf, apurge_scope, apurge_ids

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post(
    "/{org_id}/ingest",
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a PDF into the RAG store for a given org and tab.",
)
async def rag_ingest_pdf(
    org_id: str,
    tab: str = Query(..., description="Target tab scope, e.g., 'impacts' or 'threats'."),
    file: UploadFile = File(..., description="PDF file to ingest."),
    _auth: None = Depends(require_auth),
):
    """
    Accepts a PDF (multipart/form-data) and ingests it into the vector store under (org_id, tab).
    Returns a summary: scope, detected title, chunk count, and upserted IDs.
    """
    # Some clients set octet-stream; allow it if filename ends with .pdf
    content_type = (file.content_type or "").lower()
    if content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=415, detail="Unsupported media type; expected a PDF.")

    try:
        data = await file.read()
        result = await aingest_pdf(
            org_id=org_id,
            tab=tab,
            pdf=data,
            source=file.filename or "upload.pdf",
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("RAG ingest failed")
        raise HTTPException(status_code=500, detail=f"RAG ingest failed: {e}") from e


@router.delete(
    "/{org_id}",
    status_code=status.HTTP_200_OK,
    summary="Purge RAG chunks for a given org and tab, or specific doc_ids.",
)
async def rag_purge(
    org_id: str,
    tab: str = Query(..., description="Target tab scope to purge, e.g., 'impacts' or 'threats'."),
    doc_id: Optional[List[str]] = Query(
        None,
        description="Optional list of logical doc_ids to delete. If omitted, the whole scope is purged.",
    ),
    _auth: None = Depends(require_auth),
):
    """
    If `doc_id` is provided, removes only those documents (logical IDs).
    Otherwise, purges the entire (org_id, tab) scope from the vector store.
    """
    try:
        if doc_id:
            deleted = await apurge_ids(org_id=org_id, tab=tab, doc_ids=doc_id)
            return {
                "scope": {"org_id": org_id, "tab": tab},
                "requested_doc_ids": len(doc_id),
                "deleted": deleted,
            }
        deleted = await apurge_scope(org_id=org_id, tab=tab)
        return {"scope": {"org_id": org_id, "tab": tab}, "deleted": deleted}
    except Exception as e:
        logger.exception("RAG purge failed")
        raise HTTPException(status_code=500, detail=f"RAG purge failed: {e}") from e
