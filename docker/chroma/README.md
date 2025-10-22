# File: docker/chroma/README.md

# Chroma — local vector index (dev)

This folder documents how to run a local **Chroma** server client-side (embedded DB on disk) and where to mount volumes so your RAG indices persist across restarts.

## Where Chroma stores data
- By default, our app uses an **embedded Chroma client** with on-disk persistence (no separate server).
- The data directory is controlled by:
  - `WORKSHOPS_AI_RAG__DATA_DIR` → default `./workshops_ai/data/chroma`
- Collections are namespaced as: `<WORKSHOPS_AI_RAG__NAMESPACE>:<org_id>:<tab>` (default namespace `rag`).

## Recommended docker-compose mounts
Add a named volume and mount it to the app container path that matches `WORKSHOPS_AI_RAG__DATA_DIR`:

```yaml
services:
  api:
    image: your/image
    environment:
      - WORKSHOPS_AI_RAG__STORE=chroma
      - WORKSHOPS_AI_RAG__DATA_DIR=/workshops_ai/data/chroma    
      - WORKSHOPS_AI_RAG__NAMESPACE=rag
      - WORKSHOPS_AI_RAG__EMBED_PROVIDER=fastembed
      - WORKSHOPS_AI_RAG__EMBED_MODEL=BAAI/bge-small-en-v1.5
    volumes:
      - chroma_data:/workshops_ai/data/chroma

volumes:
  chroma_data:
If you run the app locally without Docker, set WORKSHOPS_AI_RAG__DATA_DIR to an absolute path (e.g., ${PWD}/data/chroma).

Back up & restore
Backup: tar/zip the directory at WORKSHOPS_AI_RAG__DATA_DIR.

Restore: stop the app, replace the directory with your backup, then start the app.

Backups are safe to copy while the app is stopped; for hot copies, prefer filesystem snapshots.

Cleaning the index
API: DELETE /rag/{org_id}?tab=<tab> removes all chunks for that scope.

CLI (dev): you can also delete the on-disk directory under WORKSHOPS_AI_RAG__DATA_DIR, but prefer the API for consistency.

Troubleshooting
Index not persisting: verify the container volume mount matches WORKSHOPS_AI_RAG__DATA_DIR.

Slow ingest: reduce chunk size (WORKSHOPS_AI_RAG__CHUNK_SIZE), or switch embeddings (OpenAI) if you have a key.

High disk usage: periodically purge unused scopes via API; archive old indices.

EF mismatch errors: older Chroma versions may lock an embedding function per collection. If you switch EF, purge the collection first.

Useful env (summary)
WORKSHOPS_AI_RAG__STORE=chroma

WORKSHOPS_AI_RAG__DATA_DIR=/workshops_ai/data/chroma

WORKSHOPS_AI_RAG__NAMESPACE=rag

WORKSHOPS_AI_RAG__EMBED_PROVIDER=fastembed|openai

WORKSHOPS_AI_RAG__EMBED_MODEL=BAAI/bge-small-en-v1.5

OPENAI_API_KEY=... (if using OpenAI embeddings)

rust
Zkopírovat kód

**.env additions**  
– None beyond what we’ve already introduced for RAG/Chroma.