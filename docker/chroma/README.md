# Chroma — local vector index (dev)

> **Status: experimental.** The RAG router is not mounted in the served application
> (`workshops_ai.app.main:app`), so the `/rag/*` endpoints referenced below are not currently
> reachable. This document describes the intended setup for the RAG/Chroma path and is kept for
> development reference.

This folder documents how to run a local **Chroma** instance (embedded, on-disk) and where to mount
volumes so RAG indices persist across restarts.

## Where Chroma stores data

- The app uses an **embedded Chroma client** with on-disk persistence — no separate server.
- The data directory is controlled by `WORKSHOPS_AI_RAG__DATA_DIR`
  (default `./workshops_ai/data/chroma`).
- Collections are namespaced as `<WORKSHOPS_AI_RAG__NAMESPACE>:<org_id>:<tab>`
  (default namespace `rag`).

## Recommended docker-compose mounts

Add a named volume and mount it at the container path matching `WORKSHOPS_AI_RAG__DATA_DIR`:

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
```

Running locally without Docker, set `WORKSHOPS_AI_RAG__DATA_DIR` to an absolute path — for example
`${PWD}/data/chroma`.

## Backup and restore

- **Backup**: archive the directory at `WORKSHOPS_AI_RAG__DATA_DIR`.
- **Restore**: stop the app, replace the directory with your backup, start the app.

Backups are safe to copy while the app is stopped. For hot copies, prefer filesystem snapshots.

## Cleaning the index

- **API**: `DELETE /rag/{org_id}?tab=<tab>` removes all chunks for that scope.
  *(Not currently mounted — see the status note above.)*
- **Manually (dev)**: delete the on-disk directory under `WORKSHOPS_AI_RAG__DATA_DIR`. Prefer the API
  where available, for consistency.

## Troubleshooting

| Symptom | Check |
|---|---|
| Index not persisting | Verify the container volume mount matches `WORKSHOPS_AI_RAG__DATA_DIR`. |
| Slow ingest | Reduce `WORKSHOPS_AI_RAG__CHUNK_SIZE`, or switch to OpenAI embeddings if you have a key. |
| High disk usage | Periodically purge unused scopes; archive old indices. |
| Embedding-function mismatch | Older Chroma versions lock an embedding function per collection. If you switch EF, purge the collection first. |

## Environment variables

```bash
WORKSHOPS_AI_RAG__STORE=chroma
WORKSHOPS_AI_RAG__DATA_DIR=/workshops_ai/data/chroma
WORKSHOPS_AI_RAG__NAMESPACE=rag
WORKSHOPS_AI_RAG__EMBED_PROVIDER=fastembed        # fastembed | openai
WORKSHOPS_AI_RAG__EMBED_MODEL=BAAI/bge-small-en-v1.5
OPENAI_API_KEY=...                                # only if using OpenAI embeddings
```

No `.env` additions are required beyond the RAG/Chroma variables already listed above.
