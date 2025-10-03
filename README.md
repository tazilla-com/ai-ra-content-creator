# WorkshopsAI — Quick Start

A tiny demo to chat with an **AssetAgent** (LLM + tool-calls) over **HTTP `/chat`** and **WS `/ws`**. Uses CSV as the backend.

---

1) Prereqs

- Python **3.11+**
- (Optional) LangSmith account for tracing
- (Windows) Use **PowerShell** not CMD

2) Install

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1

pip install -U pip
pip install --no-cache-dir -r requirements.txt -c constraints.txt


3) Environment

Create .env in project root (or export in your shell):

# ====================== Runtime / App ======================
WORKSHOPS_AI_ENV=dev
WORKSHOPS_AI_SERVICE_NAME=workshops_ai
WORKSHOPS_AI_VERSION=1.0.0
WORKSHOPS_AI_LOG_LEVEL=DEBUG
WORKSHOPS_AI_APP_HOST=0.0.0.0
WORKSHOPS_AI_APP_PORT=8000
WORKSHOPS_AI_ENABLE_DOCS=false
WORKSHOPS_AI_RELOAD=true

# CORS
WORKSHOPS_AI_CORS_ORIGINS=["*"]

# ======================== Security =========================
# API-key auth
WORKSHOPS_AI_SECURITY__ENABLE_API_KEY=true
WORKSHOPS_AI_SECURITY__API_KEYS=["PKHYk5Fex7GNMOaDVqtowwebiYc9o3TfqNM4KROmVVp2p1VjcILtqpBhMSNi1YD7","waPxELBmZfoP0gVP8USh6OMWOdMn_D4iNRIa3j-lDhOGZWtg11IsmdH3Cqi1oW3a"]

# JWT auth (HS256)
WORKSHOPS_AI_SECURITY__ENABLE_JWT=true
WORKSHOPS_AI_SECURITY__JWT_ALGORITHM=HS256
WORKSHOPS_AI_SECURITY__JWT_SECRET=Oa5pK7qZ3wY8fT1nV4rH9mD2sC6xB0uP3jL7eQ1yN5gR8tW2kF6vA9zJ4iM0oU5
WORKSHOPS_AI_SECURITY__JWT_ISSUER=https://auth.example.com
WORKSHOPS_AI_SECURITY__JWT_AUDIENCE=workshops-ai
# WORKSHOPS_AI_SECURITY__JWKS_URL=

# ====================== Paths (CSV data) ===================
WORKSHOPS_AI_PATHS__DATA_DIR=./workshops_ai/data
WORKSHOPS_AI_PATHS__STATE_DIR=./workshops_ai/data/state
WORKSHOPS_AI_PATHS__CATALOGS_DIR=./workshops_ai/data/catalogs

# ========== Export ChangeSets (Tazila) ==========
WORKSHOPS_AI_TAZILA__MODE=file         # file | http | disabled
WORKSHOPS_AI_TAZILA__OUTBOX_PATH=/workshops_ai/data/outbox
WORKSHOPS_AI_TAZILA__ENDPOINT=https://backend.example.com/changesets
WORKSHOPS_AI_TAZILA__AUTH_TOKEN=change-me-bearer-token
WORKSHOPS_AI_TAZILA__TIMEOUT_SEC=30

# ===================== Redis / Ratelimit ===================
WORKSHOPS_AI_REDIS__URL=redis://redis:6379/0
WORKSHOPS_AI_RATELIMIT__ENABLED=true
WORKSHOPS_AI_RATELIMIT__PER_MINUTE=120

# ======================= HTTP / Chat =======================
WORKSHOPS_AI_CHAT__TIMEOUT_SEC=180
WORKSHOPS_AI_CHAT__MAX_MESSAGE_LEN=4000

# ====================== WebSockets =========================
WORKSHOPS_AI_WS__TIMEOUT_SEC=75
# WORKSHOPS_AI_WS__COOKIE_NAME=wsai.sid

# ==================== LLM / OpenAI =========================
OPENAI_API_KEY=
WORKSHOPS_AI_LLM__MODEL=gpt-4o-mini
WORKSHOPS_AI_LLM__TEMPERATURE=0.2
WORKSHOPS_AI_LLM__HTTP_TIMEOUT_SEC=30
WORKSHOPS_AI_LLM__OPENAI_MAX_RETRIES=3

# ======== LangChain / LangSmith tracing ====================
WORKSHOPS_AI_LANGCHAIN__TRACING_V2=false
# WORKSHOPS_AI_LANGCHAIN__ENDPOINT=
WORKSHOPS_AI_LANGCHAIN__API_KEY=
WORKSHOPS_AI_LANGCHAIN__PROJECT=workshops-ai

# =============== LangGraph checkpointing ===================
WORKSHOPS_AI_CHECKPOINTER=redis           # redis | sqlite | memory
WORKSHOPS_AI_REDIS_URL=redis://redis:6379/0
WORKSHOPS_AI_CHECKPOINT_NS=lg:ckpt
WORKSHOPS_AI_CHECKPOINT_TTL_SEC=1209600   # 14 dní
WORKSHOPS_AI_SQLITE_CKPT_PATH=/mnt/data/checkpoints.db

# ======= Per-tab agent timeouts (flat overrides) ==========
WORKSHOPS_AI_AGENT_TIMEOUT_SEC_ASSETS=60
WORKSHOPS_AI_AGENT_TIMEOUT_SEC_IMPACTS=90
WORKSHOPS_AI_AGENT_TIMEOUT_SEC_THREATS=90

# ============================ RAG ==========================
WORKSHOPS_AI_RAG__ENABLED=true
WORKSHOPS_AI_RAG__STORE=chroma
WORKSHOPS_AI_RAG__DATA_DIR=./data/chroma
WORKSHOPS_AI_RAG__NAMESPACE=rag

# Embeddings
WORKSHOPS_AI_RAG__EMBED_PROVIDER=fastembed   # fastembed | openai
WORKSHOPS_AI_RAG__EMBED_MODEL=BAAI/bge-small-en-v1.5
# Pokud chceš OpenAI embeddings:
# WORKSHOPS_AI_RAG__EMBED_PROVIDER=openai
# WORKSHOPS_AI_RAG__EMBED_MODEL=text-embedding-3-small

# Retriever
WORKSHOPS_AI_RAG__TOP_K=5
WORKSHOPS_AI_RAG__MMR_ENABLED=false
WORKSHOPS_AI_RAG__MMR_LAMBDA=0.5
WORKSHOPS_AI_RAG__MMR_POOL=12

# Ingest
WORKSHOPS_AI_RAG__CHUNK_SIZE=1200
WORKSHOPS_AI_RAG__CHUNK_OVERLAP=200
WORKSHOPS_AI_RAG__MAX_PAGES=0
WORKSHOPS_AI_RAG__UUID5_NAMESPACE=00000000-0000-5000-a000-000000000000






4) Run
python -m uvicorn workshops_ai.app.main:app --reload --host 127.0.0.1 --port 8000

Open: http://127.0.0.1:8000/demo

Chat button posts to /chat (LLM talks + returns proposals).

WS auto-connects to /ws and shows AgentEvent broadcasts (e.g., after create/update).


5) Data folders
./data/catalogs   # read-only catalogs (CSV)
./data/state      # org state (CSV)
./data/outbox     # (if configured) changeset exports

6) Troubleshooting

LLM offline / “I found 0 item(s)…” → The server does not see OPENAI_API_KEY. Export it in the SAME shell where you run Uvicorn (PowerShell: $env:OPENAI_API_KEY="sk-...").

CORS → set CORS_ORIGINS="*" (quoted).

WS close error on client leave → it’s harmless; we ignore sends after close.

No proposals → check your CSV paths (DATA_ROOT, CSV_*).