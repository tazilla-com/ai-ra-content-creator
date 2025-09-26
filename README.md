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
WORKSHOPS_AI_ENV=dev # prod or dev
WORKSHOPS_AI_VERSION=1.0.0
WORKSHOPS_AI_LOG_LEVEL=INFO

WORKSHOPS_AI_CORS_ORIGINS=["*"]

# Timeouts / limits 
WORKSHOPS_AI_CHAT_TIMEOUT_SEC=20
WORKSHOPS_AI_MAX_MESSAGE_LEN=4000
WORKSHOPS_AI_WS_TIMEOUT_SEC=75

# ======================== Security =========================
# API-key autentization (for HTTP and WS)
WORKSHOPS_AI_SECURITY__ENABLE_API_KEY=true
WORKSHOPS_AI_SECURITY__API_KEYS=["PKHYk5Fex7GNMOaDVqtowwebiYc9o3TfqNM4KROmVVp2p1VjcILtqpBhMSNi1YD7","waPxELBmZfoP0gVP8USh6OMWOdMn_D4iNRIa3j-lDhOGZWtg11IsmdH3Cqi1oW3a"]

# JWT autentization (HTTP Bearer)
WORKSHOPS_AI_SECURITY__ENABLE_JWT=true
WORKSHOPS_AI_SECURITY__JWT_ALGORITHM=HS256
WORKSHOPS_AI_SECURITY__JWT_SECRET=Oa5pK7qZ3wY8fT1nV4rH9mD2sC6xB0uP3jL7eQ1yN5gR8tW2kF6vA9zJ4iM0oU5
WORKSHOPS_AI_SECURITY__JWT_ISSUER=https://auth.example.com
WORKSHOPS_AI_SECURITY__JWT_AUDIENCE=workshops-ai

# ====================== Data roots (CSV) ===================
WORKSHOPS_AI_DATA_DIR=./data
WORKSHOPS_AI_CAT_DIR=./data/catalogs
WORKSHOPS_AI_STATE_DIR=./data/state

# ========== Export ChangeSets (Tazila) ==============
WORKSHOPS_AI_TAZILA__MODE=file   # file or http
WORKSHOPS_AI_TAZILA__OUTBOX_PATH=./data/outbox
WORKSHOPS_AI_TAZILA__ENDPOINT=https://backend.example.com/changesets
WORKSHOPS_AI_TAZILA__AUTH_TOKEN=change-me-bearer-token
WORKSHOPS_AI_TAZILA__TIMEOUT_SEC=30

# ====================== Redis / Ratelimit ==================
# Redis for sessions/ratelimit 
WORKSHOPS_AI_REDIS__URL=redis://redis:6379/0
WORKSHOPS_AI_RATELIMIT__ENABLED=true
WORKSHOPS_AI_RATELIMIT__PER_MINUTE=120

# ===================== FastAPI server ======================
APP_HOST=0.0.0.0
APP_PORT=8000
ENABLE_DOCS=false

# ==================== LLM / OpenAI =========================
OPENAI_API_KEY=
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.0

# ======== LangChain / LangSmith tracing (prod default off) ========
LANGCHAIN_TRACING_V2=false
# LANGCHAIN_ENDPOINT=   
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=workshops-ai



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