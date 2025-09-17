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

# === LLM / OpenAI ===
OPENAI_API_KEY=sk-your-real-key
LLM_MODEL=gpt-4o-mini
OPENAI_TIMEOUT_SEC=60
OPENAI_MAX_RETRIES=3

# === Backend (CSV) ===
BACKEND_KIND=csv
DATA_ROOT=./data
CSV_CATALOG_DIR=./data/catalogs
CSV_STATE_DIR=./data/state
CSV_DELIMITER=;
CSV_ENCODING=utf-8

# === FastAPI ===
APP_HOST=127.0.0.1
APP_PORT=8000
CORS_ORIGINS="*"
LOG_LEVEL=INFO
WS_EVENT_TIMEOUT_S=75

# === LangChain / LangSmith (optional) ===
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2-...             # optional
LANGCHAIN_PROJECT=workshops-ai          # optional


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