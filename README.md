# AI RA Content Creator

AI-powered **risk analysis (RA)** content generation. The service offers two ways to build a risk
analysis, over one shared session and one shared catalog of security knowledge.

**1. The wizard** — a batch, six-step REST flow. Point it at an organisation's website and it works
through the chain:

```
Website → Services → Assets → Impacts → Risks → Controls
```

Each step is an independent HTTP endpoint consuming the previous step's output.

**2. The Asset workshop** — a conversational agent over WebSocket. Instead of generating an asset
register in one shot, you talk to it: *"create assets for the IT department"*, *"mark the database as
high availability"*. It lists, creates, updates and deletes assets with CIA classification, grounded
in the wizard's services, the company's scraped website, and a knowledge base.

The two are **not separate products**. The workshop reads the services the wizard generated and
writes assets back into the same session, so you can start in either and finish in the other. Risks
and controls are selected from curated catalogs rather than invented by the model.

- **Version**: `1.0.0` (see [CHANGELOG.md](CHANGELOG.md))
- **Stack**: FastAPI · LangChain · LangGraph · Redis · OpenAI / Azure OpenAI · Langfuse
- **Docs**: [WebSocket API](docs/websocket-api.md) · live demo UI at `/demo`

## Features

- **Six-step wizard flow** — services, assets, impacts, risks, controls, plus a combined
  `/wizard/evaluation` that runs the whole chain in one call.
- **Conversational Asset workshop** — a LangGraph ReAct agent over WebSocket that edits the asset
  register through chat, returning RFC 6902 JSON Patches your client applies to its own state.
  Ships with a working demo UI at `/demo`.
- **Two LLM providers** — direct **OpenAI** or **Azure OpenAI**, selected by one env var.
- **Reasoning-effort control** — per-request `reasoning_effort` for reasoning models (`gpt-5*`, `o1`,
  `o3`, `o4`), with an env-level default.
- **Catalog-grounded output** — risks and controls are selected from curated catalogs
  (controls, threats, control×threat and asset-type×threat matrices), not invented by the model.
- **Web content retrieval** — traditional scraping (BeautifulSoup) or AI-powered web search.
- **Session management** — Redis-backed state across wizard steps, keyed by an `X-Session-Id` header.
- **Rate limiting & resilience** — per-model tokens-per-minute throttling, per-request token limits,
  and a circuit breaker around the LLM client.
- **Cost tracking** — token usage and USD cost per request, including recovery of usage from
  cancelled calls.
- **Observability** — Langfuse tracing and remote prompt management.
- **Multi-language output** — English, Slovak, Czech.

## Requirements

- **Python 3.11+**
- **Redis** — required in practice; wizard steps chain through it
- **An LLM key** — either an OpenAI API key or an Azure OpenAI deployment

## Quick start (Docker)

The fastest path. `docker-compose` starts two services: the API and Redis.

```bash
cp env.example .env
# edit .env — at minimum set your LLM key (see Configuration below)

docker compose up --build
```

The API is then on `http://localhost:8000`. Check it:

```bash
curl http://localhost:8000/health
```

## Quick start (local)

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -U pip
pip install -r requirements.txt

cp env.example .env      # then edit .env

uvicorn workshops_ai.app.main:app --host 0.0.0.0 --port 8000
```

Redis must be reachable at `WORKSHOPS_AI_REDIS__URL` (default `redis://localhost:6379/0`).

## Configuration

All settings are environment variables prefixed `WORKSHOPS_AI_`, with `__` marking nesting.
[`env.example`](env.example) is the complete annotated reference — the essentials are below.

### Choosing an LLM provider

The wizard supports **OpenAI** and **Azure OpenAI**. Pick one with `WORKSHOPS_AI_LLM__PROVIDER`.

**Option A — direct OpenAI:**

```bash
WORKSHOPS_AI_LLM__PROVIDER=openai
OPENAI_API_KEY=sk-your-openai-api-key-here
```

**Option B — Azure OpenAI:**

```bash
WORKSHOPS_AI_LLM__PROVIDER=azure
WORKSHOPS_AI_LLM__API_KEY=your-azure-openai-api-key
WORKSHOPS_AI_LLM__AZURE_ENDPOINT=https://your-resource.openai.azure.com/
WORKSHOPS_AI_LLM__AZURE_API_VERSION=2025-03-01-preview
```

> **Azure deployment names must match the model names.** The resolved model name is passed straight
> through as `azure_deployment`, so a deployment named `gpt-5-mini` is required for the default
> model to work.

### Wizard model settings

> **Note:** `WORKSHOPS_AI_LLM__MODEL` and `WORKSHOPS_AI_LLM__TEMPERATURE` configure the **chat/agent**
> path, **not** the wizard. The wizard has its own settings, below.

```bash
# Default model for all wizard steps
WORKSHOPS_AI_WIZARD__DEFAULT_MODEL=gpt-5-mini

# Reasoning effort: minimal | low | medium | high
# Cascade: request body > this env var > default ("medium")
WORKSHOPS_AI_WIZARD__REASONING_EFFORT=medium

# Only applies to NON-reasoning models — ignored for gpt-5*, o1, o3, o4
WORKSHOPS_AI_WIZARD__TEMPERATURE=0.2
```

**Allowed models:** `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.1-chat`, `gpt-5.4-mini`,
`gpt-5.4-nano`, `gpt-4.1`, `gpt-4o`, `gpt-4o-mini`. A model outside this list is rejected.

`minimal` reasoning effort is only supported by `gpt-5`, `gpt-5-mini` and `gpt-5-nano`;
`low`/`medium`/`high` work for all reasoning models.

### App, security and storage

```bash
WORKSHOPS_AI_ENV=production
WORKSHOPS_AI_LOG_LEVEL=INFO
WORKSHOPS_AI_APP_HOST=0.0.0.0
WORKSHOPS_AI_APP_PORT=8000
WORKSHOPS_AI_ENABLE_DOCS=false          # true exposes /docs

# CORS — "*" is rejected at startup unless WORKSHOPS_AI_ENV=dev
WORKSHOPS_AI_CORS_ORIGINS=["https://yourdomain.com"]

# API key auth
WORKSHOPS_AI_SECURITY__ENABLE_API_KEY=true
WORKSHOPS_AI_SECURITY__API_KEYS=["your-secure-api-key-here"]

# JWT auth (optional)
WORKSHOPS_AI_SECURITY__ENABLE_JWT=false

# Redis
WORKSHOPS_AI_REDIS__URL=redis://localhost:6379/0
WORKSHOPS_AI_WIZARD__SESSION_TTL_HOURS=8
```

### Web retrieval

```bash
WORKSHOPS_AI_WIZARD__WEB_SEARCH_ENABLED=true
WORKSHOPS_AI_WIZARD__WEB_SEARCH_MODEL=gpt-5-nano
WORKSHOPS_AI_WIZARD__WEB_SEARCH_TIMEOUT_SEC=120

WORKSHOPS_AI_WIZARD__SCRAPE_ENABLED=true
WORKSHOPS_AI_WIZARD__SCRAPE_TIMEOUT_SEC=10
WORKSHOPS_AI_WIZARD__SCRAPE_MAX_RETRIES=2
```

### Rate limiting

`WORKSHOPS_AI_WIZARD__OPENAI_CAPACITY_LIMITS` sets per-model tokens-per-minute capacity, matching
your deployment quotas. Calls that would exceed the rolling 60-second window wait for the window to
slide rather than failing. Per-request token limits are separate and raise a `RequestTooLargeError`
with a batch-split hint.

## Sessions

Wizard steps chain through a server-side session identified by the **`X-Session-Id`** header.

- Omit the header on your first call — the server generates an id and returns it in the
  `X-Session-Id` response header.
- Send that id on every subsequent call to chain the steps.
- State is stored in Redis and expires after `WORKSHOPS_AI_WIZARD__SESSION_TTL_HOURS` (default 8).

```bash
# 1. First call — no header; capture the returned id
curl -i -X POST http://localhost:8000/wizard/services \
  -H "X-API-Key: your-api-key" -H "Content-Type: application/json" \
  -d '{"website_url": "https://example.com", "language": "en"}'
# → response header: X-Session-Id: kJ8fQ2mNp4rT7vX1yZ3aBc

# 2. Subsequent calls — reuse it
curl -X POST http://localhost:8000/wizard/assets \
  -H "X-Session-Id: kJ8fQ2mNp4rT7vX1yZ3aBc" \
  -H "X-API-Key: your-api-key" -H "Content-Type: application/json" \
  -d '{"language": "en", "max_assets": 10}'
```

## API

All endpoints require authentication via an `X-API-Key` header or a JWT bearer token.

> **Unknown request fields are rejected with HTTP 422.** Wizard inputs are strict
> (`extra="forbid"`) — a typo in a field name is an error, not a silently ignored value.

Every wizard step accepts these optional fields:

| Field | Type | Description |
|---|---|---|
| `language` | `en` \| `sk` \| `cs` | Output language (default `en`) |
| `ai_model` | string | Model override for this call; must be in the allowed list |
| `reasoning_effort` | `minimal` \| `low` \| `medium` \| `high` | Reasoning-effort override |

### `POST /scrape/website`

```json
{ "url": "https://example.com", "search_method": "scrape" }
```

`search_method` is `scrape` (BeautifulSoup) or `web_search` (AI-powered). Returns the combined
content of the homepage plus services and about pages.

### `POST /wizard/services`

```json
{
  "website_url": "https://example.com",
  "search_method": "web_search",
  "language": "en",
  "min_services": 1,
  "max_services": 3
}
```

`min_services` / `max_services` are optional, range 1–10. `organization_profile` may be supplied for
better grounding.

**Response:**
```json
{
  "services": [
    {
      "name": "Cloud Hosting",
      "description": "Enterprise cloud infrastructure services",
      "sector": "Digital infrastructure"
    }
  ]
}
```

### `POST /wizard/assets`

```json
{ "language": "en", "min_assets": 5, "max_assets": 10 }
```

Optional, range 1–100. Assets are generated from the services and website content in the session.

### `POST /wizard/impacts`

```json
{ "language": "en" }
```

Reads assets from the session.

### `POST /wizard/risks`

```json
{ "language": "en" }
```

Reads assets and impacts from the session. Threats are selected from the threat catalog.

### `POST /wizard/controls`

```json
{ "language": "en", "max_controls_per_asset": 5 }
```

Reads assets and risks from the session. Controls are selected from the control catalog and enriched
with catalog metadata after generation.

### `POST /wizard/evaluation`

```json
{
  "website_url": "https://example.com",
  "search_method": "web_search",
  "language": "en",
  "min_services": 1,
  "max_services": 3,
  "min_assets": 5,
  "max_assets": 10,
  "max_controls_per_asset": 5
}
```

Runs all steps in sequence and returns the combined result.

### Session

```http
GET    /session      # current session state    (X-Session-Id header)
DELETE /session      # clear session state      (X-Session-Id header)
```

### Catalogs

`GET /api/catalogs/*` exposes the underlying reference data — controls, threats, impacts, sectors,
and the control×threat and asset-type×threat mappings.

### Health

```http
GET /health
```

## Asset workshop (chat agent)

The workshop is a **LangGraph ReAct agent** you talk to over a WebSocket. Rather than generating an
asset register in one batch call, you build it conversationally — and the agent edits the same
session the wizard uses.

**The full protocol is documented in [docs/websocket-api.md](docs/websocket-api.md).** In short:

```
ws(s)://<host>/ws?session_id=<id>&workshop_id=<assets|impacts|threats|controls>
```

Authenticated with the same credentials as the REST API — `X-API-Key` header, or `?api_key=` for
browser clients, which cannot set headers on a WebSocket handshake. A failed auth closes the socket
with code **4401** rather than returning an HTTP status.

Frames are `{"type": ..., "payload": {...}}` in both directions. You send `chat`, `ping`,
`sync_history` or `cancel`; the server sends `hello`, `progress`, `ui_response`, `history_sync`,
`pong` or `error`. Replies arrive whole — there is no token streaming — and carry **RFC 6902 JSON
Patches** to apply to your client-side asset state, alongside the Markdown reply, the tools used, and
token/cost usage.

### Try it

A working demo client ships at **`/demo`** (`/` redirects there): a chat pane plus a raw event
inspector, speaking the protocol over the WebSocket. With `docker compose up` running:

```
http://localhost:8000/demo
```

> `/` and `/demo` are **unauthenticated** and served at the application root. Consider whether to
> expose them in a deployed environment.

Its source in [workshops_ai/chatui/router.py](workshops_ai/chatui/router.py) is a compact reference
implementation of the client side.

### Agent tools

| Tool | Purpose |
|---|---|
| `list_assets` | Read the assets supplied by the client |
| `list_services` | Read the business services generated by the wizard |
| `create_assets` | Batch-create assets with CIA classification |
| `update_assets` | Partial batch update, matched by name |
| `delete_assets` | Batch delete by name |
| `scrape_company_info` | Scrape the company site; cached into the session |
| `lookup_knowledge` | RAG lookup over the internal knowledge base (no web search, by design) |

The system prompt is managed in Langfuse (`workshops/asset-agent-system`) with a local fallback.

### Shared session

The workshop and the wizard are two front-ends onto one session:

- Assets live in `wizard_steps["assets"]["assets"]` — the single source of truth for both halves.
- The agent reads the wizard's services from `wizard_steps["services"]["services"]`.
- Asset-type vocabulary comes from the same catalogs the wizard uses.

Note the session is carried differently on each half: the wizard uses an **`X-Session-Id` header**,
the WebSocket uses a **`session_id` query parameter**.

## Cost tracking

Every LLM operation reports usage and cost:

```json
{
  "debug": {
    "usage": {
      "prompt_tokens": 1234,
      "completion_tokens": 567,
      "total_tokens": 1801,
      "cost_usd": 0.00123,
      "model": "gpt-5-mini",
      "includes_recovered": false,
      "recovered_cost_usd": 0.0
    }
  }
}
```

If a client disconnects mid-call, the tokens Azure/OpenAI already consumed are recorded and rolled
into the next successful response — `includes_recovered` marks that, and `recovered_cost_usd` is
already included in `cost_usd`.

Controlled by `WORKSHOPS_AI_WIZARD__EXPOSE_DEBUG_INFO`.

## Observability

[Langfuse](https://langfuse.com) provides tracing and **remote prompt management** — wizard prompts
are fetched from Langfuse at runtime rather than being hard-coded. Traces include the resolved
`reasoning_effort` for each generator.

> Because prompts load remotely, changing a prompt in Langfuse changes behaviour without a code
> deploy. Prompt variables are part of the API contract — see [CHANGELOG.md](CHANGELOG.md).

## Architecture

| Concern | Technology |
|---|---|
| API framework | FastAPI |
| LLM orchestration | LangChain, LangGraph |
| LLM providers | OpenAI, Azure OpenAI |
| Session & state | Redis |
| Prompt management, tracing | Langfuse |
| Scraping | BeautifulSoup, lxml, publicsuffix2 |
| Validation | Pydantic, pydantic-settings |
| Resilience | Tenacity (retries), pybreaker (circuit breaker), pyrate-limiter |
| Token accounting | tiktoken |
| Logging | structlog |
| Auth | PyJWT |

### Scraper

```
workshops_ai/wizard/scraper/
├── core.py              # get_website() orchestration
├── http_client.py       # shared HTTP client with connection reuse
├── text_processing.py   # unicode normalisation, encoding fixes
├── sanitizer.py         # prompt-injection detection
├── validation.py        # content quality checks
├── link_extractor.py    # same-domain link extraction (PSL-based)
├── constants.py         # URL patterns for services/about pages
└── exceptions.py
```

Two retrieval methods:

1. **Traditional scraping** (`search_method="scrape"`) — BeautifulSoup HTML parsing across homepage,
   services and about pages, with sanitisation, validation and prompt-injection protection.
2. **AI web search** (`search_method="web_search"`) — model-driven, domain-filtered search.

### Catalogs

`workshops_ai/data/catalogs/` holds the reference data grounding the output: security controls,
threats, impact types, sectors, and weighted control×threat and asset-type×threat matrices.
Categories align with ISO/IEC 27002 domains. Exposed over `GET /api/catalogs/*`.

`WORKSHOPS_AI_CATALOG__MODE` selects the source:

- **`file`** (default) — read the bundled CSVs directly. No external dependency.
- **`http`** — fetch from the catalog service at `WORKSHOPS_AI_CATALOG__ENDPOINT`.

For local development against `http` mode without a real catalog service, a reference
implementation serving the bundled CSVs is included:

```bash
python scripts/mock_tazilla_api.py --port 9000

# point the app at it
WORKSHOPS_AI_CATALOG__MODE=http
WORKSHOPS_AI_CATALOG__ENDPOINT=http://localhost:9000
```

### Session flow

1. **Scrape website** → content cached in the session
2. **Services** → from cached website content
3. **Assets** → from services + website content
4. **Impacts** → from assets
5. **Risks** → from assets + impacts, grounded in the threat catalog
6. **Controls** → from assets + risks, grounded in the control catalog

Each step persists its result for the next.

## Security notes

- **`WORKSHOPS_AI_SECURITY__DISABLE_ALL_AUTH`** disables all authentication. It is a development
  convenience and must never be enabled in a deployed environment.
- **`WORKSHOPS_AI_CORS_ORIGINS=["*"]`** is rejected at startup unless `WORKSHOPS_AI_ENV=dev`.
- **`WORKSHOPS_AI_ENABLE_DOCS`** should stay `false` in production; it exposes `/docs`.
- Scraped content is sanitised for prompt-injection patterns before reaching the model.

## Tests

```bash
pytest workshops_ai/tests/
```

## Test & evidence harness

[`harness/`](harness/) is a separate, opt-in **live test harness** that drives the real six-step
wizard chain and the Asset-workshop agent against the configured provider and catalog, verifies the
results, and renders a testing/deployment evidence report. It runs as its own container (it is **not**
part of `docker compose up`) and is **config-driven** — it exercises whatever provider (OpenAI/Azure)
and catalog (`http`/`file`) the service is configured with, using real calls only (no mocked
responses).

```bash
# full evidence suite: normal operation, provider error, retry/fallback, user validation
docker compose -f docker-compose.yaml \
  -f harness/docker-compose.harness.yml run --rm harness
```

It also serves as an on-demand correctness check after changes. See
**[harness/README.md](harness/README.md)** for its purpose, all subcommands, configuration, and how
each result maps to the evidence appendix.

## License

Licensed under the [Apache License 2.0](LICENSE).
