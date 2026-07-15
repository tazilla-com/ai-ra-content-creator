# WebSocket API — Asset workshop chat agent

The Asset workshop is a conversational agent for building an organisation's information-asset
register. A client connects over a WebSocket, sends natural-language messages, and receives the
agent's reply together with structured changes to apply to its own asset state.

The agent shares the wizard's session. Assets it creates land in the same session the
[`/wizard/*`](../README.md#api) endpoints read and write, so the two halves of the API operate on one
source of truth.

This protocol is not described by the OpenAPI schema — WebSocket routes cannot be. This document is
the contract.

## Endpoint

```
ws(s)://<host>/ws?session_id=<id>&workshop_id=<id>
```

| Param | Required | Type | Description |
|---|---|---|---|
| `session_id` | yes | string | Session identifier. Reuse the same id to keep conversation context. Must match `^[a-zA-Z0-9_-]{1,40}$`. |
| `workshop_id` | yes | string | One of `assets`, `impacts`, `threats`, `controls`. |

Both are query parameters, not headers. Note this differs from the wizard endpoints, which carry the
session in an `X-Session-Id` **header**.

## Authentication

**The WebSocket is authenticated.** Auth is checked *before* the connection is accepted — a failure
closes the socket with code **4401** rather than returning an HTTP status.

Credentials are accepted as either:

- `X-API-Key: <key>` header, or
- `?api_key=<key>` query parameter.

Browsers cannot set custom headers on a WebSocket handshake, so browser clients must use the
`api_key` query parameter:

```
ws://localhost:8000/ws?session_id=demo_123&workshop_id=assets&api_key=<key>
```

> Passing a key in a query string exposes it to proxy and server logs. Prefer the header for
> server-to-server clients, and prefer short-lived keys for browser clients.

## Connection lifecycle

1. Open the WebSocket with `session_id` and `workshop_id` (and credentials).
2. Wait for the `hello` frame before sending anything.
3. Exchange JSON frames (below).
4. On disconnect, reconnect with the same `session_id` and send `sync_history` to restore context.

## Envelope

Every frame, in both directions, is:

```json
{ "type": "<string>", "payload": { } }
```

---

# 1. Client → server

Four message types are accepted: `chat`, `ping`, `sync_history`, `cancel`. Unknown types and unknown
payload fields are rejected — payloads are strict (`extra="forbid"`).

## 1.1 `chat` — send a user message

```json
{
    "type": "chat",
    "payload": {
        "message": "Vytvor assety pre IT oddelenie",
        "assets": [
            {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Existing Asset",
                "description": "Some description",
                "isPrimary": true,
                "confidentiality": "INTERNAL",
                "integrity": "MEDIUM",
                "availability": "HIGH",
                "assetType": "Server infrastructure",
                "selected": false
            }
        ],
        "organization": {
            "organizationName": "Acme Corporation",
            "legalForm": "s.r.o.",
            "country": "SK",
            "numberOfEmployees": 150,
            "organizationalUnits": [
                {"name": "IT oddelenie", "description": "Správa infraštruktúry"},
                {"name": "Financie", "description": "Účtovníctvo a reporting"}
            ],
            "jobPositions": [
                {"title": "IT Manager", "description": "Vedúci IT"},
                {"title": "Sieťový administrátor"}
            ]
        },
        "model": "gpt-5-mini"
    }
}
```

| Field | Required | Description |
|---|---|---|
| `message` | yes | User's message text. Must be non-empty after trimming. |
| `assets` | no | Current assets. Defaults to `[]` if omitted — but see below; send it. |
| `organization` | no | Organisation context. Omit if unavailable. |
| `model` | **yes** | Model name. Must be in the server's allowed-models list. |

**Always send your current assets.** The agent uses them to avoid creating duplicates and to target
updates at existing entries. Set `selected: true` on assets the user has explicitly selected.

**`model` is required on every message,** which means the model can change mid-conversation without
reconnecting.

## 1.2 `ping` — keepalive

```json
{ "type": "ping" }
```

Server replies with `pong`.

## 1.3 `sync_history` — request conversation history

```json
{ "type": "sync_history" }
```

Use after reconnecting. Server replies with `history_sync`.

## 1.4 `cancel` — stop the running request

```json
{ "type": "cancel" }
```

Cancels the in-flight agent task and clears the busy flag. No confirmation frame is sent.

---

# 2. Server → client

Six frame types are emitted: `hello`, `progress`, `ui_response`, `history_sync`, `pong`, `error`.

## 2.1 `hello` — connection established

Sent immediately after the connection is accepted.

```json
{
    "type": "hello",
    "payload": {
        "ok": true,
        "session_id": "abc123",
        "workshop_id": "assets",
        "message": "Hello! I'm ready to help with your assets."
    }
}
```

## 2.2 `progress` — agent working

Sent when an agent run starts and again when it finishes. Use it to drive a busy indicator.

```json
{ "type": "progress", "payload": { "status": "running", "session_id": "abc123" } }
```

`status` is `running` or `done`. There is no per-tool progress and no token streaming — the agent
replies in one `ui_response` when it is finished. Which tools ran is reported afterwards, in
`ui_response.tools_used`.

## 2.3 `ui_response` — agent response

Sent once the agent finishes.

```json
{
    "type": "ui_response",
    "payload": {
        "message": "Vytvoril som **4 assety**...",
        "action": "created",
        "patches": [
            {
                "op": "add",
                "path": "/assets/-",
                "value": {
                    "name": "Serverová infraštruktúra",
                    "description": "3 fyzické servery",
                    "isPrimary": true,
                    "confidentiality": "INTERNAL"
                }
            }
        ],
        "assets": [],
        "tools_used": ["create_assets"],
        "usage": {
            "prompt_tokens": 1250,
            "completion_tokens": 850,
            "total_tokens": 2100,
            "model": "gpt-5-mini",
            "cost_usd": 0.0525
        }
    }
}
```

| Field | Type | Description |
|---|---|---|
| `message` | string | Agent's reply, in Markdown. Render as rich text. |
| `patches` | array \| null | JSON Patch (RFC 6902) operations to apply to your asset state. |
| `assets` | array \| null | Full objects with reasoning. The field name tracks the workshop — `assets`, `impacts`, `threats`, `controls`. |
| `tools_used` | array \| null | Tools the agent called. For display or analytics. |
| `usage` | object \| null | Token usage and cost. |
| `action` | string \| null | `created`, `updated`, or `deleted`. Null when nothing changed — e.g. the agent only answered a question. |

## 2.4 `history_sync` — conversation history

Sent in response to `sync_history`.

```json
{
    "type": "history_sync",
    "payload": {
        "messages": [
            { "role": "user", "content": "Vytvor assety...", "timestamp": "2026-01-06T10:30:00Z" },
            { "role": "assistant", "content": "Vytvoril som 4 assety...", "timestamp": "2026-01-06T10:30:15Z", "assets": [] }
        ],
        "total_messages": 10,
        "returned_messages": 10
    }
}
```

History is capped at the **last 20 messages** — 10 user plus 10 assistant turns — and trimmed FIFO.

## 2.5 `pong`

```json
{ "type": "pong", "payload": {} }
```

## 2.6 `error`

```json
{
    "type": "error",
    "payload": {
        "message": "Agent is busy, please wait",
        "code": "BUSY",
        "details": "Previous request still processing"
    }
}
```

| Code | Meaning | Emitted |
|---|---|---|
| `BUSY` | A request is already running for this session | yes |
| `VALIDATION_ERROR` | Invalid message or payload | yes |
| `AGENT_ERROR` | Agent execution failed | yes |
| `RATE_LIMITED` | Too many requests | yes |
| `TIMEOUT` | Request took too long | see [Timeouts](#timeouts) |
| `SESSION_EXPIRED` | Session expired | **not currently emitted** |
| `TOOL_FAILED` | A tool failed | **not currently emitted** |

`SESSION_EXPIRED` and `TOOL_FAILED` are defined in the protocol but no code path emits them today.
Clients may handle them defensively; they should not be relied on.

---

# 3. Limits and timeouts

All values below are configurable. The defaults quoted are those in the shipped
`docker-compose.yaml` / `env.example`, which differ from the bare code defaults where noted.

## Rate limits

| Limit | Value | Scope | Behaviour |
|---|---|---|---|
| Requests per minute | 120 | per `session_id` | 121st gets `RATE_LIMITED` |
| Concurrent requests | 1 | per `session_id` | 2nd gets `BUSY` |

Set by `WORKSHOPS_AI_RATELIMIT__ENABLED=true` and `WORKSHOPS_AI_RATELIMIT__PER_MINUTE=120`.

> The **code** defaults are `enabled=False` and `per_minute=60`. Rate limiting is therefore *off*
> unless enabled by configuration. The shipped compose file and `env.example` both enable it and set
> 120, so a standard deployment gets the values above — but a bare clone running without them does
> not.

## Timeouts

`WORKSHOPS_AI_WS__TIMEOUT_SEC` defaults to **75** seconds and is reported in the `TIMEOUT` error
message.

> **Known gap:** there is currently no timeout enforced around the agent run at the WebSocket layer —
> the task is awaited without a deadline. In practice the effective limit is the LLM client's own
> timeout (`WORKSHOPS_AI_WIZARD__LLM_TIMEOUT_SEC`), not this value. A long agent run may therefore
> exceed 75 seconds without producing a `TIMEOUT` frame.

## Size limits

| Limit | Value |
|---|---|
| Conversation history | last 20 messages (10 turns), FIFO |

`message` has no server-side maximum length beyond the WebSocket frame limits of the deployment's
proxy and ASGI server.

---

# 4. Catalog integration

## `GET /api/catalogs/scraped-content`

Returns website content previously scraped in a session, so the agent can reuse it instead of
re-fetching. When the service runs with `WORKSHOPS_AI_CATALOG__MODE=http`, this endpoint is expected
to be provided by the configured catalog backend at `WORKSHOPS_AI_CATALOG__ENDPOINT`; the definition
below is the contract that backend must satisfy. With the default `MODE=file`, catalogs are read
directly from the CSVs in `workshops_ai/data/catalogs/` instead.

**Query parameters:**

| Param | Required | Type | Description |
|---|---|---|---|
| `session_id` | yes | string | Session identifier |

**Response `200`:**

```json
{
  "url": "https://example.com",
  "content": "Combined scraped content from all pages or company profile"
}
```

Both fields are `null` when nothing has been scraped for the session yet.

A reference implementation of the full catalog backend, serving this and the other
`/catalogs/*` endpoints from the bundled CSVs, is included at
[`scripts/mock_tazilla_api.py`](../scripts/mock_tazilla_api.py):

```bash
python scripts/mock_tazilla_api.py --port 9000
# then point the app at it:
#   WORKSHOPS_AI_CATALOG__MODE=http
#   WORKSHOPS_AI_CATALOG__ENDPOINT=http://localhost:9000
```

---

# 5. Not yet implemented

Declared in the protocol schema but not emitted by any current code path. Do not build against these
without checking the server first:

| Item | Status |
|---|---|
| `tool_status` frame (per-tool `started`/`completed`/`failed`) | `ToolStatusPayload` is defined and the type is in the protocol Literal, but no code path sends it. Use `progress` plus `ui_response.tools_used` instead. |
| `SESSION_EXPIRED`, `TOOL_FAILED` error codes | Defined, never emitted. |
| WebSocket-level request timeout | Configurable but not enforced — see [Timeouts](#timeouts). |

---

# 6. Try it

A working demo client ships with the service at **`/demo`** (redirected from `/`). It is a
single-page UI with a chat pane and a raw event inspector, and it speaks this protocol over the
WebSocket. Read its source at [`workshops_ai/chatui/router.py`](../workshops_ai/chatui/router.py) for
a minimal reference implementation.

> `/` and `/demo` are **not** authenticated and are served at the application root. Consider whether
> to expose them in a deployed environment.
