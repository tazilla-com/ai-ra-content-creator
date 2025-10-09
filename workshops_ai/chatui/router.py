# File: workshops_ai/chatui/router.py
# Purpose: Standalone demo UI router for WorkshopsAI.
# Exposes:
#   - GET /demo     → minimal HTML page to talk to /chat and /ws
#   - GET /         → redirects to /demo (when mounted at root)
#
# Notes:
# - Pure FastAPI; no templates or static deps. Safe to include in any app.
# - UI lets you set org_id, user_id, active_tab, and optional X-API-Key header.
# - WebSocket connects to /ws and prints incoming AgentEvent frames.
# - Keep this file UI-only; all auth/logic is handled by /chat and /ws.
#
# Usage (in main.py):
#   from workshops_ai.chatui.router import router as chatui_router
#   app.include_router(chatui_router, tags=["_demo"])

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect bare mount path to /demo."""
    return RedirectResponse(url="/demo", status_code=302)


@router.get("/demo", include_in_schema=False, response_class=HTMLResponse)
async def demo_page() -> str:
    """Simple HTML page to exercise /chat and /ws endpoints."""
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>WorkshopsAI — Demo Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <style>
    :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
    body { margin: 0; background: #0b1020; color: #e6edf3; }
    header { padding: 14px 16px; background: #101830; border-bottom: 1px solid #1d2750; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    header input, header select, header button { background: #0f1a38; color: #e6edf3; border: 1px solid #2a3a72; border-radius: 8px; padding: 8px 10px; }
    header input::placeholder { color: #93a1c6; }
    main { display: grid; grid-template-columns: 1fr 360px; gap: 12px; padding: 12px; height: calc(100vh - 64px); box-sizing: border-box; }
    .chat, .events { background: #0f1730; border: 1px solid #1d2750; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; }
    .chat-log { flex: 1; overflow: auto; padding: 12px; }
    .msg { margin: 8px 0; line-height: 1.4; }
    .msg.me { color: #c8e1ff; }
    .msg.ai { color: #a7f3d0; }
    .msg.sys { color: #facc15; }
    .input-row { display: flex; gap: 8px; padding: 10px; border-top: 1px solid #1d2750; }
    .input-row textarea { flex: 1; min-height: 44px; resize: vertical; background: #0f1a38; color: #e6edf3; border: 1px solid #2a3a72; border-radius: 8px; padding: 8px 10px; }
    .events-log { flex: 1; overflow: auto; padding: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }
    .pill { font-size: 12px; padding: 2px 8px; border-radius: 999px; border: 1px solid #334155; }
    a { color: #8ab4ff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; height: auto; }
      .events { min-height: 200px; }
    }
  </style>
</head>
<body>
  <header>
    <strong>WorkshopsAI Demo</strong>
    <span class="pill">/chat + /ws</span>
    <span style="margin-left:auto"></span>
    <label>org_id <input id="org" value="demo-org" size="10" /></label>
    <label>user_id <input id="user" value="demo-user" size="10" /></label>
    <label>tab
      <select id="tab">
        <option value="assets" selected>assets</option>
        <option value="impacts">impacts</option>
        <option value="threats">threats</option>
        <option value="controls">controls</option>
      </select>
    </label>
    <label>API key <input id="apiKey" value="" size="16" placeholder="X-API-Key (optional)" /></label>
    <button id="ws-btn">Connect WS</button>
    <a href="/docs" target="_blank" rel="noreferrer" style="margin-left:8px">/docs</a>
  </header>

  <main>
    <section class="chat">
      <div id="chat" class="chat-log"></div>
      <div class="input-row">
        <textarea id="msg" placeholder="Type and press Enter to send. Shift+Enter for newline. E.g., 'List primary Application software', or 'Create asset CRM as primary type Application software'"></textarea>
        <button id="send">Send</button>
      </div>
    </section>

    <aside class="events">
      <div class="events-log" id="events"></div>
    </aside>
  </main>

  <script>
    const chat = document.getElementById('chat');
    const eventsEl = document.getElementById('events');
    const sendBtn = document.getElementById('send');
    const wsBtn = document.getElementById('ws-btn');
    const org = document.getElementById('org');
    const user = document.getElementById('user');
    const tab = document.getElementById('tab');
    const msg = document.getElementById('msg');
    const apiKey = document.getElementById('apiKey');

    let ws = null;

    // Chat lines: text only. Any non-string is hinted and logged to Events.
    function line(kind, content) {
      const wrap = document.createElement('div');
      wrap.className = `msg ${kind}`;

      let text;
      if (content === null || content === undefined) {
        text = String(content);
      } else if (typeof content === 'string') {
        text = content;
      } else {
        text = '[JSON attached → see Events]';
        // Also mirror the raw object to the right panel for inspection
        try { ev({ type: 'chat_message_json', payload: content }); } catch {}
      }

      const el = document.createElement('div');
      el.textContent = text;
      wrap.appendChild(el);

      chat.appendChild(wrap);
      chat.scrollTop = chat.scrollHeight;
    }

    function ev(obj) {
      const pre = document.createElement('pre');
      pre.textContent = JSON.stringify(obj, null, 2);
      eventsEl.appendChild(pre);
      eventsEl.scrollTop = eventsEl.scrollHeight;
    }

    async function sendMessage() {
      const text = msg.value;
      if (!text.trim()) return;
      const payload = {
        org_id: org.value.trim(),
        active_tab: tab.value,
        selection: {},   // pass asset_id, etc. if needed
        message: text,
        staging: false,
        user_id: user.value.trim(),
      };
      line('me', payload.message || '(empty)');
      msg.value = '';

      const headers = { 'Content-Type': 'application/json' };
      const key = apiKey.value.trim();
      if (key) headers['X-API-Key'] = key;

      try {
        // Add api_key as query param as well (proxies/CORS tolerant)
        const chatUrl = new URL('/api/chat', window.location.origin);
        if (key) chatUrl.searchParams.set('api_key', key);

        const res = await fetch(chatUrl.toString(), {
          method: 'POST',
          headers,
          body: JSON.stringify(payload),
        });

        let data = null;
        const ct = res.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
          data = await res.json();
        } else {
          data = { message: await res.text() };
        }

        if (res.ok) {
          // Chat bubble shows only strings; JSONs go to Events
          (data.messages || []).forEach(item => {
            if (typeof item === 'string') {
              line('ai', item);
            } else {
              ev({ type: 'chat_message_json', payload: item });
              // Optional short hint in chat:
              // line('sys', '[JSON attached → see Events]');
            }
          });

          if (data.proposals) {
            line('sys', '[proposals attached]');
            ev({ type: 'proposals', proposals: data.proposals });
          }
          if (data.staging) {
            ev({ type: 'staging', staging: data.staging });
          }
          if (data.agent_events) {
            // Only in the right panel (no JSON in chat)
            ev({ type: 'agent_events', events: data.agent_events });
          }
        } else {
          line('sys', `ERROR ${res.status}: ${data?.message || 'Request failed'}`);
          ev({ type: 'error', data });
        }
      } catch (e) {
        line('sys', `Network error: ${e}`);
      }
    }

    function connectWs() {
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close(1000, 'reconnect');
      }
      const url = new URL(window.location.href);
      url.protocol = url.protocol.replace('http', 'ws'); // http → ws, https → wss
      url.pathname = '/ws';
      // Send both for compatibility (server commonly expects `tab`)
      url.searchParams.set('tab', tab.value);
      url.searchParams.set('active_tab', tab.value);
      url.searchParams.set('org_id', org.value.trim());
      url.searchParams.set('user_id', user.value.trim());
      const key = apiKey.value.trim();
      if (key) url.searchParams.set('api_key', key);

      ws = new WebSocket(url.toString());
      ws.onopen = () => { line('sys', 'WS connected'); wsBtn.textContent = 'Reconnect WS'; };
      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);
          // AgentEvent frames are raw JSON; chat frames may be wrapped {type, payload}
          if (data && typeof data === 'object') {
            if (data.type && data.payload) {
              // Structured chat/info frames from server
              ev({ ws_type: data.type, payload: data.payload });
              if (data.type === 'info' && data.payload && data.payload.message) {
                // Only plain text goes into the chat
                line('sys', data.payload.message);
              }
            } else {
              // Raw AgentEvent (asset.created / impact.set / etc.) → Events only
              ev(data);
              // No JSON echo into chat
            }
          }
        } catch (e) {
          ev({ parse_error: String(e), raw: evt.data });
        }
      };
      ws.onclose = (evt) => { line('sys', `WS closed (${evt.code})`); };
      ws.onerror = () => { line('sys', 'WS error'); };
    }

    // Click to send
    sendBtn.addEventListener('click', sendMessage);

    // ENTER to send, SHIFT+ENTER for newline
    msg.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });

    wsBtn.addEventListener('click', connectWs);
    window.addEventListener('load', () => {
      const url = new URL(location.href);
      const apiKeyEl = document.getElementById('apiKey');
      const initialKey = url.searchParams.get('api_key') || '';
      if (initialKey && apiKeyEl) apiKeyEl.value = initialKey; // prefill the input

      if (initialKey || (apiKeyEl && apiKeyEl.value.trim())) {
        connectWs();
      } else {
        line('sys', 'API key missing: add ?api_key=… to URL or fill the box and click Connect.');
      }
    });
    window.addEventListener('beforeunload', () => { if (ws) try { ws.close(1000, 'leaving'); } catch {} });
  </script>
</body>
</html>
    """
