# File: workshops_ai/chatui/router.py
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter()


@router.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/demo", status_code=302)


@router.get("/demo", include_in_schema=False, response_class=HTMLResponse)
async def demo_page() -> str:
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
    .msg { margin: 8px 0; line-height: 1.4; white-space: pre-wrap; }
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
    .kbd { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#0b1228; border:1px solid #233266; border-radius:6px; padding:1px 6px; }
  </style>
</head>
<body>
  <header>
    <strong>WorkshopsAI Demo</strong>
    <span class="pill">/chat + /ws</span>
    <span style="margin-left:auto"></span>
    <label>session_id <input id="sessionId" value="" placeholder="Auto-generated" size="24" /></label>
    <label>tab
      <select id="tab">
        <option value="assets" selected>assets</option>
        <option value="impacts">impacts</option>
        <option value="threats">threats</option>
      </select>
    </label>
    <label>API key <input id="apiKey" value="" size="16" placeholder="Leave empty (dev mode)" /></label>
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
    const sessionId = document.getElementById('sessionId');
    const tab = document.getElementById('tab');
    const msg = document.getElementById('msg');
    const apiKey = document.getElementById('apiKey');

    let ws = null;

    // Generate session ID
    function generateSessionId() {
      return 'demo_' + Math.random().toString(36).substring(2, 15);
    }

    function getOrGenerateSessionId() {
      let sid = sessionId.value.trim();
      if (!sid) {
        sid = generateSessionId();
        sessionId.value = sid;
      }
      return sid;
    }

    // --- Chat UI helpers ------------------------------------------------------
    function line(kind, content) {
      const wrap = document.createElement('div');
      wrap.className = 'msg ' + kind;
      const el = document.createElement('div');
      el.textContent = (content === null || content === undefined)
        ? String(content)
        : (typeof content === 'string' ? content : '[JSON attached → see Events]');
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

    function browserAcceptLanguage() {
      const nav = navigator.languages && navigator.languages.length
        ? navigator.languages.join(', ')
        : (navigator.language || 'en');
      return nav;
    }

    // --- Humanizers -----------------------------------------------------------
    function humanizeAssetList(payload) {
      try {
        const n = (payload && payload.count != null)
          ? payload.count
          : ((payload && payload.assets) ? payload.assets.length : 0);
        const names = ((payload && payload.assets) || []).map(a => a.name).filter(Boolean);
        const head = 'Assets (' + n + '):';
        if (!names.length) return head;
        const shown = names.slice(0, 12).join(', ');
        const more = names.length > 12 ? ' … +' + (names.length - 12) + ' more' : '';
        return head + ' ' + shown + more;
      } catch { return null; }
    }
    function humanizeChangesetQueued(payload) {
      const p = (payload && payload.outbox_path) || '(outbox)';
      return 'Changeset queued → ' + p;
    }
    function humanizeChangesetSubmitted(payload) {
      const n = (payload && payload.actions) || '?';
      const t = (payload && payload.transport) || 'unknown';
      return 'Changeset submitted (' + n + ') via ' + t + '.';
    }
    function humanizeAssetDetail(payload) {
      try {
        const a = (payload && payload.asset) || (payload && payload.preview) || payload;
        if (!a || typeof a !== 'object') return null;

        const id = a.id || a.asset_id || 'n/a';
        const name = a.name || 'n/a';
        const typ = a.assetType || 'n/a';
        const primary = (a.isPrimary === true || a.isPrimary === 'true');
        const primTxt = primary ? 'primary' : 'supporting';
        const c = (a.confidentiality != null ? a.confidentiality : (primary ? '-' : 'n/a'));
        const i = (a.integrity != null ? a.integrity : (primary ? '-' : 'n/a'));
        const av = (a.availability != null ? a.availability : (primary ? '-' : 'n/a'));
        const desc = a.description || '—';

        let lines = [];
        lines.push('Asset detail');
        lines.push('Name: ' + name);
        lines.push('Type: ' + typ);
        lines.push('Primary: ' + (primary ? 'Yes' : 'No'));
        lines.push('CIA: C=' + c + '  I=' + i + '  A=' + av);
        lines.push('Description: ' + desc);
        lines.push('ID: ' + id);
        return lines.join('\\n');
      } catch { return null; }
    }

    function renderMessageObject(obj, source) {
      source = source || 'http';
      const kind = (obj && (obj.kind || obj.type)) || null;

      if ((kind === 'text' || kind === 'llm.say') && obj && typeof obj.text === 'string') {
        line('ai', obj.text);
        ev({ type: source === 'ws' ? 'ws_message_json' : 'chat_message_json', payload: obj });
        if (Array.isArray(obj.messages)) {
          obj.messages.forEach(function(m) {
            if (typeof m === 'string') return line('ai', m);
            if (m && (m.text || m.message)) return line('ai', m.text || m.message);
            line('ai', JSON.stringify(m));
          });
        }
        return;
      }

      let text = null;
      if (kind === 'asset.list') text = humanizeAssetList(obj);
      if (kind === 'asset.get' || kind === 'asset.preview') text = humanizeAssetDetail(obj);
      if (kind === 'changeset.queued') text = humanizeChangesetQueued(obj);
      if (kind === 'changeset.submitted') text = humanizeChangesetSubmitted(obj);

      ev({ type: source === 'ws' ? 'ws_message_json' : 'chat_message_json', payload: obj });
      if (text) {
        line('ai', text);
      } else if (obj && obj.text) {
        line('ai', String(obj.text));
      }
    }

    // --- Send message via WebSocket -------------------------------------------
    function sendMessage() {
      const text = msg.value;
      if (!text.trim()) return;

      // Check if WebSocket is connected
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        line('sys', 'WebSocket not connected. Click "Connect WS" first.');
        return;
      }

      // Send via WebSocket
      const payload = {
        type: 'chat',
        payload: {
          message: text,
          assets: [],
          // No organization - agent will use scrape_company_info to get wizard_metadata.website_content
          model: 'gpt-5-mini'
        }
      };

      line('me', text);
      msg.value = '';

      try {
        ws.send(JSON.stringify(payload));
      } catch (e) {
        line('sys', 'Failed to send message: ' + e);
      }
    }

    // --- WebSocket /ws --------------------------------------------------------
    function connectWs() {
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close(1000, 'reconnect');
      }

      const sid = getOrGenerateSessionId();

      const url = new URL(window.location.href);
      url.protocol = url.protocol.replace('http', 'ws'); // http → ws, https → wss
      url.pathname = '/ws';
      url.searchParams.set('session_id', sid);  // NEW: session_id instead of org_id/user_id
      url.searchParams.set('workshop_id', tab.value);  // Fixed: workshop_id instead of active_tab
      const key = apiKey.value.trim();
      if (key) url.searchParams.set('api_key', key);

      ws = new WebSocket(url.toString());
      ws.onopen = function() { line('sys', 'WS connected'); wsBtn.textContent = 'Reconnect WS'; };
      ws.onmessage = function(evt) {
        try {
          const data = JSON.parse(evt.data);

          if (data && typeof data === 'object') {
            if (data.type && data.payload !== undefined) {
              ev({ ws_type: data.type, payload: data.payload });

              if (data.type === 'hello' && data.payload && data.payload.message) {
                line('ai', data.payload.message);
              }
              if (data.type === 'message' && data.payload && data.payload.content) {
                line('ai', data.payload.content);
              }
              if (data.type === 'assets_generated' && data.payload) {
                if (data.payload.message) {
                  line('ai', data.payload.message);
                }
                var action = data.payload.action || 'created';
                if (data.payload.assets && data.payload.assets.length > 0) {
                  var label = action === 'updated' ? 'Assets updated: ' : 'Assets created: ';
                  line('sys', label + data.payload.assets.length);
                  data.payload.assets.forEach(function(asset) {
                    line('sys', '  → ' + asset.name + ' (' + (asset.confidentiality || 'null') + '/' + (asset.integrity || 'null') + '/' + (asset.availability || 'null') + ')');
                  });
                }
                if (action === 'deleted' && data.payload.patches && data.payload.patches.length > 0) {
                  var removePatches = data.payload.patches.filter(function(p) { return p.op === 'remove'; });
                  if (removePatches.length > 0) {
                    line('sys', 'Assets deleted: ' + removePatches.length);
                  }
                }
              }
              if (data.type === 'progress' && data.payload) {
                line('sys', 'Status: ' + data.payload.status);
              }
              if (data.type === 'error' && data.payload && data.payload.message) {
                line('sys', 'ERROR: ' + data.payload.message);
              }
              if (data.type === 'info' && data.payload && data.payload.message) {
                line('sys', data.payload.message);
              }
              if (data.type === 'changeset.submitted') {
                line('sys', humanizeChangesetSubmitted(data.payload));
              }
              if (data.type === 'changeset.queued') {
                line('sys', humanizeChangesetQueued(data.payload));
              }

              if (data.type === 'chat_message_json' && data.payload) {
                renderMessageObject(data.payload, 'ws');
              }
              if (data.type === 'chat_message' && typeof data.payload === 'string') {
                line('ai', data.payload);
              }
            } else {
              ev(data);
            }
          }
        } catch (e) {
          ev({ parse_error: String(e), raw: evt.data });
        }
      };
      ws.onclose = function(evt) { line('sys', 'WS closed (' + evt.code + ')'); };
      ws.onerror = function() { line('sys', 'WS error'); };
    }

    // --- Wire UI --------------------------------------------------------------
    sendBtn.addEventListener('click', sendMessage);
    msg.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    wsBtn.addEventListener('click', connectWs);
    window.addEventListener('load', function() {
      const url = new URL(location.href);
      const apiKeyEl = document.getElementById('apiKey');
      const initialKey = url.searchParams.get('api_key') || '';
      if (initialKey && apiKeyEl) apiKeyEl.value = initialKey;

      // Auto-connect (no API key needed in dev mode with DISABLE_AUTH=true)
      line('sys', 'Welcome! Type a message or click "Connect WS" for real-time updates.');
      line('sys', 'DEV MODE: Auth disabled (DISABLE_AUTH=true)');

      // Auto-generate session ID
      if (!sessionId.value.trim()) {
        sessionId.value = generateSessionId();
        line('sys', 'Session ID: ' + sessionId.value);
      }
    });
    window.addEventListener('beforeunload', function() { if (ws) try { ws.close(1000, 'leaving'); } catch (e) {} });
  </script>
</body>
</html>
    """
