# File: workshops_ai/agents/impact_agent.py
# -----------------------------------------------------------------------------
# ImpactAgent — LangChain tools agent for the "impacts" tab (SEND-NOW semantics)
#
# What it does
# - Runs an OpenAI Tools agent (via LangChain) bound to the impacts domain only.
# - Exposes tools:
#     1) list_impacts()              -> read catalog (canonical EN codes)
#     2) asset_get_impacts(asset_id) -> read current impact levels for an asset
#     3) set_impact_level(...)       -> SEND-NOW: WS optimistic + async ChangeSet submit
#     4) bulk_set_impact_levels(...) -> SEND-NOW across multiple assets (guarded, concurrent)
#     5) explain_impacts(...)        -> LLM explanation (optionally with retriever, READ-ONLY)
# - Adds telemetry logs, graceful errors, idempotency (UUIDv5), and i18n (EN/SK with CZ→SK).
#
# Contracts
# - UIRequest/UIResponse in workshops_ai/schemas/ui_contract.py
# - WsHub.aemit_impact_set_level(payload) or generic broadcast("impact.set_level", payload)
# - tazila_client.asubmit_changeset(ChangeSet) — file|http per settings.tazila.mode
# - backend_port must provide: list_impacts(), asset_get_impacts(asset_id)
#
# Python 3.11+, Pydantic v2, LangChain 0.3.x
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal

# LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse
from workshops_ai.app.services.ws_hub import WsHub
from workshops_ai.app.services.tazila_client import asubmit_changeset
from workshops_ai.schemas.impact_models import (
    ImpactUpdate,
    ImpactChangeItem,
    ImpactChangeSet,
    WSImpactSetEvent,
    LEVEL_INDEX_TO_VALUE,
)
from workshops_ai.tools.telemetry import telemetry_log
from workshops_ai.settings import get_settings


# -----------------------------------------------------------------------------
# Constants / tuning
# -----------------------------------------------------------------------------

ALLOWED_LEVELS = {0, 1, 2, 3, 4}
MAX_BULK = 100                 # hard cap for safety
BULK_CONCURRENCY = 10          # parallel in-flight emits/submits


# -----------------------------------------------------------------------------
# Minimal i18n (EN + SK with CZ→SK mapping). We prefer req.language, fallback req.locale.
# -----------------------------------------------------------------------------

I18N = {
    "en": {
        "select_asset": "Please select an asset to show its impacts.",
        "no_impacts": "No impacts have been set for the selected asset yet.",
        "current_impacts_header": "Current impacts:",
        "impact_line": "- {impact_code}: {level}",
        "set_done": "Set {impact_code}={level} for asset {asset_id} (sent immediately).",
        "bulk_done_header": "Bulk changes (send-now):",
        "bulk_line": "- {code_eq_level} × {count}",
        "unknown_code": "Unknown impact_code: {impact_code}",
        "invalid_level": "Invalid level {level} (allowed 0..4).",
        "fallback": "I didn't understand the request on the Impacts tab.",
        "error_generic": "The request couldn't be processed. Please try again.",
        "bulk_capped": "Bulk request capped to {n} items for safety.",
        "explain_intro": "Rationale for selected impact levels:",
    },
    "sk": {
        "select_asset": "Vyberte, prosím, aktívum, pre ktoré mám zobraziť dopady.",
        "no_impacts": "Pre vybrané aktívum zatiaľ nie sú nastavené žiadne dopady.",
        "current_impacts_header": "Aktuálne dopady:",
        "impact_line": "- {impact_code}: {level}",
        "set_done": "Nastavil som {impact_code}={level} pre aktívum {asset_id} (odoslané okamžite).",
        "bulk_done_header": "Hromadné zmeny (send-now):",
        "bulk_line": "- {code_eq_level} × {count}",
        "unknown_code": "Neznámy impact_code: {impact_code}",
        "invalid_level": "Neplatná úroveň {level} (povolené 0..4).",
        "fallback": "Nerozumiem požiadavke na karte Dopady.",
        "error_generic": "Požiadavku sa nepodarilo spracovať. Skúste znova.",
        "bulk_capped": "Hromadný príkaz bol z bezpečnostných dôvodov skrátený na {n} položiek.",
        "explain_intro": "Odôvodnenie zvolených úrovní dopadov:",
    },
}

def _pick_lang(req: UIRequest) -> str:
    loc = (getattr(req, "language", None) or getattr(req, "locale", "") or "").lower()
    if loc.startswith(("sk", "cs")):
        return "sk"
    return "en"

def _t(req: UIRequest, key: str, **kwargs) -> str:
    lang = _pick_lang(req)
    return I18N.get(lang, I18N["en"])[key].format(**kwargs)


# -----------------------------------------------------------------------------
# Intent hint (optional; the LC agent mostly routes by itself)
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class IntentHint:
    kind: Literal["read", "set", "bulk", "explain"] | None = None


# -----------------------------------------------------------------------------
# ImpactAgent (LangChain tools)
# -----------------------------------------------------------------------------

class ImpactAgent:
    """
    LangChain tools agent for the 'impacts' tab (SEND-NOW).
    - Uses ChatOpenAI (from settings) + tools below.
    - No staging/diff/commit; each change is emitted and submitted immediately.
    """

    def __init__(
        self,
        backend_port: Any,
        ws_hub: Any | None = None,
        llm: ChatOpenAI | None = None,
        retriever: Any | None = None,  # optional RAG retriever for explain
    ) -> None:
        self.backend = backend_port
        self.ws_hub = ws_hub or WsHub
        self._settings = get_settings()
        self._impact_codes_cache: set[str] | None = None
        self.retriever = retriever

        # Initialize LLM from settings when not provided — robust to both flat & nested settings
        if llm is not None:
            self.llm = llm
        else:
            s = self._settings
            # Prefer nested `llm.*` if available, otherwise fall back to flat fields.
            try:
                model = getattr(s.llm, "model")              # type: ignore[attr-defined]
                temperature = float(getattr(s.llm, "temperature"))  # type: ignore[attr-defined]
                timeout = int(getattr(s.llm, "http_timeout_sec"))   # type: ignore[attr-defined]
                max_retries = int(getattr(s.llm, "openai_max_retries"))  # type: ignore[attr-defined]
            except Exception:
                model = getattr(s, "llm_model", "gpt-4o-mini")
                temperature = float(getattr(s, "llm_temperature", 0.1))
                timeout = int(getattr(s, "openai_timeout_sec", 60))
                max_retries = int(getattr(s, "openai_max_retries", 3))

            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
            )

        # Build tools registry and the agent
        self.tools = self._build_tools()
        self.agent = self._build_agent(self.llm, self.tools)

    # -- Public entrypoint -------------------------------------------------------

    async def run(self, req: UIRequest, hint: IntentHint | None = None) -> UIResponse:
        """
        Invoke the LangChain tools agent. Returns UIResponse with policy.emit_changes_enabled=True.
        """
        self._assert_tab(req)
        telemetry_log("impact_agent.in", {
            "org_id": req.org_id,
            "user_id": req.user_id,
            "asset_id": (req.selection or {}).get("asset_id"),
            "message": (req.message or "")[:512],
            "language": _pick_lang(req),
        })

        agent_input = {
            "input": req.message or "",
            "org_id": req.org_id,
            "user_id": req.user_id,
            "asset_id": (req.selection or {}).get("asset_id"),
            "language": _pick_lang(req),
        }

        try:
            result = await self.agent.ainvoke(agent_input)
            text = result if isinstance(result, str) else str(result)
            if not text.strip():
                text = _t(req, "fallback")
            resp = UIResponse(messages=[text], policy={"emit_changes_enabled": True})
            telemetry_log("impact_agent.out", {"ok": True})
            return resp
        except Exception as e:
            telemetry_log("impact_agent.error", {"error": str(e)})
            return UIResponse(messages=[_t(req, "error_generic")], policy={"emit_changes_enabled": True})

    # -- Tools registry ----------------------------------------------------------

    def _build_tools(self) -> list[StructuredTool]:
        """
        Register only 'impacts' tools (tool-gating).
        All tool functions capture self (backend/ws/tazila_client).
        """

        async def _codes() -> set[str]:
            if self._impact_codes_cache is None:
                catalog = await self.backend.list_impacts()
                self._impact_codes_cache = {self._canon_code(i["name"]) for i in catalog}
            return self._impact_codes_cache

        # list_impacts -----------------------------------------------------------
        async def list_impacts_tool() -> list[dict]:
            """Return the impacts catalog (canonical EN names/codes)."""
            items = await self.backend.list_impacts()
            return [{"code": self._canon_code(i["name"]), "name": i["name"], "category": i.get("category")} for i in items]

        # asset_get_impacts ------------------------------------------------------
        async def asset_get_impacts_tool(asset_id: str) -> list[dict]:
            """
            Return current impacts for the given asset_id.
            Each item: {impact_code, level, note?}
            """
            if not asset_id:
                return []
            return await self.backend.asset_get_impacts(asset_id)

        # set_impact_level (SEND-NOW) -------------------------------------------
        async def set_impact_level_tool(
            org_id: str,
            user_id: str,
            asset_id: str,
            impact_code: str,
            level: int,
            note: str | None = None,
        ) -> str:
            """Validate → WS optimistic → async submit to BE (send-now)."""
            try:
                if level not in ALLOWED_LEVELS:
                    return _t(self._mk_req(org_id, user_id), "invalid_level", level=level)

                code = self._canon_code(impact_code)
                if code not in await _codes():
                    return _t(self._mk_req(org_id, user_id), "unknown_code", impact_code=impact_code)

                upd = ImpactUpdate(
                    org_id=org_id, user_id=user_id,
                    asset_id=asset_id, impact_code=code, level=level, note=note
                )
                await self._emit_and_submit(upd)
                telemetry_log("impact_agent.set.ok", {"asset_id": asset_id, "impact_code": code, "level": level})
                return f"OK: {impact_code}={level} on {asset_id}"
            except Exception as e:
                telemetry_log("impact_agent.set.error", {"error": str(e)})
                return _t(self._mk_req(org_id, user_id), "error_generic")

        # bulk_set_impact_levels (SEND-NOW) -------------------------------------
        async def bulk_set_impact_levels_tool(
            org_id: str,
            user_id: str,
            asset_ids: list[str],
            impact_code: str,
            level: int,
            note: str | None = None,
        ) -> str:
            """Apply the same impact level to multiple assets (send-now, guarded concurrency)."""
            try:
                if level not in ALLOWED_LEVELS:
                    return _t(self._mk_req(org_id, user_id), "invalid_level", level=level)

                code = self._canon_code(impact_code)
                if code not in await _codes():
                    return _t(self._mk_req(org_id, user_id), "unknown_code", impact_code=impact_code)

                ids = list(asset_ids or [])
                capped = False
                if len(ids) > MAX_BULK:
                    ids = ids[:MAX_BULK]
                    capped = True

                sem = asyncio.Semaphore(BULK_CONCURRENCY)

                async def _one(aid: str) -> None:
                    async with sem:
                        upd = ImpactUpdate(
                            org_id=org_id, user_id=user_id,
                            asset_id=aid, impact_code=code, level=level, note=note
                        )
                        await self._emit_and_submit(upd)

                await asyncio.gather(*(_one(a) for a in ids))
                telemetry_log("impact_agent.bulk.ok", {"impact_code": code, "level": level, "count": len(ids)})

                msg = f"OK: {impact_code}={level} on {len(ids)} assets"
                if capped:
                    msg += f" | {_t(self._mk_req(org_id, user_id), 'bulk_capped', n=MAX_BULK)}"
                return msg
            except Exception as e:
                telemetry_log("impact_agent.bulk.error", {"error": str(e)})
                return _t(self._mk_req(org_id, user_id), "error_generic")

        # explain_impacts --------------------------------------------------------
        async def explain_impacts_tool(
            prompt: str,
            org_id: str | None = None,
            asset_id: str | None = None,
        ) -> str:
            """LLM-only explanation (optionally with retriever). READ-ONLY (no changes)."""
            sys = (
                "You are an impact analysis assistant. Use the internal 0..4 impact scale. "
                f"Map levels to {LEVEL_INDEX_TO_VALUE} when reasoning about semi-quantitative risk (R = P × D). "
                "Be concise and specific. If a methodology concept is unclear, be conservative."
            )
            try:
                if self.retriever:
                    docs = await self.retriever.ainvoke({"query": prompt})
                    context = "\n\n".join(getattr(d, "page_content", str(d)) for d in (docs or [])[:5])
                    content = f"{sys}\n\nContext:\n{context}\n\nUser:\n{prompt}"
                else:
                    content = f"{sys}\n\nUser:\n{prompt}"

                llm_resp = await self.llm.ainvoke(content)
                out = str(getattr(llm_resp, "content", llm_resp))
                telemetry_log("impact_agent.explain.ok", {"len": len(out or "")})
                return out
            except Exception as e:
                telemetry_log("impact_agent.explain.error", {"error": str(e)})
                return _t(self._mk_req(org_id or "", ""), "error_generic")

        # Wrap as StructuredTool (async)
        return [
            StructuredTool.from_function(
                coroutine=list_impacts_tool,
                name="list_impacts",
                description="List the impacts catalog with canonical EN codes.",
            ),
            StructuredTool.from_function(
                coroutine=asset_get_impacts_tool,
                name="asset_get_impacts",
                description="Get current impacts for a given asset_id.",
            ),
            StructuredTool.from_function(
                coroutine=set_impact_level_tool,
                name="set_impact_level",
                description="Set a single impact level for a specific asset (send-now, WS+submit).",
            ),
            StructuredTool.from_function(
                coroutine=bulk_set_impact_levels_tool,
                name="bulk_set_impact_levels",
                description="Set the same impact level for multiple assets (send-now, guarded concurrency).",
            ),
            StructuredTool.from_function(
                coroutine=explain_impacts_tool,
                name="explain_impacts",
                description="Explain/justify impact levels (read-only; may use RAG).",
            ),
        ]

    def _build_agent(self, llm: ChatOpenAI, tools: list[StructuredTool]) -> AgentExecutor:
        """
        Build an OpenAI Tools agent with a guardrail prompt.
        - The agent MUST use only provided tools to read/change impacts.
        - No staging; changes must be applied immediately using set_impact_level / bulk_set_impact_levels.
        """
        system = (
            "You are ImpactAgent, operating on the 'impacts' tab.\n"
            "- Use ONLY the provided tools. Do not invent APIs.\n"
            "- All writes MUST go through set_impact_level or bulk_set_impact_levels, "
            "which immediately emit WS updates and submit a ChangeSet to BE.\n"
            "- For reading, use list_impacts and asset_get_impacts.\n"
            "- For rationale, use explain_impacts (read-only).\n"
            "- Internal scale: 0..4. Also map values to 5/20/50/70/100.\n"
            "- Respond concisely in the user's language if provided."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
                ("system", "org_id={org_id} user_id={user_id} asset_id={asset_id} language={language}"),
            ]
        )

        agent = create_openai_tools_agent(llm, tools, prompt)
        return AgentExecutor(agent=agent, tools=tools, verbose=False)

    # -- Core send-now path ------------------------------------------------------

    async def _emit_and_submit(self, upd: ImpactUpdate) -> None:
        """
        1) WS optimistic event.
        2) Async submit ChangeSet (HTTP/file). Retry/backoff handled inside tazila_client.
        """
        level_value = LEVEL_INDEX_TO_VALUE[upd.level]

        # WS optimistic
        ws_payload = WSImpactSetEvent(
            org_id=upd.org_id,
            asset_id=upd.asset_id,
            impact_code=upd.impact_code,
            level_index=upd.level,
            level_value=level_value,
            note=upd.note,
            actor=upd.user_id,
            ts=self._now_iso(),
        ).model_dump()
        try:
            await self._ws_emit(ws_payload)
            telemetry_log("impact_agent.ws_emit", {"asset_id": upd.asset_id, "impact_code": upd.impact_code, "level": upd.level})
        except Exception as e:
            telemetry_log("impact_agent.ws_emit.error", {"error": str(e)})

        # Submit ChangeSet (fire-and-forget)
        item = ImpactChangeItem(
            asset_id=upd.asset_id,
            impact_code=upd.impact_code,
            level_index=upd.level,
            level_value=level_value,
            note=upd.note,
        )
        cs = ImpactChangeSet(
            org_id=upd.org_id,
            items=[item],
            idempotency_key=self._idem_key(upd),
            audit={"source": "AI", "user_id": upd.user_id, "ts": ws_payload["ts"]},
        )
        try:
            # Pass the Pydantic model directly (tazila_client handles BaseModel)
            asyncio.create_task(asubmit_changeset(cs))
        except Exception as e:
            telemetry_log("impact_agent.submit.schedule_error", {"error": str(e)})

    async def _ws_emit(self, payload: dict) -> None:
        """
        Prefer WsHub.aemit_impact_set_level(payload); fallback to broadcast.
        """
        hub = getattr(self.ws_hub, "get", None)
        if callable(hub):
            hub = hub()
        else:
            hub = self.ws_hub

        if hasattr(hub, "aemit_impact_set_level"):
            await hub.aemit_impact_set_level(payload)
            return
        if hasattr(hub, "broadcast"):
            await hub.broadcast("impact.set_level", payload)
            return
        telemetry_log("impact_agent.ws_emit.noop", {"reason": "WsHub has no emitter"})

    # -- Helpers -----------------------------------------------------------------

    @staticmethod
    def _canon_code(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).lower()

    @staticmethod
    def _idem_key(upd: ImpactUpdate) -> str:
        ns = uuid.UUID("00000000-0000-0000-0000-000000000001")
        raw = f"{upd.org_id}:{upd.asset_id}:{ImpactAgent._canon_code(upd.impact_code)}:{upd.level}"
        return str(uuid.uuid5(ns, raw))

    @staticmethod
    def _now_iso() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _assert_tab(req: UIRequest) -> None:
        if (req.active_tab or "").lower() != "impacts":
            raise ValueError("ImpactAgent can only run on the 'impacts' tab.")

    @staticmethod
    def _mk_req(org_id: str, user_id: str) -> UIRequest:
        # Tiny helper to reuse i18n strings in tool error paths without a full UIRequest.
        # We set language="en" as a neutral default; callers should pass real UIRequest for proper i18n.
        return UIRequest(
            org_id=org_id,
            user_id=user_id,
            active_tab="impacts",
            message="",
            selection={},
            staging=False,
            language="en",
        )
