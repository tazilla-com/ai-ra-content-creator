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
#     5) explain_impacts(...)        -> LLM explanation (optionally with catalog context)
#
# READ path
# - Uses search_catalogs.generic_search(...) to read Impact catalogs (robust CSV autodetect),
#   enriches with category mapping from "Impact x Impact category.csv".
# - Keeps EN codes canonical (derived from EN Name); tolerant to user input (name vs. code).
#
# WRITE path (SEND-NOW)
# - Optimistic WS broadcast to update UI quickly (schema-validated emitter when available).
# - Asynchronous ChangeSet submission via asubmit_changeset(...) (no commit step).
#
# Contracts
# - UIRequest/UIResponse in workshops_ai/schemas/ui_contract.py
# - WsHub.aemit_impact_set_level(payload) or generic broadcast("impact.set_level", payload)
# - tazila_client.asubmit_changeset(ChangeSet) — file|http per settings.tazila.mode
# - Impact models in workshops_ai/schemas/impact_models.py (strict)
#
# Python 3.11+, Pydantic v2, LangChain 0.3.x
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

# Outbox overlay (to reflect pending ChangeSets in reads)
import json
import os
import glob

# LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool

from workshops_ai.schemas.ui_contract import UIRequest, UIResponse, StagingState
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

# Read-only catalogs (robust CSV readers; zero writes)
from workshops_ai.tools import search_catalogs as cat

# Centralized i18n (supports "en", "sk", "cs"; we answer in the user's language)
from workshops_ai.i18n import t as i18n_t, lang_from as i18n_lang_from


# -----------------------------------------------------------------------------
# Constants / tuning
# -----------------------------------------------------------------------------

ALLOWED_LEVELS = {0, 1, 2, 3, 4}
MAX_BULK = 100                 # hard cap for safety
BULK_CONCURRENCY = 10          # parallel in-flight emits/submits

# Catalog filenames (relative to settings.paths.catalogs_dir)
IMPACT_FILE = "Impact.csv"
IMPACT_CAT_FILE = "Impact category.csv"
IMPACT_X_CAT_FILE = "Impact x Impact category.csv"


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
        self.retriever = retriever

        # Catalog caches (built from CSVs via search_catalogs.generic_search)
        self._impact_idx: dict[str, dict] | None = None  # code -> detail
        self._impact_codes: set[str] | None = None       # canonical codes

        # Last-request language (used by tool closures to localize responses)
        self._last_lang: str = "en"

        # Initialize LLM from settings when not provided — robust to both flat & nested settings
        if llm is not None:
            self.llm = llm
        else:
            s = self._settings
            try:
                model = getattr(s.llm, "model")                       # type: ignore[attr-defined]
                temperature = float(getattr(s.llm, "temperature"))    # type: ignore[attr-defined]
                timeout = int(getattr(s.llm, "call_timeout_sec"))     # type: ignore[attr-defined]
                max_retries = int(getattr(s.llm, "openai_max_retries"))  # type: ignore[attr-defined]
            except Exception:
                model = getattr(s, "llm_model", "gpt-4o-mini")
                temperature = float(getattr(s, "llm_temperature", 0.0))
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
        Invoke the LangChain tools agent. Returns UIResponse.

        Guard behavior:
        - Single-asset writes still require an asset_id (either in selection or inferred).
        - Bulk writes DO NOT require selection.asset_id (asset IDs are taken from the message).
        """
        self._assert_tab(req)
        lang = i18n_lang_from(language=getattr(req, "language", None), locale=None)
        self._last_lang = lang

        asset_id = self._asset_id_from_req(req)

        telemetry_log("impact_agent.in", {
            "org_id": req.org_id,
            "user_id": req.user_id,
            "asset_id": asset_id,
            "message": (req.message or "")[:512],
            "language": lang,
        })

        agent_input = {
            "input": req.message or "help",
            "org_id": req.org_id,
            "user_id": req.user_id or "anon",
            "asset_id": asset_id,
            "language": lang,
        }

        # Intent classifiers -------------------------------------------------
        def _is_change_intent(msg: str) -> bool:
            m = (msg or "").strip().lower()

            # Explicit explanations are not changes
            if re.search(r"^\s*(explain|why)\b", m):
                return False

            # Real change keywords
            if re.search(r"\b(set|update|change|bulk)\b", m):
                return True

            # Patterns like “to <0-4>” or “level <0-4>”
            if re.search(r"\b(to\s*[0-4]\b|\blevel\s*[0-4]\b)", m):
                return True

            return False

        def _is_bulk_intent(msg: str) -> bool:
            m = (msg or "").strip().lower()
            # Heuristics: presence of 'bulk' OR "for asset(s) X,Y,..." list
            if re.search(r"\bbulk\b", m):
                return True
            if re.search(r"\bfor\s+assets?\s+[a-z0-9._:-]+(?:\s*,\s*[a-z0-9._:-]+)+", m):
                return True
            return False

        # Guard: only block single-asset writes without asset_id; allow bulk without it
        if _is_change_intent(agent_input["input"]) and not _is_bulk_intent(agent_input["input"]) and not agent_input["asset_id"]:
            telemetry_log("impact_agent.guard.no_asset", {"intent": "single-change-like", "asset_id": None})
            return UIResponse(
                messages=[
                    i18n_t("impacts", "need_asset", lang=lang)
                    if i18n_t("impacts", "need_asset", lang=lang) != "impacts.need_asset"
                    else "Please select an asset (selection.asset_id) before changing impact levels."
                ],
                staging=StagingState(needs_commit=False),
            )

        try:
            # In LC 0.3 the agent executor returns a dict with 'output' (or a raw string)
            result = await self.agent.ainvoke(agent_input)
            if isinstance(result, dict) and "output" in result:
                text = str(result["output"] or "")
            else:
                text = str(result or "")
            if not text.strip():
                text = i18n_t("impacts", "fallback", lang=lang)

            telemetry_log("impact_agent.out", {"ok": True})
            return UIResponse(messages=[text], staging=StagingState(needs_commit=False))
        except Exception as e:
            telemetry_log("impact_agent.error", {"error": str(e)})
            return UIResponse(
                messages=[i18n_t("impacts", "error_generic", lang=lang), f"(details: {e})".strip()],
                staging=StagingState(needs_commit=False),
            )

    # -- Tools registry ----------------------------------------------------------

    def _build_tools(self) -> list[StructuredTool]:
        """
        Register only 'impacts' tools (tool-gating).
        All tool functions capture self (backend/ws/tazila_client).
        """

        async def _ensure_idx() -> dict[str, dict]:
            if self._impact_idx is None:
                self._impact_idx = self._load_impact_index()
                self._impact_codes = set(self._impact_idx.keys())
            return self._impact_idx

        # list_impacts -----------------------------------------------------------
        async def list_impacts_tool() -> list[dict]:
            """
            Return the impacts catalog (canonical EN codes) enriched with category (if mapped).
            Uses CSV catalogs via search_catalogs.generic_search(); tolerant headers/encodings.
            """
            idx = await _ensure_idx()
            items = [idx[c] for c in sorted(idx.keys())]  # stable order by code
            return [
                {
                    "code": it["code"],
                    "name": it["name"],
                    "category": it.get("category"),
                    "weight": it.get("weight"),
                    "levels": it.get("levels"),           # EN Level 0..4 (if present)
                    "description": it.get("description"), # EN Description (if present)
                }
                for it in items
            ]

        # asset_get_impacts ------------------------------------------------------
        async def asset_get_impacts_tool(asset_id: str) -> list[dict]:
            """
            Return current impacts for the given asset_id.
            Each item: {impact_code, level, note?}
            - Base is read from backend (CSV).
            - Overlay any pending ChangeSets in outbox (kind=impact.set_level); last wins.
            """
            if not asset_id:
                telemetry_log("impact_agent.get_impacts.no_asset", {})
                return []

            # 1) base from backend (best-effort)
            try:
                base = list(self.backend.asset_get_impacts(asset_id) or [])
            except Exception:
                base = []

            merged: dict[str, dict] = {}
            for it in base:
                code_key = ((it.get("impact_code") or "")).strip().lower()
                if not code_key:
                    continue
                lvl = it.get("level")
                if lvl is None:
                    lvl = it.get("level_index")
                merged[code_key] = {
                    "impact_code": it.get("impact_code"),
                    "level": lvl,
                    "note": it.get("note"),
                }

            # 2) overlay from outbox (SEND-NOW)
            try:
                outdir = getattr(self._settings.tazila, "outbox_path", "./data/outbox")
                files = sorted(glob.glob(os.path.join(outdir, "*.json")))
                for p in files:
                    try:
                        with open(p, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        if data.get("kind") != "impact.set_level":
                            continue
                        for it in data.get("items", []):
                            if (it.get("asset_id") or "") != asset_id:
                                continue
                            code_key = ((it.get("impact_code") or "")).strip().lower()
                            if not code_key:
                                continue
                            merged[code_key] = {
                                "impact_code": it.get("impact_code"),
                                "level": it.get("level_index"),
                                "note": it.get("note"),
                            }
                    except Exception:
                        continue
            except Exception:
                pass

            out = list(merged.values())
            out.sort(key=lambda x: (x.get("impact_code") or "").casefold())
            return out

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
            lang = self._last_lang
            try:
                if not asset_id:
                    return (
                        i18n_t("impacts", "need_asset", lang=lang)
                        if i18n_t("impacts", "need_asset", lang=lang) != "impacts.need_asset"
                        else "Missing asset_id."
                    )
                if level not in ALLOWED_LEVELS:
                    return i18n_t("impacts", "invalid_level", lang=lang, level=level)

                idx = await _ensure_idx()
                code = self._resolve_impact_code(impact_code, idx)
                if not code:
                    return i18n_t("impacts", "unknown_code", lang=lang, impact_code=impact_code)

                upd = ImpactUpdate(
                    org_id=org_id, user_id=user_id or "anon",
                    asset_id=asset_id, impact_code=code, level=level, note=note
                )
                await self._emit_and_submit(upd)
                telemetry_log("impact_agent.set.ok", {"asset_id": asset_id, "impact_code": code, "level": level})
                return i18n_t("impacts", "set_done", lang=lang, impact_code=code, level=level, asset_id=asset_id)
            except Exception as e:
                telemetry_log("impact_agent.set.error", {"error": str(e)})
                return i18n_t("impacts", "error_generic", lang=lang)

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
            lang = self._last_lang
            try:
                if level not in ALLOWED_LEVELS:
                    return i18n_t("impacts", "invalid_level", lang=lang, level=level)

                idx = await _ensure_idx()
                code = self._resolve_impact_code(impact_code, idx)
                if not code:
                    return i18n_t("impacts", "unknown_code", lang=lang, impact_code=impact_code)

                ids = list(asset_ids or [])
                capped = False
                if len(ids) > MAX_BULK:
                    ids = ids[:MAX_BULK]
                    capped = True

                sem = asyncio.Semaphore(BULK_CONCURRENCY)

                async def _one(aid: str) -> None:
                    async with sem:
                        upd = ImpactUpdate(
                            org_id=org_id, user_id=user_id or "anon",
                            asset_id=aid, impact_code=code, level=level, note=note
                        )
                        await self._emit_and_submit(upd)

                await asyncio.gather(*(_one(a) for a in ids))
                telemetry_log("impact_agent.bulk.ok", {"impact_code": code, "level": level, "count": len(ids)})

                msg = f"OK: {code}={level} on {len(ids)} assets"
                if capped:
                    msg += f" | {i18n_t('impacts', 'bulk_capped', lang=lang, n=MAX_BULK)}"
                return msg
            except Exception as e:
                telemetry_log("impact_agent.bulk.error", {"error": str(e)})
                return i18n_t("impacts", "error_generic", lang=lang)

        # explain_impacts --------------------------------------------------------
        async def explain_impacts_tool(
            prompt: str,
            org_id: str | None = None,
            asset_id: str | None = None,
        ) -> str:
            """
            LLM explanation with optional catalog context (Impact.csv + mappings).
            READ-ONLY (no changes).
            """
            sys = (
                "You are an impact analysis assistant. Use the internal 0..4 impact scale. "
                f"Map levels to {LEVEL_INDEX_TO_VALUE} when reasoning about semi-quantitative risk (R = P × D). "
                "Be concise and specific. If a methodology concept is unclear, be conservative."
            )
            try:
                # Build a small context window from the CSV catalogs
                idx = await _ensure_idx()
                ctx_lines: list[str] = []
                for c, it in list(idx.items())[:20]:  # keep context bounded
                    cat_label = it.get("category") or "Uncategorized"
                    levels = it.get("levels") or {}
                    lv = ", ".join(f"{k}:{v}" for k, v in levels.items()) if isinstance(levels, dict) else ""
                    ctx_lines.append(f"[{c}] {it.get('name')} | category={cat_label} | levels={lv} | weight={it.get('weight')}")
                catalog_context = "\n".join(ctx_lines)

                if self.retriever:
                    docs = await self.retriever.ainvoke({"query": prompt})
                    rag = "\n\n".join(getattr(d, "page_content", str(d)) for d in (docs or [])[:5])
                else:
                    rag = ""

                content = f"{sys}\n\nCatalog:\n{catalog_context}\n\n"
                if rag:
                    content += f"Context:\n{rag}\n\n"
                content += f"User:\n{prompt}"
                llm_resp = await self.llm.ainvoke(content)
                out = str(getattr(llm_resp, "content", llm_resp))
                telemetry_log("impact_agent.explain.ok", {"len": len(out or "")})
                return out
            except Exception as e:
                telemetry_log("impact_agent.explain.error", {"error": str(e)})
                return i18n_t("impacts", "error_generic", lang=self._last_lang)

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
                description="Explain/justify impact levels using catalog context (read-only).",
            ),
        ]

    def _build_agent(self, llm: ChatOpenAI, tools: list[StructuredTool]) -> AgentExecutor:
        """
        Build an OpenAI Tools agent with a guardrail prompt.
        - The agent MUST use only provided tools to read/change impacts.
        - No staging; changes must be applied immediately using set_impact_level / bulk_set_impact_levels.
        - IMPORTANT: include MessagesPlaceholder('agent_scratchpad') to avoid LC template errors.
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
                # (Optional) future chat history hook
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                ("system", "org_id={org_id} user_id={user_id} asset_id={asset_id} language={language}"),
                # Required so LC can inject tool traces
                MessagesPlaceholder("agent_scratchpad"),
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

    # -- Catalog helpers (read-only) --------------------------------------------

    def _load_impact_index(self) -> dict[str, dict]:
        """
        Build an impact index from Impact.csv + Impact x Impact category.csv:
          code (canonical), name, category?, weight?, levels{0..4}?, description?
        Tolerant to header variants (EN Name, Name, Description, EN Description, EN Level 0..4).
        """
        # Load primary catalog
        rows = cat.generic_search(IMPACT_FILE, q=None, limit=10_000)
        # Load category mapping
        map_rows = cat.generic_search(IMPACT_X_CAT_FILE, q=None, limit=10_000)
        # (Categories are optional for labels; still load for completeness)
        _ = cat.generic_search(IMPACT_CAT_FILE, q=None, limit=10_000)

        # Helper: find category label for a given EN impact name (best-effort)
        def _cat_of(impact_en_name: str) -> Optional[str]:
            if not map_rows:
                return None
            imp_norm = self._canon_code(impact_en_name)
            for mr in map_rows:
                imp_label = mr.get("Impact") or mr.get("Impact name") or mr.get("Name") or ""
                if self._canon_code(imp_label) == imp_norm:
                    return (mr.get("Category") or mr.get("category") or mr.get("EN Name") or mr.get("Name") or "").strip() or None
            return None

        idx: dict[str, dict] = {}
        for r in rows:
            name = (r.get("EN Name") or r.get("Name") or r.get("Impact") or "").strip()
            if not name:
                continue
            code = self._canon_code(name)
            description = (r.get("EN Description") or r.get("Description") or "").strip() or None
            weight = (r.get("Weight") or r.get("weight") or "").strip() or None
            levels = {
                "L0": (r.get("EN Level 0") or r.get("Level 0") or "").strip(),
                "L1": (r.get("EN Level 1") or r.get("Level 1") or "").strip(),
                "L2": (r.get("EN Level 2") or r.get("Level 2") or "").strip(),
                "L3": (r.get("EN Level 3") or r.get("Level 3") or "").strip(),
                "L4": (r.get("EN Level 4") or r.get("Level 4") or "").strip(),
            }
            # Trim empty level labels
            levels = {k: v for k, v in levels.items() if v}

            idx[code] = {
                "code": code,
                "name": name,
                "category": _cat_of(name),
                "weight": weight or None,
                "levels": levels or None,
                "description": description,
            }
        return idx

    def _resolve_impact_code(self, user_code_or_name: str, idx: dict[str, dict]) -> Optional[str]:
        """
        Accept either canonical code or human EN name; return canonical code if resolvable.
        """
        if not user_code_or_name:
            return None
        raw = user_code_or_name.strip()
        canon = self._canon_code(raw)
        if canon in idx:
            return canon
        # Name matching (case/space tolerant)
        for c, it in idx.items():
            if self._canon_code(it.get("name", "")) == canon:
                return c
        return None

    # -- Helpers ----------------------------------------------------------------

    @staticmethod
    def _canon_code(text: str) -> str:
        # Normalize to a stable, code-ish key (lower + single spaces)
        return re.sub(r"\s+", " ", (text or "").strip().lower())

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
    def _asset_id_from_req(req: UIRequest) -> Optional[str]:
        sel = getattr(req, "selection", None)
        if isinstance(sel, dict):
            return sel.get("asset_id")
        return getattr(sel, "asset_id", None)

    @staticmethod
    def _assert_tab(req: UIRequest) -> None:
        if (req.active_tab or "").lower() != "impacts":
            raise ValueError("ImpactAgent can only run on the 'impacts' tab.")

    @staticmethod
    def _mk_req(org_id: str, user_id: str) -> UIRequest:
        # Tiny helper (kept for backward compatibility).
        return UIRequest(
            org_id=org_id,
            user_id=user_id,
            active_tab="impacts",
            message="",
            selection={},
            staging=False,
            language="en",
        )
