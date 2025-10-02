# File: workshops_ai/graph/asset_graph.py
# -----------------------------------------------------------------------------
# Purpose
# LangGraph graph for the ASSETS tab (LLM-first NLU + SEND-NOW semantics).
#
# High-level flow
#   start
#    └─► nlu                   # extract action/slots from natural language (LLM → JSON)
#         └─► route            # infer action: list | get | create | set_meta | explain
#              ├─► assets_list           # read-only
#              ├─► asset_get             # read-only
#              ├─► explain               # read-only, RAG→LLM fallback
#              ├─► create_asset  ──► submit_changeset   # write (multi supported)
#              └─► set_meta      ──► submit_changeset   # write (multi supported)
#
# Output (UIResponse-like dict)
# {
#   "messages": ["assistant text..."],
#   "proposals": null | {...},
#   "staging": {"needs_commit": false, "batch_id": null, "diff": null},
#   "policy": {"emit_changes_enabled": true}
# }
#
# Notes
# - History/checkpoints are enabled by compiling with the injected checkpointer.
# - This graph keeps catalogs out of scope; lists/selects assets and performs "safe"
#   meta updates only (CIA/type/isPrimary).
# - ChangeSet payloads are built using the shared builder (multi-action).
#   When BE submit is unavailable, we fall back to writing JSON into outbox.
# - LLM is used ONLY for NLU (action+slots) and explain (RAG/LLM); write logic is deterministic.
# - Idempotence and WS optimistic emits are preserved.
# - IMPORTANT: All asset payloads returned in `state.result` are the canonical
#   `workshops_ai.api.schema.asset.Asset` DTO dumps.
# -----------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass, asdict, is_dataclass
from typing import Any, Callable, Literal, Optional, Sequence, cast

from pydantic import BaseModel, Field, TypeAdapter

from workshops_ai.graph.factory import GraphDeps
from workshops_ai.api.schema.asset import Asset  # ← canonical DTO

try:
    from langgraph.graph import StateGraph, END  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("langgraph must be installed to use asset_graph") from e

# Optional LLM (LangChain OpenAI); runtime-safe fallbacks
_HAS_LLM = True
try:
    from langchain_openai import ChatOpenAI  # type: ignore
    from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore
except Exception:  # pragma: no cover
    _HAS_LLM = False
    ChatOpenAI = object  # type: ignore
    SystemMessage = HumanMessage = object  # type: ignore

# ChangeSet builder (shared schema)
try:
    from workshops_ai.models.changeset import ChangeSet  # type: ignore
    from workshops_ai.agents.utils.changeset_builder import build_changeset  # type: ignore
except Exception:  # pragma: no cover
    ChangeSet = Any  # type: ignore

    def build_changeset(**kwargs):  # type: ignore
        # Fallback: return a dict that looks like a ChangeSet
        return {
            "id": kwargs.get("id") or str(uuid.uuid4()),
            "org_id": kwargs.get("org_id"),
            "user_id": kwargs.get("user_id"),
            "intent": kwargs.get("intent", "apply"),
            "actions": kwargs.get("actions", []),
            "audit": {"source": "AI", "timestamp": dt.datetime.now().isoformat()},
        }

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# State & helpers
# -----------------------------------------------------------------------------

Intent = Literal["list", "get", "create", "set_meta", "explain"]


class AssetState(BaseModel):
    # Routing context / inputs
    org_id: str
    active_tab: str = Field(default="assets")
    message: Optional[str] = None
    selection: Optional[dict] = None  # e.g., {"asset_id": "..."} or {"asset_ids": [...]} from UI selection
    user_id: Optional[str] = None
    locale: Optional[str] = None

    # Wizard / structured inputs (may be empty)
    action: Optional[Intent] = None
    new_asset: Optional[dict] = None   # single create seed when action == "create"
    update: Optional[dict] = None      # single update seed when action == "set_meta"

    # Working fields
    intent: Optional[Intent] = None
    # Multi-write staging (list of action dicts for the builder)
    actions: Optional[list[dict]] = None
    # Submission result/metadata
    change: Optional[Any] = None
    result: Optional[dict] = None


# -----------------------------------------------------------------------------
# UI + utility helpers
# -----------------------------------------------------------------------------

def _ui_response(text: str, *, extra: Optional[dict] = None) -> dict:
    """Build a minimal UIResponse-like payload; emit-only policy enforced."""
    base = {
        "messages": [text],
        "proposals": None,
        "staging": {"needs_commit": False, "batch_id": None, "diff": None},
        "policy": {"emit_changes_enabled": True},
    }
    if extra:
        base.update(extra)
    return base


def _safe_emit(ws_hub: Any, event: str, payload: dict) -> None:
    """Try to broadcast an optimistic WS event; ignore failures."""
    try:
        if hasattr(ws_hub, "emit"):
            ws_hub.emit(event, payload)
        elif hasattr(ws_hub, "broadcast"):
            ws_hub.broadcast(event, payload)
    except Exception as e:  # pragma: no cover
        logger.debug("WS emit failed (%s): %s", event, e)


async def _amemory_set(deps: GraphDeps, key: str, value: Any) -> None:
    func = getattr(deps, "memory_set", None)
    if callable(func):
        try:
            res = func(key, value)  # may return coroutine
            if asyncio.iscoroutine(res):
                await res  # type: ignore[func-returns-value]
        except Exception:
            pass


async def _amemory_get(deps: GraphDeps, key: str) -> Any:
    func = getattr(deps, "memory_get", None)
    if callable(func):
        try:
            res = func(key)  # may return coroutine
            return await res if asyncio.iscoroutine(res) else res  # type: ignore[return-value]
        except Exception:
            return None
    return None


def _schedule_coro(coro: "asyncio.Future[Any]" | "asyncio.coroutines"):  # type: ignore[name-defined]
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)  # fire-and-forget
    except RuntimeError:
        # no running loop (e.g., sync invoke in thread) — ignore
        pass


def _to_asset_dict(a: Any) -> dict:
    """Normalize any asset-like object to dict (legacy/internal helper)."""
    if a is None:
        return {}
    if isinstance(a, dict):
        return a
    # pydantic BaseModel
    try:
        from pydantic import BaseModel as _BM  # lazy import safe
        if isinstance(a, _BM):  # type: ignore[truthy-bool]
            return a.model_dump()
    except Exception:
        pass
    # dataclass
    if is_dataclass(a):
        try:
            return asdict(a)
        except Exception:
            pass
    # generic object with attributes
    for key in ("id", "asset_id", "name", "assetType", "isPrimary", "confidentiality", "integrity", "availability"):
        if hasattr(a, key):
            try:
                # build dict from known attrs
                return {
                    k: getattr(a, k, None)
                    for k in ("id", "asset_id", "name", "assetType", "isPrimary", "confidentiality", "integrity", "availability")
                }
            except Exception:
                break
    # treat as simple id-like
    try:
        s = str(a)
        return {"id": s}
    except Exception:
        return {}

# --- Canonical Asset DTO normalization (numbers → ISO labels) -----------------

_CONF_INDEX_TO_LABEL = {1: "PUBLIC", 2: "INTERNAL", 3: "PROTECTED", 4: "STRICTLY_PROTECTED"}
_IA_INDEX_TO_LABEL = {1: "LOW", 2: "MEDIUM", 3: "HIGH"}
TA_ASSET = TypeAdapter(Asset)

def _first_nonempty(*vals: Any) -> Any | None:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None

def _as_bool(v: Any) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return None

def _num_conf_to_label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return _CONF_INDEX_TO_LABEL.get(int(v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null"}:
            return None
        return s.upper().replace("-", "_")
    return None

def _num_ia_to_label(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return _IA_INDEX_TO_LABEL.get(int(v))
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() in {"none", "null"}:
            return None
        return s.upper()
    return None

def _to_asset_dto(item: Any, *, org_id: str | None = None) -> Asset:
    """
    Convert a backend item (dict or Pydantic model) into canonical `Asset`.
    Accepts multiple legacy key variants and numeric CIA scales and normalizes them.
    """
    data = item.model_dump() if hasattr(item, "model_dump") else dict(item)

    payload: dict[str, Any] = {}
    _org = _first_nonempty(org_id, data.get("org_id"))
    if _org is not None and str(_org).strip():
        payload["org_id"] = str(_org)
    # id
    _id = _first_nonempty(data.get("id"), data.get("asset_id"))
    if _id is not None:
        payload["id"] = str(_id)

    # name
    _name = _first_nonempty(data.get("name"), data.get("title"))
    if _name is not None and str(_name).strip():
        payload["name"] = str(_name)

    # description (optional)
    _desc = _first_nonempty(data.get("description"), data.get("desc"))
    if _desc is not None:
        payload["description"] = _desc

    # isPrimary (optional, coerce)
    _prim = _as_bool(_first_nonempty(data.get("isPrimary"), data.get("primary"), data.get("is_primary")))
    if _prim is not None:
        payload["isPrimary"] = _prim

    # assetType
    _atype = _first_nonempty(data.get("assetType"), data.get("type"), data.get("asset_type"))
    if _atype is not None and str(_atype).strip():
        payload["assetType"] = str(_atype)

    # CIA mappings (optional; numeric → labels)
    conf = _num_conf_to_label(_first_nonempty(data.get("confidentiality"), data.get("c")))
    if conf is not None:
        payload["confidentiality"] = conf

    integ = _num_ia_to_label(_first_nonempty(data.get("integrity"), data.get("i")))
    if integ is not None:
        payload["integrity"] = integ

    avail = _num_ia_to_label(_first_nonempty(data.get("availability"), data.get("a")))
    if avail is not None:
        payload["availability"] = avail

    # Validate against canonical DTO
    return TA_ASSET.validate_python(payload)

# -----------------------------------------------------------------------------
# ChangeSet submit + outbox fallback
# -----------------------------------------------------------------------------

async def _asubmit_changeset_or_outbox(deps: GraphDeps, cs: ChangeSet | dict) -> None:
    """
    Prefer async submit via app service if present; fallback to writing JSON into outbox.
    Accepts a ChangeSet model or a plain dict.
    """
    payload: dict
    if hasattr(cs, "model_dump"):  # pydantic model support
        payload = cs.model_dump()  # type: ignore[attr-defined]
    else:
        payload = cast(dict, cs)

    # Submit via service if available
    try:
        from workshops_ai.app.services import tazila_client  # type: ignore
    except Exception:
        tazila_client = None  # type: ignore

    try:
        if tazila_client is not None:
            if hasattr(tazila_client, "asubmit_changeset"):
                await tazila_client.asubmit_changeset(payload)  # type: ignore[arg-type]
                return
            if hasattr(tazila_client, "submit_changeset"):
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, tazila_client.submit_changeset, payload)  # type: ignore[arg-type]
                return
    except Exception as e:  # pragma: no cover
        logger.warning("Tazila submit failed, falling back to outbox: %s", e)

    # Fallback: write to outbox directory (settings-aware)
    settings = getattr(deps, "settings", None)
    outbox_dir = None
    try:
        outbox_dir = getattr(getattr(settings, "tazila", None), "outbox_path", None)
    except Exception:
        outbox_dir = None
    outbox_dir = outbox_dir or os.getenv("WORKSHOPS_AI_TAZILA__OUTBOX_PATH") or "/app/workshops_ai/data/outbox"

    try:
        os.makedirs(outbox_dir, exist_ok=True)
        cs_id = payload.get("id") or str(uuid.uuid4())
        path = os.path.join(outbox_dir, f"changeset_{cs_id}.json")

        def _write():
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

        await asyncio.to_thread(_write)
        logger.info("ChangeSet written to outbox: %s", path)
    except Exception as e:  # pragma: no cover
        logger.warning("Outbox write failed: %s", e)


# -----------------------------------------------------------------------------
# NLU (LLM-first) + regex fallback
# -----------------------------------------------------------------------------

def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    return t.lower()


def _get_llm_from_deps(deps: GraphDeps):
    llm = getattr(deps, "llm", None)
    if llm is not None:
        return llm
    if not _HAS_LLM:
        return None
    # Safe default
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, timeout=12, max_retries=1)  # type: ignore[call-arg]


class NLUResult(BaseModel):
    """
    Compatibility shim used inside asset_graph.py.

    - Keeps the original fields + sanitized() API for minimal diffs.
    - Adds from_shared_dict() to normalize/validate raw LLM JSON
      via shared AssetsNLU (workshops_ai.graph.nlu).
    """
    action: Optional[Intent] = None
    language: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    new_asset: Optional[dict] = None
    update: Optional[dict] = None
    notes: Optional[str] = None

    def sanitized(self) -> "NLUResult":
        # normalize action that may come as ["list"] etc.
        action = self.action
        if isinstance(action, list) and action:
            action = cast(Intent, str(action[0]).strip())

        def _only(d: Optional[dict], allowed: Sequence[str]) -> Optional[dict]:
            if not d:
                return None
            return {k: d.get(k) for k in allowed if k in d}

        na = _only(
            self.new_asset,
            ("id", "name", "assetType", "isPrimary", "confidentiality", "integrity", "availability"),
        )
        up = _only(
            self.update,
            ("asset_id", "assetType", "isPrimary", "confidentiality", "integrity", "availability"),
        )
        return NLUResult(
            action=cast(Optional[Intent], action),
            language=self.language,
            confidence=self.confidence,
            new_asset=na,
            update=up,
            notes=self.notes,
        )

    @classmethod
    def from_shared_dict(cls, data: dict, *, selection_asset_id: Optional[str] = None) -> "NLUResult":
        """
        Validate/normalize raw LLM JSON via the shared AssetsNLU schema
        without forcing callers to import it at module top-level.
        """
        # Lazy import to avoid hard dependency if graph is imported elsewhere
        from workshops_ai.graph.nlu import AssetsNLU, ensure_action_is_string  # type: ignore

        # 1) Fix occasional action=["list"] → "list"
        data = ensure_action_is_string(data)

        # 2) Strict validation against shared schema
        shared = AssetsNLU.model_validate(data)

        # 3) Extract compatible fields (keep unknowns out)
        action: Optional[Intent] = cast(Optional[Intent], getattr(shared, "action", None))
        language = getattr(shared, "language", None)

        # These may/not exist depending on the LLM output; pass-through safely.
        new_asset = cast(Optional[dict], getattr(shared, "new_asset", None))
        update = cast(Optional[dict], getattr(shared, "update", None)) or {}

        # If caller passed current UI selection, propagate it into update.asset_id
        if selection_asset_id and isinstance(update, dict) and not update.get("asset_id"):
            update["asset_id"] = str(selection_asset_id)

        inst = cls(
            action=action,
            language=language,
            confidence=None,   # shared model does not define confidence
            new_asset=new_asset,
            update=update,
            notes=getattr(shared, "notes", None),
        )
        return inst.sanitized()


CREATE_RX = re.compile(r"\b(create|new|add|vytvor|zaloz|pridej|pridaj)\b")
UPDATE_RX = re.compile(r"\b(update|set|change|zmen|nastav|uprav)\b")
GET_RX    = re.compile(r"\b(get|show|detail|ukaz|zobraz|co mi reknes|co mi povies|povedz|rekni)\b")
LIST_RX   = re.compile(r"\b(list|seznam|vypis|all assets)\b")
EXPLAIN_RX= re.compile(r"\b(explain|vysvetli|popis|povedz viac|řekni víc|rekni vic)\b")

CIA_RX = {
    "confidentiality": re.compile(r"\b(c|confidentiality)\s*([0-4])\b"),
    "integrity":       re.compile(r"\b(i|integrity)\s*([0-4])\b"),
    "availability":    re.compile(r"\b(a|availability)\s*([0-4])\b"),
}
PRIMARY_TRUE_RX  = re.compile(r"\b(primarni|hlavni|primary|make primary|oznac jako hlavni)\b")
PRIMARY_FALSE_RX = re.compile(r"\b(neprimarni|zrusit hlavni|unprimary|odebrat hlavni)\b")
TYPE_RX = re.compile(r"\b(typ|type)\s*[:=]?\s*([a-z0-9 _\-]+)")
NAME_QUOTED_RX = re.compile(r"[\"“”']\s*([^\"“”']+?)\s*[\"“”']")

_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)


async def _nlu(deps: GraphDeps, state: AssetState) -> AssetState:
    """LLM-first NLU → fills state.action/new_asset/update from natural language."""
    raw = (state.message or "").strip()
    if not raw:
        return state

    selection_id = (state.selection or {}).get("asset_id")
    llm = _get_llm_from_deps(deps)

    if llm is not None:
        system = (
            "You are a strict NLU for an enterprise 'assets' assistant. "
            "Extract intent and slots from user text in ANY language (CS/SK/EN; tolerate typos). "
            "Return ONLY a JSON object with fields: action (list|get|create|set_meta|explain), "
            "selection (optional with asset_id), new_asset (optional), update (optional).\n"
            "Rules:\n"
            "- NEVER invent IDs.\n"
            "- If selection.asset_id is provided, you MAY copy it into update.asset_id for set_meta/get/explain.\n"
            "- new_asset fields: id(optional), name(required for create), assetType, isPrimary (bool), "
            "  confidentiality, integrity, availability (0..4).\n"
            "- update fields: asset_id(optional), assetType, isPrimary (bool), confidentiality, integrity, availability (0..4).\n"
            "- If unsure, leave fields null. Output STRICT JSON, no prose."
        )
        human = {
            "message": raw,
            "selection": {"asset_id": selection_id} if selection_id else None,
            "hints": {
                "allowed_actions": ["list", "get", "create", "set_meta", "explain"],
                "cia_scale": [0, 1, 2, 3, 4],
                "safe_fields": ["assetType", "isPrimary", "confidentiality", "integrity", "availability"],
            },
        }
        try:
            resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=json.dumps(human))])  # type: ignore[arg-type]
            txt = (getattr(resp, "content", None) or "").strip()
            start = txt.find("{"); end = txt.rfind("}")
            data = json.loads(txt[start:end + 1]) if start != -1 and end != -1 else json.loads(txt)

            # Normalize & validate against the shared schema via the shim
            nlu_res = NLUResult.from_shared_dict(data, selection_asset_id=selection_id)

            if nlu_res.action and not state.action:
                state.action = cast(Intent, nlu_res.action)

            if nlu_res.new_asset and not state.new_asset:
                state.new_asset = nlu_res.new_asset

            maybe_update = dict(nlu_res.update or {})
            if selection_id and not maybe_update.get("asset_id"):
                maybe_update["asset_id"] = selection_id
            if maybe_update and not state.update:
                state.update = maybe_update

        except Exception as e:
            logger.warning("NLU LLM failed: %s", e)

    # Fallback regex — keeps things moving if LLM is missing/failing
    if not state.action:
        t = _norm(raw)
        if EXPLAIN_RX.search(t):
            state.action = "explain"
        elif CREATE_RX.search(t):
            state.action = "create"
        elif UPDATE_RX.search(t) or any(rx.search(t) for rx in CIA_RX.values()) or TYPE_RX.search(t):
            state.action = "set_meta"
        elif GET_RX.search(t):
            state.action = "get"
        elif LIST_RX.search(t):
            state.action = "list"

        # Minimal slot filling
        if state.action == "create" and not state.new_asset:
            na = {}
            m_name = NAME_QUOTED_RX.search(raw)
            if m_name:
                na["name"] = m_name.group(1).strip()
            m_type = TYPE_RX.search(t)
            if m_type:
                na["assetType"] = m_type.group(2).strip()
            if na:
                state.new_asset = na

        if state.action == "set_meta" and not state.update:
            fields: dict[str, Any] = {}
            for k, rx in CIA_RX.items():
                m = rx.search(t)
                if m:
                    fields[k] = int(m.group(2))
            m_type = TYPE_RX.search(t)
            if m_type:
                fields["assetType"] = m_type.group(2).strip()
            if PRIMARY_TRUE_RX.search(t):
                fields["isPrimary"] = True
            elif PRIMARY_FALSE_RX.search(t):
                fields["isPrimary"] = False
            if fields:
                if selection_id:
                    fields["asset_id"] = selection_id
                state.update = fields

    return state

# -----------------------------------------------------------------------------
# Intent node
# -----------------------------------------------------------------------------

def _infer_intent(state: AssetState) -> AssetState:
    """Infer intent from structured inputs or free-form message (post-NLU)."""
    if state.action in ("list", "get", "create", "set_meta", "explain"):
        state.intent = cast(Intent, state.action)
        return state

    text = (state.message or "").lower()
    if (state.selection or {}).get("asset_id") and re.search(r"\b(explain|detail|get|ukaz|zobraz|vysvetli)\b", text):
        state.intent = "explain" if re.search(r"\b(explain|vysvetli)\b", text) else "get"
        return state

    if re.search(r"\b(create|new|add)\b", text):
        state.intent = "create"
    elif re.search(r"\b(update|set|change|cia|confidentiality|integrity|availability|type)\b", text):
        state.intent = "set_meta"
    elif re.search(r"\b(explain|vysvetli|popis)\b", text):
        state.intent = "explain"
    elif re.search(r"\b(detail|show|get)\b", text):
        state.intent = "get"
    else:
        state.intent = "list"
    return state


# -----------------------------------------------------------------------------
# Read-only nodes (returning canonical Asset DTO dumps)
# -----------------------------------------------------------------------------

def _assets_list(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        try:
            items = deps.backend_port.assets_list(org_id=state.org_id)
        except Exception as e:
            state.result = _ui_response(f"Failed to list assets: {e}")
            return state

        items = items or []
        # Normalize to canonical DTOs
        dtos: list[Asset] = []
        for a in items:
            try:
                dtos.append(_to_asset_dto(a, org_id=state.org_id))
            except Exception as ex:
                logger.debug("Failed to normalize asset item: %s", ex)

        # Human preview
        lines = [f"Found {len(dtos)} assets."]
        preview = []
        for dto in dtos[:10]:
            d = dto.model_dump()
            nid = d.get("id") or "?"
            name = d.get("name") or "Unnamed"
            a_type = d.get("assetType") or "-"
            primary = " (primary)" if d.get("isPrimary") else ""
            preview.append(f"- {nid}: {name} [{a_type}]{primary}")
        if preview:
            lines += preview

        state.result = _ui_response(
            "\n".join(lines),
            extra={"assets": [a.model_dump() for a in dtos[:50]]},  # canonical DTO dumps
        )
        return state
    return _inner


def _asset_get(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        asset_id = (state.selection or {}).get("asset_id")
        if not asset_id:
            state.result = _ui_response("No asset selected. Provide selection.asset_id.")
            return state

        try:
            asset = deps.backend_port.asset_get(asset_id)
        except Exception as e:
            state.result = _ui_response(f"Failed to load asset '{asset_id}': {e}")
            return state

        if not asset:
            state.result = _ui_response(f"Asset '{asset_id}' not found.")
            return state

        try:
            dto = _to_asset_dto(asset, org_id=state.org_id)
        except Exception as ex:
            state.result = _ui_response(f"Asset normalization failed: {ex}")
            return state

        d = dto.model_dump()
        name = d.get("name") or asset_id
        a_type = d.get("assetType") or "-"
        cia = {"C": d.get("confidentiality"), "I": d.get("integrity"), "A": d.get("availability")}
        is_primary = bool(d.get("isPrimary"))
        note = f"Asset {asset_id}: {name} [{a_type}], CIA={cia}, primary={is_primary}"
        state.result = _ui_response(note, extra={"asset": d})  # canonical DTO dump
        return state
    return _inner


# -----------------------------------------------------------------------------
# Explain node (RAG → LLM fallback)
# -----------------------------------------------------------------------------

async def _asset_explain(deps: GraphDeps, state: AssetState) -> AssetState:
    # Determine target asset id
    asset_id = (state.selection or {}).get("asset_id") \
        or (state.update or {}).get("asset_id") \
        or (await _amemory_get(deps, "assets.last_hint") or {}).get("asset_id")

    asset_dto_dump: dict | None = None
    if asset_id:
        try:
            asset = deps.backend_port.asset_get(asset_id)
            if asset:
                asset_dto_dump = _to_asset_dto(asset).model_dump()
        except Exception:
            asset_dto_dump = None

    # Try to get retriever from deps (GraphDeps.make_retriever(org_id, tab))
    retriever = None
    try:
        if hasattr(deps, "make_retriever"):
            retriever = deps.make_retriever(state.org_id, "assets")  # type: ignore[arg-type]
    except Exception:
        retriever = None

    context_chunks: list[dict] = []
    if retriever is not None:
        try:
            query = "Explain the asset" + (f" {asset_dto_dump.get('name')}" if asset_dto_dump and asset_dto_dump.get("name") else "")
            if hasattr(retriever, "asearch"):
                docs = await retriever.asearch(query, k=4)  # type: ignore
            else:
                docs = retriever.search(query, k=4)  # type: ignore
            for d in (docs or [])[:5]:
                if isinstance(d, dict):
                    context_chunks.append(d)
                else:
                    context_chunks.append({"text": str(d)})
        except Exception as e:
            logger.debug("Retriever failed: %s", e)

    # Build explanation text
    llm = _get_llm_from_deps(deps) if _HAS_LLM else None
    if llm is not None:
        sys = (
            "You are a helpful security workshop assistant. Explain the selected asset to a non-expert stakeholder. "
            "Use the provided context if relevant; otherwise reason from the asset metadata. "
            "Keep it concise (5–8 short sentences). Avoid hallucinating unknown fields."
        )
        human_payload = {
            "asset": asset_dto_dump or {},
            "selection": (state.selection or {}),
            "org_id": state.org_id,
            "context": context_chunks or None,
            "question": state.message or "Explain this asset",
        }
        try:
            resp = await llm.ainvoke([SystemMessage(content=sys), HumanMessage(content=json.dumps(human_payload))])  # type: ignore
            text = (getattr(resp, "content", None) or "").strip()
            if text:
                state.result = _ui_response(text, extra={"asset": asset_dto_dump, "sources": context_chunks or None})
                return state
        except Exception as e:
            logger.debug("Explain LLM failed: %s", e)

    # Plain fallback without LLM
    if asset_dto_dump:
        name = asset_dto_dump.get("name") or "Unnamed asset"
        a_type = asset_dto_dump.get("assetType") or "-"
        c = asset_dto_dump.get("confidentiality"); i = asset_dto_dump.get("integrity"); a = asset_dto_dump.get("availability")
        cia_txt = ", ".join([f"C={c}", f"I={i}", f"A={a}"])
        text = f"{name} [{a_type}]. From a security perspective: {cia_txt}. A deeper explanation will be added once the knowledge base is available."
    else:
        text = "I need a specific asset selected (selection.asset_id) to explain it."
    state.result = _ui_response(text, extra={"asset": asset_dto_dump})
    return state


# -----------------------------------------------------------------------------
# Multi-write helpers
# -----------------------------------------------------------------------------

_ALLOWED_CREATE = {"id", "name", "assetType", "isPrimary", "confidentiality", "integrity", "availability"}
_ALLOWED_UPDATE = {"asset_id", "assetType", "isPrimary", "confidentiality", "integrity", "availability"}


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "item"


def _json_array_from_text(s: str) -> list[Any]:
    """Best-effort: try to parse a JSON array embedded anywhere in text."""
    s = (s or "").strip()
    if not s:
        return []
    # Fast path: whole text is JSON array
    try:
        data = json.loads(s)
        return data if isinstance(data, list) else []
    except Exception:
        pass
    # Slow scan: first bracket pair that parses
    try:
        start = s.find("[")
        end = s.rfind("]")
        if start != -1 and end != -1 and end > start:
            data = json.loads(s[start:end + 1])
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def _collect_selected_ids(selection: Optional[dict], message: Optional[str]) -> list[str]:
    """Collect asset ids from selection (asset_ids/assets) and message (UUIDs)."""
    ids: list[str] = []
    sel = selection or {}
    if sel.get("asset_id"):
        ids.append(str(sel["asset_id"]))
    if isinstance(sel.get("asset_ids"), (list, tuple)):
        ids += [str(x) for x in sel["asset_ids"] if x]
    if isinstance(sel.get("assets"), (list, tuple)):
        for a in sel["assets"]:
            try:
                ad = _to_asset_dict(a)
                aid = ad.get("id") or ad.get("asset_id")
                if aid:
                    ids.append(str(aid))
            except Exception:
                continue
    msg = message or ""
    ids += _UUID_RE.findall(msg)
    # Deduplicate, keep order
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def _derive_create_items(state: AssetState) -> list[dict]:
    """
    Derive a list of create items from:
    - JSON array in message (objects with name + optional fields),
    - quoted name list in message + fields from state.new_asset,
    - single new_asset from wizard.
    """
    items: list[dict] = []
    raw = state.message or ""

    # 1) JSON array in message
    arr = _json_array_from_text(raw)
    for it in arr:
        if not isinstance(it, dict):
            continue
        obj = {k: it.get(k) for k in _ALLOWED_CREATE if k in it}
        if (obj.get("name") or "").strip():
            items.append(obj)

    if items:
        return items

    # 2) Quoted names + shared fields from new_asset
    names = [m.strip() for m in NAME_QUOTED_RX.findall(raw) if m and m.strip()]
    if names:
        shared = {k: v for k, v in (state.new_asset or {}).items() if k in _ALLOWED_CREATE and k != "name"}
        for nm in names:
            obj = {"name": nm}
            obj.update(shared)
            items.append(obj)
        return items

    # 3) Single from wizard
    if state.new_asset:
        obj = {k: state.new_asset.get(k) for k in _ALLOWED_CREATE if k in state.new_asset}
        if (obj.get("name") or "").strip():
            items.append(obj)

    return items


def _derive_update_items(state: AssetState) -> list[dict]:
    """
    Derive a list of update items from:
    - JSON array in message (each item has asset_id + fields),
    - selection/message ids + shared fields from state.update.
    """
    items: list[dict] = []
    raw = state.message or ""

    # 1) JSON array with per-item payloads
    arr = _json_array_from_text(raw)
    for it in arr:
        if not isinstance(it, dict):
            continue
        aid = it.get("asset_id") or it.get("id")
        if not aid:
            continue
        fields = {k: it.get(k) for k in _ALLOWED_UPDATE if k in it and k != "asset_id"}
        fields = {k: v for k, v in fields.items() if v is not None}
        if fields:
            items.append({"asset_id": str(aid), **fields})

    if items:
        return items

    # 2) Collect id list from selection/message, apply shared fields from state.update
    ids = _collect_selected_ids(state.selection, state.message)
    shared_fields = {
        k: v for k, v in (state.update or {}).items()
        if k in _ALLOWED_UPDATE and k != "asset_id" and v is not None
    }
    for aid in ids:
        if shared_fields:
            items.append({"asset_id": aid, **shared_fields})

    # 3) Single update seed (if nothing else)
    if not items and state.update and state.update.get("asset_id"):
        fields = {k: v for k, v in state.update.items() if k in _ALLOWED_UPDATE and k != "asset_id" and v is not None}
        if fields:
            items.append({"asset_id": str(state.update["asset_id"]), **fields})

    return items


# -----------------------------------------------------------------------------
# Write nodes (create / set_meta) — SEND-NOW, multi supported
# -----------------------------------------------------------------------------

def _create_asset(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        # Derive list of create items
        create_items = _derive_create_items(state)
        if not create_items:
            state.result = _ui_response("Nothing to create. Provide at least a name for each asset.")
            return state

        actions: list[dict] = []
        for it in create_items:
            name = (it.get("name") or "").strip()
            if not name:
                continue
            fields = {
                "id": it.get("id") or None,  # server may assign id
                "name": name,
                "assetType": it.get("assetType"),
                "isPrimary": bool(it.get("isPrimary")) if it.get("isPrimary") is not None else None,
                "confidentiality": it.get("confidentiality"),
                "integrity": it.get("integrity"),
                "availability": it.get("availability"),
            }
            # Drop Nones for cleaner payloads
            fields = {k: v for k, v in fields.items() if v is not None}

            client_token = f"tmp-{_slug(name)}-{uuid.uuid4().hex[:8]}"
            action = {
                "op": "create_asset",
                "client_token": client_token,
                "fields": fields,
            }
            actions.append(action)

            # Optimistic WS emit per item
            emit_payload = {"org_id": state.org_id, **fields}
            _safe_emit(getattr(deps, "ws_hub", None), "asset.created", emit_payload)

        if not actions:
            state.result = _ui_response("Nothing to create after validation.")
            return state

        state.actions = actions
        # Lightweight hint for last item (for cross-tab follow-ups)
        last_name = create_items[-1].get("name")
        last_id = create_items[-1].get("id")
        _schedule_coro(_amemory_set(deps, "assets.last_hint", {"asset_name": last_name, "asset_id": last_id}))
        return state
    return _inner


def _set_meta(deps: GraphDeps) -> Callable[[AssetState], AssetState]:
    def _inner(state: AssetState) -> AssetState:
        update_items = _derive_update_items(state)
        if not update_items:
            state.result = _ui_response("Nothing to update. Provide target asset ids and at least one field.")
            return state

        actions: list[dict] = []
        for it in update_items:
            asset_id = it.get("asset_id")
            if not asset_id:
                continue
            fields = {k: v for k, v in it.items() if k in _ALLOWED_UPDATE and k != "asset_id" and v is not None}
            if not fields:
                continue
            action = {
                "op": "set_asset_meta",
                "asset_id": asset_id,
                "fields": fields,
            }
            actions.append(action)

            # Optimistic WS emit per item
            emit_payload = {"org_id": state.org_id, "asset_id": asset_id, **fields}
            _safe_emit(getattr(deps, "ws_hub", None), "asset.updated", emit_payload)

        if not actions:
            state.result = _ui_response("Nothing to update after validation.")
            return state

        state.actions = actions
        # Update lightweight hint to last affected asset
        _schedule_coro(_amemory_set(deps, "assets.last_hint", {"asset_id": update_items[-1].get("asset_id")}))
        return state
    return _inner


async def _submit_changeset_async(deps: GraphDeps, state: AssetState) -> AssetState:
    """Async submit node — builds a multi-action ChangeSet and submits it."""
    acts = state.actions or []
    if not acts:
        state.result = _ui_response("Nothing to submit.")
        return state

    # Build multi-action ChangeSet via shared builder
    cs: ChangeSet | dict = build_changeset(
        org_id=state.org_id,
        user_id=state.user_id,
        actions=acts,
        intent="apply",
    )
    state.change = cs

    # Submit (or outbox fallback)
    await _asubmit_changeset_or_outbox(deps, cs)

    # Batch memory hints (for cross-tab orchestration)
    created_names = [a.get("fields", {}).get("name") for a in acts if a.get("op") == "create_asset"]
    updated_ids = [a.get("asset_id") for a in acts if a.get("op") == "set_asset_meta"]
    batch_hint = {
        "created_names": [n for n in created_names if n],
        "updated_ids": [i for i in updated_ids if i],
        "count": len(acts),
        "timestamp": dt.datetime.utcnow().isoformat(),
    }
    _schedule_coro(_amemory_set(deps, "assets.last_batch", batch_hint))
    # Keep also the last_hint for quick references (prefer last updated id, else last created name)
    if updated_ids and updated_ids[-1]:
        _schedule_coro(_amemory_set(deps, "assets.last_hint", {"asset_id": updated_ids[-1]}))
    elif created_names and created_names[-1]:
        _schedule_coro(_amemory_set(deps, "assets.last_hint", {"asset_name": created_names[-1]}))

    # Human summary
    n_create = sum(1 for a in acts if a.get("op") == "create_asset")
    n_update = sum(1 for a in acts if a.get("op") == "set_asset_meta")
    parts = []
    if n_create:
        parts.append(f"{n_create} created")
    if n_update:
        parts.append(f"{n_update} updated")
    summary = " and ".join(parts) if parts else f"{len(acts)} actions submitted"

    state.result = _ui_response(
        f"{summary}.",
        extra={"changeset_actions": len(acts)}
    )
    return state


# -----------------------------------------------------------------------------
# Graph assembly
# -----------------------------------------------------------------------------

def create_graph(deps: GraphDeps):
    """
    Build and compile the Asset graph with an injected checkpointer.
    Supports: list | get | create | set_meta | explain
    Natural language → NLU (LLM) → deterministic actions (multi supported).
    """
    graph = StateGraph(AssetState)

    # Nodes
    async def _nlu_node(state: AssetState) -> AssetState:
        return await _nlu(deps, state)

    graph.add_node("nlu", _nlu_node)
    graph.add_node("route", _infer_intent)
    graph.add_node("assets_list", _assets_list(deps))
    graph.add_node("asset_get", _asset_get(deps))

    async def _explain_node(state: AssetState) -> AssetState:
        return await _asset_explain(deps, state)
    graph.add_node("explain", _explain_node)

    graph.add_node("create_asset", _create_asset(deps))
    graph.add_node("set_meta", _set_meta(deps))

    async def _submit_node(state: AssetState) -> AssetState:
        return await _submit_changeset_async(deps, state)
    graph.add_node("submit_changeset", _submit_node)

    # Entry
    graph.set_entry_point("nlu")
    graph.add_edge("nlu", "route")

    # Routing from intent
    def _route(state: AssetState) -> str:
        intent = state.intent or "list"
        if intent == "list":
            return "assets_list"
        if intent == "get":
            return "asset_get"
        if intent == "create":
            return "create_asset"
        if intent == "set_meta":
            return "set_meta"
        if intent == "explain":
            return "explain"
        return "assets_list"

    graph.add_conditional_edges(
        "route",
        _route,
        {
            "assets_list": "assets_list",
            "asset_get": "asset_get",
            "create_asset": "create_asset",
            "set_meta": "set_meta",
            "explain": "explain",
        },
    )

    # Write paths continue to async submit
    graph.add_edge("create_asset", "submit_changeset")
    graph.add_edge("set_meta", "submit_changeset")

    # Read-only nodes end
    graph.add_edge("assets_list", END)
    graph.add_edge("asset_get", END)
    graph.add_edge("explain", END)
    graph.add_edge("submit_changeset", END)

    # Compile with injected checkpointer for per-org/tab history
    return graph.compile(checkpointer=deps.checkpointer)


# -----------------------------------------------------------------------------
# (Optional) tiny struct used in list responses (kept local to this module)
# -----------------------------------------------------------------------------

@dataclass
class AssetPreview:
    id: str
    name: str
    assetType: str
    isPrimary: bool | None = None
    confidentiality: int | None = None
    integrity: int | None = None
    availability: int | None = None
