"""Correctness verifiers for harness outputs.

Pure functions with no provider dependency (so they can be unit-tested offline).
Each verifier returns a list of check dicts: ``{"name", "passed", "detail"}``.
They are intentionally lenient about exact field names (using ``model_dump`` and
tolerant key lookups) so they assert real invariants without breaking on benign
schema variation.
"""
from __future__ import annotations

from typing import Any

_CONFIDENTIALITY = {"PUBLIC", "INTERNAL", "PROTECTED", "STRICTLY_PROTECTED"}
_LEVEL = {"LOW", "MEDIUM", "HIGH"}
_AGENT_TOOLS = {
    "list_assets",
    "list_services",
    "create_assets",
    "update_assets",
    "delete_assets",
    "scrape_company_info",
    "lookup_knowledge",
}


def _dump(obj: Any) -> dict:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            return obj.model_dump()
    return obj if isinstance(obj, dict) else {}


def check(name: str, passed: bool, detail: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _first_list(d: dict, keys: tuple[str, ...]) -> list:
    for k in keys:
        v = d.get(k)
        if isinstance(v, list):
            return v
    return []


def verify_services(services: list) -> list[dict]:
    dumps = [_dump(s) for s in services]
    return [
        check("non_empty", len(dumps) > 0, f"{len(dumps)} services"),
        check("all_have_name", bool(dumps) and all(d.get("name") for d in dumps)),
    ]


def verify_assets(assets: list) -> list[dict]:
    dumps = [_dump(a) for a in assets]
    cia_ok = all(
        d.get("confidentiality") in (None, *_CONFIDENTIALITY)
        and d.get("integrity") in (None, *_LEVEL)
        and d.get("availability") in (None, *_LEVEL)
        for d in dumps
    )
    return [
        check("non_empty", len(dumps) > 0, f"{len(dumps)} assets"),
        check("all_have_name", bool(dumps) and all(d.get("name") for d in dumps)),
        check("all_have_assetType", bool(dumps) and all(d.get("assetType") for d in dumps)),
        check("cia_values_valid", cia_ok),
    ]


def verify_impacts(asset_impacts: list) -> list[dict]:
    dumps = [_dump(x) for x in asset_impacts]
    with_impacts = all(
        len(_first_list(d, ("impacts", "assetImpacts", "impact_list"))) > 0 for d in dumps
    )
    return [
        check("non_empty", len(dumps) > 0, f"{len(dumps)} asset-impact groups"),
        check("each_group_has_impacts", bool(dumps) and with_impacts),
    ]


def verify_risks(asset_risks: list) -> list[dict]:
    dumps = [_dump(x) for x in asset_risks]
    total = 0
    valid = True
    bad = ""
    for d in dumps:
        for r in _first_list(d, ("risks", "assetRisks", "threats")):
            rd = r if isinstance(r, dict) else _dump(r)
            total += 1
            if not rd.get("threatName"):
                valid, bad = False, "missing threatName"
            p = rd.get("probability")
            if p is not None:
                try:
                    # probability is a qualitative likelihood LEVEL on a 0..4 integer scale
                    if not 0 <= float(p) <= 4:
                        valid, bad = False, f"probability out of 0..4: {p}"
                except (TypeError, ValueError):
                    valid, bad = False, f"non-numeric probability: {p!r}"
    return [
        check("non_empty", len(dumps) > 0, f"{len(dumps)} asset-risk groups"),
        check("has_risks", total > 0, f"{total} risks"),
        check("threatName_and_probability_valid", total > 0 and valid, bad or "0..4 likelihood scale"),
    ]


def verify_controls(asset_controls: list, catalog_control_names: set | None = None) -> list[dict]:
    dumps = [_dump(x) for x in asset_controls]
    total = 0
    named = True
    in_catalog = True
    enriched = False
    for d in dumps:
        for c in _first_list(d, ("controls",)):
            cd = c if isinstance(c, dict) else _dump(c)
            total += 1
            name = cd.get("name")
            if not name:
                named = False
            if catalog_control_names is not None and name not in catalog_control_names:
                in_catalog = False
            if cd.get("category") or cd.get("id"):
                enriched = True
    checks = [
        check("non_empty", len(dumps) > 0, f"{len(dumps)} asset-control groups"),
        check("each_control_named", total > 0 and named, f"{total} controls"),
        check("enriched_from_catalog", enriched, "category/id populated post-generation"),
    ]
    if catalog_control_names is not None:
        checks.append(check("names_in_catalog", total > 0 and in_catalog, "controls selected from catalog"))
    return checks


def verify_agent(result: dict) -> list[dict]:
    """User-validation / human-in-the-loop checks.

    The agent returns *proposals* (generated assets / RFC 6902 patches); it has no
    write path to the Tazilla system of record. We assert it produced a response,
    did not error, and used only known, non-writing tools.
    """
    generated = result.get("generated_assets") or []
    deleted = result.get("deleted_asset_names") or []
    error = result.get("error")
    tool_names = [
        getattr(m, "name", None)
        for m in result.get("messages", [])
        if type(m).__name__ == "ToolMessage"
    ]
    tool_names = [t for t in tool_names if t]
    only_known = all(t in _AGENT_TOOLS for t in tool_names)
    return [
        check("no_error", not error, str(error) if error else ""),
        check(
            "produced_proposal",
            bool(generated or deleted or result.get("messages")),
            f"{len(generated)} proposed asset(s), {len(deleted)} to delete",
        ),
        check("only_known_non_writing_tools", only_known, f"tools={tool_names}"),
        check(
            "no_system_of_record_write",
            True,
            "agent has no Tazilla write path; changes are proposals applied by the client",
        ),
    ]


def summarize(checks: list[dict]) -> tuple[int, int]:
    """Return ``(passed, total)`` for a list of checks."""
    return sum(1 for c in checks if c.get("passed")), len(checks)
