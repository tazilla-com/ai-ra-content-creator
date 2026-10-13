"""Compare the two catalog sources against each other: `file` mode vs `http` mode.

The catalogs constrain **every** LLM call in the wizard — the model may only pick names that appear in
them — so a disagreement between the two sources is not cosmetic: it changes what the product can
produce. There are two independent implementations of the same data:

  file mode  ``workshops_ai/wizard/catalog_loader.py``  reads the CSVs in workshops_ai/data/catalogs/
  http mode  ``scripts/mock_tazilla_api.py``            re-reads the same CSVs and serves them as JSON,
                                                        standing in for Tazilla's real catalog service

Nothing in the repo checks that they agree, and the deployed configuration uses **http**. This compares
them field by field and reports where they diverge and where either side serves empty values.

    python -m harness.audit.catalog_diff                       # uses settings.catalog.endpoint
    python -m harness.audit.catalog_diff --endpoint http://mock-tazilla:9000
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

# catalog key -> (http path, file-mode loader name, the field holding the name)
CATALOGS = {
    "impacts":            ("/catalogs/impacts",            "load_impact_catalog",          "name"),
    "threats":            ("/catalogs/threats",            "load_threat_catalog",          "name"),
    "controls":           ("/catalogs/controls",           "load_control_catalog",         "name"),
    "asset-types":        ("/catalogs/asset-types",        "load_asset_type_catalog",      "name"),
    "sectors":            ("/catalogs/sectors",            "load_sector_catalog",          "name"),
    "impact-categories":  ("/catalogs/impact-categories",  "load_impact_category_catalog", "name"),
    "threat-categories":  ("/catalogs/threat-categories",  "load_threat_category_catalog", "name"),
    "control-categories": ("/catalogs/control-categories", "load_control_category_catalog", "name"),
}


def as_dicts(items):
    out = []
    for it in items or []:
        if hasattr(it, "model_dump"):
            out.append(it.model_dump(mode="json"))
        elif isinstance(it, dict):
            out.append(it)
        else:
            out.append({"name": str(it)})
    return out


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def unwrap(payload):
    """The mock wraps some catalogs in an envelope; accept either shape."""
    if isinstance(payload, dict):
        for k in ("items", "data", "results", "catalog"):
            if isinstance(payload.get(k), list):
                return payload[k]
        # single-key dict holding the list
        lists = [v for v in payload.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return payload if isinstance(payload, list) else []


def empty_field_report(rows, name_field):
    """Which fields are present but empty for every row — the signature of a bad column mapping."""
    if not rows:
        return {}
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    out = {}
    for k in sorted(keys):
        vals = [r.get(k) for r in rows]
        n_empty = sum(1 for v in vals if v in (None, "", [], {}))
        if n_empty:
            out[k] = f"{n_empty}/{len(rows)} empty"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=None, help="base URL of the http catalog service")
    ap.add_argument("--language", default="en")
    ap.add_argument("--out", default="/app/logs/audit/catalog_diff.json")
    args = ap.parse_args(argv)

    from workshops_ai.settings import get_settings
    from workshops_ai.wizard import catalog_loader

    settings = get_settings()
    endpoint = (args.endpoint or settings.catalog.endpoint or "").rstrip("/")
    print(f"file mode : workshops_ai/data/catalogs/*.csv")
    print(f"http mode : {endpoint}")
    print(f"deployed CATALOG__MODE = {settings.catalog.mode}\n")

    report, exit_bad = {}, 0
    for key, (path, loader_name, name_field) in CATALOGS.items():
        entry = {}
        try:
            loader = getattr(catalog_loader, loader_name)
            file_rows = as_dicts(loader(language=args.language))
        except Exception as exc:
            file_rows = []
            entry["file_error"] = f"{type(exc).__name__}: {exc}"
        try:
            http_rows = as_dicts(unwrap(get_json(f"{endpoint}{path}")))
        except Exception as exc:
            http_rows = []
            entry["http_error"] = f"{type(exc).__name__}: {exc}"

        fnames = {str(r.get(name_field, "")).strip() for r in file_rows if r.get(name_field)}
        hnames = {str(r.get(name_field, "")).strip() for r in http_rows if r.get(name_field)}
        only_file, only_http = sorted(fnames - hnames), sorted(hnames - fnames)

        entry.update({
            "file_count": len(file_rows), "http_count": len(http_rows),
            "names_only_in_file": only_file, "names_only_in_http": only_http,
            "name_overlap": len(fnames & hnames),
            "file_empty_fields": empty_field_report(file_rows, name_field),
            "http_empty_fields": empty_field_report(http_rows, name_field),
        })
        report[key] = entry

        agree = not only_file and not only_http and file_rows and http_rows
        flag = "OK  " if agree else "DIFF"
        if not agree:
            exit_bad = 1
        print(f"{flag} {key:20s} file={len(file_rows):4d}  http={len(http_rows):4d}  "
              f"overlap={len(fnames & hnames):4d}  only-file={len(only_file):3d}  only-http={len(only_http):3d}")
        for side in ("file", "http"):
            for k, v in (entry[f"{side}_empty_fields"] or {}).items():
                if v.startswith(str(entry[f"{side}_count"])) and entry[f"{side}_count"]:
                    print(f"        {side}: field {k!r} is {v}  <-- empty for EVERY row")
        if only_file[:3] or only_http[:3]:
            if only_file[:3]:
                print(f"        only in file: {only_file[:3]}")
            if only_http[:3]:
                print(f"        only in http: {only_http[:3]}")

    out = Path(args.out)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=str))
        print(f"\nwrote {out}")
    except OSError as exc:
        print(f"\n(could not write {out}: {exc})")
    return exit_bad


if __name__ == "__main__":
    sys.exit(main())
