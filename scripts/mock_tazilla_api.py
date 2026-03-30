#!/usr/bin/env python3
"""
Mock Tazilla Backend API for testing HTTP catalog mode.

Serves catalog data from CSV files in the exact format that
TazillaApiCatalogProvider expects.

Usage:
    python scripts/mock_tazilla_api.py [--port 9000]

Then set in .env:
    CATALOG__MODE=http
    CATALOG__ENDPOINT=http://localhost:9000
"""
import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, Header
from fastapi.responses import JSONResponse
import uvicorn

# Add project root to path so we can import catalog_loader
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CATALOG_BASE = PROJECT_ROOT / "workshops_ai" / "data" / "catalogs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mock-tazilla")

app = FastAPI(title="Mock Tazilla Backend", version="1.0.0")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Data loaders (from CSV, same as catalog_loader but return raw dicts)
# ---------------------------------------------------------------------------

def load_impacts() -> List[Dict[str, Any]]:
    rows = _read_csv(CATALOG_BASE / "Impact.csv")
    # Load categories for mapping
    cat_rows = _read_csv(CATALOG_BASE / "Impact category.csv")
    cat_map = {}
    for r in cat_rows:
        cat_map[r.get("Name", "").strip()] = (r.get("EN Name") or r.get("Name", "")).strip()

    # Load impact-to-category mapping
    mapping_rows = _read_csv(CATALOG_BASE / "Impact x Impact category.csv")
    impact_cat = {}
    for r in mapping_rows:
        impact_name = (r.get("Impact") or "").strip()
        cat_name = (r.get("Impact category") or "").strip()
        if impact_name and cat_name:
            impact_cat[impact_name] = cat_name

    impacts = []
    for r in rows:
        name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not name:
            continue
        desc = (r.get("EN Notes") or r.get("Notes") or "").strip()
        category = impact_cat.get(name, "Other")
        impacts.append({
            "name": name,
            "description": desc,
            "impactCategory": category,
        })
    return impacts


def load_threats() -> List[Dict[str, Any]]:
    rows = _read_csv(CATALOG_BASE / "Threat.csv")

    # Load categories
    cat_rows = _read_csv(CATALOG_BASE / "Threat category.csv")
    cat_map = {}
    for r in cat_rows:
        cat_map[r.get("Name", "").strip()] = (r.get("EN Name") or r.get("Name", "")).strip()

    # Load threat-to-category mapping
    mapping_rows = _read_csv(CATALOG_BASE / "Threat x Threat category.csv")
    threat_cat = {}
    for r in mapping_rows:
        threat_name = (r.get("Threat") or "").strip()
        cat_name = (r.get("Threat category") or "").strip()
        if threat_name and cat_name:
            threat_cat[threat_name] = cat_name

    # Load threat-to-asset-type mapping
    at_mapping_rows = _read_csv(CATALOG_BASE / "Asset type x Threat.csv")
    threat_asset_types: Dict[str, List[str]] = {}
    for r in at_mapping_rows:
        threat_name = (r.get("Threat") or "").strip()
        asset_type = (r.get("Asset type") or "").strip()
        if threat_name and asset_type:
            threat_asset_types.setdefault(threat_name, []).append(asset_type)

    threats = []
    for r in rows:
        name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not name:
            continue
        desc = (r.get("EN Notes") or r.get("Notes") or "").strip()
        category = threat_cat.get(name, "Other")

        # Parse boolean flags
        unintentional = (r.get("Unintentional") or "").strip().upper() in ("TRUE", "1", "YES")
        intentional = (r.get("Intentional") or "").strip().upper() in ("TRUE", "1", "YES")
        environmental = (r.get("Environmental") or "").strip().upper() in ("TRUE", "1", "YES")

        threats.append({
            "name": name,
            "description": desc,
            "category": category,
            "unintentional": unintentional,
            "intentional": intentional,
            "environmental": environmental,
            "assetTypes": threat_asset_types.get(name, []),
        })
    return threats


def load_controls() -> List[Dict[str, Any]]:
    rows = _read_csv(CATALOG_BASE / "Control.csv")

    # Load control-to-category mapping
    cat_mapping_rows = _read_csv(CATALOG_BASE / "Control x Control category.csv")
    ctrl_cat = {}
    for r in cat_mapping_rows:
        ctrl_name = (r.get("Control") or "").strip()
        cat_name = (r.get("Control category") or "").strip()
        if ctrl_name and cat_name:
            ctrl_cat[ctrl_name] = cat_name

    # Load control-to-threat coverage
    cov_rows = _read_csv(CATALOG_BASE / "Control x Threat.csv")
    ctrl_threats: Dict[str, List[Dict[str, Any]]] = {}
    for r in cov_rows:
        ctrl_name = (r.get("Control") or "").strip()
        threat_name = (r.get("Threat") or "").strip()
        value_str = (r.get("Value") or "0").strip()
        if not ctrl_name or not threat_name:
            continue
        try:
            value = int(value_str)
        except ValueError:
            continue
        if value > 0:
            ctrl_threats.setdefault(ctrl_name, []).append({
                "threatName": threat_name,
                "riskCoverage": value,
            })

    controls = []
    for r in rows:
        name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not name:
            continue
        desc = (r.get("EN Notes") or r.get("Notes") or "").strip()
        category = ctrl_cat.get(name, "Other")
        controls.append({
            "id": None,
            "name": name,
            "description": desc,
            "category": category,
            "implementationCost": None,
            "estimatedAnnualCost": None,
            "threats": ctrl_threats.get(name, []),
        })
    return controls


def load_asset_types() -> List[Dict[str, Any]]:
    rows = _read_csv(CATALOG_BASE / "Asset type.csv")
    asset_types = []
    for r in rows:
        name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not name:
            continue
        desc = (r.get("EN Notes") or r.get("Notes") or "").strip()
        is_dep = (r.get("Is dependent") or "").strip().upper() in ("TRUE", "1", "YES")
        asset_types.append({
            "name": name,
            "description": desc,
            "isDependent": is_dep,
        })
    return asset_types


def load_sectors() -> List[Dict[str, str]]:
    rows = _read_csv(CATALOG_BASE / "Sector.csv")
    sectors = []
    for r in rows:
        name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not name:
            continue
        desc = (r.get("EN Notes") or r.get("Notes") or "").strip()
        sectors.append({"name": name, "description": desc})
    return sectors


def load_categories(csv_name: str) -> List[Dict[str, str]]:
    rows = _read_csv(CATALOG_BASE / csv_name)
    categories = []
    for r in rows:
        name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not name:
            continue
        desc = (r.get("EN Notes") or r.get("Notes") or "").strip()
        categories.append({"name": name, "description": desc})
    return categories


# ---------------------------------------------------------------------------
# Endpoints — match exactly what TazillaApiCatalogProvider expects
# ---------------------------------------------------------------------------

@app.get("/catalogs/impacts")
async def get_impacts(x_session_id: str = Header(None, alias="X-Session-Id")):
    """Returns {"impacts": [...]}"""
    data = load_impacts()
    logger.info(f"[impacts] session={x_session_id} count={len(data)}")
    return {"impacts": data}


@app.get("/catalogs/threats")
async def get_threats(x_session_id: str = Header(None, alias="X-Session-Id")):
    """Returns {"threats": [...]}"""
    data = load_threats()
    logger.info(f"[threats] session={x_session_id} count={len(data)}")
    return {"threats": data}


@app.get("/catalogs/controls")
async def get_controls(x_session_id: str = Header(None, alias="X-Session-Id")):
    """Returns {"controls": [...]}"""
    data = load_controls()
    logger.info(f"[controls] session={x_session_id} count={len(data)}")
    return {"controls": data}


@app.get("/catalogs/asset-types")
async def get_asset_types(x_session_id: str = Header(None, alias="X-Session-Id")):
    """Returns {"assetTypes": [...]}"""
    data = load_asset_types()
    logger.info(f"[asset-types] session={x_session_id} count={len(data)}")
    return {"assetTypes": data}


@app.get("/catalogs/sectors")
async def get_sectors(x_session_id: str = Header(None, alias="X-Session-Id")):
    """Returns {"sectors": [...]}"""
    data = load_sectors()
    logger.info(f"[sectors] session={x_session_id} count={len(data)}")
    return {"sectors": data}


@app.get("/catalogs/impact-categories")
async def get_impact_categories(x_session_id: str = Header(None, alias="X-Session-Id")):
    """Returns {"categories": [...]}"""
    data = load_categories("Impact category.csv")
    logger.info(f"[impact-categories] session={x_session_id} count={len(data)}")
    return {"categories": data}


@app.get("/catalogs/threat-categories")
async def get_threat_categories(x_session_id: str = Header(None, alias="X-Session-Id")):
    """Returns {"categories": [...]}"""
    data = load_categories("Threat category.csv")
    logger.info(f"[threat-categories] session={x_session_id} count={len(data)}")
    return {"categories": data}


@app.get("/catalogs/control-categories")
async def get_control_categories(x_session_id: str = Header(None, alias="X-Session-Id")):
    """Returns {"categories": [...]}"""
    data = load_categories("Control category.csv")
    logger.info(f"[control-categories] session={x_session_id} count={len(data)}")
    return {"categories": data}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock Tazilla Backend API")
    parser.add_argument("--port", type=int, default=9000, help="Port (default: 9000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host (default: 0.0.0.0)")
    args = parser.parse_args()

    logger.info(f"Starting Mock Tazilla API on {args.host}:{args.port}")
    logger.info(f"Catalog data from: {CATALOG_BASE}")
    logger.info(f"Set CATALOG__MODE=http and CATALOG__ENDPOINT=http://localhost:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)
